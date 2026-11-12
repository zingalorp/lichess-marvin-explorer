"""Opt-in real-model test through native framing (requires installed dependencies)."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'host'))
import chess
from native_host import read_message, write_message

process = subprocess.Popen([sys.executable, str(ROOT / 'host/native_host.py')],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
deadline = threading.Timer(120, process.kill)
deadline.start()
try:
    write_message(process.stdin, {'id': 'info', 'type': 'info'})
    info = read_message(process.stdout)
    assert info['ok'] and info['result']['ready'], info
    print('Native host handshake passed.', flush=True)
    for i, moves in enumerate([[], ['e2e4'], []]):
        board = chess.Board()
        for move in moves:
            board.push_uci(move)
        start = time.perf_counter()
        write_message(process.stdin, {'id': str(i), 'client': 'smoke', 'type': 'analyse',
                                     'position': {'initialFen': chess.STARTING_FEN, 'moves': moves, 'fen': board.fen()},
                                     'filters': {'rating': 1800, 'timeControl': '600+0'}})
        # Policy rows arrive first, then WDL after each batch, then the final reply.
        first_rows, partials = None, 0
        while (reply := read_message(process.stdout)).get('partial'):
            assert reply['id'] == str(i) and reply['ok'], reply
            assert {m['uci'] for m in reply['result']['moves']} == {m.uci() for m in board.legal_moves}
            first_rows = first_rows or time.perf_counter() - start
            partials += 1
        assert reply['ok'], reply
        result = reply['result']
        assert result['fen'] == board.fen()
        assert {m['uci'] for m in result['moves']} == {m.uci() for m in board.legal_moves}
        assert math.isclose(sum(m['probability'] for m in result['moves']), 1, abs_tol=1e-7)
        for wdl in [result['wdl']] + [m['wdl'] for m in result['moves']]:
            assert math.isclose(sum(wdl.values()), 1, abs_tol=1e-7)
        for outcome in ('white', 'draws', 'black'):
            assert math.isclose(result['wdl'][outcome],
                                math.fsum(m['probability'] * m['wdl'][outcome] for m in result['moves']),
                                abs_tol=1e-7)
        print(json.dumps({'position': ' '.join(moves) or 'start', 'seconds': round(time.perf_counter()-start, 3),
                          'firstRowsSeconds': first_rows and round(first_rows, 3), 'partials': partials,
                          'device': result['context']['device'], 'legalMoves': len(result['moves']),
                          'top': [(m['san'], round(m['probability'], 4)) for m in result['moves'][:3]]}), flush=True)
finally:
    deadline.cancel()
    process.stdin.close()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
