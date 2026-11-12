"""Development-only UI harness using real inference; never loaded by the extension."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'host'))
import chess
from explorer import Explorer

engine = Explorer()
lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, mime='application/json'):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        if self.path.startswith('/analysis') or self.path.startswith('/study/') or self.path.split('?')[0] in ['/abcd1234', '/abcd1234/white', '/abcd1234/black/analysis']:
            file = ROOT / 'tests/harness.html'
        elif self.path in ['/tests/harness.js', '/tests/storage-shim.js', '/extension/content.js', '/extension/page.js', '/extension/explorer.css', '/extension/popup.html', '/extension/popup.js', '/extension/popup.css']:
            file = ROOT / self.path.lstrip('/')
        else:
            return self.send(404, {})
        mime = {'.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css'}[file.suffix]
        body = file.read_bytes()
        if self.path == '/extension/popup.html':
            body = body.replace(b'<script src="popup.js"', b'<script src="/tests/storage-shim.js"></script><script src="popup.js"')
        self.send(200, body, mime)

    def do_POST(self):
        if self.headers.get('Origin') != f'http://127.0.0.1:{self.server.server_port}':
            return self.send(403, {})
        if self.path != '/api':
            return self.send(404, {})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 256*1024:
                return self.send(413, {})
            data = json.loads(self.rfile.read(length))
            kind = data.get('type')
            if kind == 'info':
                result = engine.info()
            elif kind == 'cancel':
                result = {}
            elif kind == 'position':
                board = chess.Board(data.get('initialFen', chess.STARTING_FEN))
                nodes = [{'fen': board.fen()}]
                for uci in data.get('moves', []):
                    move = board.parse_uci(uci)
                    san = board.san(move)
                    board.push(move)
                    nodes.append({'fen': board.fen(), 'uci': uci, 'san': san})
                result = {'nodes': nodes, 'pieces': {chess.square_name(s): p.unicode_symbol() for s, p in board.piece_map().items()}}
            elif kind == 'analyse':
                with lock:
                    result = engine.analyse(data['position'], data['filters'])
            else:
                raise ValueError('Unknown test message')
            self.send(200, {'ok': True, 'result': result})
        except Exception as exc:
            self.send(200, {'ok': False, 'error': str(exc)})


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8766), Handler)
    print('Marvin test harness: http://127.0.0.1:8766/analysis', flush=True)
    server.serve_forever()
