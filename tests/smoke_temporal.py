"""Opt-in real-model temporal/navigation audit. Pass the ONNX weights path."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
import chess
import numpy as np
from explorer import Explorer, DEFAULTS

# Pass weights/marvin_<size>.onnx; its folder and model name select the network.
weights = Path(sys.argv[1]).resolve()
engine = Explorer(dict(DEFAULTS) | {'weightsDir': str(weights.parent),
                                     'defaultModel': weights.stem.removeprefix('marvin_')})
def position(moves=(), initial=chess.STARTING_FEN):
    b=chess.Board(initial)
    for uci in moves: b.push_uci(uci)
    return dict(initialFen=initial,moves=list(moves),fen=b.fen())

report=[]
for control in DEFAULTS['timeControls']:
    filters={'rating':1800,'timeControl':control}
    parent=engine.analyse(position(),filters)
    child=engine.analyse(position(['e2e4']),filters)
    row=next(m for m in parent['moves'] if m['uci']=='e2e4')
    # FP16 CUDA kernels can differ slightly between batch sizes.
    np.testing.assert_allclose(list(row['wdl'].values()),list(child['wdl'].values()),atol=2e-4)
    for result in [parent,child]:
        assert abs(sum(m['probability'] for m in result['moves'])-1) < 1e-9
        assert all(abs(sum(m['wdl'].values())-1)<1e-9 for m in result['moves'])
    report.append({'control':control,'start':parent['wdl'],'afterE4':child['wdl'],
                   'context':child['context']})

white=chess.Board('7k/8/6K1/8/8/8/8/1Q6 w - - 0 1')
for board,colour in [(white,'white'),(white.mirror(),'black')]:
    result=engine.analyse(position(initial=board.fen()),{'rating':1800,'timeControl':'600+0'})
    assert result['wdl'][colour] > .8, (colour,result['wdl'])
    report.append({'queenAdvantage':colour,'wdl':result['wdl']})
print(json.dumps(report,indent=2))
