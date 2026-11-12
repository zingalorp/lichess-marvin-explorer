"""Population temporal conditioning, independent of position predictions."""
import json
from pathlib import Path


class TemporalProfile:
    def __init__(self, path=None):
        path = path or Path(__file__).with_name('temporal-profile.json')
        self.data = json.loads(Path(path).read_text(encoding='utf-8'))

    def cell(self, base, inc, rating, completed, field):
        controls = self.data['profiles'].get(f'{base}+{inc}')
        if not controls:
            raise ValueError('No temporal profile for this time control.')
        bucket = str(max(1200, min(2400, rating // 200 * 200)))
        count_field = 'count' if field == 'clock' else 'timeCount'
        minimum = self.data['minimumSamples']
        for group in [controls.get(bucket, {}), controls['0']]:
            cell = group.get(str(completed))
            if cell and cell[count_field] >= minimum and cell[field] is not None:
                return float(cell[field])
        # At sparse/late plies use the closest sufficiently sampled pooled cell.
        candidates = [(int(n), cell) for n, cell in controls['0'].items()
                      if cell[count_field] >= minimum and cell[field] is not None]
        if not candidates:
            raise ValueError('Insufficient temporal samples for this time control.')
        _, cell = min(candidates, key=lambda item: (abs(item[0]-completed), item[0]))
        return float(cell[field])

    def context(self, board, initial, moves, rating, base, inc):
        import chess
        completed = {chess.WHITE: (board.ply()+1)//2, chess.BLACK: board.ply()//2}
        clocks = {colour: (float(base) if n == 0 else self.cell(base, inc, rating, n, 'clock'))
                  for colour, n in completed.items()}
        # Only actual selected-branch history slots have temporal observations.
        # Each historical move gets a population mean, newest first.
        start_ply = chess.Board(initial).ply()
        timing = [self.cell(base, inc, rating, (start_ply+i)//2, 'time')
                  for i in range(max(0, len(moves)-8), len(moves))]
        timing.reverse()
        timing += [0.0] * (8-len(timing))
        return clocks, timing
