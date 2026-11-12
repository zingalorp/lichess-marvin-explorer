import io
import json
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
import tempfile
import chess
import numpy as np
from explorer import DEFAULTS, Cancelled, Explorer, parse_position, white_wdl, time_summary
from marvin.timing import time_stats
from native_host import read_message, write_message, MAX_MESSAGE


def whole_second_stats(probabilities):
    """Time statistics for a whole-second distribution (no sub-second head)."""
    p = np.asarray(probabilities + [0.] * (121 - len(probabilities)), dtype=np.float64)
    return time_stats(np.log(np.maximum(p, 1e-30)))


def position(moves=(), initial=chess.STARTING_FEN):
    board = chess.Board(initial)
    for move in moves:
        board.push_uci(move)
    return {'initialFen': initial, 'moves': list(moves), 'fen': board.fen()}


class FakeModel:
    device = 'test'

    def __init__(self):
        self.calls = []
        self.batches = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        moves = list(kwargs['board'].legal_moves)
        # Equal policy exposes truncation/renormalization mistakes.
        return SimpleNamespace(moves=[(move, 1 / len(moves)) for move in moves],
                               wdl=(.6, .3, .1), resign_prob=.2, move_time_stats=lambda _: whole_second_stats([0, .25, .75]))

    def predict_many(self, specs):
        self.batches.append(specs)
        return [SimpleNamespace(wdl=(.2, .3, .5)) for _ in specs]


class ExplorerTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.explorer = Explorer(dict(DEFAULTS), self.model)
        self.filters = {'rating': 1800, 'timeControl': '600+0'}

    def test_complete_policy_and_white_perspective(self):
        result = self.explorer.analyse(position(), self.filters)
        self.assertEqual(len(result['moves']), 20)
        self.assertAlmostEqual(sum(m['probability'] for m in result['moves']), 1)
        for outcome, expected in {'white': .5, 'draws': .3, 'black': .2}.items():
            self.assertAlmostEqual(result['wdl'][outcome], expected)
        self.assertEqual(result['moves'][0]['wdl'], {'white': .5, 'draws': .3, 'black': .2})
        self.assertEqual([len(b) for b in self.model.batches], [8, 8, 4])
        for m in result['moves']:
            self.assertEqual(len(m['time']['bins']), 40)
            self.assertAlmostEqual(sum(m['time']['bins']), 1)
            self.assertAlmostEqual(m['time']['meanSeconds'], 1.75, places=6)
            self.assertTrue(1.5 <= m['time']['modeSeconds'] <= 2.5)
            self.assertLess(m['time']['typicalSeconds'], 2.5)
            self.assertFalse(m['time']['subsecond'])

    def test_black_root_and_white_child_perspectives(self):
        result = self.explorer.analyse(position(['e2e4']), self.filters)
        self.assertAlmostEqual(result['wdl']['white'], .2)
        self.assertEqual(result['moves'][0]['wdl'], {'white': .2, 'draws': .3, 'black': .5})

    def test_history_and_filter_cache_keys(self):
        first = position(['g1f3', 'g8f6', 'f3g1', 'f6g8'])
        self.explorer.analyse(first, self.filters)
        self.explorer.analyse(first, self.filters)
        self.assertEqual(len(self.model.calls), 1)
        self.explorer.analyse(first, self.filters | {'rating': 1900})
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(self.model.calls[0]['moves_uci'], first['moves'])

    def test_summary_weights_unequal_policy_for_both_colours(self):
        class UnequalPolicyModel:
            device = 'test'

            def predict(self, **kwargs):
                moves = list(kwargs['board'].legal_moves)
                main = 'e2e4' if kwargs['board'].turn == chess.WHITE else 'e7e5'
                return SimpleNamespace(
                    moves=[(move, .8 if move.uci() == main else .2 / (len(moves)-1)) for move in moves],
                    wdl=(.33, .33, .34), resign_prob=.2,
                    move_time_stats=lambda _: whole_second_stats([1]),
                )

            def predict_many(self, specs):
                return [SimpleNamespace(wdl=(.05, .05, .9) if spec['moves_uci'][-1] in ('e2e4', 'e7e5')
                                        else (.7, .2, .1)) for spec in specs]

        for moves, expected in [
            ([], {'white': .74, 'draws': .08, 'black': .18}),
            (['e2e4'], {'white': .18, 'draws': .08, 'black': .74}),
        ]:
            with self.subTest(moves=moves):
                result = Explorer(dict(DEFAULTS), UnequalPolicyModel()).analyse(position(moves), self.filters)
                for outcome, value in expected.items():
                    self.assertAlmostEqual(result['wdl'][outcome], value)
                self.assertEqual(len(result['moves']), 20)
                self.assertAlmostEqual(sum(result['wdl'].values()), 1)

    def test_resignation_is_current_player_loss_and_cached_separately(self):
        for line, winner in [([], 'black'), (['e2e4'], 'white')]:
            result = self.explorer.analyse(position(line), self.filters)
            self.assertEqual(result['resignation']['probability'], .2)
            self.assertEqual(result['resignation']['wdl'][winner], 1)
            self.assertAlmostEqual(sum(row['probability'] for row in result['moves']), 1)
            self.assertEqual(len(result['moves'][0]['time']['bins']), 40)
            self.assertIs(self.explorer.analyse(position(line), self.filters), result)

    def test_model_choice_is_validated_defaulted_and_cached_separately(self):
        small = FakeModel()
        self.explorer.models['23m'] = small
        default = self.explorer.analyse(position(), self.filters)
        self.assertEqual(default['context']['model'], '79m')
        self.assertIs(self.explorer.analyse(position(), self.filters | {'model': '79m'}), default)
        other = self.explorer.analyse(position(), self.filters | {'model': '23m'})
        self.assertEqual(other['context']['model'], '23m')
        self.assertIsNot(other, default)
        self.assertEqual((len(self.model.calls), len(small.calls)), (1, 1))
        with self.assertRaisesRegex(ValueError, 'Unknown Marvin model'):
            self.explorer.analyse(position(), self.filters | {'model': '7b'})

    def test_info_lists_only_models_with_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            explorer = Explorer(dict(DEFAULTS, weightsDir=folder))
            self.assertEqual((explorer.info()['models'], explorer.info()['ready']), ([], False))
            (Path(folder) / 'marvin_23m.onnx').write_bytes(b'')
            self.assertEqual((explorer.info()['models'], explorer.info()['ready']), (['23m'], False))
            (Path(folder) / 'marvin_79m.onnx').write_bytes(b'')
            info = explorer.info()
            self.assertEqual((info['models'], info['defaultModel'], info['ready']), (['79m', '23m'], '79m', True))

    def test_invalid_resignation_head_is_rejected(self):
        original = self.model.predict
        def predict(**kwargs):
            prediction = original(**kwargs)
            prediction.resign_prob = float('nan')
            return prediction
        self.model.predict = predict
        with self.assertRaisesRegex(ValueError, 'resignation'):
            self.explorer.analyse(position(), self.filters)

    def test_history_mismatch_rejected(self):
        data = position(['e2e4'])
        data['fen'] = chess.STARTING_FEN
        with self.assertRaisesRegex(ValueError, 'do not match'):
            self.explorer.analyse(data, self.filters)

    def test_lichess_castling_notation_and_en_passant(self):
        data = position(['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'e1g1'])
        data['moves'][-1] = 'e1h1'
        board, _, moves = parse_position(data)
        self.assertEqual(moves[-1], 'e1g1')
        self.assertEqual(board.king(chess.WHITE), chess.G1)
        ep = position(['e2e4', 'a7a6', 'e4e5', 'd7d5', 'e5d6'])
        self.assertEqual(parse_position(ep)[0].fen(), ep['fen'])

    def test_all_four_promotions(self):
        result = self.explorer.analyse(position(initial='8/P6k/8/8/8/8/7p/K7 w - - 0 1'), self.filters)
        promotions = {m['uci'] for m in result['moves'] if m['uci'].startswith('a7a8')}
        self.assertEqual(promotions, {'a7a8q', 'a7a8r', 'a7a8b', 'a7a8n'})

    def test_terminal_root_and_mating_child(self):
        result = self.explorer.analyse(position(['f2f3', 'e7e5', 'g2g4', 'd8h4']), self.filters)
        self.assertEqual(result['terminal'], 'checkmate')
        self.assertEqual(result['wdl']['black'], 1)
        self.assertEqual(self.model.calls, [])
        result = self.explorer.analyse(position(['f2f3', 'e7e5', 'g2g4']), self.filters)
        mate = next(m for m in result['moves'] if m['uci'] == 'd8h4')
        self.assertEqual(mate['wdl'], {'white': 0, 'draws': 0, 'black': 1})
        # One exact black win plus the remaining White-to-move child predictions.
        self.assertAlmostEqual(result['wdl']['black'], (1 + .5*(len(result['moves'])-1)) / len(result['moves']))

    def test_stalemate_is_exact_draw(self):
        result = self.explorer.analyse(position(initial='7k/5K2/6Q1/8/8/8/8/8 b - - 0 1'), self.filters)
        self.assertEqual(result['wdl'], {'white': 0, 'draws': 1, 'black': 0})

    def test_filters_and_numbers_rejected(self):
        for filters in [self.filters | {'rating': 2600}, self.filters | {'rating': True},
                        self.filters | {'timeControl': '86400+0'}]:
            with self.assertRaises(ValueError): self.explorer.analyse(position(), filters)
        with self.assertRaises(ValueError): white_wdl((float('nan'), .2, .8), chess.WHITE)

    def test_child_clocks_and_history(self):
        self.explorer.analyse(position(['e2e4']), self.filters)
        child = self.model.batches[0][0]
        self.assertEqual(child['time_history_s'][2:], [0]*6)
        self.assertGreaterEqual(child['active_clock_s'], 0)
        self.assertGreaterEqual(child['opponent_clock_s'], 0)
        self.assertEqual(len(child['moves_uci']), 2)
        child_position = position(child['moves_uci'])
        self.explorer.analyse(child_position, self.filters)
        navigated = self.model.calls[-1]
        for field in ['active_clock_s', 'opponent_clock_s', 'time_history_s']:
            self.assertEqual(child[field], navigated[field])

    def test_no_phantom_temporal_history(self):
        self.explorer.analyse(position(), self.filters)
        root = self.model.calls[0]
        self.assertEqual(root['time_history_s'], [0]*8)
        self.assertEqual(root['active_clock_s'], 600)
        self.assertEqual(root['opponent_clock_s'], 600)
        self.assertEqual(self.model.batches[0][0]['time_history_s'][1:], [0]*7)

    def test_all_temporal_presets_have_population_data(self):
        for tc in DEFAULTS['timeControls']:
            base, inc = map(int, tc.split('+'))
            for rating in [1200, 1800, 2500]:
                for line in [[], ['e2e4'], ['e2e4','e7e5']]:
                    data = position(line)
                    board, initial, moves = parse_position(data)
                    clocks, timing = self.explorer.temporal.context(board, initial, moves, rating, base, inc)
                    self.assertEqual(len(timing), 8)
                    self.assertTrue(all(t >= 0 for t in timing))
                    self.assertTrue(all(c >= 0 for c in clocks.values()))

    def test_policy_is_reported_before_wdl(self):
        updates = []
        def progress(partial):
            # Snapshot now: the host serialises each update when it is sent.
            updates.append(json.loads(json.dumps(partial)))
            updates[-1]['batches'] = len(self.model.batches)
        result = self.explorer.analyse(position(), self.filters, progress=progress)
        # 20 children, batches of 8: before any batch, after the first and second.
        self.assertEqual([u['evaluated'] for u in updates], [0, 8, 16])
        self.assertEqual([u['batches'] for u in updates], [0, 1, 2])
        first = updates[0]
        self.assertTrue(first['partial'])
        self.assertIsNone(first['wdl'])
        self.assertEqual([m['uci'] for m in first['moves']], [m['uci'] for m in result['moves']])
        self.assertEqual([m['probability'] for m in first['moves']], [m['probability'] for m in result['moves']])
        self.assertTrue(all(m['wdl'] is None for m in first['moves']))
        self.assertTrue(all(m['time'] is None for m in first['moves']), 'think time is deferred too')
        self.assertEqual(sum(m['wdl'] is not None for m in updates[1]['moves']), 8)
        self.assertEqual([m['time'] is not None for m in updates[1]['moves']], [True] * 8 + [False] * 12)
        self.assertTrue(all(m['time'] for m in result['moves']))
        self.assertNotIn('partial', result)
        self.assertTrue(all(m['wdl'] for m in result['moves']))
        # Cached results are complete; no progress is needed.
        again = []
        self.assertIs(self.explorer.analyse(position(), self.filters, progress=again.append), result)
        self.assertEqual(again, [])

    def test_terminal_children_need_no_progress_rows(self):
        updates = []
        # Every child of this position is evaluated by the model except mate.
        self.explorer.analyse(position(['f2f3', 'e7e5', 'g2g4']), self.filters,
                              progress=lambda partial: updates.append(json.loads(json.dumps(partial))))
        mate = next(m for m in updates[0]['moves'] if m['uci'] == 'd8h4')
        self.assertEqual(mate['wdl'], {'white': 0.0, 'draws': 0.0, 'black': 1.0})
        self.assertIsNone(mate['time'])

    def test_cancelled_between_batches_and_not_cached(self):
        calls = []
        def cancelled():
            calls.append(len(self.model.batches))
            return len(self.model.batches) >= 1
        with self.assertRaises(Cancelled):
            self.explorer.analyse(position(), self.filters, cancelled=cancelled)
        self.assertEqual(len(self.model.batches), 1)
        self.assertEqual(len(self.explorer.cache), 0)
        result = self.explorer.analyse(position(), self.filters)
        self.assertTrue(all(m['wdl'] for m in result['moves']))


class ProtocolTests(unittest.TestCase):
    def test_round_trip_and_eof(self):
        stream = io.BytesIO()
        message = {'id': 'test', 'result': '½-½'}
        write_message(stream, message)
        stream.seek(0)
        self.assertEqual(read_message(stream), message)
        self.assertIsNone(read_message(stream))

    def test_invalid_frames(self):
        for frame in [struct.pack('<I', MAX_MESSAGE+1), b'\x01', struct.pack('<I', 8)+b'{}', struct.pack('<I', 2)+b'[]']:
            with self.assertRaises((ValueError, EOFError)): read_message(io.BytesIO(frame))


if __name__ == '__main__':
    unittest.main()
