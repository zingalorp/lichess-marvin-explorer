import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'host'))
import chess
import numpy as np
from explorer import time_summary
from marvin.inference import HumanModel
from marvin.timing import BAR_EDGES, TimeStats, time_stats


def whole_seconds(entries):
    p = np.full(121, 1e-30)
    for second, probability in entries:
        p[second] = probability
    return np.log(p)


class MoveTimeTests(unittest.TestCase):
    def test_fixed_log_bars_and_statistics(self):
        self.assertEqual(len(BAR_EDGES), 41)
        self.assertEqual((BAR_EDGES[0], round(BAR_EDGES[-1], 9)), (0.0, 120.0))
        self.assertTrue(np.all(np.diff(BAR_EDGES) > 0))
        for second in (0, 2, 12, 75, 120):
            result = time_summary(time_stats(whole_seconds([(second, 1)])), subsecond=False)
            self.assertEqual(len(result['bins']), 40)
            self.assertAlmostEqual(sum(result['bins']), 1)
            # Without a sub-second head, a whole second is spread evenly over +-0.5 s.
            self.assertLess(abs(result['meanSeconds'] - second), 0.5)
            self.assertLess(abs(result['modeSeconds'] - second), 0.75)
            self.assertLess(abs(result['typicalSeconds'] - second), 0.5)
            self.assertFalse(result['subsecond'])

    def test_long_tail_and_overflow_keep_all_probability(self):
        stats = time_stats(whole_seconds([(2, .8), (3, .195), (100, .004), (120, .001)]))
        result = time_summary(stats, subsecond=False)
        self.assertAlmostEqual(sum(result['bins']), 1)
        self.assertAlmostEqual(sum(result['bins'][-3:]), .005, places=3)  # 100 s and 120 s+ are in the last bars
        self.assertGreater(result['meanSeconds'], 2.6)
        self.assertLess(result['typicalSeconds'], result['meanSeconds'])

    def test_invalid_outputs_are_rejected(self):
        good = time_stats(whole_seconds([(2, 1)]))
        for bad in [TimeStats(good.bins[:39], good.mode, good.typical, good.mean),
                    TimeStats(good.bins * 2, good.mode, good.typical, good.mean),
                    TimeStats(np.where(np.arange(40) == 0, np.nan, good.bins), good.mode, good.typical, good.mean),
                    TimeStats(good.bins, -1.0, good.typical, good.mean),
                    TimeStats(good.bins, good.mode, good.typical, float('inf'))]:
            with self.assertRaises(ValueError):
                time_summary(bad, subsecond=False)

    def test_move_conditioning_black_mirroring_and_no_clock_clipping(self):
        for colour, uci in [(chess.WHITE, 'e2e4'), (chess.BLACK, 'e7e5')]:
            board = chess.Board(); board.turn = colour
            baseline = np.zeros(121)
            from_head, to_head = np.zeros((64, 121)), np.zeros((64, 121))
            from_head[chess.E2, 25] = 10
            to_head[chess.E4, 25] = 10
            prediction = HumanModel._decode_single(
                board, np.zeros(4096), np.zeros(3), np.array(0), np.array(0),
                baseline, from_head, to_head, np.zeros((8, 4)),
                active_clock_s=1, tc_base_s=15)
            actual = prediction.move_time_probabilities(chess.Move.from_uci(uci))
            self.assertEqual(int(actual.argmax()), 25)
            self.assertGreater(prediction.move_time_stats(chess.Move.from_uci(uci)).mean, 24.5)
            other = prediction.move_time_probabilities(chess.Move.from_uci('d2d4' if colour else 'd7d5'))
            np.testing.assert_allclose(other, np.ones(121) / 121)

    def test_promotion_choices_use_the_same_from_to_time_heads(self):
        board = chess.Board('8/P6k/8/8/8/8/7p/K7 w - - 0 1')
        prediction = HumanModel._decode_single(
            board, np.zeros(4096), np.zeros(3), np.array(0), np.array(0),
            np.arange(121) * -.1, np.zeros((64, 121)), np.zeros((64, 121)),
            np.zeros((8, 4)), active_clock_s=30, tc_base_s=60)
        expected = prediction.move_time_probabilities(chess.Move.from_uci('a7a8q'))
        for promo in 'rbn':
            np.testing.assert_allclose(prediction.move_time_probabilities(chess.Move.from_uci('a7a8' + promo)), expected)


@unittest.skipUnless((ROOT / 'weights' / 'marvin_23m.offset.npz').is_file(), 'needs weights/marvin_23m.*')
class SubsecondHeadTests(unittest.TestCase):
    def test_real_weights_use_the_sub_second_head(self):
        model = HumanModel(ROOT / 'weights' / 'marvin_23m.onnx', device='cpu')
        self.assertIsNotNone(model.offset_head)
        board = chess.Board()
        for uci in ('e2e4', 'd7d5', 'e4d5'):
            board.push_uci(uci)
        prediction = model.predict(board=board, active_elo=1500, opponent_elo=1500, active_clock_s=55,
                                   opponent_clock_s=56, tc_base_s=60, moves_uci=['e2e4', 'd7d5', 'e4d5'],
                                   time_history_s=[1, .5, 1, 0, 0, 0, 0, 0])
        refined = prediction.move_time_stats(chess.Move.from_uci('d8d5'))
        # Black's d8d5 is d1d4 on the canonical (White-to-move) board the heads use.
        logits = prediction.time_baseline + prediction.time_from[chess.D1] + prediction.time_to[chess.D4]
        plain = time_stats(logits)
        self.assertAlmostEqual(refined.bins.sum(), 1)
        self.assertLess(abs(refined.mean - plain.mean), 1.0)
        self.assertFalse(np.allclose(refined.bins, plain.bins))


if __name__ == '__main__': unittest.main()
