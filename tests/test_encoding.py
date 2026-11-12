import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'host'))
import chess
import numpy as np
from marvin.encoding import ContextOptions, build_history_from_position, make_model_batch
from marvin.inference import HumanModel
from explorer import white_wdl


class EncodingTests(unittest.TestCase):
    def test_training_scalar_layout_and_zero_based_ply(self):
        board = chess.Board()
        for ply in range(3):
            _, hist, rep = build_history_from_position(board, [])
            ctx = ContextOptions(active_elo=1800,opponent_elo=2000,active_clock_s=302,
                                 opponent_clock_s=301,active_inc_s=1,opponent_inc_s=1,tc_base_s=300)
            batch = make_model_batch(board=board,board_history=hist,repetition_flags=rep,ctx=ctx)
            np.testing.assert_allclose(batch['scalars'][0],
                [-100/700,100/700,ply/100,302,301,1,1,0,300],rtol=1e-6)
            board.push_uci(['e2e4','e7e5','g1f3'][ply])

    def test_wdl_raw_output_order_for_both_colours(self):
        for colour in [chess.WHITE, chess.BLACK]:
            board = chess.Board(); board.turn=colour
            # The training classes are Loss, Draw, Win.
            prediction = HumanModel._decode_single(board,np.zeros(4096),
                np.log([.1,.2,.7]),np.array(0),np.array(0),np.zeros(121),
                np.zeros((64,121)),np.zeros((64,121)),np.zeros((8,4)),
                active_clock_s=300,tc_base_s=300)
            np.testing.assert_allclose(prediction.wdl,[.7,.2,.1])
            actual = white_wdl(prediction.wdl,colour)
            self.assertAlmostEqual(actual['white'], .7 if colour else .1)
            self.assertAlmostEqual(actual['black'], .1 if colour else .7)
            self.assertAlmostEqual(actual['draws'],.2)


if __name__ == '__main__': unittest.main()
