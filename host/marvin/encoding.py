"""Input encoding for the human ONNX model.

Every byte of this file exists to produce the 7 input tensors the model expects,
encoded *exactly* as during training. The semantics are copied verbatim from the
reference inference code (which itself mirrors the training-time `process_pgn_v2`
+ `dataset.py`). Do not "clean up" the normalization constants or the
canonical/real split — they are load-bearing.

The 7 model inputs (all with a leading batch dim of 1):

    board_history  (1, 8, 64) int64  — 8 CANONICAL boards, newest-first
    time_history   (1, 8)      f32    — raw seconds per ply, newest-first
    rep_flags      (1, 8)      f32    — threefold-style repetition flags
    castling       (1, 4)      f32    — [WK, WQ, BK, BQ] from the REAL board
    ep_mask        (1, 64)     f32    — en-passant target one-hot, REAL board
    scalars        (1, 9)      f32    — see layout below
    legal_mask     (1, 4096)   bool   — CANONICAL from*64+to legality

Key subtlety (matches training): board_history and legal_mask are built from the
*canonical* board (always white-to-move; the board is mirrored when black is to
move), but castling / ep / the scalar clocks come from the *real* board.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

import numpy as np
import chess


HISTORY_LEN = 8
NUM_SQUARES = 64
NUM_POLICY_OUTPUTS = 4096

# Piece -> token id. Black pieces are +6. 0 = empty. (matches process_pgn_v2)
PIECE_MAP = {
    chess.PAWN: 1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.ROOK: 4,
    chess.QUEEN: 5,
    chess.KING: 6,
}

def canonicalize(board: chess.Board) -> chess.Board:
    """White-to-move view: mirror the board when black is to move."""
    return board if board.turn == chess.WHITE else board.mirror()


def encode_board(board: chess.Board) -> List[int]:
    tokens = [0] * NUM_SQUARES
    for square in range(NUM_SQUARES):
        piece = board.piece_at(square)
        if piece is None:
            continue
        val = PIECE_MAP[piece.piece_type]
        tokens[square] = val if piece.color == chess.WHITE else val + 6
    return tokens


def encode_legal_moves(board: chess.Board) -> List[Tuple[int, int, int]]:
    # promo: 0 none, 1 q, 2 r, 3 b, 4 n (matches process_pgn_v2.PROMOTION_MAP)
    promo_map = {None: 0, chess.QUEEN: 1, chess.ROOK: 2, chess.BISHOP: 3, chess.KNIGHT: 4}
    out: List[Tuple[int, int, int]] = []
    for mv in board.legal_moves:
        out.append((mv.from_square, mv.to_square, promo_map.get(mv.promotion, 0)))
    return out


def build_history_from_position(
    start_board: chess.Board,
    moves_uci: Iterable[str],
    *,
    history_len: int = HISTORY_LEN,
) -> Tuple[chess.Board, List[List[int]], List[int]]:
    """Replay moves and return (final_board, board_history, repetition_flags).

    `board_history` is `history_len` encoded CANONICAL boards, newest-first,
    zero-padded if the game is shorter than `history_len`.
    """
    board = start_board.copy(stack=False)

    hist: Deque[List[int]] = deque(maxlen=history_len)
    raw: Deque[chess.Board] = deque(maxlen=history_len)

    # Include the initial position (training records one entry per ply).
    canonical = canonicalize(board)
    hist.appendleft(encode_board(canonical))
    raw.appendleft(canonical.copy(stack=False))

    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal move in position list: {uci}")
        board.push(mv)
        canonical = canonicalize(board)
        hist.appendleft(encode_board(canonical))
        raw.appendleft(canonical.copy(stack=False))

    board_history = list(hist)
    while len(board_history) < history_len:
        board_history.append([0] * NUM_SQUARES)
    board_history = board_history[:history_len]

    repetition = _repetition_flags(list(raw), history_len=history_len)
    return board, board_history, repetition


def _repetition_flags(history: List[chess.Board], *, history_len: int) -> List[int]:
    seen: dict[str, int] = {}
    flags: List[int] = []
    for b in history:
        key = b.board_fen()
        seen[key] = seen.get(key, 0) + 1
        flags.append(1 if seen[key] >= 2 else 0)
    while len(flags) < history_len:
        flags.insert(0, 0)
    return flags[:history_len]


@dataclass
class ContextOptions:
    """Per-position conditioning fed to the model alongside the board."""
    active_elo: int = 1900
    opponent_elo: int = 1900
    active_clock_s: float = 300.0
    opponent_clock_s: float = 300.0
    active_inc_s: float = 0.0
    opponent_inc_s: float = 0.0
    # Game base time (seconds) from the TimeControl. Training uses the *initial*
    # base time, not the remaining clock.
    tc_base_s: float | None = None
    halfmove_clock: int = 0


def whole_seconds(t: float) -> float:
    """Round a time to whole seconds, half up, as Lichess game records do."""
    return float(np.floor(max(0.0, float(t)) + 0.5))


def make_model_batch(
    *,
    board: chess.Board,
    board_history: List[List[int]],
    repetition_flags: List[int],
    ctx: ContextOptions,
    time_history_s: Optional[List[float]] = None,
) -> dict[str, np.ndarray]:
    """Build the exact numpy feed dict expected by the ONNX session.

    Each array has a leading batch dimension of 1.
    """
    if time_history_s is None:
        time_history_s = [0.0] * HISTORY_LEN
    if len(time_history_s) != HISTORY_LEN:
        raise ValueError(f"time_history_s must have len {HISTORY_LEN}")

    # Legal mask from the canonical (white-to-move) board.
    canonical = canonicalize(board)
    legal_mask = np.zeros(NUM_POLICY_OUTPUTS, dtype=np.bool_)
    for from_sq, to_sq, _promo in encode_legal_moves(canonical):
        legal_mask[from_sq * 64 + to_sq] = True

    # --- Scalars (must match dataset.py normalization) ---
    active_elo_norm = (ctx.active_elo - 1900) / 700.0
    opp_elo_norm = (ctx.opponent_elo - 1900) / 700.0
    # Training move_ply is zero-based: 0, 1, 2, ... for both colours.
    ply_norm = board.ply() / 100.0

    # Clocks and increments flow as RAW SECONDS (the model's time encoder
    # featurizes them). Training retains clocks above base after increments.
    # Lichess records clocks in whole seconds (rounded half up), so training only
    # saw whole seconds: round clocks and move times the same way.
    active_clock_s = whole_seconds(ctx.active_clock_s)
    opp_clock_s = whole_seconds(ctx.opponent_clock_s)

    clamped_active_inc = float(ctx.active_inc_s)
    clamped_opp_inc = float(ctx.opponent_inc_s)
    hmc_norm = float(ctx.halfmove_clock) / 100.0

    if ctx.tc_base_s is not None and ctx.tc_base_s > 0:
        base_time_s = float(ctx.tc_base_s)
    else:
        # Reconstruct an in-distribution base time (matches the tc_cat formula).
        base_time_s = float(ctx.active_clock_s) + 40.0 * float(clamped_active_inc)
    base_time_s = max(0.0, base_time_s)

    # Layout MUST stay in sync with dataset.py _sample_to_tensors:
    #   0 elo  1 opp_elo  2 ply
    #   3 active_clock_s  4 opp_clock_s  5 active_inc_s  6 opp_inc_s  (RAW SECONDS)
    #   7 hmc  8 base_time_s  (RAW SECONDS)
    scalars_list = [
        active_elo_norm,
        opp_elo_norm,
        ply_norm,
        active_clock_s,
        opp_clock_s,
        float(clamped_active_inc),
        float(clamped_opp_inc),
        hmc_norm,
        base_time_s,
    ]

    # Castling / ep come from the REAL board, not the canonical one.
    castling_list = [
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
    ]

    ep_mask_arr = np.zeros(64, dtype=np.float32)
    if board.ep_square is not None:
        ep_mask_arr[int(board.ep_square)] = 1.0

    batch: dict[str, np.ndarray] = {
        "board_history": np.array(board_history, dtype=np.int64),
        "time_history": np.array([whole_seconds(t) for t in time_history_s], dtype=np.float32),
        "rep_flags": np.array(repetition_flags, dtype=np.float32),
        "castling": np.array(castling_list, dtype=np.float32),
        "ep_mask": ep_mask_arr,
        "scalars": np.array(scalars_list, dtype=np.float32),
        "legal_mask": legal_mask,
    }
    return {k: np.expand_dims(v, 0) for k, v in batch.items()}
