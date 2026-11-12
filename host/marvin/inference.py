"""Minimal ONNX inference wrapper for the human chess model.

`HumanModel` owns an onnxruntime session and exposes:

    raw_forward(batch)  -> the 8 raw output arrays, batch dim kept
    predict(...)        -> a decoded `Prediction` for one position
    choose_move(...)    -> sample a single (real) move from the policy

The 8 model outputs (in the order the graph defines them):

    move_logits       (B, 4096)        policy over CANONICAL from*64+to (masked)
    wdl_logits        (B, 3)           [Loss, Draw, Win] (softmax for probs)
    resign_logit      (B,)             sigmoid -> resign probability
    flag_logit        (B,)             sigmoid -> flag probability
    time_baseline     (B, 121)         position-level move-time bin logits
    time_from_logits  (B, 64, 121)     per-from-square time bias
    time_to_logits    (B, 64, 121)     per-to-square time bias
    promo_logits      (B, 8, 4)        per-file (rank-8) promotion: [q, r, b, n]

Move-time for a given move is `baseline + from[from_sq] + to[to_sq]` over 121
linear bins where bin index == seconds (top bin = ">=120s"), clamped to the
remaining clock.

Current weight files also output `time_hidden` and `square_features` and come with
`<name>.offset.npz`, a head that refines the time to 40 ms (see timing.py). Their
`flag_logit` is calibrated (ONNX metadata flag_logit=calibrated).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import chess


# --------------------------------------------------------------------------- #
# CUDA DLL discovery (Windows)
# --------------------------------------------------------------------------- #
_NVIDIA_DLLS_REGISTERED = False


def register_nvidia_dll_dirs() -> List[str]:
    """Put pip-installed NVIDIA runtime DLLs (cuDNN, cuBLAS, ...) on the loader path.

    onnxruntime-gpu's CUDA provider links against e.g. ``cudnn64_9.dll``; when cuDNN
    is provided by pip wheels (``nvidia-cudnn-cuXX``) rather than a system install,
    those DLLs sit under ``site-packages/nvidia/*/bin`` and are not on PATH, so the
    CUDA provider silently fails to load and ORT falls back to CPU. We prepend those
    dirs to PATH and register them with the loader. Idempotent; CUDA-13 runtime/cuBLAS
    still come from the system toolkit on PATH. No-op when no such wheels are present.
    """
    global _NVIDIA_DLLS_REGISTERED
    if _NVIDIA_DLLS_REGISTERED:
        return []
    import site

    roots: List[str] = []
    if hasattr(site, "getsitepackages"):
        roots.extend(site.getsitepackages())
    if hasattr(site, "getusersitepackages"):
        roots.append(site.getusersitepackages())

    added: List[str] = []
    seen = set()
    for root in roots:
        base = os.path.join(root, "nvidia")
        if not os.path.isdir(base):
            continue
        for sub in sorted(os.listdir(base)):
            for leaf in ("bin", "lib"):
                d = os.path.join(base, sub, leaf)
                if d in seen:
                    continue
                seen.add(d)
                if os.path.isdir(d) and any(f.lower().endswith(".dll") for f in os.listdir(d)):
                    added.append(d)

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for d in added:
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
    _NVIDIA_DLLS_REGISTERED = True
    return added

from .timing import OffsetHead, TimeStats, time_stats
from .encoding import (
    ContextOptions,
    HISTORY_LEN,
    build_history_from_position,
    canonicalize,
    make_model_batch,
)

# Ordered to match the ONNX graph inputs/outputs exactly.
INPUT_NAMES = (
    "board_history", "time_history", "rep_flags", "castling",
    "ep_mask", "scalars", "legal_mask",
)
OUTPUT_NAMES = (
    "move_logits", "wdl_logits", "resign_logit", "flag_logit",
    "time_baseline", "time_from_logits", "time_to_logits", "promo_logits",
)
# Inputs to the sub-second head, present in current weight files.
OFFSET_OUTPUT_NAMES = ("time_hidden", "square_features")

NUM_TIME_BINS = 121  # TIME_SECOND_CAP (120) + 1

# Promotion head column order and the chess piece each maps to.
PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMO_CHARS = ("q", "r", "b", "n")
_PROMO_PIECE_TO_IDX = {p: i for i, p in enumerate(PROMO_PIECES)}

_ONNX_TO_NP = {
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _mirror_square(sq: int) -> int:
    return sq ^ 56


def canonical_to_real_move(move: chess.Move, real_turn: chess.Color) -> chess.Move:
    """Map a canonical (white-POV) move back to the real board's orientation."""
    if real_turn == chess.WHITE:
        return move
    return chess.Move(
        _mirror_square(move.from_square),
        _mirror_square(move.to_square),
        promotion=move.promotion,
    )


def time_bin_to_seconds(bin_idx: int, active_clock_s: float,
                        time_budget_cap: Optional[float] = None) -> float:
    """Linear raw-second bins: bin index == seconds, clamped to the clock."""
    clk = active_clock_s
    if time_budget_cap is not None and time_budget_cap > 0:
        clk = min(clk, time_budget_cap)
    return float(min(max(0.0, float(bin_idx)), max(0.0, clk)))


# --------------------------------------------------------------------------- #
# sampling (temperature + nucleus / top-p)
# --------------------------------------------------------------------------- #
def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    if top_p >= 1.0:
        return probs
    if top_p <= 0.0:
        out = np.zeros_like(probs)
        out[int(np.argmax(probs))] = 1.0
        return out
    order = np.argsort(-probs)
    cdf = np.cumsum(probs[order])
    keep = cdf <= top_p
    if not np.any(keep):
        keep[0] = True
    else:
        keep[int(np.argmax(cdf > top_p))] = True
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep]] = True
    out = np.where(mask, probs, 0.0)
    s = float(out.sum())
    return out / s if s > 0 else probs


def sample_index(logits: np.ndarray, *, temperature: float = 1.0, top_p: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> Tuple[int, float]:
    """Sample one index from logits. Returns (index, probability)."""
    x = np.asarray(logits, dtype=np.float64)
    if temperature <= 0.0:
        return int(np.argmax(x)), 1.0
    if rng is None:
        rng = np.random.default_rng()
    x = x / float(temperature)
    probs = softmax(x)
    probs = _top_p_filter(probs, float(top_p))
    idx = int(rng.choice(len(probs), p=probs))
    return idx, float(probs[idx])


# --------------------------------------------------------------------------- #
# decoded result containers
# --------------------------------------------------------------------------- #
@dataclass
class MoveTime:
    """Decoded move-time distribution (seconds) for one move."""
    expected_s: float
    mode_s: float
    dist: List[Tuple[float, float]]  # [(seconds, prob), ...] over all 121 bins


@dataclass
class Prediction:
    """All decoded model outputs for a single position."""
    # Policy over real (board-oriented) legal moves, highest prob first.
    moves: List[Tuple[chess.Move, float]]
    move_logits: np.ndarray            # raw (4096,) canonical policy logits
    # Value.
    wdl: Tuple[float, float, float]    # (win, draw, loss) probabilities
    value: float                       # win + 0.5 * draw, for side to move
    # Binary heads.
    resign_prob: float
    flag_prob: float
    resign_logit: float
    flag_logit: float
    # Raw time heads (canonical squares); use `move_time()` to decode per move.
    time_baseline: np.ndarray          # (121,)
    time_from: np.ndarray              # (64, 121)
    time_to: np.ndarray                # (64, 121)
    promo_logits: np.ndarray           # (8, 4) per rank-8 file, [q, r, b, n]
    # Bookkeeping needed to decode times / map moves.
    real_turn: chess.Color = field(default=chess.WHITE)
    active_clock_s: float = 0.0
    time_budget_cap: Optional[float] = None
    # Sub-second head and its inputs; None for weights without one.
    offset_head: Optional[OffsetHead] = None
    time_hidden: Optional[np.ndarray] = None       # (128,)
    square_features: Optional[np.ndarray] = None   # (64, D), canonical squares

    def _canonical(self, move: chess.Move) -> chess.Move:
        return move if self.real_turn == chess.WHITE else chess.Move(
            _mirror_square(move.from_square), _mirror_square(move.to_square),
            promotion=move.promotion,
        )

    def move_time_stats(self, move: chess.Move) -> TimeStats:
        """Think-time histogram, mode, typical and mean for a real-board move, to 40 ms when the
        weights have a sub-second head (else whole seconds spread evenly within each second)."""
        cmv = self._canonical(move)
        logits = self.time_baseline + self.time_from[cmv.from_square] + self.time_to[cmv.to_square]
        offset = None
        if self.offset_head is not None and self.square_features is not None:
            x_from, x_to = self.square_features[cmv.from_square], self.square_features[cmv.to_square]
            offset = lambda ks: self.offset_head.log_probs(self.time_hidden, x_from, x_to, logits, ks)
        return time_stats(logits, offset)

    def move_time_probabilities(self, move: chess.Move) -> np.ndarray:
        """Raw 0..119 second probabilities plus the open-ended 120s+ bin.

        Accept real-board coordinates. Do not clip this distribution to a clock:
        it describes the model output, independently of engine sleep limits.
        """
        cmv = self._canonical(move)
        logits = self.time_baseline + self.time_from[cmv.from_square] + self.time_to[cmv.to_square]
        return softmax(logits)

    def move_time(self, move: chess.Move) -> MoveTime:
        """Legacy clock-clipped timing for callers that simulate a move."""
        probs = self.move_time_probabilities(move)
        mode_bin = int(np.argmax(probs))
        dist = [
            (time_bin_to_seconds(int(b), self.active_clock_s, self.time_budget_cap), float(p))
            for b, p in enumerate(probs)
        ]
        # Expected time = probability-weighted decoded seconds (matches the
        # reference engine's reported expected_time_s), not round-then-decode.
        expected_s = float(sum(sec * p for sec, p in dist))
        return MoveTime(
            expected_s=expected_s,
            mode_s=time_bin_to_seconds(mode_bin, self.active_clock_s, self.time_budget_cap),
            dist=dist,
        )


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class HumanModel:
    """ONNX Runtime wrapper around the human chess model."""

    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        import onnxruntime as ort

        self.weights_path = str(weights_path)
        if not Path(self.weights_path).exists():
            raise FileNotFoundError(self.weights_path)

        device = str(device).split(":")[0].lower()
        want_cuda = device in ("cuda", "gpu", "auto")
        providers: List[str] = []
        if want_cuda:
            # Make pip-installed cuDNN/cuBLAS visible before the CUDA provider loads.
            register_nvidia_dll_dirs()
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(self.weights_path, providers=providers)
        self.device = "cuda" if "CUDAExecutionProvider" in self.session.get_providers() else "cpu"
        # "auto"/"gpu" fall back to CPU silently; explicit "cuda" must get the GPU.
        if device == "cuda" and self.device != "cuda":
            raise RuntimeError(
                "CUDA requested but CUDAExecutionProvider failed to load. Ensure "
                "onnxruntime-gpu is installed and cuDNN 9 (nvidia-cudnn-cuXX) plus a "
                "matching CUDA toolkit are available. Providers actually loaded: "
                f"{self.session.get_providers()}"
            )
        self._in_types = {i.name: i.type for i in self.session.get_inputs()}
        outputs = {o.name for o in self.session.get_outputs()}
        offset_path = Path(self.weights_path).with_suffix(".offset.npz")
        self.offset_head = (OffsetHead(offset_path)
                            if offset_path.is_file() and set(OFFSET_OUTPUT_NAMES) <= outputs else None)
        self._outputs = list(OUTPUT_NAMES) + (list(OFFSET_OUTPUT_NAMES) if self.offset_head else [])
        # ONNX Runtime CUDA sessions are not safe for concurrent .run() calls.
        self._lock = threading.Lock()

    # -- low level -------------------------------------------------------- #
    def raw_forward(self, batch: dict[str, np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Run the session: the 8 outputs, plus the sub-second head's 2 inputs when available."""
        feed = {}
        for name in INPUT_NAMES:
            arr = np.asarray(batch[name])
            want = _ONNX_TO_NP.get(self._in_types.get(name, ""))
            if want is not None and arr.dtype != want:
                arr = arr.astype(want)
            feed[name] = arr
        with self._lock:
            return tuple(self.session.run(self._outputs, feed))

    # -- encoding / decoding (shared by predict + predict_many) ----------- #
    @staticmethod
    def _build_batch(
        *,
        board: chess.Board,
        active_elo: int,
        opponent_elo: int,
        active_clock_s: float,
        opponent_clock_s: float,
        active_inc_s: float = 0.0,
        opponent_inc_s: float = 0.0,
        moves_uci: Optional[Sequence[str]] = None,
        time_history_s: Optional[List[float]] = None,
        tc_base_s: Optional[float] = None,
        initial_fen: str = chess.STARTING_FEN,
    ) -> Tuple[chess.Board, dict[str, np.ndarray]]:
        """Encode one position into the (batch-dim-1) feed dict; return (board, feed)."""
        if moves_uci is not None:
            final_board, board_history, rep_flags = build_history_from_position(
                chess.Board(initial_fen), moves_uci
            )
            # Trust the caller's board if given; otherwise use the replayed one.
            board = board or final_board
        else:
            _b, board_history, rep_flags = build_history_from_position(board, [])

        ctx = ContextOptions(
            active_elo=active_elo,
            opponent_elo=opponent_elo,
            active_clock_s=active_clock_s,
            opponent_clock_s=opponent_clock_s,
            active_inc_s=active_inc_s,
            opponent_inc_s=opponent_inc_s,
            tc_base_s=tc_base_s,
            halfmove_clock=int(board.halfmove_clock),
        )
        batch = make_model_batch(
            board=board,
            board_history=board_history,
            repetition_flags=rep_flags,
            ctx=ctx,
            time_history_s=time_history_s,
        )
        return board, batch

    @staticmethod
    def _decode_single(
        board: chess.Board,
        move_logits: np.ndarray,
        wdl_logits: np.ndarray,
        resign_logit: np.ndarray,
        flag_logit: np.ndarray,
        t_baseline: np.ndarray,
        t_from: np.ndarray,
        t_to: np.ndarray,
        promo_logits: np.ndarray,
        *,
        active_clock_s: float,
        tc_base_s: Optional[float],
        offset_head: Optional[OffsetHead] = None,
        time_hidden: Optional[np.ndarray] = None,
        square_features: Optional[np.ndarray] = None,
    ) -> Prediction:
        """Decode one position's 8 raw outputs (batch dim already stripped)."""
        move_logits = np.asarray(move_logits)
        wdl = softmax(np.asarray(wdl_logits))               # [L, D, W]
        promo = np.asarray(promo_logits)
        promo_prob = softmax(promo, axis=-1)

        # Build the real-move policy from legal moves, folding in the promo head.
        real_turn = board.turn
        canonical = canonicalize(board)
        entries: List[Tuple[chess.Move, float]] = []
        for mv in canonical.legal_moves:
            logit = float(move_logits[mv.from_square * 64 + mv.to_square])
            if mv.promotion is not None and 56 <= mv.to_square <= 63:
                file_idx = mv.to_square - 56
                p_idx = _PROMO_PIECE_TO_IDX.get(mv.promotion, 0)
                logit += float(np.log(max(1e-8, float(promo_prob[file_idx, p_idx]))))
            entries.append((canonical_to_real_move(mv, real_turn), logit))

        if entries:
            probs = softmax(np.array([e[1] for e in entries], dtype=np.float64))
            moves = sorted(
                ((mv, float(p)) for (mv, _), p in zip(entries, probs)),
                key=lambda kv: kv[1], reverse=True,
            )
        else:
            moves = []

        r_logit = float(np.asarray(resign_logit).reshape(-1)[0])
        f_logit = float(np.asarray(flag_logit).reshape(-1)[0])

        return Prediction(
            moves=moves,
            move_logits=move_logits,
            wdl=(float(wdl[2]), float(wdl[1]), float(wdl[0])),
            value=float(wdl[2] + 0.5 * wdl[1]),
            resign_prob=_sigmoid(r_logit),
            flag_prob=_sigmoid(f_logit),
            resign_logit=r_logit,
            flag_logit=f_logit,
            time_baseline=np.asarray(t_baseline),
            time_from=np.asarray(t_from),
            time_to=np.asarray(t_to),
            promo_logits=promo,
            real_turn=real_turn,
            active_clock_s=active_clock_s,
            time_budget_cap=tc_base_s,
            offset_head=offset_head,
            time_hidden=time_hidden,
            square_features=square_features,
        )

    # -- high level ------------------------------------------------------- #
    def predict(
        self,
        *,
        board: chess.Board,
        active_elo: int,
        opponent_elo: int,
        active_clock_s: float,
        opponent_clock_s: float,
        active_inc_s: float = 0.0,
        opponent_inc_s: float = 0.0,
        moves_uci: Optional[Sequence[str]] = None,
        time_history_s: Optional[List[float]] = None,
        tc_base_s: Optional[float] = None,
        initial_fen: str = chess.STARTING_FEN,
    ) -> Prediction:
        """Encode one position, run the model, and decode every output.

        `moves_uci` is the game line from `initial_fen` to the current position
        (used to build the 8-ply board history). If omitted, only the current
        `board` is used as history. `time_history_s` is per-ply seconds spent,
        newest-first; omit for a zero history.
        """
        board, batch = self._build_batch(
            board=board, active_elo=active_elo, opponent_elo=opponent_elo,
            active_clock_s=active_clock_s, opponent_clock_s=opponent_clock_s,
            active_inc_s=active_inc_s, opponent_inc_s=opponent_inc_s,
            moves_uci=moves_uci, time_history_s=time_history_s,
            tc_base_s=tc_base_s, initial_fen=initial_fen,
        )
        outs = self.raw_forward(batch)
        return self._decode_single(
            board, *(np.asarray(o[0]) for o in outs[:len(OUTPUT_NAMES)]),
            active_clock_s=active_clock_s, tc_base_s=tc_base_s, **self._offset_inputs(outs, 0),
        )

    def _offset_inputs(self, outs, i: int) -> dict:
        if not self.offset_head:
            return {}
        return {"offset_head": self.offset_head, "time_hidden": np.asarray(outs[-2][i]),
                "square_features": np.asarray(outs[-1][i])}

    def predict_many(self, specs: Sequence[dict]) -> List[Prediction]:
        """Batched predict: one ONNX forward for many positions.

        Each `spec` is a kwargs dict accepted by `predict` (board, elos, clocks,
        moves_uci, initial_fen, tc_base_s, ...). Returns predictions in order.
        This is the GPU win for search: amortizes kernel launches over the batch.
        """
        if not specs:
            return []
        boards: List[chess.Board] = []
        metas: List[Tuple[float, Optional[float]]] = []
        feeds: List[dict[str, np.ndarray]] = []
        for s in specs:
            board, batch = self._build_batch(**s)
            boards.append(board)
            metas.append((s["active_clock_s"], s.get("tc_base_s")))
            feeds.append(batch)

        stacked = {
            name: np.concatenate([f[name] for f in feeds], axis=0)
            for name in INPUT_NAMES
        }
        outs = self.raw_forward(stacked)
        return [
            self._decode_single(
                boards[i], *(np.asarray(o[i]) for o in outs[:len(OUTPUT_NAMES)]),
                active_clock_s=metas[i][0], tc_base_s=metas[i][1], **self._offset_inputs(outs, i),
            )
            for i in range(len(specs))
        ]

    def choose_move(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        **predict_kwargs,
    ) -> Tuple[chess.Move, float, Prediction]:
        """Sample a single legal (real) move. Returns (move, prob, prediction)."""
        pred = self.predict(**predict_kwargs)
        if not pred.moves:
            raise ValueError("No legal moves in position")
        logits = np.log(np.array([p for _, p in pred.moves], dtype=np.float64) + 1e-12)
        idx, prob = sample_index(logits, temperature=temperature, top_p=top_p, rng=rng)
        return pred.moves[idx][0], prob, pred
