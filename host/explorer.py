"""Marvin opening-explorer inference. All WDL sent to the UI is White/Draw/Black."""
from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path

import chess
from temporal import TemporalProfile

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    # Models are weightsDir/marvin_<name>.onnx with marvin_<name>.offset.npz beside them.
    "weightsDir": str(ROOT / "weights"),
    "models": ["79m", "23m"],
    "defaultModel": "79m",
    "device": "auto",
    "ratingMin": 1200,
    "ratingMax": 2500,
    "ratingStep": 100,
    "rangeVerified": True,
    "timeControls": ["15+0", "60+0", "120+0", "180+0", "180+2", "300+0", "300+3", "600+0", "600+5", "900+10", "1800+0"],
}


class Cancelled(Exception):
    """A newer request for the same tab replaced the one being evaluated."""


def load_config():
    config = dict(DEFAULTS)
    path = ROOT / "host-config.json"
    if path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8-sig")))
    return config


def white_wdl(wdl, turn):
    w, d, loss = map(float, wdl)
    values = [w, d, loss] if turn == chess.WHITE else [loss, d, w]
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("Marvin returned an invalid WDL distribution.")
    total = sum(values)
    return dict(zip(("white", "draws", "black"), (v / total for v in values)))


def terminal_wdl(board):
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return None
    return {"white": float(outcome.winner == chess.WHITE),
            "draws": float(outcome.winner is None),
            "black": float(outcome.winner == chess.BLACK)}


def time_summary(stats, subsecond):
    """A move's think time for the popup, as in marvin-engine's GUI: 40 bars equally wide in
    log(1 + seconds) from 0 to 120 s (marvin.timing.BAR_EDGES), the same for every move, and the
    mode (peak on that axis), typical and mean of the full distribution."""
    bins = [float(p) for p in stats.bins]
    if len(bins) != 40 or any(not math.isfinite(p) or p < 0 for p in bins) or not math.isclose(math.fsum(bins), 1, abs_tol=1e-6):
        raise ValueError("Marvin returned an invalid time distribution.")
    values = (stats.mode, stats.typical, stats.mean)
    if any(not math.isfinite(v) or not -1e-6 <= v <= 120 + 1e-6 for v in values):
        raise ValueError("Marvin returned invalid move-time statistics.")
    mode, typical, mean = (min(max(float(v), 0.0), 120.0) for v in values)  # float rounding at the ends
    return {"bins": bins, "modeSeconds": mode, "typicalSeconds": typical,
            "meanSeconds": mean, "subsecond": bool(subsecond)}


def parse_position(position):
    if not isinstance(position, dict):
        raise ValueError("Missing analysis position.")
    initial = position.get("initialFen")
    moves = position.get("moves")
    fen = position.get("fen")
    if not isinstance(initial, str) or len(initial) > 150:
        raise ValueError("Missing starting FEN.")
    if not isinstance(fen, str) or len(fen) > 150 or len(fen.split()) != 6:
        raise ValueError("Invalid current FEN.")
    if not isinstance(moves, list) or len(moves) > 2000:
        raise ValueError("Invalid move history.")
    board = chess.Board(initial)
    if not board.is_valid():
        raise ValueError("Invalid starting position; standard chess is required.")
    canonical_moves = []
    for uci in moves:
        if not isinstance(uci, str) or len(uci) > 5:
            raise ValueError("Invalid move in history.")
        # parse_uci also normalizes Lichess's king-to-rook castling notation.
        move = board.parse_uci(uci)
        canonical_moves.append(move.uci())
        board.push(move)
    expected = chess.Board(fen)
    if not expected.is_valid() or board.fen() != expected.fen():
        raise ValueError("The analysis position and move history do not match. Refresh Lichess.")
    return board, initial, canonical_moves


class Explorer:
    def __init__(self, config=None, model=None):
        self.config = config or load_config()
        self.models = {} if model is None else {self.config["defaultModel"]: model}
        self.cache = OrderedDict()
        self.temporal = TemporalProfile()

    def model_path(self, name):
        return Path(self.config["weightsDir"]) / f"marvin_{name}.onnx"

    def available_models(self):
        return [name for name in self.config["models"]
                if name in self.models or self.model_path(name).is_file()]

    def info(self):
        models = self.available_models()
        return {key: self.config[key] for key in (
            "ratingMin", "ratingMax", "ratingStep", "rangeVerified", "timeControls", "defaultModel"
        )} | {"models": models, "ready": self.config["defaultModel"] in models, "version": "0.2.1"}

    def model(self, name):
        if name not in self.models:
            from marvin.inference import HumanModel
            path = self.model_path(name)
            if not path.is_file():
                raise ValueError(f"Model weights not found: {path}. Run Setup.cmd again.")
            self.models[name] = HumanModel(path, device=self.config["device"])
        return self.models[name]

    def filters(self, filters):
        if not isinstance(filters, dict):
            raise ValueError("Missing model filters.")
        rating = filters.get("rating")
        if isinstance(rating, bool) or not isinstance(rating, int):
            raise ValueError("Rating must be an integer.")
        c = self.config
        if not c["ratingMin"] <= rating <= c["ratingMax"] or (rating-c["ratingMin"]) % c["ratingStep"]:
            raise ValueError("Rating is outside the configured Marvin range.")
        tc = filters.get("timeControl")
        if tc not in c["timeControls"]:
            raise ValueError("Time control is not supported by this Marvin configuration.")
        base, inc = map(int, tc.split("+"))
        model = filters.get("model", c["defaultModel"])
        if model not in c["models"]:
            raise ValueError("Unknown Marvin model.")
        return rating, base, inc, model

    def analyse(self, position, filters, progress=None, cancelled=None):
        """Complete analysis, cached. With `progress`, a partial result is reported as
        soon as the policy is known (rows have "time": None and non-terminal rows
        "wdl": None, as does the summary), and again after every batch of 8 rows, in
        policy order. Think times are the slow part of the policy step, so they are
        filled in with the WDL rather than delaying the moves. `cancelled()` is polled between
        batches; it raises Cancelled so a superseded position stops early."""
        board, initial, moves = parse_position(position)
        rating, base, inc, model_name = self.filters(filters)
        key = (initial, tuple(moves), rating, base, inc, model_name)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        exact = terminal_wdl(board)
        if exact is not None:
            return {"fen": position["fen"], "moves": [], "wdl": exact,
                    "terminal": board.outcome().termination.name.replace("_", " ").lower()}
        model = self.model(model_name)
        subsecond = getattr(model, "offset_head", None) is not None
        # The same population context is used for a row's child and for that
        # position after navigation. Never choose temporal inputs to alter WDL.
        clocks, timing = self.temporal.context(board, initial, moves, rating, base, inc)

        def spec(b, history, clock_values, timing):
            return dict(board=b, moves_uci=history, initial_fen=initial,
                        active_elo=rating, opponent_elo=rating,
                        active_clock_s=clock_values[b.turn], opponent_clock_s=clock_values[not b.turn],
                        active_inc_s=inc, opponent_inc_s=inc, tc_base_s=base,
                        time_history_s=timing)

        root = model.predict(**spec(board, moves, clocks, timing))
        resignation = float(root.resign_prob)
        if not math.isfinite(resignation) or not 0 <= resignation <= 1:
            raise ValueError("Marvin returned an invalid resignation score.")
        probabilities = {m.uci(): float(p) for m, p in root.moves}
        legal = list(board.legal_moves)
        if set(probabilities) != {m.uci() for m in legal}:
            raise ValueError("Marvin did not return the complete legal policy.")
        total = sum(probabilities.values())
        if total <= 0 or any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
            raise ValueError("Marvin returned an invalid policy distribution.")
        legal.sort(key=lambda m: probabilities[m.uci()], reverse=True)
        rows, work = [], []
        for move in legal:
            child = board.copy()
            child.push(move)
            row = {"uci": move.uci(), "san": board.san(move),
                   "probability": probabilities[move.uci()] / total, "wdl": terminal_wdl(child),
                   "time": None}
            rows.append(row)
            work.append((row, move, child))
        context = {"rating": rating, "timeControl": filters["timeControl"], "model": model_name,
                   "whiteClock": round(clocks[chess.WHITE], 1),
                   "blackClock": round(clocks[chess.BLACK], 1),
                   "temporalSource": self.temporal.data['source'],
                   "device": getattr(model, "device", "test")}

        def snapshot(summary):
            return {"fen": position["fen"], "moves": rows, "wdl": summary,
                    # Separate binary head, not a legal-move softmax entry. Keep
                    # raw output cached; presentation decides whether to include it.
                    "resignation": {"probability": resignation,
                                    "wdl": white_wdl((0, 0, 1), board.turn)},
                    "context": context}

        def report(done):
            if progress is not None:
                progress(snapshot(None) | {"partial": True, "evaluated": done, "pending": len(rows)})

        # Policy first: rows are shown before their think time and child WDL.
        # Rows are sorted by policy, so the visible moves are completed first.
        report(0)
        # Bound GPU memory in high-mobility positions; evaluate EVERY legal move.
        for offset in range(0, len(work), 8):
            if cancelled is not None and cancelled():
                raise Cancelled("Superseded by a newer position.")
            chunk = work[offset:offset+8]
            for row, move, _ in chunk:
                row["time"] = time_summary(root.move_time_stats(move), subsecond)
            batch = []
            for row, move, child in chunk:
                if row["wdl"] is None:
                    child_moves = moves + [move.uci()]
                    child_clocks, child_timing = self.temporal.context(child, initial, child_moves, rating, base, inc)
                    batch.append((row, child.turn, spec(child, child_moves, child_clocks, child_timing)))
            if batch:
                predictions = model.predict_many([item[2] for item in batch])
                if len(predictions) != len(batch):
                    raise ValueError("Incomplete Marvin evaluation batch.")
                for (row, turn, _), prediction in zip(batch, predictions):
                    row["wdl"] = white_wdl(prediction.wdl, turn)
            if offset + 8 < len(work):
                report(offset + len(chunk))
        # Match the explorer's total: mix every child's White/Draw/Black WDL
        # using its full, unrounded policy probability (including hidden rows).
        summary = {outcome: math.fsum(row["probability"] * row["wdl"][outcome] for row in rows)
                   for outcome in ("white", "draws", "black")}
        summary_total = math.fsum(summary.values())
        summary = {outcome: value / summary_total for outcome, value in summary.items()}
        result = snapshot(summary)
        self.cache[key] = result
        while len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result
