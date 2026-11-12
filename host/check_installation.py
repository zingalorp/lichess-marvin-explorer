"""Validate the bundled runtime and actual model before registering the host."""
from pathlib import Path
import math
import sys
import time

from explorer import DEFAULTS, Explorer


def main():
    import chess
    import numpy
    import onnxruntime

    root = Path(__file__).resolve().parents[1]
    config = dict(DEFAULTS, weightsDir=str(root / "weights"), device="cpu")
    for name in config["models"]:
        for suffix in (".onnx", ".offset.npz"):
            if not (root / "weights" / f"marvin_{name}{suffix}").is_file():
                raise FileNotFoundError(f"weights/marvin_{name}{suffix} is missing. "
                                        "Extract the entire ZIP, including the weights folder.")
    explorer = Explorer(config=config)
    for name in config["models"]:
        check_model(explorer, name)
    print(f"Bundled Python {sys.version.split()[0]}, NumPy {numpy.__version__}, ONNX Runtime {onnxruntime.__version__}.", flush=True)


def check_model(explorer, name):
    import chess
    started = time.perf_counter()
    result = explorer.analyse(
        {"initialFen": chess.STARTING_FEN, "moves": [], "fen": chess.STARTING_FEN},
        {"rating": 1800, "timeControl": "600+0", "model": name},
    )
    legal = {move.uci() for move in chess.Board().legal_moves}
    if {move["uci"] for move in result["moves"]} != legal:
        raise RuntimeError("Model did not return every legal move.")
    if not math.isclose(sum(move["probability"] for move in result["moves"]), 1, abs_tol=1e-7):
        raise RuntimeError("Model returned an invalid policy distribution.")
    for wdl in [result["wdl"]] + [move["wdl"] for move in result["moves"]]:
        if any(not math.isfinite(v) or v < 0 for v in wdl.values()) or not math.isclose(sum(wdl.values()), 1, abs_tol=1e-7):
            raise RuntimeError("Model returned an invalid WDL distribution.")
    for outcome in ("white", "draws", "black"):
        expected = math.fsum(move["probability"] * move["wdl"][outcome] for move in result["moves"])
        if not math.isclose(result["wdl"][outcome], expected, abs_tol=1e-7):
            raise RuntimeError("Model summary is not weighted by the full policy.")
    for move in result["moves"]:
        timing = move["time"]
        if len(timing["bins"]) != 40 or not math.isclose(sum(timing["bins"]), 1, abs_tol=1e-6):
            raise RuntimeError("Model returned an invalid time distribution.")
        if not all(0 <= timing[k] <= 120 for k in ("modeSeconds", "typicalSeconds", "meanSeconds")):
            raise RuntimeError("Model returned invalid move-time statistics.")
        if not timing["subsecond"]:
            raise RuntimeError(f"The {name} sub-second think-time head did not load.")
    print(f"Model check passed for {name}: {len(legal)} legal moves, sub-second think times, CPU, "
          f"{time.perf_counter() - started:.1f}s.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Model check failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
