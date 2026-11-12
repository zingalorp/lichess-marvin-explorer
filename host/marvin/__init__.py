"""Standalone copy of the Marvin human-model ONNX inference stack.

`encoding.py` and `inference.py` are vendored verbatim from the exploiter
project so this extension's native host runs with no dependency on any other
repo or on WSL. They need only numpy, python-chess and onnxruntime.

If you retrain the model and the input encoding changes, re-copy both files
together — the encoding and the graph's input names are a matched pair.
"""

from .encoding import ContextOptions, HISTORY_LEN, build_history_from_position
from .inference import HumanModel, Prediction, MoveTime

__all__ = [
    "ContextOptions",
    "HISTORY_LEN",
    "build_history_from_position",
    "HumanModel",
    "Prediction",
    "MoveTime",
]
