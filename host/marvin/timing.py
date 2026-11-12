"""Sub-second think-time distributions, as in marvin-engine (marvin/offset.py, marvin/engine.py).

The time head predicts whole seconds k (training clocks were rounded to the second). Current
weight files come with `<name>.offset.npz`, a small head trained on Lichess's centisecond
clocks that places the time within (k - 1 s, k + 1 s) in 40 ms steps, given the position and
the move. Keep this module identical to the engine so both show the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIME_BINS = 121  # bin k = k seconds; the last bin means 120 s or more
OFFSET_BINS = 49
CENTRES = (-96 + 4 * np.arange(OFFSET_BINS)) / 100.0  # seconds relative to k
GRID = 0.04  # seconds; the offset head's resolution
TYPICAL_TEMPERATURE = 0.5  # "typical" = mean of the sharpened time distribution
_FLOOR = -30.0

# Histograms use the same 40 bars for every move, equally wide in log(1 + t / 1 s) from 0 to
# 120 s: about 0.13 s wide at 0, 0.24 s at 1 s, 1.3 s at 10 s and 7 s at 60 s.
LOG_SCALE = 1.0
BAR_EDGES = LOG_SCALE * np.expm1(np.linspace(0.0, np.log1p((TIME_BINS - 1) / LOG_SCALE), 41))
# The mode is the peak of the same distribution on the same axis, smoothed (Gaussian, half a bar
# wide) and evaluated on a fine grid, so it is continuous rather than one of the 40 bar centres.
_U_MAX = float(np.log1p((TIME_BINS - 1) / LOG_SCALE))
_MODE_GRID = np.linspace(0.0, _U_MAX, 1201)
_MODE_SIGMA = _U_MAX / 40 / 2


def _layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _mish(x):
    return x * np.tanh(np.logaddexp(0.0, x))


def _softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / temperature
    p = np.exp(x - x.max())
    return p / p.sum()


class OffsetHead:
    def __init__(self, path):
        self.w = dict(np.load(Path(path)))

    def log_probs(self, hidden, x_from, x_to, time_logits, k):
        """log p(offset bin | k) for one move: (OFFSET_BINS,) for an int k, (len(k), OFFSET_BINS) for an array.

        `time_logits`: that move's 121 whole-second logits.
        """
        w = self.w
        ks = np.atleast_1d(np.asarray(k, dtype=np.int64))
        tl = time_logits - time_logits.max()
        lp = np.maximum(tl - np.log(np.exp(tl).sum()), _FLOOR)
        lp = np.concatenate([[_FLOOR], lp, [_FLOOR]])
        l3 = np.stack([lp[ks], lp[ks + 1], lp[ks + 2]], axis=1)
        local = np.concatenate([l3, l3[:, :1] - l3[:, 1:2], l3[:, 2:] - l3[:, 1:2]], axis=1) / 10.0
        shared = np.concatenate([
            _layer_norm(hidden, w["ln_hidden.weight"], w["ln_hidden.bias"]),
            _layer_norm(x_from, w["ln_from.weight"], w["ln_from.bias"]),
            _layer_norm(x_to, w["ln_to.weight"], w["ln_to.bias"]),
        ])
        w1 = w["mlp.0.weight"]
        per_k = np.concatenate([w["k_emb.weight"][ks], local], axis=1)
        z = _mish(w1[:, :len(shared)] @ shared + per_k @ w1[:, len(shared):].T + w["mlp.0.bias"])
        z = _mish(z @ w["mlp.3.weight"].T + w["mlp.3.bias"])
        logits = z @ w["mlp.6.weight"].T + w["mlp.6.bias"]
        logits = logits - logits.max(axis=1, keepdims=True)
        out = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return out[0] if np.ndim(k) == 0 else out


def time_distribution(time_logits, offset=None, temperature=1.0, min_prob=1e-5):
    """Think-time distribution as (seconds, probabilities) on the 40 ms grid from 0 to 120 s.

    With an offset head (`offset(ks)` -> log-probs), every k with probability >= `min_prob` is
    spread over (k - 1, k + 1) s as the head predicts. Otherwise (and for the rest) k is spread
    evenly over k +- 0.5 s. Times below 0 count as 0; 120 means 120 s or more. `temperature`
    sharpens the whole-second part only.
    """
    pk = _softmax(time_logits, temperature)
    per_second = round(1 / GRID)
    p = np.zeros(per_second * (TIME_BINS - 1) + 1)
    ks = np.arange(len(pk))
    head = (pk >= min_prob) & (ks < TIME_BINS - 1) if offset is not None else np.zeros(len(pk), bool)
    spread = np.arange(-(per_second // 2), per_second // 2 + 1)  # 25 points = one second
    idx = per_second * ks[~head, None] + spread[None, :]
    np.add.at(p, np.clip(idx, 0, len(p) - 1), np.repeat(pk[~head, None] / len(spread), len(spread), 1))
    if head.any():
        po = np.exp(offset(ks[head]).astype(np.float64))
        po /= po.sum(axis=1, keepdims=True)
        idx = per_second * ks[head, None] + np.arange(OFFSET_BINS)[None, :] - OFFSET_BINS // 2
        np.add.at(p, np.clip(idx, 0, len(p) - 1), pk[head, None] * po)
    return np.arange(len(p)) * GRID, p


@dataclass
class TimeStats:
    bins: np.ndarray  # probability in each BAR_EDGES bar
    mode: float  # peak of the density on the log axis
    typical: float
    mean: float


def time_stats(time_logits, offset=None):
    """Histogram and summary of a move's think time. The same for every move: no switches."""
    t, p = time_distribution(time_logits, offset)
    t_typ, p_typ = time_distribution(time_logits, offset, TYPICAL_TEMPERATURE)
    # Lichess clocks move on an 80 ms grid, which shows up as alternating 40 ms points:
    # average neighbours before binning so the narrow bars near 0 don't alias it.
    smooth = np.convolve(p, [0.5, 0.5])[: len(p)]
    smooth[-1] += 0.5 * p[-1]  # keep all of the last point (120 s or more)
    n = len(BAR_EDGES) - 1
    bins = np.bincount(np.clip(np.searchsorted(BAR_EDGES, t, side="right") - 1, 0, n - 1), smooth, n)
    return TimeStats(bins, _log_mode(t, p), float(t_typ @ p_typ), float(t @ p))


def _log_mode(t, p):
    """Peak of the think-time density on the log axis of the histogram (see _MODE_GRID)."""
    step = _MODE_GRID[1]
    hist = np.bincount(np.rint(np.log1p(t / LOG_SCALE) / step).astype(int), p, len(_MODE_GRID))
    half = int(np.ceil(4 * _MODE_SIGMA / step))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step / _MODE_SIGMA) ** 2)
    # Reflect at 0 s, so a spike at 0 isn't halved by the edge.
    padded = np.concatenate([hist[half:0:-1], [2 * hist[0]], hist[1:], np.zeros(half)])
    density = np.convolve(padded, kernel, mode="valid")[: len(_MODE_GRID)]
    return float(LOG_SCALE * np.expm1(_MODE_GRID[int(np.argmax(density))]))
