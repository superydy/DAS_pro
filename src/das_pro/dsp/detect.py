"""Vibration detection along the fiber.

Works on a block of phase scans (time x position). The activity of a
position is the standard deviation of its temporal first difference: a
quiet stretch of fiber barely moves scan-to-scan while a vibrating spot
swings, so diff+std separates them regardless of slow phase drift. A
position triggers the detector when its activity stands out from the
median of all positions by a user-set ratio — the median tracks the
ambient noise floor, so the threshold adapts to laser/fiber conditions.
"""

from __future__ import annotations

import numpy as np


def vibration_activity(block: np.ndarray) -> np.ndarray:
    """Per-position activity of a (scans, positions) phase block."""
    a = np.asarray(block, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2:
        return np.zeros(a.shape[-1] if a.ndim >= 1 else 0)
    return np.std(np.diff(a, axis=0), axis=0)


def detect_peak(
    activity: np.ndarray, threshold_ratio: float
) -> tuple[int, float, bool]:
    """Find the strongest position.

    Returns (position, threshold, triggered): the index of the most
    active position, the absolute threshold used (median * ratio), and
    whether that position exceeds it.
    """
    act = np.asarray(activity, dtype=np.float64)
    if act.size == 0:
        return 0, 0.0, False
    threshold = float(np.median(act)) * threshold_ratio
    pos = int(np.argmax(act))
    triggered = bool(act[pos] > threshold > 0.0)
    return pos, threshold, triggered
