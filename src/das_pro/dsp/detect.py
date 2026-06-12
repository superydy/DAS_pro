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


def group_events(
    activity: np.ndarray, mask: np.ndarray, max_events: int = 20
) -> list[tuple[int, float]]:
    """Group contiguous runs of a trigger mask into one event each.

    A vibration excites a few neighbouring positions; the run's peak
    represents it. Returns (position, activity) sorted strongest first.
    """
    act = np.asarray(activity, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if act.size == 0 or not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    events = []
    for lo, hi in zip(edges[::2], edges[1::2]):  # hi is exclusive
        peak = lo + int(np.argmax(act[lo:hi]))
        events.append((peak, float(act[peak])))
    events.sort(key=lambda e: e[1], reverse=True)
    return events[:max_events]


def detect_events(
    activity: np.ndarray, threshold_ratio: float, max_events: int = 20
) -> tuple[list[tuple[int, float]], float]:
    """Median-relative detection: positions above median * ratio.

    Works when the fiber is laid out and only a spot vibrates; breaks
    when the whole fiber vibrates together (the median rises with the
    signal). Prefer detect_relative for live monitoring.
    """
    act = np.asarray(activity, dtype=np.float64)
    if act.size == 0:
        return [], 0.0
    threshold = float(np.median(act)) * threshold_ratio
    if threshold <= 0.0:
        return [], threshold
    return group_events(act, act > threshold, max_events), threshold


def detect_relative(
    activity: np.ndarray,
    baseline: np.ndarray,
    threshold_ratio: float,
    max_events: int = 20,
) -> tuple[list[tuple[int, float]], np.ndarray]:
    """Self-baseline detection: each position against its own quiet level.

    Positions trigger when activity exceeds their *own* baseline by the
    ratio. This keeps permanently-noisy stretches (demod front edge,
    beyond the fiber end) silent — their baseline is high too — and
    still alarms everywhere when the whole fiber (e.g. a coiled spool)
    is shaken at once, which median-relative detection misses.

    Returns (events, ratio_per_position).
    """
    act = np.asarray(activity, dtype=np.float64)
    base = np.maximum(np.asarray(baseline, dtype=np.float64), 1e-9)
    ratio = act / base
    return group_events(act, ratio > threshold_ratio, max_events), ratio


def fiber_end_index(amplitude: np.ndarray, frac: float = 0.05) -> int | None:
    """Estimate where the light ends along the fiber (OTDR-style).

    The amplitude-monitor trace is strong where backscattered light
    returns and ~zero beyond the fiber end, so the last position above
    `frac` of the robust maximum marks the end. Returns None when the
    trace is empty or flat.
    """
    a = np.asarray(amplitude, dtype=np.float64)
    if a.size == 0:
        return None
    ref = float(np.percentile(a, 95))
    if ref <= 0.0:
        return None
    above = np.flatnonzero(a > ref * frac)
    return int(above[-1]) if above.size else None
