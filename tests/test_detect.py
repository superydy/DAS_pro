"""Vibration-detection unit tests."""

import numpy as np

from das_pro.dsp.detect import (
    detect_events,
    detect_peak,
    detect_relative,
    vibration_activity,
)


def _block_with_vibration(pos: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    block = rng.normal(scale=1.0, size=(200, 100))
    t = np.arange(200)
    block[:, pos] += 500.0 * np.sin(2 * np.pi * 0.05 * t)
    return block


def test_activity_peaks_at_vibrating_position():
    act = vibration_activity(_block_with_vibration(37))
    assert int(np.argmax(act)) == 37


def test_detect_triggers_on_vibration():
    act = vibration_activity(_block_with_vibration(37))
    pos, threshold, hit = detect_peak(act, 6.0)
    assert hit
    assert pos == 37
    assert act[pos] > threshold


def test_no_trigger_when_uniform():
    # identical time series at every position -> uniform activity
    rng = np.random.default_rng(1)
    column = rng.normal(size=(200, 1))
    act = vibration_activity(np.tile(column, (1, 100)))
    _, _, hit = detect_peak(act, 6.0)
    assert not hit


def test_no_trigger_on_silence():
    act = vibration_activity(np.zeros((50, 30)))
    _, _, hit = detect_peak(act, 6.0)
    assert not hit


def test_detect_events_finds_multiple_points():
    rng = np.random.default_rng(2)
    block = rng.normal(scale=1.0, size=(200, 100))
    t = np.arange(200)
    block[:, 20] += 500.0 * np.sin(2 * np.pi * 0.05 * t)
    block[:, 70] += 300.0 * np.sin(2 * np.pi * 0.08 * t)
    events, threshold = detect_events(vibration_activity(block), 6.0)
    positions = [p for p, _ in events]
    assert 20 in positions and 70 in positions
    assert positions[0] == 20  # strongest first
    assert all(v > threshold for _, v in events)


def test_detect_events_groups_adjacent_positions():
    # one vibration spreading over neighbouring positions = one event
    rng = np.random.default_rng(3)
    block = rng.normal(scale=1.0, size=(200, 100))
    t = np.arange(200)
    for pos, scale in ((40, 250.0), (41, 500.0), (42, 250.0)):
        block[:, pos] += scale * np.sin(2 * np.pi * 0.05 * t)
    events, _ = detect_events(vibration_activity(block), 6.0)
    assert len(events) == 1
    assert events[0][0] == 41  # the run's peak represents it


def test_detect_events_empty_when_quiet():
    events, _ = detect_events(np.zeros(50), 6.0)
    assert events == []


def test_relative_triggers_against_own_baseline():
    baseline = np.full(100, 50.0)
    baseline[:5] = 20000.0  # permanently-noisy demod front edge
    activity = baseline.copy()
    activity[40] = 800.0  # a kick: 16x its own baseline
    events, ratio = detect_relative(activity, baseline, 4.0)
    assert [p for p, _ in events] == [40]
    assert ratio[40] > 4.0
    # the hot front edge does NOT alarm: it equals its own baseline
    assert ratio[0] < 2.0


def test_relative_detects_whole_fiber_shaking():
    # a coiled spool hit by a hammer raises every position at once;
    # median-relative detection misses this, self-baseline must not
    baseline = np.full(100, 50.0)
    activity = baseline * 20.0
    med_events, _ = detect_events(activity, 4.0)
    assert med_events == []  # the old detector's blind spot
    rel_events, _ = detect_relative(activity, baseline, 4.0)
    assert len(rel_events) == 1  # one contiguous event covering the fiber


def test_relative_quiet_is_silent():
    baseline = np.full(100, 50.0)
    events, _ = detect_relative(baseline.copy(), baseline, 4.0)
    assert events == []


def test_short_block_is_safe():
    act = vibration_activity(np.zeros((1, 30)))
    assert act.shape == (30,)
    _, _, hit = detect_peak(act, 6.0)
    assert not hit


def test_fiber_end_index():
    from das_pro.dsp.detect import fiber_end_index

    amp = np.zeros(204)
    amp[:56] = 5e6 + np.arange(56) * 1000.0  # light up to position 55
    assert fiber_end_index(amp) == 55
    assert fiber_end_index(np.zeros(204)) is None
    assert fiber_end_index(np.zeros(0)) is None
