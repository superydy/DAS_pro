"""Vibration-detection unit tests."""

import numpy as np

from das_pro.dsp.detect import detect_events, detect_peak, vibration_activity


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


def test_short_block_is_safe():
    act = vibration_activity(np.zeros((1, 30)))
    assert act.shape == (30,)
    _, _, hit = detect_peak(act, 6.0)
    assert not hit
