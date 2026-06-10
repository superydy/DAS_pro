"""Byte-level checks against the wire format used by the original CVI demo."""

import struct

import numpy as np
import pytest

from das_pro.protocol import commands
from das_pro.protocol.constants import Reg
from das_pro.protocol.frames import FrameHeader, decode_payload, payload_byte_count


def test_set_trig_freq_layout():
    # 0x5A 0xA5, addr 0x0018 big-endian, value big-endian.
    assert commands.set_trig_freq(1000) == bytes((0x5A, 0xA5, 0x00, 0x18)) + struct.pack(">I", 1000)


def test_point_num_truncated_to_multiple_of_16():
    pkt = commands.set_point_num_per_scan(1000)
    value = struct.unpack(">I", pkt[4:8])[0]
    assert value == 992  # (1000 // 16) * 16


def test_start_packs_frame_num_big_endian_and_marker():
    pkt = commands.start(5)
    assert pkt[:4] == bytes((0x5A, 0xA5, 0x00, 0x14))
    assert struct.unpack(">H", pkt[4:6])[0] == 5
    assert pkt[7] == 0x01


def test_stop_is_zero_payload():
    assert commands.stop() == bytes((0x5A, 0xA5, 0x00, 0x14)) + b"\x00\x00\x00\x00"


def test_detrend_bw_scaled_by_1e5():
    pkt = commands.set_detrend_filter_bw(0.5)
    assert struct.unpack(">I", pkt[4:8])[0] == 50000


def test_conf_user_ip_octets_in_payload():
    pkt = commands.conf_user_ip(192, 168, 1, 100)
    assert pkt == bytes((0x5A, 0xA5, 0x00, int(Reg.CONF_USER_IP), 192, 168, 1, 100))


def test_invalid_channel_count_rejected():
    with pytest.raises(ValueError):
        commands.set_upload_ch_num(3)


def test_frame_header_roundtrip():
    h = FrameHeader(identifier=0x5520, data_type=4, frame_num=2, point_num_per_ch_per_scan=512)
    assert FrameHeader.parse(h.pack()) == h


def test_payload_byte_count_phase_modes():
    h = FrameHeader(0, 4, frame_num=2, point_num_per_ch_per_scan=100)
    assert payload_byte_count(h, ch_num=2, phase_bits_16=False) == 100 * 2 * 2 * 4
    assert payload_byte_count(h, ch_num=2, phase_bits_16=True) == 100 * 2 * 2 * 2


def test_decode_payload_int16():
    raw = np.array([1, -2, 3], dtype="<i2").tobytes()
    out = decode_payload(raw, data_type=0, phase_bits_16=False)
    assert list(out) == [1, -2, 3]
