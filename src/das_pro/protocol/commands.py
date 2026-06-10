"""Builders for the 8-byte control commands sent to the ETH-5520 board.

Each function returns the raw bytes to write to the socket. Keeping these pure
(no I/O) makes them trivial to unit-test against the byte layout used by the
original demo.
"""

import struct

from .constants import CMD_HEADER, Reg


def build_command(reg: int, value: int) -> bytes:
    """Build a generic command: header + 2-byte big-endian addr + 4-byte big-endian value."""
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"value out of 32-bit range: {value}")
    return CMD_HEADER + struct.pack(">HI", int(reg) & 0xFFFF, value)


def set_clock_src(clk_src: int) -> bytes:
    if clk_src not in (0, 1):
        raise ValueError("clock source must be 0 (external) or 1 (onboard)")
    return build_command(Reg.CLOCK_SRC, clk_src)


def set_trig_dir(trig_dir: int) -> bytes:
    if trig_dir not in (0, 1):
        raise ValueError("trigger direction must be 0 (receive) or 1 (send)")
    return build_command(Reg.TRIG_DIR, trig_dir)


def set_trig_freq(freq_hz: int) -> bytes:
    return build_command(Reg.TRIG_FREQ, freq_hz)


def set_trig_pulse_width(width_ns: int) -> bytes:
    return build_command(Reg.TRIG_PULSE_WIDTH, width_ns)


def set_point_num_per_scan(point_num: int) -> bytes:
    # The board requires a multiple of 16; the demo silently truncates.
    point_num = (point_num // 16) * 16
    return build_command(Reg.POINT_NUM_PER_SCAN, point_num)


def set_bypass_point_num(bypass_point_num: int) -> bytes:
    return build_command(Reg.BYPASS_POINT_NUM, bypass_point_num)


def set_upload_ch_num(ch_num: int) -> bytes:
    if ch_num not in (1, 2, 4):
        raise ValueError("upload channel number must be 1, 2 or 4")
    return build_command(Reg.UPLOAD_CH_NUM, ch_num)


def set_upload_data_src(data_src: int) -> bytes:
    if data_src not in (0, 2, 3, 4):
        raise ValueError("upload data source must be 0, 2, 3 or 4")
    return build_command(Reg.UPLOAD_DATA_SRC, data_src)


def set_upload_data_rate(data_rate_sel: int) -> bytes:
    return build_command(Reg.UPLOAD_DATA_RATE, data_rate_sel)


def set_center_freq(center_freq_hz: int) -> bytes:
    return build_command(Reg.CENTER_FREQ, center_freq_hz)


def set_phase_upload_bit(bit_sel: int) -> bytes:
    if bit_sel not in (0, 1):
        raise ValueError("phase upload bit select must be 0 (32-bit) or 1 (16-bit)")
    return build_command(Reg.PHASE_UPLOAD_BIT, bit_sel)


def set_phase_upload_dec_ratio(dec_ratio: int) -> bytes:
    if dec_ratio == 0:
        raise ValueError("decimation ratio must be >= 1")
    return build_command(Reg.PHASE_UPLOAD_DEC_RATIO, dec_ratio)


def conf_user_ip(o1: int, o2: int, o3: int, o4: int) -> bytes:
    """Configure the board's IP. The four octets occupy the payload bytes directly."""
    for o in (o1, o2, o3, o4):
        if not 0 <= o <= 255:
            raise ValueError("each IP octet must be 0..255")
    return CMD_HEADER + struct.pack(">HBBBB", int(Reg.CONF_USER_IP), o1, o2, o3, o4)


def set_do_bit_enable(bit_en: int) -> bytes:
    return build_command(Reg.DO_BIT_ENABLE, bit_en & 0xFF)


def set_do_bit_status(bit_status: int) -> bytes:
    return build_command(Reg.DO_BIT_STATUS, bit_status & 0xFF)


# --- Phase demodulation parameter group (sent as a sequence in the demo) ---

def set_data_rate_to_phase_dem(value: int) -> bytes:
    if value < 1:
        raise ValueError("data_rate_to_phase_dem must be >= 1")
    return build_command(Reg.DATA_RATE_TO_PHASE_DEM, value)


def set_space_avg_order(value: int) -> bytes:
    if value < 1:
        raise ValueError("space_avg_order must be >= 1")
    return build_command(Reg.SPACE_AVG_ORDER, value)


def set_space_merge_point_num(value: int) -> bytes:
    if value < 1:
        raise ValueError("space_merge_point_num must be >= 1")
    return build_command(Reg.SPACE_MERGE_POINT_NUM, value)


def set_space_region_diff_order(value: int) -> bytes:
    if value < 1:
        raise ValueError("space_region_diff_order must be >= 1")
    return build_command(Reg.SPACE_REGION_DIFF_ORDER, value)


def set_detrend_filter_bw(bw: float) -> bytes:
    # The demo transmits the bandwidth scaled by 1e5 as an integer.
    return build_command(Reg.DETREND_FILTER_BW, int(round(bw * 100000)))


def set_polarization_diversity_en(enabled: int) -> bytes:
    if enabled not in (0, 1):
        raise ValueError("polarization_diversity_en must be 0 or 1")
    return build_command(Reg.POLARIZATION_DIVERSITY_EN, enabled)


def start(frame_num: int) -> bytes:
    """Start acquisition.

    The demo packs frame_num as a big-endian 16-bit value in payload bytes 0..1
    and sets the final payload byte to 0x01.
    """
    if frame_num > 10000:
        raise ValueError("frame_num must be <= 10000")
    return CMD_HEADER + struct.pack(">HHBB", int(Reg.START_STOP), frame_num & 0xFFFF, 0x00, 0x01)


def stop() -> bytes:
    """Stop acquisition: same register, all-zero payload."""
    return CMD_HEADER + struct.pack(">HI", int(Reg.START_STOP), 0)
