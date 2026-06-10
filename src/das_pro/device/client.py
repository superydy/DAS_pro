"""TCP client for the ETH-5520 board.

Mirrors the request/response pattern of the original demo: every control
command writes 8 bytes and reads an 8-byte ACK whose second little-endian
uint32 word is the status (0 == success). Data acquisition reads a 16-byte
frame header followed by a length-determined payload.
"""

import socket
import struct
from dataclasses import dataclass

from ..protocol import commands
from ..protocol.constants import (
    ACK_LENGTH,
    DEFAULT_IP,
    DEFAULT_PORT,
    FRAME_HEADER_LENGTH,
)
from ..protocol.frames import FrameHeader, decode_payload, payload_byte_count


class DeviceError(Exception):
    """Raised when the board reports a non-zero status or the link fails."""


@dataclass
class AcquisitionConfig:
    """The subset of parameters needed to size and decode the data stream."""

    upload_ch_num: int = 1
    phase_bits_16: bool = False


class DasClient:
    def __init__(self, host: str = DEFAULT_IP, port: int = DEFAULT_PORT, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None

    # --- connection management ---

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "DasClient":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- low-level IO ---

    def _recv_exact(self, n: int) -> bytes:
        assert self._sock is not None
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise DeviceError("connection closed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _command(self, payload: bytes) -> int:
        """Send an 8-byte command, read the ACK and return its status word."""
        if self._sock is None:
            raise DeviceError("not connected")
        self._sock.sendall(payload)
        ack = self._recv_exact(ACK_LENGTH)
        _identifier, status = struct.unpack("<II", ack)
        return status

    def _command_checked(self, payload: bytes, what: str) -> int:
        status = self._command(payload)
        if status != 0:
            raise DeviceError(f"{what} failed, board status={status}")
        return status

    # --- high-level control surface (1:1 with the demo) ---

    def set_clock_src(self, clk_src: int) -> int:
        return self._command_checked(commands.set_clock_src(clk_src), "set_clock_src")

    def set_trig_dir(self, trig_dir: int) -> int:
        return self._command_checked(commands.set_trig_dir(trig_dir), "set_trig_dir")

    def set_trig_freq(self, freq_hz: int) -> int:
        return self._command_checked(commands.set_trig_freq(freq_hz), "set_trig_freq")

    def set_trig_pulse_width(self, width_ns: int) -> int:
        return self._command_checked(
            commands.set_trig_pulse_width(width_ns), "set_trig_pulse_width"
        )

    def set_point_num_per_scan(self, point_num: int) -> int:
        return self._command_checked(
            commands.set_point_num_per_scan(point_num), "set_point_num_per_scan"
        )

    def set_bypass_point_num(self, n: int) -> int:
        return self._command_checked(commands.set_bypass_point_num(n), "set_bypass_point_num")

    def set_upload_ch_num(self, ch_num: int) -> int:
        return self._command_checked(commands.set_upload_ch_num(ch_num), "set_upload_ch_num")

    def set_upload_data_src(self, data_src: int) -> int:
        return self._command_checked(
            commands.set_upload_data_src(data_src), "set_upload_data_src"
        )

    def set_upload_data_rate(self, rate_sel: int) -> int:
        return self._command_checked(
            commands.set_upload_data_rate(rate_sel), "set_upload_data_rate"
        )

    def set_center_freq(self, freq_hz: int) -> int:
        return self._command_checked(commands.set_center_freq(freq_hz), "set_center_freq")

    def set_phase_demod_params(
        self,
        data_rate_to_phase_dem: int,
        space_avg_order: int,
        space_merge_point_num: int,
        space_region_diff_order: int,
        detrend_filter_bw: float,
        polarization_diversity_en: int,
    ) -> None:
        self._command_checked(
            commands.set_data_rate_to_phase_dem(data_rate_to_phase_dem),
            "set_data_rate_to_phase_dem",
        )
        self._command_checked(
            commands.set_space_avg_order(space_avg_order), "set_space_avg_order"
        )
        self._command_checked(
            commands.set_space_merge_point_num(space_merge_point_num),
            "set_space_merge_point_num",
        )
        self._command_checked(
            commands.set_space_region_diff_order(space_region_diff_order),
            "set_space_region_diff_order",
        )
        self._command_checked(
            commands.set_detrend_filter_bw(detrend_filter_bw), "set_detrend_filter_bw"
        )
        self._command_checked(
            commands.set_polarization_diversity_en(polarization_diversity_en),
            "set_polarization_diversity_en",
        )

    def set_phase_upload_bit(self, bit_sel: int) -> int:
        return self._command_checked(
            commands.set_phase_upload_bit(bit_sel), "set_phase_upload_bit"
        )

    def set_phase_upload_dec_ratio(self, ratio: int) -> int:
        return self._command_checked(
            commands.set_phase_upload_dec_ratio(ratio), "set_phase_upload_dec_ratio"
        )

    def conf_user_ip(self, o1: int, o2: int, o3: int, o4: int) -> int:
        return self._command_checked(commands.conf_user_ip(o1, o2, o3, o4), "conf_user_ip")

    def set_do_bit(self, bit_en: int, bit_status: int) -> int:
        self._command_checked(commands.set_do_bit_enable(bit_en), "set_do_bit_enable")
        return self._command_checked(commands.set_do_bit_status(bit_status), "set_do_bit_status")

    def start(self, frame_num: int) -> int:
        return self._command_checked(commands.start(frame_num), "start")

    def stop(self) -> int:
        return self._command_checked(commands.stop(), "stop")

    # --- data acquisition ---

    def read_frame(self, config: AcquisitionConfig):
        """Read one upload: returns (FrameHeader, decoded numpy array)."""
        header = FrameHeader.parse(self._recv_exact(FRAME_HEADER_LENGTH))
        n_bytes = payload_byte_count(header, config.upload_ch_num, config.phase_bits_16)
        payload = self._recv_exact(n_bytes) if n_bytes else b""
        data = decode_payload(payload, header.data_type, config.phase_bits_16)
        return header, data
