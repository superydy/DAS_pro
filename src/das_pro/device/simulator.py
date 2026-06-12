"""A software stand-in for the ETH-5520 board.

It accepts the same control commands as the real hardware, replies with ACKs,
and — once started — streams synthetic data frames in the exact wire format the
client expects. This lets the full application run and be demonstrated before
real hardware is available.

Run standalone:

    python -m das_pro.device.simulator --host 127.0.0.1 --port 5000
"""

import argparse
import socket
import socketserver
import struct
import threading
import time

import numpy as np

from ..protocol.constants import (
    BASE_SAMPLE_RATE,
    CMD_HEADER,
    DEFAULT_PORT,
    DataType,
    Reg,
)
from ..protocol.frames import FrameHeader


class _BoardState:
    """Holds the register values the host writes, mirroring the board config."""

    def __init__(self) -> None:
        self.trig_freq = 1000
        self.point_num_per_scan = 1024
        self.upload_ch_num = 1
        self.upload_data_src = DataType.RAW
        self.data_rate_sel = 1
        self.phase_bits_16 = False
        self.space_merge_point_num = 1
        self.dec_ratio = 1
        self.frame_num = 1
        self.running = False

    @property
    def sample_rate(self) -> float:
        return BASE_SAMPLE_RATE / max(self.data_rate_sel, 1)

    @property
    def point_num_after_merge(self) -> int:
        return max(self.point_num_per_scan // max(self.space_merge_point_num, 1), 1)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        state = _BoardState()
        streamer: threading.Thread | None = None
        stop_flag = threading.Event()

        try:
            while True:
                cmd = self._recv_exact(sock, 8)
                if cmd is None:
                    break
                if cmd[:2] != CMD_HEADER:
                    # Unknown framing; ignore but keep the link alive.
                    continue
                addr = struct.unpack(">H", cmd[2:4])[0]
                value = struct.unpack(">I", cmd[4:8])[0]
                self._apply(state, addr, cmd, value)

                if addr == Reg.START_STOP:
                    start_requested = cmd[7] == 0x01
                    if start_requested and not state.running:
                        state.running = True
                        stop_flag.clear()
                        streamer = threading.Thread(
                            target=self._stream, args=(sock, state, stop_flag), daemon=True
                        )
                        # ACK the start before streaming begins.
                        self._ack(sock)
                        streamer.start()
                        continue
                    elif not start_requested:
                        stop_flag.set()
                        state.running = False
                        if streamer is not None:
                            streamer.join(timeout=2.0)
                            streamer = None
                        self._ack(sock)
                        continue

                self._ack(sock)
        except (ConnectionError, OSError):
            pass
        finally:
            stop_flag.set()

    # --- helpers ---

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
        buf = b""
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    @staticmethod
    def _ack(sock: socket.socket, status: int = 0) -> None:
        # 8-byte ACK: identifier word + status word, both little-endian.
        sock.sendall(struct.pack("<II", 0xA5A5A5A5, status))

    @staticmethod
    def _apply(state: _BoardState, addr: int, cmd: bytes, value: int) -> None:
        if addr == Reg.TRIG_FREQ:
            state.trig_freq = value
        elif addr == Reg.POINT_NUM_PER_SCAN:
            state.point_num_per_scan = (value // 16) * 16 or 16
        elif addr == Reg.UPLOAD_CH_NUM:
            state.upload_ch_num = value
        elif addr == Reg.UPLOAD_DATA_SRC:
            state.upload_data_src = value
        elif addr == Reg.UPLOAD_DATA_RATE:
            state.data_rate_sel = value or 1
        elif addr == Reg.PHASE_UPLOAD_BIT:
            state.phase_bits_16 = value == 1
        elif addr == Reg.SPACE_MERGE_POINT_NUM:
            state.space_merge_point_num = value or 1
        elif addr == Reg.PHASE_UPLOAD_DEC_RATIO:
            state.dec_ratio = value or 1
        elif addr == Reg.START_STOP and cmd[7] == 0x01:
            state.frame_num = struct.unpack(">H", cmd[4:6])[0] or 1

    def _stream(self, sock: socket.socket, state: _BoardState, stop_flag: threading.Event) -> None:
        """Emit synthetic frames until told to stop."""
        rng = np.random.default_rng(0xDA5)
        phase_t = 0.0
        try:
            while not stop_flag.is_set():
                data_type = self._effective_data_type(state)
                if data_type == DataType.PHASE:
                    points = state.point_num_after_merge
                else:
                    points = state.point_num_per_scan
                payload, phase_t = self._make_payload(state, data_type, points, rng, phase_t)

                header = FrameHeader(
                    identifier=0x5520,
                    data_type=int(data_type),
                    frame_num=state.frame_num,
                    point_num_per_ch_per_scan=points,
                )
                sock.sendall(header.pack())
                sock.sendall(payload)

                if data_type == DataType.PHASE:
                    # the real board interleaves an amp-monitor packet
                    # (frame_num=1) after each phase upload
                    mon_points = state.point_num_after_merge
                    mon = FrameHeader(
                        identifier=0x5520,
                        data_type=int(DataType.AMP_MONITOR),
                        frame_num=1,
                        point_num_per_ch_per_scan=mon_points,
                    )
                    mon_payload = (
                        rng.random(mon_points * state.upload_ch_num) * 4000 + 1000
                    ).astype("<u4")
                    sock.sendall(mon.pack())
                    sock.sendall(mon_payload.tobytes())

                # Pace the stream loosely by the trigger rate so the GUI stays responsive.
                time.sleep(min(max(state.frame_num / max(state.trig_freq, 1), 0.02), 0.2))
        except (ConnectionError, OSError):
            pass

    @staticmethod
    def _effective_data_type(state: _BoardState) -> int:
        # Map the requested upload source to the data type reported in the header.
        return int(state.upload_data_src)

    def _make_payload(self, state, data_type, points, rng, phase_t):
        ch = state.upload_ch_num
        total = points * ch * state.frame_num
        fs = state.sample_rate

        if data_type == DataType.PHASE:
            # Quiet fiber everywhere except one vibrating spot at 1/3 of
            # the length — lets the single-point detector be demonstrated.
            tone_hz = 50.0
            dt = 1.0 / max(state.trig_freq / max(state.dec_ratio, 1), 1.0)
            t = phase_t + np.arange(state.frame_num) * dt
            phase_t = t[-1] + dt if len(t) else phase_t
            grid = rng.normal(0, 50, size=(state.frame_num, points, ch))
            vib = points // 3
            grid[:, vib, :] += (12000.0 * np.sin(2 * np.pi * tone_hz * t))[:, None]
            grid = grid.ravel()[:total]
            if state.phase_bits_16:
                return grid.astype("<i2").tobytes(), phase_t
            return grid.astype("<i4").tobytes(), phase_t

        if data_type == DataType.AMP_MONITOR:
            vals = (rng.random(total) * 4000 + 1000).astype("<u4")
            return vals.tobytes(), phase_t

        # raw / IQ / arctan&sqrt -> int16 waveform (a tone + noise)
        x = np.arange(total)
        wave = 1500.0 * np.sin(2 * np.pi * 5.0 * x / max(points, 1))
        wave = wave + rng.normal(0, 80, size=total)
        return np.clip(wave, -2047, 2047).astype("<i2").tobytes(), phase_t


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> _Server:
    """Create and start a simulator server in a background thread; return it."""
    server = _Server((host, port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="ETH-5520 board simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    server = _Server((args.host, args.port), _Handler)
    print(f"ETH-5520 simulator listening on {args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
