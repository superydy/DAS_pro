"""Background acquisition worker.

Runs the read loop off the UI thread (mirroring the demo's ReadDataThread).

Frames can arrive faster than the GUI can plot them, so the worker conflates:
it keeps only the latest frame and emits at most one pending `frame_ready`
notification at a time. The GUI fetches the newest frame with `take_latest()`.
Without this, queued cross-thread signals pile up unboundedly and freeze the
UI. Recording is done here in the worker so disk capture never drops frames
regardless of plot speed.
"""

import threading
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..device.client import AcquisitionConfig, DasClient, DeviceError
from ..protocol.constants import DataType


@dataclass
class StreamSettings:
    upload_ch_num: int
    phase_bits_16: bool
    data_src: int


class AcquisitionWorker(QObject):
    """Owns the read loop. Move to a QThread and call run()."""

    frame_ready = Signal()  # fetch the actual frame with take_latest()
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        client: DasClient,
        settings: StreamSettings,
        record_file: BinaryIO | None = None,
    ):
        super().__init__()
        self._client = client
        self._settings = settings
        self._record_file = record_file
        self._running = True
        self._lock = threading.Lock()
        self._latest = None
        self._notified = False
        self._recv_count = 0
        self._byte_count = 0

    def stop(self) -> None:
        self._running = False

    def stats(self) -> tuple[int, int]:
        """(mass-data frames received, total bytes received) so far."""
        with self._lock:
            return self._recv_count, self._byte_count

    def take_latest(self):
        """Return ((header, data) | None, total mass-data frames received)."""
        with self._lock:
            item = self._latest
            self._notified = False
            count = self._recv_count
        return item, count

    def run(self) -> None:
        cfg = AcquisitionConfig(
            upload_ch_num=self._settings.upload_ch_num,
            phase_bits_16=self._settings.phase_bits_16,
        )
        try:
            while self._running:
                header, data = self._client.read_frame(cfg)
                if not self._running:
                    break

                if header.data_type <= DataType.PHASE:
                    self._recv_count += 1
                self._byte_count += np.asarray(data).nbytes + 16

                if self._record_file is not None:
                    self._record_file.write(np.asarray(data).tobytes())

                with self._lock:
                    self._latest = (header, data)
                    notify = not self._notified
                    self._notified = True
                if notify:
                    self.frame_ready.emit()
        except DeviceError as exc:
            if self._running:
                self.error.emit(str(exc))
        except OSError as exc:
            if self._running:
                self.error.emit(f"connection error: {exc}")
        finally:
            self.finished.emit()


def deinterleave(data: np.ndarray, ch_num: int) -> list[np.ndarray]:
    """Split interleaved channel samples into a list of per-channel arrays."""
    if ch_num <= 1:
        return [data]
    return [data[ch::ch_num] for ch in range(ch_num)]
