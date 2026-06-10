"""Background acquisition worker.

Runs the read loop off the UI thread and forwards decoded frames to the GUI via
Qt signals, mirroring the demo's ReadDataThread.
"""

from dataclasses import dataclass

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

    frame_ready = Signal(object, object)  # (FrameHeader, np.ndarray)
    error = Signal(str)
    finished = Signal()

    def __init__(self, client: DasClient, settings: StreamSettings):
        super().__init__()
        self._client = client
        self._settings = settings
        self._running = True

    def stop(self) -> None:
        self._running = False

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
                self.frame_ready.emit(header, data)
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
