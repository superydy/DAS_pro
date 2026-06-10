"""Device communication: TCP client and a board simulator."""

from .client import AcquisitionConfig, DasClient, DeviceError
from .simulator import serve as serve_simulator

__all__ = ["DasClient", "DeviceError", "AcquisitionConfig", "serve_simulator"]
