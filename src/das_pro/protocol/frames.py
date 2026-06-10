"""Parsing of the data-stream frames uploaded by the board.

Each upload begins with a 16-byte header of four little-endian uint32 words:

    [0] identifier
    [1] data_type          (see DataType)
    [2] frame_num          (number of scans bundled in this upload)
    [3] point_num_per_ch_per_scan

The payload that follows depends on data_type and, for phase data, on the
16/32-bit upload selection:

    raw / IQ / arctan&sqrt (type < 4): int16, 2 bytes per point
    phase (type == 4):                 int32 (4 bytes) if 32-bit mode,
                                        int16 (2 bytes) if 16-bit mode
    amplitude monitor (type == 5):     uint32, 4 bytes per point

The number of points per upload is point_num_per_ch_per_scan * ch_num * frame_num.
"""

import struct
from dataclasses import dataclass

import numpy as np

from .constants import FRAME_HEADER_LENGTH, DataType


@dataclass
class FrameHeader:
    identifier: int
    data_type: int
    frame_num: int
    point_num_per_ch_per_scan: int

    @classmethod
    def parse(cls, data: bytes) -> "FrameHeader":
        if len(data) < FRAME_HEADER_LENGTH:
            raise ValueError(
                f"frame header needs {FRAME_HEADER_LENGTH} bytes, got {len(data)}"
            )
        identifier, data_type, frame_num, point_num = struct.unpack(
            "<IIII", data[:FRAME_HEADER_LENGTH]
        )
        return cls(identifier, data_type, frame_num, point_num)

    def pack(self) -> bytes:
        return struct.pack(
            "<IIII",
            self.identifier,
            self.data_type,
            self.frame_num,
            self.point_num_per_ch_per_scan,
        )


def payload_byte_count(header: FrameHeader, ch_num: int, phase_bits_16: bool) -> int:
    """Number of payload bytes that follow the header for this upload."""
    points = header.point_num_per_ch_per_scan * ch_num * header.frame_num
    dt = header.data_type
    if dt < DataType.PHASE:  # raw / IQ / arctan&sqrt -> int16
        return points * 2
    if dt == DataType.PHASE:
        return points * (2 if phase_bits_16 else 4)
    if dt == DataType.AMP_MONITOR:  # uint32
        return points * 4
    raise ValueError(f"unknown data_type: {dt}")


def payload_dtype(data_type: int, phase_bits_16: bool) -> np.dtype:
    """numpy dtype for the payload of a given data_type."""
    if data_type < DataType.PHASE:
        return np.dtype("<i2")
    if data_type == DataType.PHASE:
        return np.dtype("<i2") if phase_bits_16 else np.dtype("<i4")
    if data_type == DataType.AMP_MONITOR:
        return np.dtype("<u4")
    raise ValueError(f"unknown data_type: {data_type}")


def decode_payload(raw: bytes, data_type: int, phase_bits_16: bool) -> np.ndarray:
    """Decode a payload byte buffer into a 1-D numpy array (interleaved channels)."""
    return np.frombuffer(raw, dtype=payload_dtype(data_type, phase_bits_16))
