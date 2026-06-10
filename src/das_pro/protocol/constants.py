"""Protocol constants for the ETH-5520 DAS board.

All values are reverse-engineered from the official LabWindows/CVI demo
(ETH_DAS_DEMO.c). The board acts as a TCP server; the host is the client.

Control commands are 8 bytes:

    0x5A 0xA5  <addr_hi> <addr_lo>  <val[31:24]> <val[23:16]> <val[15:8]> <val[7:0]>

The two-byte frame header is always 0x5A 0xA5. The next two bytes are a
big-endian register address (the high byte is always 0x00 in the demo). The
final four bytes are the parameter, big-endian.

The board replies with an 8-byte ACK interpreted as two little-endian uint32
words; the second word is the status (0 == success).
"""

from enum import IntEnum

# Command framing
CMD_HEADER = bytes((0x5A, 0xA5))
CMD_LENGTH = 8
ACK_LENGTH = 8

# Default network endpoint of the board
DEFAULT_IP = "192.168.1.100"
DEFAULT_PORT = 5000


class Reg(IntEnum):
    """Register addresses (low byte; high byte is 0x00)."""

    CLOCK_SRC = 0x00            # 0 = external 10 MHz ref, 1 = onboard ref
    TRIG_DIR = 0x10            # 0 = receive trigger, 1 = send trigger
    START_STOP = 0x14          # special payload, see commands.start/stop
    TRIG_FREQ = 0x18           # Hz
    TRIG_PULSE_WIDTH = 0x1C    # ns
    POINT_NUM_PER_SCAN = 0x20  # must be a multiple of 16
    UPLOAD_CH_NUM = 0x24       # 1, 2 or 4
    UPLOAD_DATA_SRC = 0x28     # see DataSrc
    UPLOAD_DATA_RATE = 0x2C    # sample_rate = 500 MHz / data_rate_sel
    CENTER_FREQ = 0x34         # Hz
    BYPASS_POINT_NUM = 0x38
    CONF_USER_IP = 0x60        # 4 payload bytes are the IP octets
    DO_BIT_ENABLE = 0x74
    DO_BIT_STATUS = 0x78
    SPACE_AVG_ORDER = 0x94
    SPACE_MERGE_POINT_NUM = 0x9C
    SPACE_REGION_DIFF_ORDER = 0xA0
    DETREND_FILTER_BW = 0xA8   # payload = bw * 100000
    POLARIZATION_DIVERSITY_EN = 0xAC
    DATA_RATE_TO_PHASE_DEM = 0xB0
    PHASE_UPLOAD_BIT = 0xB4    # 0 = 32-bit, 1 = 16-bit
    PHASE_UPLOAD_DEC_RATIO = 0xB8


class DataSrc(IntEnum):
    """Valid values for UPLOAD_DATA_SRC."""

    RAW = 0
    IQ = 2
    ARCTAN_SQRT = 3
    PHASE = 4


class DataType(IntEnum):
    """Data type reported in the received frame header (word index 1)."""

    RAW = 0
    IQ = 2
    ARCTAN_SQRT = 3
    PHASE = 4
    AMP_MONITOR = 5


class ClockSrc(IntEnum):
    EXTERNAL = 0
    ONBOARD = 1


class TrigDir(IntEnum):
    RECEIVE = 0
    SEND = 1


class PhaseBits(IntEnum):
    BITS_32 = 0
    BITS_16 = 1


# Sampling clock used to derive the effective sample rate.
BASE_SAMPLE_RATE = 500_000_000.0  # 500 MSps

# Received data frame header: 4 x uint32, little-endian on the board's x86 host.
FRAME_HEADER_WORDS = 4
FRAME_HEADER_LENGTH = FRAME_HEADER_WORDS * 4
