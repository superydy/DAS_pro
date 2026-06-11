"""Parameter state objects shared between the main window and dialogs.

Pure data, no Qt: dialogs edit a copy and hand it back, the main window
reads them when configuring the board. Keeping them here decouples the
parameter dialogs from the main window and from each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..protocol.constants import BASE_SAMPLE_RATE, DataSrc


@dataclass
class AcquisitionParams:
    trig_freq: int = 2000          # Hz
    trig_width: int = 100          # ns
    point_num: int = 5120          # multiple of 16
    bypass_point: int = 2
    data_src: int = int(DataSrc.RAW)
    upload_rate_sel: int = 1       # 500M / sel
    ch_num: int = 2                # 1 / 2 / 4
    center_freq_mhz: int = 80
    trig_dir: int = 1              # 0=IN 1=OUT
    clk_src: int = 1               # 0=ExtRef 1=OnBoard
    phase_bits_16: bool = False
    dec_ratio: int = 1

    @property
    def sample_rate(self) -> float:
        return BASE_SAMPLE_RATE / max(self.upload_rate_sel, 1)

    @property
    def phase_sample_rate(self) -> float:
        return self.trig_freq / max(self.dec_ratio, 1)

    @property
    def is_phase(self) -> bool:
        return self.data_src == int(DataSrc.PHASE)


@dataclass
class PhaseDemodParams:
    space_avg: int = 25
    space_merge: int = 25
    region_diff: int = 2
    detrend_bw: float = 20.0       # Hz
    polar_div: int = 0             # 0=DIS 1=EN
    rate2phase_sel: int = 2        # 500M / sel


def merged_points(acq: AcquisitionParams, demod: PhaseDemodParams) -> int:
    """Positions per scan after spatial merging (phase mode)."""
    return max(acq.point_num // max(demod.space_merge, 1), 1)


def throughput_mb_s(
    acq: AcquisitionParams, demod: PhaseDemodParams
) -> float:
    if not acq.is_phase:
        return acq.trig_freq * acq.point_num * 2 * acq.ch_num / 1024 / 1024
    byte_per_phase = 2 if acq.phase_bits_16 else 4
    return (
        acq.phase_sample_rate
        * merged_points(acq, demod)
        * byte_per_phase
        * acq.ch_num
        / 1024
        / 1024
    )


def fiber_len_km(acq: AcquisitionParams, demod: PhaseDemodParams) -> float:
    # 0.2 m per sample at 500 MSps (round trip), as computed by the demo.
    if not acq.is_phase:
        return acq.point_num * acq.upload_rate_sel * 0.2 / 1000.0
    return acq.point_num * demod.rate2phase_sel * 0.4 / 1000.0
