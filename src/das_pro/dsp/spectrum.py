"""Power-spectrum / PSD analysis matching the original demo.

The CVI demo uses NI's AutoPowerSpectrum with a scaled (Hann) window and reports
the result in dBm or dBm/Hz, referenced to a 50-ohm load. We reproduce the same
chain with numpy so plots line up with the legacy tool.

Conventions taken from ETH_DAS_DEMO.c:
    * int16 samples are scaled by 1/2048 before analysis (12-bit full scale).
    * int32 (phase) samples are used as-is.
    * power = single-sided autopower spectrum (V^2), divided by 50 ohm -> W,
      x1000 -> mW, then 10*log10.
    * PSD additionally divides by the frequency resolution df.
"""

import numpy as np

# Matches the tiny epsilon the demo adds before log10 to avoid log(0).
_EPS = 1e-19
_LOAD_OHM = 50.0
_INT16_FULL_SCALE = 2048.0


def auto_power_spectrum(signal: np.ndarray, sample_rate: float):
    """Single-sided autopower spectrum in V^2 and the frequency resolution df.

    Reproduces NI AutoPowerSpectrum: a Hann window with amplitude correction,
    single-sided scaling (DC and Nyquist not doubled).
    """
    n = len(signal)
    if n == 0:
        return np.zeros(0), 0.0

    window = np.hanning(n)
    # Coherent gain correction so a full-scale sinusoid reads its true amplitude.
    coherent_gain = window.sum() / n
    windowed = signal * window / coherent_gain

    spectrum = np.fft.rfft(windowed)
    df = sample_rate / n

    # |X|^2 scaled to a single-sided amplitude spectrum, then to power (V^2 rms).
    mag = np.abs(spectrum) / n
    mag[1:] *= 2.0  # fold the negative frequencies onto the positive ones
    power_v2 = (mag ** 2) / 2.0  # rms power for a sinusoid of amplitude `mag`
    return power_v2, df


def _to_volts(signal: np.ndarray) -> np.ndarray:
    if np.issubdtype(signal.dtype, np.integer) and signal.dtype.itemsize <= 2:
        return signal.astype(np.float64) / _INT16_FULL_SCALE
    return signal.astype(np.float64)


def power_spectrum_dbm(signal: np.ndarray, sample_rate: float, psd: bool = False):
    """Return (spectrum_dbm, df).

    spectrum_dbm is in dBm when psd is False, dBm/Hz when psd is True.
    """
    volts = _to_volts(signal)
    power_v2, df = auto_power_spectrum(volts, sample_rate)
    if len(power_v2) == 0:
        return np.zeros(0), 0.0

    watts = power_v2 / _LOAD_OHM
    if psd and df > 0:
        watts = watts / df
    milliwatts = watts * 1000.0
    return 10.0 * np.log10(milliwatts + _EPS), df
