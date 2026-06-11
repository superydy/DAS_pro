# DAS_pro

A modern, cross-platform host application for the **ETH-5520** gigabit-Ethernet
DAS (Distributed Acoustic Sensing) demodulation board.

This is a clean-room reimplementation of the vendor's LabWindows/CVI demo. It
keeps the exact TCP wire protocol but replaces the closed, Windows-only NI
toolchain with Python + Qt so the tool is easy to run, extend and maintain.

## What the board does

ETH-5520 is a dual-channel, 500 MSps, 12-bit DAS acquisition/demodulation board.
It performs IQ demodulation, coherent-fading suppression, phase unwrapping,
detrend filtering and polarization-diversity processing on-board, and uploads
the result over TCP. The host configures the board with small control commands
and then reads a continuous data stream.

## Protocol summary

The board is a **TCP server** (default `192.168.1.100:5000`); the host is the
client.

**Control command (8 bytes):**

```
0x5A 0xA5 | addr_hi addr_lo | val[31:24] val[23:16] val[15:8] val[7:0]
```

`0x5A 0xA5` is the fixed header, followed by a big-endian 2-byte register
address and a big-endian 4-byte value. The board replies with an 8-byte ACK
(two little-endian uint32 words); the second word is the status, `0` = success.
The register map lives in `src/das_pro/protocol/constants.py`.

**Data stream:** each upload starts with a 16-byte header (four little-endian
uint32 words: identifier, data type, frame count, points-per-channel-per-scan),
followed by the payload. Payload element size depends on the data type:

| data_type | meaning            | element            |
|-----------|--------------------|--------------------|
| 0         | raw                | int16              |
| 2         | IQ                 | int16              |
| 3         | arctan & sqrt      | int16              |
| 4         | phase              | int32 or int16     |
| 5         | amplitude monitor  | uint32             |

## Project layout

```
src/das_pro/
  protocol/   wire protocol: register map, command builders, frame parsing
  device/     TCP client + a board simulator (fake hardware over TCP)
  dsp/        power-spectrum / PSD analysis matching the original demo
  gui/        PySide6 control panel and real-time plots
  app.py      entry point
tests/        byte-level protocol tests
```

## Running without hardware

A simulator implements the full protocol and streams synthetic data, so the app
is fully usable before a real board is on the bench:

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m das_pro.app --simulator
```

The connection field is pre-filled with the simulator's address; press
**开始采集** to start streaming.

To run the simulator as a separate process (e.g. on another machine):

```bash
PYTHONPATH=src python -m das_pro.device.simulator --host 0.0.0.0 --port 5000
```

## Connecting to real hardware

Launch without `--simulator`, set the board's IP/port, and start. The protocol
is identical; only the endpoint changes.

```bash
PYTHONPATH=src python -m das_pro.app
```

## Recorded data format

Enabling **SaveData** writes the raw stream to `save_data/<n>-<time>_D.bin`
exactly as received (no conversion, bit-exact), plus a sidecar
`<n>-<time>_D.json` capturing every acquisition parameter — data source,
dtype, channel count, points per scan, sample rate, start/stop time and the
number of frames received — so any recording is self-describing. The JSON
includes a ready-to-paste numpy expression, e.g.:

```python
import numpy as np
data = np.fromfile("save_data/1-12-00-00_D.bin", dtype="<i2")
data = data.reshape(-1, 204, 2)   # (scans, points, channels)
```

## Building a standalone .exe

The app ships as a single-file desktop executable — no Python required on the
target machine.

**Option A — download from CI:** every push builds `DAS_pro.exe` on GitHub
Actions (workflow "Build Windows EXE"). Open the run on the Actions tab and
download the `DAS_pro-windows` artifact.

**Option B — build locally on Windows:** double-click `build_exe.bat` (or run
`pyinstaller das_pro.spec --noconfirm`). The result is `dist\DAS_pro.exe`.

Run it directly, or with the built-in simulator when no board is attached:

```bat
DAS_pro.exe --simulator
```

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=src pytest
```

## Single-point monitor (AudioEN)

In Phase mode the **AudioEN** checkbox opens a dedicated monitor window:

- **automatic vibration detection** — per-position activity (std of the
  temporal phase difference) with an adaptive median-based threshold;
  the strongest position is flagged and can be tracked automatically;
- live plots of the monitored point: activity vs position, time waveform,
  spectrum;
- **audio playback** of the point through the speakers (QtMultimedia,
  upsampled when the phase rate is below what the sound card accepts);
- **single-point recording** to a user-chosen path as WAV / CSV / float32
  BIN, each with a JSON sidecar — records just one fiber position instead
  of the whole stream.

The simulator injects a 50 Hz vibration at 1/3 of the fiber so the
detector can be exercised without hardware.

## Status

Milestone 2 — validated against real hardware (protocol confirmed
byte-exact); single-point monitor with auto detection, audio and per-point
recording. Next: waterfall view and data playback.
