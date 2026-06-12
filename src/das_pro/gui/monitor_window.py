"""Vibration monitor / audio window (振动监测).

Receives every decoded phase frame from the main window and provides:

* automatic vibration detection — per-position activity along the fiber
  with an adaptive threshold; the strongest position is flagged and can
  be tracked automatically;
* three live plots: activity vs position, the monitored point's time
  waveform, and its spectrum;
* audio playback of the monitored point through the PC speakers;
* single-point recording — only the monitored position's time series is
  written, to a user-chosen path, as WAV audio / CSV text / float32 BIN
  (BIN/CSV get a JSON sidecar describing the recording).
"""

from __future__ import annotations

import json
import os
import time
import wave
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

try:  # QtMultimedia may be missing from stripped-down installs
    from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices

    _HAS_AUDIO = True
except ImportError:  # pragma: no cover
    _HAS_AUDIO = False

from ..dsp.detect import detect_relative, vibration_activity
from ..dsp.spectrum import power_spectrum_dbm
from .plotutil import make_zoomable, set_labels

_WAVE_SECONDS = 8.0  # rolling time-waveform window
_SPECTRUM_SAMPLES = 4096
_WARMUP_FEEDS = 5       # frames spent learning the baseline before alarming
_ALPHA_QUIET = 0.05     # baseline tracking speed for quiet positions
_ALPHA_TRIGGERED = 0.005  # much slower while alarming, so alarms persist


class MonitorWindow(QWidget):
    closed = Signal()

    def __init__(self, save_dir: str, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("振动监测 / 音频")
        self.resize(960, 780)

        self._default_dir = save_dir
        self._fs = 2000.0
        self._positions = 0
        self._baseline: np.ndarray | None = None
        self._warmup = 0
        self._wave_buf = np.zeros(0)
        self._last_sel = -1
        self._last_draw = 0.0

        # accumulated event log: ongoing vibrations update their row
        # instead of spamming one row per frame
        self._feed_count = 0
        self._active_events: dict[int, dict] = {}
        self._hint_item = None

        # audio
        self._sink = None
        self._sink_io = None
        self._out_rate = 0
        self._agc = 1.0

        # recording
        self._rec_path = ""
        self._rec_wav: wave.Wave_write | None = None
        self._rec_file = None
        self._rec_meta_path = ""
        self._rec_meta: dict = {}
        self._rec_pos = 0
        self._rec_samples = 0
        self._rec_gain = 1.0

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        col = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.ch_combo = QComboBox()
        self.ch_combo.addItems(["通道0", "通道1"])
        self.pos_spin = QSpinBox()
        self.pos_spin.setRange(0, 1_000_000)
        self.pos_spin.setMinimumWidth(85)
        self.auto_track = QCheckBox("自动跟踪振动点")
        self.auto_track.setChecked(True)
        self.thresh = QDoubleSpinBox()
        self.thresh.setRange(1.1, 1000.0)
        self.thresh.setValue(4.0)
        self.thresh.setToolTip(
            "每个位置和它自己安静时的基线比：超过基线的几倍判定为振动。\n"
            "常噪声区（光纤前端、终点之外）基线本身就高，不会误报"
        )
        self.range_lo = QSpinBox()
        self.range_lo.setRange(0, 1_000_000)
        self.range_hi = QSpinBox()
        self.range_hi.setRange(0, 1_000_000)
        for w in (self.range_lo, self.range_hi):
            w.setMinimumWidth(75)
            w.setToolTip("只在该位置区间内检测，用于排除光纤末端以外的噪声区")
        for label, w in (
            ("通道", self.ch_combo),
            ("监测位置", self.pos_spin),
            ("", self.auto_track),
            ("阈值×", self.thresh),
            ("范围", self.range_lo),
            ("~", self.range_hi),
        ):
            if label:
                bar.addWidget(QLabel(label))
            bar.addWidget(w)
        bar.addStretch(1)
        col.addLayout(bar)

        self.det_label = QLabel("等待数据…")
        self.det_label.setStyleSheet("font-weight:bold;color:#808080")
        col.addWidget(self.det_label)

        self.graph_act = pg.PlotWidget(title="振动强度分布")
        self.graph_wave = pg.PlotWidget(title="监测点时域波形")
        self.graph_spec = pg.PlotWidget(title="监测点频谱")
        set_labels(self.graph_act, "光纤位置序号", "活动强度（相位帧间变化）")
        set_labels(self.graph_wave, "时间 (秒)", "相位（已去均值）")
        set_labels(self.graph_spec, "频率 (Hz)", "功率 (dBm)")
        for gph, title in (
            (self.graph_act, "振动强度分布"),
            (self.graph_wave, "监测点时域波形"),
            (self.graph_spec, "监测点频谱"),
        ):
            gph.showGrid(x=True, y=True, alpha=0.3)
            col.addWidget(gph, 1)
            make_zoomable(gph, title, col, 1)
        self._peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff3030"))
        self.graph_act.addItem(self._peak_line)
        self._peak_line.hide()
        self.graph_act.setTitle("振动强度分布（黄=当前，红虚线=报警线；只画监测范围）")

        ev_box = QGroupBox("振动事件记录（累计，最新在上；点击查看该点波形）")
        eb = QVBoxLayout(ev_box)
        self.event_list = QListWidget()
        self.event_list.setMaximumHeight(96)
        self.event_list.itemClicked.connect(self._on_event_clicked)
        eb.addWidget(self.event_list)
        col.addWidget(ev_box)

        bottom = QHBoxLayout()

        audio_box = QGroupBox("音频")
        ab = QHBoxLayout(audio_box)
        self.play_chk = QCheckBox("播放声音")
        self.play_chk.toggled.connect(self._on_play_toggled)
        if not _HAS_AUDIO:
            self.play_chk.setEnabled(False)
            self.play_chk.setToolTip("当前环境缺少 QtMultimedia，无法播放")
        ab.addWidget(self.play_chk)
        bottom.addWidget(audio_box)

        rec_box = QGroupBox("单点录制（只保存监测位置的数据）")
        rb = QHBoxLayout(rec_box)
        choose_btn = QPushButton("选择保存文件…")
        choose_btn.clicked.connect(self._choose_file)
        self.rec_btn = QPushButton("开始录制")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._on_record_toggled)
        self.rec_label = QLabel("未选择文件（默认 save_data 目录）")
        rb.addWidget(choose_btn)
        rb.addWidget(self.rec_btn)
        rb.addWidget(self.rec_label, 1)
        bottom.addWidget(rec_box, 1)

        col.addLayout(bottom)

    # ------------------------------------------------------------ stream

    def set_stream(self, sample_rate: float) -> None:
        """Called by the main window when acquisition (re)starts."""
        self._fs = max(float(sample_rate), 1.0)
        self._baseline = None
        self._warmup = 0
        self._wave_buf = np.zeros(0)
        self._agc = 1.0

    def feed(self, scans: np.ndarray) -> None:
        """One decoded phase frame, shaped (n_scans, positions, channels)."""
        ch = min(self.ch_combo.currentIndex(), scans.shape[2] - 1)
        block = scans[:, :, ch].astype(np.float64)
        n_scans, points = block.shape
        if points != self._positions:
            self._configure_positions(points)

        act = vibration_activity(block)
        if self._baseline is None or self._baseline.shape != act.shape:
            self._baseline = act.copy()
            self._warmup = 0
        self._warmup += 1

        events, ratio = detect_relative(act, self._baseline, self.thresh.value())

        lo = min(self.range_lo.value(), points - 1)
        hi = min(self.range_hi.value(), points - 1)
        if hi < lo:
            lo, hi = hi, lo
        events = [(p, v) for p, v in events if lo <= p <= hi]
        hit = bool(events) and self._warmup > _WARMUP_FEEDS
        peak = events[0][0] if events else lo

        alpha = np.where(
            ratio > self.thresh.value(), _ALPHA_TRIGGERED, _ALPHA_QUIET
        )
        self._baseline = (1.0 - alpha) * self._baseline + alpha * act

        if self._warmup <= _WARMUP_FEEDS:
            self.det_label.setText("正在学习背景基线…（几秒后开始检测）")
            self.det_label.setStyleSheet("font-weight:bold;color:#c08000")
        elif hit:
            head = "、".join(str(p) for p, _ in events[:6])
            more = f" 等{len(events)}处" if len(events) > 6 else ""
            self.det_label.setText(
                f"⚠ 检测到 {len(events)} 处振动：{head}{more}"
                f"（跟踪位置 {peak}，超基线 {ratio[peak]:.1f} 倍）"
            )
            self.det_label.setStyleSheet("font-weight:bold;color:#ff3030")
            if self.auto_track.isChecked() and not self.rec_btn.isChecked():
                self.pos_spin.setValue(peak)
        else:
            self.det_label.setText("无振动（背景安静）")
            self.det_label.setStyleSheet("font-weight:bold;color:#30a030")

        sel = min(self.pos_spin.value(), points - 1)
        if sel != self._last_sel:
            self._wave_buf = np.zeros(0)  # don't mix two positions' history
            self._last_sel = sel
            self.graph_wave.setTitle(f"监测点时域波形 — 位置 {sel}")
            self.graph_spec.setTitle(f"监测点频谱 — 位置 {sel}")
        series = block[:, sel]

        if self.rec_btn.isChecked():
            rec_series = series if self._rec_pos == sel else block[:, self._rec_pos]
            self._write_record(rec_series)

        centered = series - series.mean()
        maxlen = int(self._fs * _WAVE_SECONDS)
        self._wave_buf = np.concatenate([self._wave_buf, centered])[-maxlen:]

        if self._sink_io is not None:
            self._play(centered)

        self._update_event_list(events if self._warmup > _WARMUP_FEEDS else [])

        # cap redraw rate: at high frame rates plotting every frame
        # saturates the GUI thread and the whole app stutters; audio and
        # recording above still run for every delivered frame
        now = time.monotonic()
        if now - self._last_draw >= 0.1:
            self._last_draw = now
            self._plot(act, lo, hi, peak, hit)

    def _configure_positions(self, points: int) -> None:
        self._positions = points
        self._baseline = None
        self.pos_spin.setMaximum(points - 1)
        for w in (self.range_lo, self.range_hi):
            w.setMaximum(points - 1)
        if self.range_hi.value() == 0:
            self.range_hi.setValue(points - 1)

    # --- accumulated event log ---

    def _update_event_list(self, events: list[tuple[int, float]]) -> None:
        """Append new vibrations as history rows; ongoing ones update in place."""
        self._feed_count += 1

        if self._hint_item is None and self.event_list.count() == 0 and not events:
            self._hint_item = QListWidgetItem("（暂无振动事件）")
            self._hint_item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.event_list.addItem(self._hint_item)

        for pos, val in events:
            rep = next(
                (r for r in self._active_events if abs(r - pos) <= 3), None
            )
            if rep is None:
                if self._hint_item is not None:
                    self.event_list.takeItem(self.event_list.row(self._hint_item))
                    self._hint_item = None
                stamp = datetime.now().strftime("%H:%M:%S")
                item = QListWidgetItem(f"{stamp}  位置 {pos}（强度 {val:.0f}）")
                item.setData(Qt.ItemDataRole.UserRole, pos)
                self.event_list.insertItem(0, item)
                self._active_events[pos] = {
                    "item": item, "last": self._feed_count, "peak": val,
                    "stamp": stamp,
                }
            else:
                entry = self._active_events[rep]
                entry["last"] = self._feed_count
                if val > entry["peak"]:
                    entry["peak"] = val
                    entry["item"].setText(
                        f"{entry['stamp']}  位置 {rep}（峰值强度 {val:.0f}）"
                    )

        # an event ends after a few quiet frames; its row stays as history
        for rep in list(self._active_events):
            if self._feed_count - self._active_events[rep]["last"] > 5:
                del self._active_events[rep]

        while self.event_list.count() > 200:
            self.event_list.takeItem(200)

    def _on_event_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return
        # inspecting a历史 position: stop auto-track so it stays put
        self.auto_track.setChecked(False)
        self.pos_spin.setValue(int(data))

    def _plot(self, act: np.ndarray, lo: int, hi: int, peak: int, hit: bool) -> None:
        # only the watched range is drawn, so the scale isn't dominated
        # by the always-noisy stretches outside it
        x = np.arange(lo, hi + 1)
        self.graph_act.clear()
        self.graph_act.addItem(self._peak_line)
        self.graph_act.plot(x, act[lo : hi + 1], pen="#ffff00")
        self.graph_act.plot(
            x,
            self._baseline[lo : hi + 1] * self.thresh.value(),
            pen=pg.mkPen("#ff3030", style=Qt.PenStyle.DashLine),
        )
        if hit:
            self._peak_line.setValue(peak)
            self._peak_line.show()
        else:
            self._peak_line.hide()

        self.graph_wave.clear()
        t = np.arange(len(self._wave_buf)) / self._fs
        self.graph_wave.plot(t, self._wave_buf, pen="#ffffff")

        self.graph_spec.clear()
        seg = self._wave_buf[-_SPECTRUM_SAMPLES:]
        if len(seg) >= 16:
            spec, df = power_spectrum_dbm(seg, self._fs, False)
            self.graph_spec.plot(np.arange(len(spec)) * df, spec, pen="#30c030")

    # ------------------------------------------------------------- audio

    def _on_play_toggled(self, on: bool) -> None:
        if not on:
            self._stop_audio()
            return
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            QMessageBox.warning(self, "音频", "未找到音频输出设备")
            self.play_chk.setChecked(False)
            return
        fmt = QAudioFormat()
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        rate = int(self._fs)
        fmt.setSampleRate(rate)
        if rate < 8000 or not device.isFormatSupported(fmt):
            # phase sample rates (e.g. 2 kHz) are below what sound cards
            # accept; upsample to the device's preferred rate instead
            self._out_rate = max(device.preferredFormat().sampleRate(), 8000)
            fmt.setSampleRate(self._out_rate)
        else:
            self._out_rate = rate
        self._sink = QAudioSink(device, fmt)
        # ~1 s of int16 mono: frames arrive in chunks (e.g. 0.25 s every
        # 0.25 s); the default buffer is far smaller and drains to silence
        # between chunks
        self._sink.setBufferSize(self._out_rate * 2)
        self._sink_io = self._sink.start()
        if self._sink_io is None:
            QMessageBox.warning(self, "音频", "音频输出启动失败")
            self.play_chk.setChecked(False)
            self._sink = None

    def _stop_audio(self) -> None:
        if self._sink is not None:
            self._sink.stop()
        self._sink = None
        self._sink_io = None

    def _play(self, centered: np.ndarray) -> None:
        # slow RMS-tracked gain: the background sets the level over a few
        # seconds, so a knock is genuinely louder than the hiss (per-chunk
        # peak normalization made every sound equally loud)
        rms = float(np.sqrt(np.mean(centered**2))) if centered.size else 0.0
        self._agc = max(0.97 * self._agc + 0.03 * (rms * 5.0), 1.0)
        scaled = centered / self._agc * 8000.0
        if self._out_rate != int(self._fs) and centered.size > 1:
            n_out = max(int(len(scaled) * self._out_rate / self._fs), 1)
            scaled = np.interp(
                np.linspace(0.0, len(scaled) - 1.0, n_out),
                np.arange(len(scaled)),
                scaled,
            )
        data = np.clip(scaled, -32767, 32767).astype("<i2").tobytes()
        free = self._sink.bytesFree()
        if free > 0:
            self._sink_io.write(data[:free])

    # --------------------------------------------------------- recording

    def _choose_file(self) -> None:
        os.makedirs(self._default_dir, exist_ok=True)
        stamp = datetime.now().strftime("%H-%M-%S")
        path, selected = QFileDialog.getSaveFileName(
            self,
            "选择保存文件",
            os.path.join(self._default_dir, f"point-{stamp}"),
            "WAV 音频 (*.wav);;CSV 文本 (*.csv);;二进制 float32 (*.bin)",
        )
        if not path:
            return
        ext = {"WAV": ".wav", "CSV": ".csv", "二进制": ".bin"}[selected.split()[0]]
        if not path.lower().endswith(ext):
            path += ext
        self._rec_path = path
        self.rec_label.setText(path)

    def _on_record_toggled(self, on: bool) -> None:
        if on:
            if not self._rec_path:
                self._choose_file()
            if not self._rec_path:
                self.rec_btn.setChecked(False)
                return
            try:
                self._open_record()
            except OSError as exc:
                QMessageBox.warning(self, "录制", f"无法创建文件：{exc}")
                self.rec_btn.setChecked(False)
                return
            self.rec_btn.setText("停止录制")
            self.rec_label.setText(f"录制中: {self._rec_path}")
        else:
            self._close_record()
            self.rec_btn.setText("开始录制")

    def _open_record(self) -> None:
        # the recorded position is frozen at record start so one file is
        # always one fiber location, even with auto-track enabled
        self._rec_pos = self.pos_spin.value()
        self._rec_samples = 0
        ext = os.path.splitext(self._rec_path)[1].lower()
        if ext == ".wav":
            peak = float(np.max(np.abs(self._wave_buf))) if self._wave_buf.size else 0.0
            self._rec_gain = 30000.0 / max(peak, 1.0)
            self._rec_wav = wave.open(self._rec_path, "wb")
            self._rec_wav.setnchannels(1)
            self._rec_wav.setsampwidth(2)
            self._rec_wav.setframerate(max(int(self._fs), 1))
        elif ext == ".csv":
            self._rec_file = open(self._rec_path, "w", encoding="utf-8")
            self._rec_file.write("phase\n")
        else:
            self._rec_file = open(self._rec_path, "wb")
        self._rec_meta_path = os.path.splitext(self._rec_path)[0] + ".json"
        self._rec_meta = {
            "software": "DAS_pro",
            "kind": "single-point recording",
            "position_index": self._rec_pos,
            "channel": self.ch_combo.currentText(),
            "sample_rate_hz": self._fs,
            "format": ext.lstrip("."),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        if ext == ".wav":
            self._rec_meta["wav_gain"] = self._rec_gain
            self._rec_meta["note"] = "WAV 为去均值并按 wav_gain 缩放后的 int16"
        elif ext == ".bin":
            self._rec_meta["dtype"] = "<f4"
            self._rec_meta["numpy_example"] = "np.fromfile(f, dtype='<f4')"

    def _write_record(self, series: np.ndarray) -> None:
        if self._rec_wav is not None:
            centered = series - series.mean()
            samples = np.clip(centered * self._rec_gain, -32767, 32767)
            self._rec_wav.writeframes(samples.astype("<i2").tobytes())
        elif self._rec_file is not None:
            if self._rec_path.lower().endswith(".csv"):
                self._rec_file.write("\n".join(f"{v:.1f}" for v in series) + "\n")
            else:
                self._rec_file.write(series.astype("<f4").tobytes())
        self._rec_samples += len(series)

    def _close_record(self) -> None:
        if self._rec_wav is not None:
            self._rec_wav.close()
            self._rec_wav = None
        if self._rec_file is not None:
            self._rec_file.close()
            self._rec_file = None
        if self._rec_meta:
            self._rec_meta["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self._rec_meta["samples"] = self._rec_samples
            with open(self._rec_meta_path, "w", encoding="utf-8") as f:
                json.dump(self._rec_meta, f, ensure_ascii=False, indent=2)
            self._rec_meta = {}
            self.rec_label.setText(f"已保存: {self._rec_path}（{self._rec_samples} 点）")

    # ------------------------------------------------------------- close

    def closeEvent(self, event) -> None:
        if self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)  # flush + finalize files
        self._stop_audio()
        self.closed.emit()
        super().closeEvent(event)
