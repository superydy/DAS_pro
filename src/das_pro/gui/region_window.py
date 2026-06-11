"""Region monitor window (区域监测) with a waterfall view.

Answers "which positions are vibrating, and when" for a chosen stretch
of fiber. Independent of the single-point window: it only consumes the
same decoded phase frames via feed() and shares the detection helpers.

* multi-point detection — every above-threshold position is reported,
  grouped into events (one per vibrating spot);
* waterfall — position × time × intensity heat map, the standard DAS
  display: a vibrating spot is a bright vertical streak, something
  moving along the fiber draws a slanted line;
* event list — click an entry to inspect that position's waveform;
* region recording — saves all positions inside the range (2-D block)
  as float32 with a JSON sidecar, at a user-chosen path.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..dsp.detect import detect_relative, vibration_activity
from .plotutil import make_zoomable, set_labels

_WATERFALL_ROWS = 240   # history depth (newest at the bottom)
_WAVE_SECONDS = 8.0
_WARMUP_FEEDS = 5       # frames spent learning the baseline before alarming
_ALPHA_QUIET = 0.05     # baseline tracking speed for quiet positions
_ALPHA_TRIGGERED = 0.005  # much slower while alarming, so alarms persist


class RegionWindow(QWidget):
    closed = Signal()

    def __init__(self, save_dir: str, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("区域监测 — 瀑布图")
        self.resize(1000, 820)

        self._default_dir = save_dir
        self._fs = 2000.0
        self._positions = 0
        self._baseline: np.ndarray | None = None
        self._warmup = 0
        self._waterfall: np.ndarray | None = None
        self._wave_buf = np.zeros(0)
        self._wave_pos = -1
        self._last_sel = -1

        self._rec_path = ""
        self._rec_file = None
        self._rec_meta_path = ""
        self._rec_meta: dict = {}
        self._rec_lo = 0
        self._rec_hi = 0
        self._rec_scans = 0

        # accumulated event log: ongoing vibrations update their row
        # instead of spamming one row per frame
        self._feed_count = 0
        self._active_events: dict[int, dict] = {}
        self._hint_item = None

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        col = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.ch_combo = QComboBox()
        self.ch_combo.addItem("通道0", 0)
        self.ch_combo.addItem("通道1", 1)
        self.range_lo = QSpinBox()
        self.range_lo.setRange(0, 1_000_000)
        self.range_hi = QSpinBox()
        self.range_hi.setRange(0, 1_000_000)
        for w in (self.range_lo, self.range_hi):
            w.setMinimumWidth(80)
            w.setToolTip("只监测该位置区间，终点设在光纤实际终点以内")
        self.thresh = QDoubleSpinBox()
        self.thresh.setRange(1.1, 1000.0)
        self.thresh.setValue(4.0)
        self.thresh.setToolTip(
            "每个位置和它自己安静时的基线比：超过基线的几倍判定为振动。\n"
            "常噪声区（光纤前端、终点之外）基线本身就高，不会误报"
        )
        for label, w in (
            ("通道", self.ch_combo),
            ("范围", self.range_lo),
            ("~", self.range_hi),
            ("阈值×", self.thresh),
        ):
            bar.addWidget(QLabel(label))
            bar.addWidget(w)
        bar.addStretch(1)
        col.addLayout(bar)

        self.det_label = QLabel("等待数据…")
        self.det_label.setStyleSheet("font-weight:bold;color:#808080")
        col.addWidget(self.det_label)

        self.graph_act = pg.PlotWidget(
            title="振动强度分布（黄=当前，红虚线=报警线；只画监测范围）"
        )
        self.graph_act.showGrid(x=True, y=True, alpha=0.3)
        set_labels(self.graph_act, "光纤位置序号", "活动强度（相位帧间变化）")
        self._marks = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("#ff3030"), pen=None, symbol="t1"
        )
        col.addWidget(self.graph_act, 2)

        self.graph_fall = pg.PlotWidget(title="瀑布图（亮=超过自身基线的倍数）")
        set_labels(self.graph_fall, "光纤位置序号", "时间（行，新数据在下）")
        self.graph_fall.getPlotItem().getViewBox().invertY(True)  # newest at bottom
        self._img = pg.ImageItem(axisOrder="row-major")
        self._img.setColorMap(pg.colormap.get("inferno"))
        self.graph_fall.addItem(self._img)
        col.addWidget(self.graph_fall, 3)

        self.graph_wave = pg.PlotWidget(title="选中点时域波形")
        self.graph_wave.showGrid(x=True, y=True, alpha=0.3)
        set_labels(self.graph_wave, "时间 (秒)", "相位（已去均值）")
        col.addWidget(self.graph_wave, 2)

        make_zoomable(self.graph_act, "振动强度分布", col, 2)
        make_zoomable(self.graph_fall, "瀑布图", col, 3)
        make_zoomable(self.graph_wave, "选中点时域波形", col, 2)

        bottom = QHBoxLayout()

        ev_box = QGroupBox("振动事件记录（累计，最新在上；点击查看该点波形）")
        eb = QVBoxLayout(ev_box)
        self.event_list = QListWidget()
        self.event_list.setMaximumHeight(110)
        self.event_list.itemClicked.connect(self._on_event_clicked)
        eb.addWidget(self.event_list)
        bottom.addWidget(ev_box, 1)

        rec_box = QGroupBox("区域录制（保存范围内全部位置）")
        rb = QVBoxLayout(rec_box)
        row = QHBoxLayout()
        choose_btn = QPushButton("选择保存文件…")
        choose_btn.clicked.connect(self._choose_file)
        self.rec_btn = QPushButton("开始录制")
        self.rec_btn.setCheckable(True)
        self.rec_btn.toggled.connect(self._on_record_toggled)
        row.addWidget(choose_btn)
        row.addWidget(self.rec_btn)
        rb.addLayout(row)
        self.rec_label = QLabel("未选择文件（默认 save_data 目录）")
        rb.addWidget(self.rec_label)
        bottom.addWidget(rec_box, 1)

        col.addLayout(bottom)

    # ------------------------------------------------------------ stream

    def set_stream(self, sample_rate: float) -> None:
        self._fs = max(float(sample_rate), 1.0)
        self._baseline = None
        self._warmup = 0
        self._waterfall = None
        self._wave_buf = np.zeros(0)

    def feed(self, scans: np.ndarray) -> None:
        """One decoded phase frame, shaped (n_scans, positions, channels)."""
        ch = min(self.ch_combo.currentData(), scans.shape[2] - 1)
        block = scans[:, :, ch].astype(np.float64)
        _, points = block.shape
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
        if self._warmup <= _WARMUP_FEEDS:
            events = []

        # baseline learns the quiet level of each position; it adapts
        # slowly where an alarm is active so ongoing vibrations keep
        # alarming instead of being absorbed into "normal"
        alpha = np.where(
            ratio > self.thresh.value(), _ALPHA_TRIGGERED, _ALPHA_QUIET
        )
        self._baseline = (1.0 - alpha) * self._baseline + alpha * act

        if self._warmup <= _WARMUP_FEEDS:
            self.det_label.setText("正在学习背景基线…（几秒后开始检测）")
            self.det_label.setStyleSheet("font-weight:bold;color:#c08000")
        elif events:
            head = "、".join(str(p) for p, _ in events[:6])
            more = f" 等{len(events)}处" if len(events) > 6 else ""
            self.det_label.setText(f"⚠ 检测到 {len(events)} 个振动点：{head}{more}")
            self.det_label.setStyleSheet("font-weight:bold;color:#ff3030")
        else:
            self.det_label.setText("无振动（背景安静）")
            self.det_label.setStyleSheet("font-weight:bold;color:#30a030")
        self._update_event_list(events)

        # waterfall row: how many times each position exceeds its own
        # baseline (1 = quiet); independent of absolute noise levels
        row = np.zeros(points, dtype=np.float32)
        row[lo : hi + 1] = ratio[lo : hi + 1]
        self._waterfall = np.vstack([self._waterfall, row[None, :]])[-_WATERFALL_ROWS:]

        if self.rec_btn.isChecked():
            self._write_record(block)

        sel = self._wave_pos if 0 <= self._wave_pos < points else (
            events[0][0] if events else lo
        )
        if sel != self._last_sel:
            self._wave_buf = np.zeros(0)  # don't mix two positions' history
            self._last_sel = sel
        series = block[:, sel] - block[:, sel].mean()
        maxlen = int(self._fs * _WAVE_SECONDS)
        self._wave_buf = np.concatenate([self._wave_buf, series])[-maxlen:]

        self._plot(events, act, lo, hi, sel)

    def _configure_positions(self, points: int) -> None:
        self._positions = points
        self._baseline = None
        self._waterfall = np.zeros((0, points), dtype=np.float32)
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
        self._wave_pos = int(data)
        self.graph_wave.setTitle(f"选中点时域波形 — 位置 {self._wave_pos}")

    def _plot(self, events, act, lo, hi, sel) -> None:
        # only the watched range is drawn, so the scale isn't dominated
        # by the always-noisy stretches outside it
        x = np.arange(lo, hi + 1)
        self.graph_act.clear()
        self.graph_act.addItem(self._marks)
        self.graph_act.plot(x, act[lo : hi + 1], pen="#ffff00")
        self.graph_act.plot(
            x,
            self._baseline[lo : hi + 1] * self.thresh.value(),
            pen=pg.mkPen("#ff3030", style=Qt.PenStyle.DashLine),
        )
        self._marks.setData([p for p, _ in events], [act[p] for p, _ in events])

        if len(self._waterfall):
            self._img.setImage(
                self._waterfall,
                autoLevels=False,
                levels=(0.0, max(self.thresh.value() * 2.0, 2.0)),
            )

        self.graph_wave.clear()
        t = np.arange(len(self._wave_buf)) / self._fs
        self.graph_wave.plot(t, self._wave_buf, pen="#ffffff")

    # --------------------------------------------------------- recording

    def _choose_file(self) -> None:
        os.makedirs(self._default_dir, exist_ok=True)
        stamp = datetime.now().strftime("%H-%M-%S")
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择保存文件",
            os.path.join(self._default_dir, f"region-{stamp}.bin"),
            "二进制 float32 (*.bin)",
        )
        if not path:
            return
        if not path.lower().endswith(".bin"):
            path += ".bin"
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
        # the range is frozen at record start so the file keeps one shape
        self._rec_lo = min(self.range_lo.value(), max(self._positions - 1, 0))
        self._rec_hi = min(self.range_hi.value(), max(self._positions - 1, 0))
        if self._rec_hi < self._rec_lo:
            self._rec_lo, self._rec_hi = self._rec_hi, self._rec_lo
        self._rec_scans = 0
        self._rec_file = open(self._rec_path, "wb")
        width = self._rec_hi - self._rec_lo + 1
        self._rec_meta_path = os.path.splitext(self._rec_path)[0] + ".json"
        self._rec_meta = {
            "software": "DAS_pro",
            "kind": "region recording",
            "position_range": [self._rec_lo, self._rec_hi],
            "positions": width,
            "channel": self.ch_combo.currentText(),
            "sample_rate_hz": self._fs,
            "dtype": "<f4",
            "layout": "scan-major: (scans, positions)",
            "numpy_example": f"np.fromfile(f, dtype='<f4').reshape(-1, {width})",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _write_record(self, block: np.ndarray) -> None:
        if self._rec_file is None:
            return
        hi = min(self._rec_hi, block.shape[1] - 1)
        seg = block[:, self._rec_lo : hi + 1].astype("<f4")
        self._rec_file.write(seg.tobytes())
        self._rec_scans += block.shape[0]

    def _close_record(self) -> None:
        if self._rec_file is not None:
            self._rec_file.close()
            self._rec_file = None
        if self._rec_meta:
            self._rec_meta["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self._rec_meta["scans"] = self._rec_scans
            with open(self._rec_meta_path, "w", encoding="utf-8") as f:
                json.dump(self._rec_meta, f, ensure_ascii=False, indent=2)
            self._rec_meta = {}
            self.rec_label.setText(
                f"已保存: {self._rec_path}（{self._rec_scans} 次扫描）"
            )

    # ------------------------------------------------------------- close

    def closeEvent(self, event) -> None:
        if self.rec_btn.isChecked():
            self.rec_btn.setChecked(False)
        self.closed.emit()
        super().closeEvent(event)
