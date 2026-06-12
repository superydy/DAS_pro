"""Main application window for DAS_pro (全中文界面).

布局（参照手绘设计稿）：

* 中部 —— 顶部工具条 + 三张大图（图1 通道0、图2 通道1/频谱、图3 幅度监测）
* 右栏 —— 一列功能按钮（采集参数 / 相位解调 / 振动监测 / 修改板卡IP /
          数字输出）和两个常驻面板（帧信息、频谱开关）
* 底部 —— 板卡地址、连接灯、开始采集 / 退出

参数本体放在 params.py 的数据类里，由 dialogs.py 的弹窗编辑；监测窗口
（monitor_window）只通过 feed() 接收解码后的相位帧。
各模块互不引用，便于单独扩展。
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..device.client import DasClient, DeviceError
from ..dsp.spectrum import power_spectrum_dbm
from ..protocol.constants import DEFAULT_PORT, DataSrc, DataType
from .dialogs import (
    AcquisitionDialog,
    ConfUserIpDialog,
    DigitalOutDialog,
    PhaseDemodDialog,
)
from .monitor_window import MonitorWindow
from .params import (
    AcquisitionParams,
    PhaseDemodParams,
    fiber_len_km,
    throughput_mb_s,
)
from .plotutil import make_zoomable, set_labels as _label
from .worker import AcquisitionWorker, StreamSettings, deinterleave

# Antialiasing off: live waveforms have up to ~100k points per refresh.
pg.setConfigOptions(antialias=False, background="k", foreground="#d0d0d0")

# Same plot colors as the demo: yellow, white, red, blue (scan 0..3).
_PLOT_PENS = ["#ffff00", "#ffffff", "#ff3030", "#4060ff"]
_MON_PENS = ["#ffff00", "#30c030"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DAS_pro — ETH-5520 分布式声波传感上位机")
        self.resize(1400, 860)
        self.setMinimumSize(1000, 640)

        self.port = DEFAULT_PORT
        self.acq = AcquisitionParams()
        self.demod = PhaseDemodParams()
        self._do_bit_en = 0
        self._do_bit = 0

        self._client: DasClient | None = None
        self._thread: QThread | None = None
        self._worker: AcquisitionWorker | None = None
        self._recording_file = None
        self._file_index = 0
        self._frame_count = 0
        self._monitor: MonitorWindow | None = None
        self._last_draw = 0.0

        # measured (not estimated) stream rate, polled once a second
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(1000)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_prev = (0, 0, 0.0)

        self._build_ui()
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("font-weight:bold")
        self.statusBar().addPermanentWidget(self._stats_label)
        self._after_params_changed()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        body = QHBoxLayout()
        body.addLayout(self._build_center(), 1)
        right = QWidget()
        right.setLayout(self._build_right())
        right.setMinimumWidth(250)
        right.setMaximumWidth(310)
        body.addWidget(right, 0)
        outer.addLayout(body, 1)
        outer.addLayout(self._build_bottom())

    def _build_center(self) -> QVBoxLayout:
        col = QVBoxLayout()

        self.frame_num = self._spin(1, 10000, 500)
        self.save_en = QCheckBox("保存数据")
        self.display_index = QComboBox()
        self.display_index.setMinimumWidth(70)
        self.display_index.addItem("0-1", 0)
        self.display_index.addItem("2-3", 1)
        self.throughput_label = QLabel("0.00 MB/s")
        self.throughput_label.setMinimumWidth(90)
        self.space_time = QCheckBox("空间视图")
        self.region_index = self._spin(0, 1_000_000, 100)

        bar = QHBoxLayout()
        for label, w in (
            ("每包扫描数", self.frame_num),
            ("", self.save_en),
            ("显示通道", self.display_index),
            ("网络速率", self.throughput_label),
            ("", self.space_time),
            ("观察位置", self.region_index),
        ):
            if label:
                bar.addWidget(QLabel(label))
            bar.addWidget(w)
            bar.addSpacing(14)
        bar.addStretch(1)
        col.addLayout(bar)

        self.graph1 = pg.PlotWidget(title="图1 · 通道0 波形")
        self.graph2 = pg.PlotWidget(title="图2 · 通道1 波形 / 频谱")
        for gph in (self.graph1, self.graph2):
            gph.showGrid(x=True, y=True, alpha=0.3)
            _label(gph, "采样点序号（对应光纤位置）", "幅度（ADC 码值）")
        col.addWidget(self.graph1, 3)
        col.addWidget(self.graph2, 3)
        make_zoomable(self.graph1, "图1 · 通道0 波形", col, 3)
        make_zoomable(self.graph2, "图2 · 通道1 波形 / 频谱", col, 3)

        mon_bar = QHBoxLayout()
        mon_bar.addWidget(QLabel("幅度监测"))
        self.ch0_amp_disp = QCheckBox("通道0显示")
        self.ch0_amp_disp.setChecked(True)
        self.ch1_amp_disp = QCheckBox("通道1显示")
        mon_bar.addWidget(self.ch0_amp_disp)
        mon_bar.addWidget(self.ch1_amp_disp)
        mon_bar.addStretch(1)
        col.addLayout(mon_bar)

        self.graph_mon = pg.PlotWidget(title="图3 · 光纤回波强度")
        self.graph_mon.showGrid(x=True, y=True, alpha=0.3)
        _label(self.graph_mon, "光纤位置序号", "回波强度")
        col.addWidget(self.graph_mon, 2)
        make_zoomable(self.graph_mon, "图3 · 光纤回波强度", col, 2)
        return col

    def _build_right(self) -> QVBoxLayout:
        col = QVBoxLayout()

        self.acq_btn = QPushButton("采集参数…")
        self.acq_btn.clicked.connect(self._on_acq_dialog)
        self.demod_btn = QPushButton("相位解调…")
        self.demod_btn.clicked.connect(self._on_demod_dialog)
        for b in (self.acq_btn, self.demod_btn):
            b.setMinimumHeight(34)
            col.addWidget(b)

        info = QGroupBox("帧信息（板卡上报，只读）")
        f = QFormLayout(info)
        self.lbl_identifier = self._readout()
        self.lbl_data_type = self._readout()
        self.lbl_frame_num = self._readout()
        self.lbl_point_num = self._readout()
        self.lbl_read_points = self._readout()
        self.lbl_frame_cnt = self._readout()
        f.addRow("设备标识", self.lbl_identifier)
        f.addRow("数据类型", self.lbl_data_type)
        f.addRow("每包扫描数", self.lbl_frame_num)
        f.addRow("每扫描点数", self.lbl_point_num)
        f.addRow("实收点数", self.lbl_read_points)
        f.addRow("累计收包", self.lbl_frame_cnt)
        col.addWidget(info)

        spec = QGroupBox("频谱")
        s = QHBoxLayout(spec)
        self.spectrum_en = QCheckBox("频谱显示")
        self.psd_en = QCheckBox("功率谱密度")
        s.addWidget(self.spectrum_en)
        s.addWidget(self.psd_en)
        col.addWidget(spec)

        self.monitor_btn = QPushButton("振动监测/音频…")
        self.monitor_btn.clicked.connect(self._on_monitor)
        self.conf_btn = QPushButton("修改板卡IP…")
        self.conf_btn.clicked.connect(self._on_conf_user_ip)
        self.setdo_btn = QPushButton("数字输出…")
        self.setdo_btn.clicked.connect(self._on_set_do)
        for b in (self.monitor_btn, self.conf_btn, self.setdo_btn):
            b.setMinimumHeight(34)
            col.addWidget(b)

        col.addStretch(1)
        return col

    def _build_bottom(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("板卡地址"))
        self.ip_octets = [self._spin(0, 255, v) for v in (192, 168, 1, 88)]
        for s in self.ip_octets:
            s.setMinimumWidth(64)
            bar.addWidget(s)
        bar.addWidget(QLabel(f"端口:{DEFAULT_PORT}"))
        self.led = QLabel("●")
        self.led.setStyleSheet("color:#103010;font-size:18px")
        bar.addWidget(self.led)
        bar.addSpacing(20)

        self.start_btn = QPushButton("开始采集")
        self.start_btn.setCheckable(True)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(160)
        self.start_btn.setStyleSheet(
            "QPushButton{background:#22aa22;color:white;font-weight:bold}"
            "QPushButton:checked{background:#777777}"
        )
        self.start_btn.toggled.connect(self._on_start_toggled)
        quit_btn = QPushButton("退出")
        quit_btn.setMinimumHeight(40)
        quit_btn.setMinimumWidth(100)
        quit_btn.setStyleSheet("background:#cc3322;color:white;font-weight:bold")
        quit_btn.clicked.connect(self.close)
        bar.addWidget(self.start_btn, 1)
        bar.addWidget(quit_btn)
        return bar

    @staticmethod
    def _spin(lo, hi, val, suffix="", step=1) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        if suffix:
            s.setSuffix(suffix)
        s.setMinimumWidth(85)
        return s

    @staticmethod
    def _readout() -> QLineEdit:
        e = QLineEdit("0")
        e.setReadOnly(True)
        e.setAlignment(Qt.AlignmentFlag.AlignRight)
        e.setMaximumWidth(120)
        e.setStyleSheet("background:#f4f4f4")
        return e

    # ----------------------------------------------------- param dialogs

    def _locked(self) -> bool:
        return self._client is not None

    def _on_acq_dialog(self) -> None:
        dlg = AcquisitionDialog(self.acq, self.demod, self._locked(), self)
        if dlg.exec() == QDialog.DialogCode.Accepted and not self._locked():
            self.acq = dlg.values()
            self._after_params_changed()

    def _on_demod_dialog(self) -> None:
        dlg = PhaseDemodDialog(self.demod, self._locked(), self)
        if dlg.exec() == QDialog.DialogCode.Accepted and not self._locked():
            self.demod = dlg.values()
            self._after_params_changed()

    def _after_params_changed(self) -> None:
        is_phase = self.acq.is_phase
        self.space_time.setEnabled(is_phase)
        self.monitor_btn.setEnabled(is_phase)
        tip = "" if is_phase else "数据源选“相位解调”后可用（采集参数里设置）"
        self.monitor_btn.setToolTip(tip)
        if not is_phase:
            self.space_time.setChecked(False)
            if self._monitor is not None:
                self._monitor.close()
        self._update_throughput()
        self.statusBar().showMessage(
            f"光纤长度（计算值）: {fiber_len_km(self.acq, self.demod):.2f} Km"
        )

    def _update_throughput(self) -> None:
        self.throughput_label.setText(
            f"{throughput_mb_s(self.acq, self.demod):.2f} MB/s"
        )

    # ------------------------------------------------- monitor windows

    def _on_monitor(self) -> None:
        if self._monitor is None:
            self._monitor = MonitorWindow(self._save_dir(), self)
        self._monitor.set_stream(self.acq.phase_sample_rate)
        self._monitor.show()
        self._monitor.raise_()

    def _feed_targets(self):
        if self._monitor is not None and self._monitor.isVisible():
            yield self._monitor

    # ----------------------------------------------------------- behavior

    def _board_address(self) -> str:
        return ".".join(str(s.value()) for s in self.ip_octets)

    def _set_led(self, on: bool) -> None:
        self.led.setStyleSheet(
            f"color:{'#30ff30' if on else '#103010'};font-size:18px"
        )

    def _on_start_toggled(self, checked: bool) -> None:
        if checked:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        try:
            client = DasClient(self._board_address(), self.port, timeout=5.0)
            client.connect()
            self._configure_board(client)
            client.start(self.frame_num.value())
        except (DeviceError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "启动失败", str(exc))
            self.start_btn.setChecked(False)
            return

        self._client = client
        self._frame_count = 0
        self._set_led(True)
        self._update_throughput()
        for w in self._feed_targets():
            w.set_stream(self.acq.phase_sample_rate)

        if self.save_en.isChecked():
            self._open_recording()

        settings = StreamSettings(
            upload_ch_num=self.acq.ch_num,
            phase_bits_16=self.acq.phase_bits_16,
            data_src=self.acq.data_src,
        )
        self._thread = QThread()
        self._worker = AcquisitionWorker(
            client, settings, record_file=self._recording_file
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()
        self.start_btn.setText("停止采集")
        self._set_adhoc_enabled(False)
        self._stats_prev = (0, 0, time.monotonic())
        self._stats_timer.start()

    def _configure_board(self, client: DasClient) -> None:
        a, d = self.acq, self.demod
        client.set_clock_src(a.clk_src)
        client.set_trig_dir(a.trig_dir)
        client.set_trig_freq(a.trig_freq)
        client.set_trig_pulse_width(a.trig_width)
        client.set_point_num_per_scan(a.point_num)
        client.set_bypass_point_num(a.bypass_point)
        client.set_upload_ch_num(a.ch_num)
        client.set_upload_data_src(a.data_src)
        client.set_upload_data_rate(a.upload_rate_sel)
        client.set_center_freq(a.center_freq_mhz * 1_000_000)
        client.set_phase_demod_params(
            data_rate_to_phase_dem=d.rate2phase_sel,
            space_avg_order=d.space_avg,
            space_merge_point_num=d.space_merge,
            space_region_diff_order=d.region_diff,
            detrend_filter_bw=d.detrend_bw,
            polarization_diversity_en=d.polar_div,
        )
        client.set_phase_upload_bit(1 if a.phase_bits_16 else 0)
        client.set_phase_upload_dec_ratio(a.dec_ratio)

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            _, self._frame_count = self._worker.take_latest()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        if self._client is not None:
            try:
                self._client.stop()
            except (DeviceError, OSError):
                pass
            self._client.close()
            self._client = None
        self._stats_timer.stop()
        self._stats_label.setText("")
        self._close_recording()
        self._set_led(False)
        self.start_btn.setText("开始采集")
        if self.start_btn.isChecked():
            self.start_btn.setChecked(False)
        self._set_adhoc_enabled(True)

    def _update_stats(self) -> None:
        """Show what is actually arriving — separates network problems
        (numbers far below the estimate) from display problems."""
        if self._worker is None:
            return
        frames, nbytes = self._worker.stats()
        f0, b0, t0 = self._stats_prev
        now = time.monotonic()
        dt = max(now - t0, 1e-6)
        self._stats_label.setText(
            f"实测 {(frames - f0) / dt:.1f} 包/秒 · "
            f"{(nbytes - b0) / dt / 1024 / 1024:.2f} MB/s"
        )
        self._stats_prev = (frames, nbytes, now)

    # --- ad-hoc commands (connect, send, disconnect — like the demo) ---

    def _adhoc_client(self) -> DasClient | None:
        try:
            client = DasClient(self._board_address(), self.port, timeout=3.0)
            client.connect()
            return client
        except OSError as exc:
            QMessageBox.critical(self, "连接失败", str(exc))
            return None

    def _set_adhoc_enabled(self, enabled: bool) -> None:
        tip = "" if enabled else "采集运行中不可用，请先停止采集"
        for btn in (self.conf_btn, self.setdo_btn):
            btn.setEnabled(enabled)
            btn.setToolTip(tip)

    def _on_conf_user_ip(self) -> None:
        octets = tuple(s.value() for s in self.ip_octets[:2]) + (2, 100)
        dlg = ConfUserIpDialog(octets, self._locked(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted or self._locked():
            return
        client = self._adhoc_client()
        if client is None:
            return
        try:
            client.conf_user_ip(*dlg.values())
        except (DeviceError, OSError) as exc:
            QMessageBox.warning(self, "修改板卡IP", str(exc))
        else:
            new_ip = ".".join(str(v) for v in dlg.values())
            QMessageBox.information(
                self,
                "修改板卡IP",
                f"修改命令已发送，板卡新地址：{new_ip}\n\n"
                "请注意：\n"
                "1. 新 IP 一般在板卡重新上电后生效；\n"
                "2. 底部的连接地址需同步改为新 IP；\n"
                "3. 新 IP 不能与电脑自身 IP 相同。",
            )
        finally:
            client.close()

    def _on_set_do(self) -> None:
        dlg = DigitalOutDialog(self._do_bit_en, self._do_bit, self._locked(), self)
        if dlg.exec() != QDialog.DialogCode.Accepted or self._locked():
            return
        self._do_bit_en, self._do_bit = dlg.values()
        client = self._adhoc_client()
        if client is None:
            return
        try:
            client.set_do_bit(self._do_bit_en, self._do_bit)
        except (DeviceError, OSError) as exc:
            QMessageBox.warning(self, "数字输出", str(exc))
        finally:
            client.close()

    # --- recording ---

    @staticmethod
    def _save_dir() -> str:
        """save_data anchored next to the executable, not the launch directory."""
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            base = os.getcwd()
        return os.path.join(base, "save_data")

    def _open_recording(self) -> None:
        save_dir = self._save_dir()
        os.makedirs(save_dir, exist_ok=True)
        self._file_index += 1
        stamp = datetime.now().strftime("%H-%M-%S")
        base = os.path.join(save_dir, f"{self._file_index}-{stamp}_D")
        self._recording_file = open(base + ".bin", "wb")
        self._meta_path = base + ".json"
        self._meta = self._recording_metadata()
        self._write_meta()
        self.statusBar().showMessage(f"正在录制: {os.path.abspath(base + '.bin')}")

    def _recording_metadata(self) -> dict:
        """Everything needed to interpret the .bin afterwards."""
        a, d = self.acq, self.demod
        if not a.is_phase:
            dtype = "<i2"
            points_per_scan = a.point_num
            sample_rate_hz = a.sample_rate
        else:
            dtype = "<i2" if a.phase_bits_16 else "<i4"
            points_per_scan = max(a.point_num // max(d.space_merge, 1), 1)
            sample_rate_hz = a.phase_sample_rate
        return {
            "software": "DAS_pro",
            "board": "ETH-5520",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "data_src": {0: "raw", 2: "iq", 3: "arctan_sqrt", 4: "phase"}.get(
                a.data_src, str(a.data_src)
            ),
            "data_src_value": a.data_src,
            "dtype": dtype,
            "channels": a.ch_num,
            "points_per_scan": points_per_scan,
            "scans_per_upload": self.frame_num.value(),
            "sample_rate_hz": sample_rate_hz,
            "trig_freq_hz": a.trig_freq,
            "trig_pulse_width_ns": a.trig_width,
            "total_point_num": a.point_num,
            "bypass_point_num": a.bypass_point,
            "upload_rate_sel": a.upload_rate_sel,
            "center_freq_hz": a.center_freq_mhz * 1_000_000,
            "phase_bits": "16Bit" if a.phase_bits_16 else "32Bit",
            "trig_freq_dec_ratio": a.dec_ratio,
            "space_avg_order": d.space_avg,
            "space_merge_points": d.space_merge,
            "region_diff_order": d.region_diff,
            "detrend_filter_bw_hz": d.detrend_bw,
            "polarization_diversity": "EN" if d.polar_div else "DIS",
            "fiber_len_km": fiber_len_km(a, d),
            "layout": "scan-major; channels interleaved per point",
            "numpy_example": (
                f"np.fromfile(f, dtype='{dtype}')"
                f".reshape(-1, {points_per_scan}, {a.ch_num})"
            ),
        }

    def _write_meta(self) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    def _close_recording(self) -> None:
        if self._recording_file is not None:
            self._recording_file.close()
            self._recording_file = None
            self._meta["stopped_at"] = datetime.now().isoformat(timespec="seconds")
            self._meta["frames_received"] = self._frame_count
            self._write_meta()
            self.statusBar().showMessage(
                f"录制完成: {os.path.abspath(self._meta_path).replace('.json', '.bin')}"
            )

    # --- frame handling ---

    def _on_frame_ready(self) -> None:
        if self._worker is None:
            return
        item, recv_count = self._worker.take_latest()
        if item is None:
            return
        self._frame_count = recv_count
        self._on_frame(*item)

    def _on_frame(self, header, data) -> None:
        ch = self.acq.ch_num

        self.lbl_identifier.setText(str(header.identifier))
        self.lbl_data_type.setText(str(header.data_type))
        self.lbl_frame_num.setText(str(header.frame_num))
        self.lbl_point_num.setText(str(header.point_num_per_ch_per_scan))
        self.lbl_read_points.setText(str(len(data) // max(ch, 1)))
        if header.data_type <= DataType.PHASE:
            self.lbl_frame_cnt.setText(str(self._frame_count))

        if header.data_type == DataType.PHASE:
            points = header.point_num_per_ch_per_scan
            expected = header.frame_num * points * ch
            arr = np.asarray(data)
            if arr.size >= expected:
                scans = arr[:expected].reshape(header.frame_num, points, ch)
                for w in self._feed_targets():
                    w.feed(scans)

        if self._recording_file is not None:
            return  # recording runs in the worker; skip plotting for throughput

        # cap redraw rate: at high frame rates plotting every frame
        # saturates the GUI thread and the whole window stutters
        now = time.monotonic()
        if now - self._last_draw < 0.07:
            return
        self._last_draw = now

        if header.data_type == DataType.AMP_MONITOR:
            self._plot_monitor(data, ch)
        elif header.data_type == DataType.PHASE and self.space_time.isChecked():
            self._plot_space(data, header, ch)
        elif header.data_type == DataType.PHASE:
            self._plot_phase_time(data, header, ch)
        else:
            self._plot_waveform(data, header, ch)

    def _plot_waveform(self, data, header, ch) -> None:
        """Raw / IQ / arctan&sqrt: CH0 on graph1, CH1 on graph2."""
        points = header.point_num_per_ch_per_scan
        channels = deinterleave(np.asarray(data), ch)
        if ch == 4:
            # 显示通道 selects channel pair 0/1 or 2/3 (4-channel IQ mode).
            base = self.display_index.currentData() * 2
            channels = [channels[base], channels[base + 1]]

        self.graph1.clear()
        _label(self.graph1, "采样点序号（对应光纤位置）", "幅度（ADC 码值，约±2048）")
        self.graph1.plot(channels[0][:points], pen=_PLOT_PENS[0])

        self.graph2.clear()
        if self.spectrum_en.isChecked():
            spec, df = power_spectrum_dbm(
                channels[0][:points], self.acq.sample_rate, self.psd_en.isChecked()
            )
            unit = "功率谱密度 (dBm/Hz)" if self.psd_en.isChecked() else "功率 (dBm)"
            _label(self.graph2, "频率 (Hz)", unit)
            self.graph2.plot(np.arange(len(spec)) * df, spec, pen=_PLOT_PENS[0])
        elif len(channels) > 1:
            _label(self.graph2, "采样点序号（对应光纤位置）", "幅度（ADC 码值，约±2048）")
            self.graph2.plot(channels[1][:points], pen=_PLOT_PENS[1])

    def _plot_phase_time(self, data, header, ch) -> None:
        """Phase, time mode: up to 4 consecutive scans overlaid per channel."""
        points = header.point_num_per_ch_per_scan
        arr = np.asarray(data)
        n_scans = min(header.frame_num, 4)
        expected = header.frame_num * points * ch
        if arr.size < expected:
            return
        scans = arr[:expected].reshape(header.frame_num, points, ch)

        self.graph1.clear()
        self.graph2.clear()
        for gph in (self.graph1, self.graph2):
            _label(gph, "光纤位置序号（合并后）", "相位（原始码值）")
        for i in range(n_scans):
            self.graph1.plot(scans[i, :, 0], pen=_PLOT_PENS[i])
            if ch > 1:
                self.graph2.plot(scans[i, :, 1], pen=_PLOT_PENS[i])

    def _plot_space(self, data, header, ch) -> None:
        """Phase, space mode: time series at 观察位置 across scans."""
        points = header.point_num_per_ch_per_scan
        arr = np.asarray(data)
        expected = header.frame_num * points * ch
        if arr.size < expected:
            return
        region = min(self.region_index.value(), points - 1)
        series = arr[:expected].reshape(header.frame_num, points, ch)[:, region, :]

        self.graph1.clear()
        self.graph2.clear()
        _label(self.graph1, "扫描序号（时间方向）", "相位（原始码值）")
        self.graph1.plot(series[:, 0], pen=_PLOT_PENS[0])
        if ch > 1:
            _label(self.graph2, "扫描序号（时间方向）", "相位（原始码值）")
            self.graph2.plot(series[:, 1], pen=_PLOT_PENS[1])
        else:
            spec, df = power_spectrum_dbm(
                series[:, 0], self.acq.phase_sample_rate, self.psd_en.isChecked()
            )
            unit = "功率谱密度 (dBm/Hz)" if self.psd_en.isChecked() else "功率 (dBm)"
            _label(self.graph2, "频率 (Hz)", unit)
            self.graph2.plot(np.arange(len(spec)) * df, spec, pen=_PLOT_PENS[1])

    def _plot_monitor(self, data, ch) -> None:
        self.graph_mon.clear()
        _label(self.graph_mon, "光纤位置序号", "回波强度")
        channels = deinterleave(np.asarray(data), ch)
        if self.ch0_amp_disp.isChecked():
            self.graph_mon.plot(channels[0], pen=_MON_PENS[0])
        if ch > 1 and self.ch1_amp_disp.isChecked():
            self.graph_mon.plot(channels[1], pen=_MON_PENS[1])

    def _on_worker_error(self, message: str) -> None:
        QMessageBox.warning(self, "采集错误", message)
        self._stop()

    def closeEvent(self, event) -> None:
        self._stop()
        if self._monitor is not None:
            self._monitor.close()
        super().closeEvent(event)
