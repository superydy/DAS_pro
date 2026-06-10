"""Main application window for DAS_pro.

A modern reimplementation of the ETH-5520 CVI demo's control panel: connection,
acquisition and phase-demodulation parameters, three real-time plots (waveform,
spectrum, amplitude monitor) and data recording.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
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
from ..protocol.constants import (
    BASE_SAMPLE_RATE,
    DEFAULT_IP,
    DEFAULT_PORT,
    DataSrc,
    DataType,
)
from .worker import AcquisitionWorker, StreamSettings, deinterleave

pg.setConfigOptions(antialias=True, background="#101418", foreground="#d0d0d0")

_PLOT_PENS = ["#f2c14e", "#ffffff", "#e8505b", "#4d9de0"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DAS_pro — ETH-5520 上位机")
        self.resize(1280, 820)

        self._client: DasClient | None = None
        self._thread: QThread | None = None
        self._worker: AcquisitionWorker | None = None
        self._recording_file = None
        self._file_index = 0
        self._frame_count = 0

        self._build_ui()
        self._on_data_src_changed()

    # --- UI construction ---

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        controls = self._build_controls()
        root.addLayout(controls, 0)
        root.addLayout(self._build_plots(), 1)

    def _build_controls(self) -> QVBoxLayout:
        col = QVBoxLayout()

        # Connection
        conn = QGroupBox("连接")
        form = QFormLayout(conn)
        self.ip_edit = QLineEdit(DEFAULT_IP)
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(DEFAULT_PORT)
        self.status_label = QLabel("● 未连接")
        self.status_label.setStyleSheet("color:#e8505b;font-weight:bold")
        form.addRow("板卡 IP", self.ip_edit)
        form.addRow("端口", self.port_spin)
        form.addRow("状态", self.status_label)
        col.addWidget(conn)

        # Acquisition
        acq = QGroupBox("采集参数")
        af = QFormLayout(acq)
        self.clock_src = QComboBox()
        self.clock_src.addItems(["外部参考 (0)", "内置参考 (1)"])
        self.trig_dir = QComboBox()
        self.trig_dir.addItems(["接收触发 (0)", "发送触发 (1)"])
        self.trig_freq = self._spin(1, 10_000_000, 1000, " Hz")
        self.pulse_width = self._spin(1, 100000, 100, " ns")
        self.point_num = self._spin(16, 1_000_000, 1024, step=16)
        self.bypass_point = self._spin(0, 1_000_000, 0)
        self.ch_num = QComboBox()
        self.ch_num.addItems(["1", "2", "4"])
        self.data_src = QComboBox()
        self.data_src.addItem("原始数据 (0)", DataSrc.RAW)
        self.data_src.addItem("IQ 数据 (2)", DataSrc.IQ)
        self.data_src.addItem("arctan&sqrt (3)", DataSrc.ARCTAN_SQRT)
        self.data_src.addItem("相位数据 (4)", DataSrc.PHASE)
        self.data_src.currentIndexChanged.connect(self._on_data_src_changed)
        self.data_rate = self._spin(1, 1024, 1)
        self.center_freq = self._spin(0, 250, 80, " MHz")
        self.frame_num = self._spin(1, 10000, 1)
        af.addRow("时钟源", self.clock_src)
        af.addRow("触发方向", self.trig_dir)
        af.addRow("触发频率", self.trig_freq)
        af.addRow("脉冲宽度", self.pulse_width)
        af.addRow("每次采样点数", self.point_num)
        af.addRow("旁路点数", self.bypass_point)
        af.addRow("上传通道数", self.ch_num)
        af.addRow("上传数据源", self.data_src)
        af.addRow("降采样比", self.data_rate)
        af.addRow("中心频率", self.center_freq)
        af.addRow("帧数", self.frame_num)
        col.addWidget(acq)

        # Phase demodulation
        self.phase_box = QGroupBox("相位解调参数")
        pf = QFormLayout(self.phase_box)
        self.rate2phase = self._spin(1, 1024, 1)
        self.space_avg = self._spin(1, 1024, 1)
        self.space_merge = self._spin(1, 1024, 1)
        self.region_diff = self._spin(1, 1024, 1)
        self.detrend_bw = QDoubleSpinBox()
        self.detrend_bw.setRange(0.0, 1000.0)
        self.detrend_bw.setValue(1.0)
        self.detrend_bw.setSuffix(" Hz")
        self.polar_div = QCheckBox("偏振分集")
        self.phase_bits = QComboBox()
        self.phase_bits.addItems(["32 bit (0)", "16 bit (1)"])
        self.dec_ratio = self._spin(1, 1024, 1)
        self.space_time = QCheckBox("空间模式 (否则时间模式)")
        pf.addRow("解调降采样比", self.rate2phase)
        pf.addRow("空间平均阶数", self.space_avg)
        pf.addRow("空间合并点数", self.space_merge)
        pf.addRow("区域差分阶数", self.region_diff)
        pf.addRow("去趋势带宽", self.detrend_bw)
        pf.addRow("", self.polar_div)
        pf.addRow("相位上传位宽", self.phase_bits)
        pf.addRow("相位降采样比", self.dec_ratio)
        pf.addRow("", self.space_time)
        col.addWidget(self.phase_box)

        # Display / output
        disp = QGroupBox("显示与输出")
        dg = QGridLayout(disp)
        self.spectrum_en = QCheckBox("频谱分析")
        self.psd_en = QCheckBox("功率谱密度 (PSD)")
        self.save_en = QCheckBox("保存数据 (.bin)")
        dg.addWidget(self.spectrum_en, 0, 0)
        dg.addWidget(self.psd_en, 0, 1)
        dg.addWidget(self.save_en, 1, 0)
        col.addWidget(disp)

        # Readouts + start/stop
        out = QGroupBox("运行状态")
        of = QFormLayout(out)
        self.throughput_label = QLabel("0.00 MB/s")
        self.samplerate_label = QLabel("—")
        self.framecount_label = QLabel("0")
        of.addRow("网口吞吐", self.throughput_label)
        of.addRow("采样率", self.samplerate_label)
        of.addRow("接收帧数", self.framecount_label)
        col.addWidget(out)

        self.start_btn = QPushButton("开始采集")
        self.start_btn.setCheckable(True)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.toggled.connect(self._on_start_toggled)
        col.addWidget(self.start_btn)

        col.addStretch(1)
        return col

    def _build_plots(self):
        layout = QVBoxLayout()
        self.graph_wave = pg.PlotWidget(title="波形 / 相位")
        self.graph_spec = pg.PlotWidget(title="频谱")
        self.graph_mon = pg.PlotWidget(title="幅度监测")
        for g in (self.graph_wave, self.graph_spec, self.graph_mon):
            g.showGrid(x=True, y=True, alpha=0.3)
            g.addLegend(offset=(-10, 10))
            layout.addWidget(g)
        return layout

    @staticmethod
    def _spin(lo, hi, val, suffix="", step=1) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        if suffix:
            s.setSuffix(suffix)
        return s

    # --- behavior ---

    def _on_data_src_changed(self) -> None:
        is_phase = self.data_src.currentData() == DataSrc.PHASE
        self.phase_box.setEnabled(is_phase)
        if not is_phase:
            self.space_time.setChecked(False)

    def _current_sample_rate(self) -> float:
        return BASE_SAMPLE_RATE / max(self.data_rate.value(), 1)

    def _on_start_toggled(self, checked: bool) -> None:
        if checked:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        try:
            client = DasClient(self.ip_edit.text().strip(), self.port_spin.value(), timeout=5.0)
            client.connect()
            self._configure_board(client)
            frame_num = self.frame_num.value()
            client.start(frame_num)
        except (DeviceError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "启动失败", str(exc))
            self.start_btn.setChecked(False)
            return

        self._client = client
        self._frame_count = 0
        self.status_label.setText("● 已连接")
        self.status_label.setStyleSheet("color:#5cb85c;font-weight:bold")
        self.samplerate_label.setText(f"{self._current_sample_rate()/1e6:.3f} MSps")
        self._update_throughput()

        if self.save_en.isChecked():
            self._open_recording()

        settings = StreamSettings(
            upload_ch_num=int(self.ch_num.currentText()),
            phase_bits_16=self.phase_bits.currentIndex() == 1,
            data_src=int(self.data_src.currentData()),
        )
        self._thread = QThread()
        self._worker = AcquisitionWorker(client, settings)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()
        self.start_btn.setText("停止采集")
        self._set_controls_enabled(False)

    def _configure_board(self, client: DasClient) -> None:
        client.set_clock_src(self.clock_src.currentIndex())
        client.set_trig_dir(self.trig_dir.currentIndex())
        client.set_trig_freq(self.trig_freq.value())
        client.set_trig_pulse_width(self.pulse_width.value())
        client.set_point_num_per_scan(self.point_num.value())
        client.set_bypass_point_num(self.bypass_point.value())
        client.set_upload_ch_num(int(self.ch_num.currentText()))
        client.set_upload_data_src(int(self.data_src.currentData()))
        client.set_upload_data_rate(self.data_rate.value())
        client.set_center_freq(self.center_freq.value() * 1_000_000)
        client.set_phase_demod_params(
            data_rate_to_phase_dem=self.rate2phase.value(),
            space_avg_order=self.space_avg.value(),
            space_merge_point_num=self.space_merge.value(),
            space_region_diff_order=self.region_diff.value(),
            detrend_filter_bw=self.detrend_bw.value(),
            polarization_diversity_en=int(self.polar_div.isChecked()),
        )
        client.set_phase_upload_bit(self.phase_bits.currentIndex())
        client.set_phase_upload_dec_ratio(self.dec_ratio.value())

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
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
        self._close_recording()
        self.status_label.setText("● 未连接")
        self.status_label.setStyleSheet("color:#e8505b;font-weight:bold")
        self.start_btn.setText("开始采集")
        if self.start_btn.isChecked():
            self.start_btn.setChecked(False)
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (self.ip_edit, self.port_spin, self.data_src, self.ch_num, self.point_num):
            w.setEnabled(enabled)

    def _update_throughput(self) -> None:
        ch = int(self.ch_num.currentText())
        if self.data_src.currentData() != DataSrc.PHASE:
            mbps = self.trig_freq.value() * self.point_num.value() * 2 * ch / 1024 / 1024
        else:
            byte_per_phase = 2 if self.phase_bits.currentIndex() == 1 else 4
            merged = max(self.point_num.value() // max(self.space_merge.value(), 1), 1)
            rate = self.trig_freq.value() / max(self.dec_ratio.value(), 1)
            mbps = rate * merged * byte_per_phase * ch / 1024 / 1024
        self.throughput_label.setText(f"{mbps:.2f} MB/s")

    # --- recording ---

    def _open_recording(self) -> None:
        os.makedirs("save_data", exist_ok=True)
        self._file_index += 1
        stamp = datetime.now().strftime("%H-%M-%S")
        path = os.path.join("save_data", f"{self._file_index}-{stamp}_D.bin")
        self._recording_file = open(path, "wb")

    def _close_recording(self) -> None:
        if self._recording_file is not None:
            self._recording_file.close()
            self._recording_file = None

    # --- frame handling ---

    def _on_frame(self, header, data) -> None:
        self._frame_count += 1
        self.framecount_label.setText(str(self._frame_count))

        if self._recording_file is not None:
            self._recording_file.write(np.asarray(data).tobytes())
            return

        ch = int(self.ch_num.currentText())
        if header.data_type == DataType.AMP_MONITOR:
            self._plot_monitor(data, ch)
        elif header.data_type == DataType.PHASE and self.space_time.isChecked():
            self._plot_space(data, header, ch)
        else:
            self._plot_time(data, header, ch)

    def _plot_time(self, data, header, ch) -> None:
        self.graph_wave.clear()
        channels = deinterleave(np.asarray(data), ch)
        points = header.point_num_per_ch_per_scan
        for idx, chan in enumerate(channels):
            seg = chan[:points]
            self.graph_wave.plot(seg, pen=_PLOT_PENS[idx % len(_PLOT_PENS)], name=f"CH{idx}")

        if self.spectrum_en.isChecked() and len(channels):
            self.graph_spec.clear()
            spec, df = power_spectrum_dbm(
                channels[0][:points], self._current_sample_rate(), self.psd_en.isChecked()
            )
            freqs = np.arange(len(spec)) * df
            self.graph_spec.plot(freqs, spec, pen=_PLOT_PENS[0], name="CH0")

    def _plot_space(self, data, header, ch) -> None:
        # In space mode a fixed spatial index is tracked across frames.
        self.graph_wave.clear()
        channels = deinterleave(np.asarray(data), ch)
        for idx, chan in enumerate(channels):
            self.graph_wave.plot(chan, pen=_PLOT_PENS[idx % len(_PLOT_PENS)], name=f"CH{idx}")

    def _plot_monitor(self, data, ch) -> None:
        self.graph_mon.clear()
        channels = deinterleave(np.asarray(data), ch)
        for idx, chan in enumerate(channels):
            self.graph_mon.plot(chan, pen=_PLOT_PENS[idx % len(_PLOT_PENS)], name=f"CH{idx}")

    def _on_worker_error(self, message: str) -> None:
        QMessageBox.warning(self, "采集错误", message)
        self._stop()

    def closeEvent(self, event) -> None:
        self._stop()
        super().closeEvent(event)
