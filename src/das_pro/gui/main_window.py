"""Main application window for DAS_pro.

Layout mirrors the original ETH_DAS_DEMO control panel:

* left column   — acquisition + phase-demodulation parameters, board IP,
                  START / QUIT buttons
* center column — top bar (frame num, save, display index, throughput,
                  space/time, region index) and three plots: waveform,
                  waveform2/spectrum, amplitude monitor
* right column  — received frame header readouts, spectrum switches,
                  ConfUserIP and digital-output controls
"""

from __future__ import annotations

import json
import os
import sys
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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..device.client import DasClient, DeviceError
from ..dsp.spectrum import power_spectrum_dbm
from ..protocol.constants import (
    BASE_SAMPLE_RATE,
    DEFAULT_PORT,
    DataSrc,
    DataType,
)
from .worker import AcquisitionWorker, StreamSettings, deinterleave

# Antialiasing off: live waveforms have up to ~100k points per refresh.
pg.setConfigOptions(antialias=False, background="k", foreground="#d0d0d0")

# Same plot colors as the demo: yellow, white, red, blue (scan 0..3).
_PLOT_PENS = ["#ffff00", "#ffffff", "#ff3030", "#4060ff"]
_MON_PENS = ["#ffff00", "#30c030"]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DAS_pro — ETH_DAS")
        self.resize(1400, 860)
        self.setMinimumSize(1000, 640)

        self.port = DEFAULT_PORT
        self._client: DasClient | None = None
        self._thread: QThread | None = None
        self._worker: AcquisitionWorker | None = None
        self._recording_file = None
        self._file_index = 0
        self._frame_count = 0

        self._build_ui()
        self._on_data_src_changed()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        left_inner = QWidget()
        left_inner.setLayout(self._build_left())
        left = QScrollArea()
        left.setWidget(left_inner)
        left.setWidgetResizable(True)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Width derived from actual content so fonts/DPI never clip it.
        need = left_inner.minimumSizeHint().width() + 24  # room for scrollbar
        left.setMinimumWidth(need)
        left.setMaximumWidth(need + 50)
        right = QWidget()
        right.setLayout(self._build_right())
        right.setMinimumWidth(240)
        right.setMaximumWidth(300)
        root.addWidget(left, 0)
        root.addLayout(self._build_center(), 1)
        root.addWidget(right, 0)

    # --- left column ---

    def _build_left(self) -> QVBoxLayout:
        col = QVBoxLayout()

        acq = QGroupBox("采集参数")
        g = QGridLayout(acq)

        self.trig_freq = self._spin(1, 10_000_000, 2000, " Hz")
        self.trig_width = self._spin(1, 100000, 100, " ns")
        self.point_num = self._spin(16, 1_000_000, 5120, step=16)
        self.bypass_point = self._spin(0, 1_000_000, 2)

        self.data_src = QComboBox()
        self.data_src.addItem("RawData", DataSrc.RAW)
        self.data_src.addItem("IQ", DataSrc.IQ)
        self.data_src.addItem("Arctan&Sqrt", DataSrc.ARCTAN_SQRT)
        self.data_src.addItem("Phase", DataSrc.PHASE)
        self.data_src.currentIndexChanged.connect(self._on_data_src_changed)

        self.upload_rate = QComboBox()
        for sel in (1, 2, 4, 5, 10):
            self.upload_rate.addItem(f"{int(500 / sel)}MSps", sel)

        self.ch_num = QComboBox()
        self.ch_num.addItem("One", 1)
        self.ch_num.addItem("Two", 2)
        self.ch_num.addItem("Four", 4)
        self.ch_num.setCurrentIndex(1)

        self.fiber_len = QLabel("0.00 Km")

        self.center_freq = self._spin(0, 250, 80, " MHz")

        self.trig_dir = QComboBox()
        self.trig_dir.addItem("IN", 0)
        self.trig_dir.addItem("OUT", 1)
        self.trig_dir.setCurrentIndex(1)

        self.clk_src = QComboBox()
        self.clk_src.addItem("ExtRef", 0)
        self.clk_src.addItem("OnBoard", 1)
        self.clk_src.setCurrentIndex(1)

        self.phase_bits = QComboBox()
        self.phase_bits.addItem("32Bit", 0)
        self.phase_bits.addItem("16Bit", 1)

        self.dec_ratio = self._spin(1, 1024, 1)

        rows = [
            ("TrigFreq", self.trig_freq, "TrigWidth", self.trig_width),
            ("TotalPointNum", self.point_num, "BypassPointNum", self.bypass_point),
            ("DataSrc", self.data_src, "UploadRate", self.upload_rate),
            ("UploadChNum", self.ch_num, "FiberLen", self.fiber_len),
            ("CenterFreq", self.center_freq, "", None),
            ("TrigDir", self.trig_dir, "ClkSrc", self.clk_src),
            ("PhaseBit", self.phase_bits, "TrigFreqDecRatio", self.dec_ratio),
        ]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            g.addWidget(QLabel(l1), r * 2, 0)
            g.addWidget(w1, r * 2 + 1, 0)
            if w2 is not None:
                g.addWidget(QLabel(l2), r * 2, 1)
                g.addWidget(w2, r * 2 + 1, 1)
        col.addWidget(acq)

        demod = QGroupBox("相位解调")
        d = QGridLayout(demod)
        self.space_avg = self._spin(1, 1024, 25)
        self.space_merge = self._spin(1, 1024, 25)
        self.region_diff = self._spin(1, 1024, 2)
        self.detrend_bw = QDoubleSpinBox()
        self.detrend_bw.setRange(0.0, 100000.0)
        self.detrend_bw.setValue(20.0)
        self.detrend_bw.setSuffix(" Hz")
        self.polar_div = QComboBox()
        self.polar_div.addItem("DIS", 0)
        self.polar_div.addItem("EN", 1)
        self.rate2phase = QComboBox()
        for sel in (1, 2, 4, 5, 10):
            self.rate2phase.addItem(f"{int(500 / sel)}M", sel)
        self.rate2phase.setCurrentIndex(1)
        self.audio_en = QCheckBox("AudioEN")
        self.audio_en.setEnabled(False)
        self.audio_en.setToolTip("音频播放将在后续版本提供")

        drows = [
            ("SpaceAvgOrder", self.space_avg, "SpaceMergePoints", self.space_merge),
            ("RegionDiffOrder", self.region_diff, "DetrendFilterBW", self.detrend_bw),
            ("Polarization", self.polar_div, "Rate2PhaseDem", self.rate2phase),
        ]
        for r, (l1, w1, l2, w2) in enumerate(drows):
            d.addWidget(QLabel(l1), r * 2, 0)
            d.addWidget(w1, r * 2 + 1, 0)
            d.addWidget(QLabel(l2), r * 2, 1)
            d.addWidget(w2, r * 2 + 1, 1)
        d.addWidget(self.audio_en, 6, 0, 1, 2)
        col.addWidget(demod)

        conn = QGroupBox("板卡地址")
        c = QHBoxLayout(conn)
        self.ip_octets = [self._spin(0, 255, v) for v in (192, 168, 1, 88)]
        for s in self.ip_octets:
            c.addWidget(s)
        c.addWidget(QLabel(f"PortNum:{DEFAULT_PORT}"))
        self.led = QLabel("●")
        self.led.setStyleSheet("color:#103010;font-size:18px")
        c.addWidget(self.led)
        col.addWidget(conn)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("START")
        self.start_btn.setCheckable(True)
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setStyleSheet(
            "QPushButton{background:#22aa22;color:white;font-weight:bold}"
            "QPushButton:checked{background:#777777}"
        )
        self.start_btn.toggled.connect(self._on_start_toggled)
        quit_btn = QPushButton("QUIT")
        quit_btn.setMinimumHeight(42)
        quit_btn.setStyleSheet("background:#cc3322;color:white;font-weight:bold")
        quit_btn.clicked.connect(self.close)
        btns.addWidget(self.start_btn)
        btns.addWidget(quit_btn)
        col.addLayout(btns)

        col.addStretch(1)
        return col

    # --- center column ---

    def _build_center(self) -> QVBoxLayout:
        col = QVBoxLayout()

        self.frame_num = self._spin(1, 10000, 500)
        self.save_en = QCheckBox("SaveData")
        self.display_index = QComboBox()
        self.display_index.setMinimumWidth(60)
        self.display_index.addItem("01", 0)
        self.display_index.addItem("23", 1)
        self.throughput_label = QLabel("0.00 MB/s")
        self.throughput_label.setMinimumWidth(90)
        self.space_time = QCheckBox("Space")
        self.region_index = self._spin(0, 1_000_000, 100)

        # Two rows; one row does not fit once Windows display scaling
        # shrinks the logical screen width.
        rows = (
            (("FrameNum", self.frame_num), ("", self.save_en),
             ("DisplayIndex", self.display_index)),
            (("ETH_Throught", self.throughput_label), ("", self.space_time),
             ("RegionIndex", self.region_index)),
        )
        for row_items in rows:
            bar = QHBoxLayout()
            for label, w in row_items:
                if label:
                    bar.addWidget(QLabel(label))
                bar.addWidget(w)
                bar.addSpacing(18)
            bar.addStretch(1)
            col.addLayout(bar)

        self.graph1 = pg.PlotWidget()
        self.graph2 = pg.PlotWidget()
        for gph in (self.graph1, self.graph2):
            gph.showGrid(x=True, y=True, alpha=0.3)
        col.addWidget(self.graph1, 3)
        col.addWidget(self.graph2, 3)

        mon_bar = QHBoxLayout()
        mon_bar.addWidget(QLabel("AMP Monitor"))
        self.ch0_amp_disp = QCheckBox("CH0_Amp_Disp")
        self.ch0_amp_disp.setChecked(True)
        self.ch1_amp_disp = QCheckBox("CH1_Amp_Disp")
        mon_bar.addWidget(self.ch0_amp_disp)
        mon_bar.addWidget(self.ch1_amp_disp)
        mon_bar.addStretch(1)
        col.addLayout(mon_bar)

        self.graph_mon = pg.PlotWidget()
        self.graph_mon.showGrid(x=True, y=True, alpha=0.3)
        col.addWidget(self.graph_mon, 2)
        return col

    # --- right column ---

    def _build_right(self) -> QVBoxLayout:
        col = QVBoxLayout()

        info = QGroupBox("帧信息 (板卡上报, 只读)")
        f = QFormLayout(info)
        self.lbl_identifier = self._readout()
        self.lbl_data_type = self._readout()
        self.lbl_frame_num = self._readout()
        self.lbl_point_num = self._readout()
        self.lbl_read_points = self._readout()
        self.lbl_frame_cnt = self._readout()
        f.addRow("Identifier", self.lbl_identifier)
        f.addRow("DataType", self.lbl_data_type)
        f.addRow("FrameNum", self.lbl_frame_num)
        f.addRow("PointNumPerScan", self.lbl_point_num)
        f.addRow("ReadPointsNum", self.lbl_read_points)
        f.addRow("RecvFrameCnt", self.lbl_frame_cnt)
        col.addWidget(info)

        spec = QGroupBox("频谱")
        s = QHBoxLayout(spec)
        self.spectrum_en = QCheckBox("SpectrumEn")
        self.psd_en = QCheckBox("PSD EN")
        s.addWidget(self.spectrum_en)
        s.addWidget(self.psd_en)
        col.addWidget(spec)

        ipbox = QGroupBox("ConfUserIP")
        i = QGridLayout(ipbox)
        self.conf_ip_octets = [self._spin(0, 255, v) for v in (192, 168, 2, 100)]
        for idx, sb in enumerate(self.conf_ip_octets):
            i.addWidget(sb, 0, idx)
        self.conf_btn = QPushButton("ConfUserIP")
        self.conf_btn.clicked.connect(self._on_conf_user_ip)
        i.addWidget(self.conf_btn, 1, 0, 1, 4)
        col.addWidget(ipbox)

        dobox = QGroupBox("数字输出")
        do = QFormLayout(dobox)
        self.do_bit_en = self._spin(0, 255, 0)
        self.do_bit = self._spin(0, 255, 0)
        do.addRow("DOBitEN", self.do_bit_en)
        do.addRow("DOBit", self.do_bit)
        self.setdo_btn = QPushButton("SetDO")
        self.setdo_btn.clicked.connect(self._on_set_do)
        do.addRow(self.setdo_btn)
        col.addWidget(dobox)

        col.addStretch(1)
        return col

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
        e.setMaximumWidth(110)
        e.setStyleSheet("background:#f4f4f4")
        return e

    # ----------------------------------------------------------- behavior

    def _board_address(self) -> str:
        return ".".join(str(s.value()) for s in self.ip_octets)

    def _set_led(self, on: bool) -> None:
        self.led.setStyleSheet(
            f"color:{'#30ff30' if on else '#103010'};font-size:18px"
        )

    def _on_data_src_changed(self) -> None:
        is_phase = self.data_src.currentData() == DataSrc.PHASE
        self.space_time.setEnabled(is_phase)
        if not is_phase:
            self.space_time.setChecked(False)

    def _current_sample_rate(self) -> float:
        return BASE_SAMPLE_RATE / max(self.upload_rate.currentData(), 1)

    def _phase_sample_rate(self) -> float:
        return self.trig_freq.value() / max(self.dec_ratio.value(), 1)

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
        self._update_fiber_len()

        if self.save_en.isChecked():
            self._open_recording()

        settings = StreamSettings(
            upload_ch_num=self.ch_num.currentData(),
            phase_bits_16=self.phase_bits.currentData() == 1,
            data_src=int(self.data_src.currentData()),
        )
        self._thread = QThread()
        self._worker = AcquisitionWorker(client, settings, record_file=self._recording_file)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()
        self.start_btn.setText("STOP")
        self._set_adhoc_enabled(False)

    def _configure_board(self, client: DasClient) -> None:
        client.set_clock_src(self.clk_src.currentData())
        client.set_trig_dir(self.trig_dir.currentData())
        client.set_trig_freq(self.trig_freq.value())
        client.set_trig_pulse_width(self.trig_width.value())
        client.set_point_num_per_scan(self.point_num.value())
        client.set_bypass_point_num(self.bypass_point.value())
        client.set_upload_ch_num(self.ch_num.currentData())
        client.set_upload_data_src(int(self.data_src.currentData()))
        client.set_upload_data_rate(self.upload_rate.currentData())
        client.set_center_freq(self.center_freq.value() * 1_000_000)
        client.set_phase_demod_params(
            data_rate_to_phase_dem=self.rate2phase.currentData(),
            space_avg_order=self.space_avg.value(),
            space_merge_point_num=self.space_merge.value(),
            space_region_diff_order=self.region_diff.value(),
            detrend_filter_bw=self.detrend_bw.value(),
            polarization_diversity_en=self.polar_div.currentData(),
        )
        client.set_phase_upload_bit(self.phase_bits.currentData())
        client.set_phase_upload_dec_ratio(self.dec_ratio.value())

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
        self._close_recording()
        self._set_led(False)
        self.start_btn.setText("START")
        if self.start_btn.isChecked():
            self.start_btn.setChecked(False)
        self._set_adhoc_enabled(True)

    def _update_throughput(self) -> None:
        ch = self.ch_num.currentData()
        if self.data_src.currentData() != DataSrc.PHASE:
            mbps = self.trig_freq.value() * self.point_num.value() * 2 * ch / 1024 / 1024
        else:
            byte_per_phase = 2 if self.phase_bits.currentData() == 1 else 4
            merged = max(self.point_num.value() // max(self.space_merge.value(), 1), 1)
            mbps = self._phase_sample_rate() * merged * byte_per_phase * ch / 1024 / 1024
        self.throughput_label.setText(f"{mbps:.2f} MB/s")

    def _update_fiber_len(self) -> None:
        # 0.2 m per sample at 500 MSps (round trip), as computed by the demo.
        if self.data_src.currentData() != DataSrc.PHASE:
            km = self.point_num.value() * self.upload_rate.currentData() * 0.2 / 1000.0
        else:
            km = self.point_num.value() * self.rate2phase.currentData() * 0.4 / 1000.0
        self.fiber_len.setText(f"{km:.2f} Km")

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
        tip = "" if enabled else "采集运行中不可用，请先 STOP"
        for btn in (self.conf_btn, self.setdo_btn):
            btn.setEnabled(enabled)
            btn.setToolTip(tip)

    def _on_conf_user_ip(self) -> None:
        client = self._adhoc_client()
        if client is None:
            return
        try:
            client.conf_user_ip(*(s.value() for s in self.conf_ip_octets))
        except (DeviceError, OSError) as exc:
            QMessageBox.warning(self, "ConfUserIP", str(exc))
        else:
            new_ip = ".".join(str(s.value()) for s in self.conf_ip_octets)
            QMessageBox.information(
                self,
                "ConfUserIP",
                f"修改命令已发送，板卡新地址：{new_ip}\n\n"
                "请注意：\n"
                "1. 新 IP 一般在板卡重新上电后生效；\n"
                "2. 左下角的连接地址需同步改为新 IP；\n"
                "3. 新 IP 不能与电脑自身 IP 相同。",
            )
        finally:
            client.close()

    def _on_set_do(self) -> None:
        client = self._adhoc_client()
        if client is None:
            return
        try:
            client.set_do_bit(self.do_bit_en.value(), self.do_bit.value())
        except (DeviceError, OSError) as exc:
            QMessageBox.warning(self, "SetDO", str(exc))
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
        is_phase = self.data_src.currentData() == DataSrc.PHASE
        if not is_phase:
            dtype = "<i2"
            points_per_scan = self.point_num.value()
            sample_rate_hz = self._current_sample_rate()
        else:
            dtype = "<i2" if self.phase_bits.currentData() == 1 else "<i4"
            points_per_scan = max(
                self.point_num.value() // max(self.space_merge.value(), 1), 1
            )
            sample_rate_hz = self._phase_sample_rate()
        channels = self.ch_num.currentData()
        return {
            "software": "DAS_pro",
            "board": "ETH-5520",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "data_src": self.data_src.currentText(),
            "data_src_value": int(self.data_src.currentData()),
            "dtype": dtype,
            "channels": channels,
            "points_per_scan": points_per_scan,
            "scans_per_upload": self.frame_num.value(),
            "sample_rate_hz": sample_rate_hz,
            "trig_freq_hz": self.trig_freq.value(),
            "trig_pulse_width_ns": self.trig_width.value(),
            "total_point_num": self.point_num.value(),
            "bypass_point_num": self.bypass_point.value(),
            "upload_rate_sel": self.upload_rate.currentData(),
            "center_freq_hz": self.center_freq.value() * 1_000_000,
            "phase_bits": self.phase_bits.currentText(),
            "trig_freq_dec_ratio": self.dec_ratio.value(),
            "space_avg_order": self.space_avg.value(),
            "space_merge_points": self.space_merge.value(),
            "region_diff_order": self.region_diff.value(),
            "detrend_filter_bw_hz": self.detrend_bw.value(),
            "polarization_diversity": self.polar_div.currentText(),
            "fiber_len_km": float(self.fiber_len.text().split()[0]),
            "layout": "scan-major; channels interleaved per point",
            "numpy_example": (
                f"np.fromfile(f, dtype='{dtype}')"
                f".reshape(-1, {points_per_scan}, {channels})"
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
            self.statusBar().showMessage(f"录制完成: {os.path.abspath(self._meta_path).replace('.json', '.bin')}")

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
        ch = self.ch_num.currentData()

        self.lbl_identifier.setText(str(header.identifier))
        self.lbl_data_type.setText(str(header.data_type))
        self.lbl_frame_num.setText(str(header.frame_num))
        self.lbl_point_num.setText(str(header.point_num_per_ch_per_scan))
        self.lbl_read_points.setText(str(len(data) // max(ch, 1)))
        if header.data_type <= DataType.PHASE:
            self.lbl_frame_cnt.setText(str(self._frame_count))

        if self._recording_file is not None:
            return  # recording runs in the worker; skip plotting for throughput

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
            # DisplayIndex selects channel pair 0/1 or 2/3 (4-channel IQ mode).
            base = self.display_index.currentData() * 2
            channels = [channels[base], channels[base + 1]]

        self.graph1.clear()
        self.graph1.plot(channels[0][:points], pen=_PLOT_PENS[0])

        self.graph2.clear()
        if self.spectrum_en.isChecked():
            spec, df = power_spectrum_dbm(
                channels[0][:points], self._current_sample_rate(), self.psd_en.isChecked()
            )
            self.graph2.plot(np.arange(len(spec)) * df, spec, pen=_PLOT_PENS[0])
        elif len(channels) > 1:
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
        for i in range(n_scans):
            self.graph1.plot(scans[i, :, 0], pen=_PLOT_PENS[i])
            if ch > 1:
                self.graph2.plot(scans[i, :, 1], pen=_PLOT_PENS[i])

    def _plot_space(self, data, header, ch) -> None:
        """Phase, space mode: time series at RegionIndex across scans."""
        points = header.point_num_per_ch_per_scan
        arr = np.asarray(data)
        expected = header.frame_num * points * ch
        if arr.size < expected:
            return
        region = min(self.region_index.value(), points - 1)
        series = arr[:expected].reshape(header.frame_num, points, ch)[:, region, :]

        self.graph1.clear()
        self.graph2.clear()
        self.graph1.plot(series[:, 0], pen=_PLOT_PENS[0])
        if ch > 1:
            self.graph2.plot(series[:, 1], pen=_PLOT_PENS[1])
        else:
            spec, df = power_spectrum_dbm(
                series[:, 0], self._phase_sample_rate(), self.psd_en.isChecked()
            )
            self.graph2.plot(np.arange(len(spec)) * df, spec, pen=_PLOT_PENS[1])

    def _plot_monitor(self, data, ch) -> None:
        self.graph_mon.clear()
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
        super().closeEvent(event)
