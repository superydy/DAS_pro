"""Parameter dialogs (采集参数 / 相位解调 / 修改板卡IP / 数字输出).

Each dialog is self-contained: it builds its own widgets from a params
object (or plain values), and the caller reads the result back after
exec(). While acquisition is running the dialogs open in read-only mode
so the displayed values always match what the board is doing.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..protocol.constants import DataSrc
from .params import AcquisitionParams, PhaseDemodParams, fiber_len_km


def _spin(lo: int, hi: int, val: int, suffix: str = "", step: int = 1) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumWidth(110)
    return s


class _ParamDialog(QDialog):
    """Shared frame: a grid of label/widget pairs + OK/Cancel."""

    def __init__(self, title: str, locked: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._locked = locked
        self._root = QVBoxLayout(self)
        self._form = QWidget()
        self.grid = QGridLayout(self._form)
        self._root.addWidget(self._form)

    def finish(self) -> None:
        """Add the lock hint and buttons; call after the grid is filled."""
        if self._locked:
            self._form.setEnabled(False)
            hint = QLabel("采集运行中，参数只读 — 先停止采集才能修改")
            hint.setStyleSheet("color:#c06000;font-weight:bold")
            self._root.addWidget(hint)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.reject)
            buttons.clicked.connect(lambda _b: self.reject())
        else:
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok
                | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确定")
            buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self._root.addWidget(buttons)

    def add_row(self, row: int, col: int, label: str, widget) -> None:
        self.grid.addWidget(QLabel(label), row * 2, col)
        self.grid.addWidget(widget, row * 2 + 1, col)


class AcquisitionDialog(_ParamDialog):
    def __init__(
        self,
        params: AcquisitionParams,
        demod: PhaseDemodParams,
        locked: bool,
        parent=None,
    ) -> None:
        super().__init__("采集参数", locked, parent)
        self._demod = demod

        self.trig_freq = _spin(1, 10_000_000, params.trig_freq, " Hz")
        self.trig_width = _spin(1, 100000, params.trig_width, " ns")
        self.point_num = _spin(16, 1_000_000, params.point_num, step=16)
        self.bypass_point = _spin(0, 1_000_000, params.bypass_point)

        self.data_src = QComboBox()
        self.data_src.addItem("原始波形", int(DataSrc.RAW))
        self.data_src.addItem("IQ", int(DataSrc.IQ))
        self.data_src.addItem("反正切&幅值", int(DataSrc.ARCTAN_SQRT))
        self.data_src.addItem("相位解调", int(DataSrc.PHASE))
        self.data_src.setCurrentIndex(self.data_src.findData(params.data_src))

        self.upload_rate = QComboBox()
        for sel in (1, 2, 4, 5, 10):
            self.upload_rate.addItem(f"{int(500 / sel)}MSps", sel)
        self.upload_rate.setCurrentIndex(
            self.upload_rate.findData(params.upload_rate_sel)
        )

        self.ch_num = QComboBox()
        self.ch_num.addItem("单通道", 1)
        self.ch_num.addItem("双通道", 2)
        self.ch_num.addItem("四通道", 4)
        self.ch_num.setCurrentIndex(self.ch_num.findData(params.ch_num))

        self.center_freq = _spin(0, 250, params.center_freq_mhz, " MHz")

        self.trig_dir = QComboBox()
        self.trig_dir.addItem("输入（外触发）", 0)
        self.trig_dir.addItem("输出（板卡主控）", 1)
        self.trig_dir.setCurrentIndex(self.trig_dir.findData(params.trig_dir))

        self.clk_src = QComboBox()
        self.clk_src.addItem("外部10M参考", 0)
        self.clk_src.addItem("板载晶振", 1)
        self.clk_src.setCurrentIndex(self.clk_src.findData(params.clk_src))

        self.phase_bits = QComboBox()
        self.phase_bits.addItem("32位", False)
        self.phase_bits.addItem("16位", True)
        self.phase_bits.setCurrentIndex(1 if params.phase_bits_16 else 0)

        self.dec_ratio = _spin(1, 1024, params.dec_ratio)

        self.fiber_len = QLabel()
        for w in (self.point_num, self.upload_rate, self.data_src):
            sig = w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
            sig.connect(self._update_fiber_len)
        self._update_fiber_len()

        rows = [
            ("触发频率", self.trig_freq, "脉冲宽度", self.trig_width),
            ("总采样点数", self.point_num, "旁路点数", self.bypass_point),
            ("数据源", self.data_src, "上传速率", self.upload_rate),
            ("上传通道数", self.ch_num, "光纤长度（计算值）", self.fiber_len),
            ("中心频率", self.center_freq, "上传抽取比", self.dec_ratio),
            ("触发方向", self.trig_dir, "时钟源", self.clk_src),
            ("相位位宽", self.phase_bits, "", None),
        ]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            self.add_row(r, 0, l1, w1)
            if w2 is not None:
                self.add_row(r, 1, l2, w2)
        self.finish()

    def _update_fiber_len(self) -> None:
        km = fiber_len_km(self.values(), self._demod)
        self.fiber_len.setText(f"{km:.2f} Km")

    def values(self) -> AcquisitionParams:
        return AcquisitionParams(
            trig_freq=self.trig_freq.value(),
            trig_width=self.trig_width.value(),
            point_num=self.point_num.value(),
            bypass_point=self.bypass_point.value(),
            data_src=self.data_src.currentData(),
            upload_rate_sel=self.upload_rate.currentData(),
            ch_num=self.ch_num.currentData(),
            center_freq_mhz=self.center_freq.value(),
            trig_dir=self.trig_dir.currentData(),
            clk_src=self.clk_src.currentData(),
            phase_bits_16=self.phase_bits.currentData(),
            dec_ratio=self.dec_ratio.value(),
        )


class PhaseDemodDialog(_ParamDialog):
    def __init__(self, params: PhaseDemodParams, locked: bool, parent=None) -> None:
        super().__init__("相位解调参数", locked, parent)

        self.space_avg = _spin(1, 1024, params.space_avg)
        self.space_merge = _spin(1, 1024, params.space_merge)
        self.region_diff = _spin(1, 1024, params.region_diff)
        self.detrend_bw = QDoubleSpinBox()
        self.detrend_bw.setRange(0.0, 100000.0)
        self.detrend_bw.setValue(params.detrend_bw)
        self.detrend_bw.setSuffix(" Hz")
        self.polar_div = QComboBox()
        self.polar_div.addItem("关", 0)
        self.polar_div.addItem("开", 1)
        self.polar_div.setCurrentIndex(self.polar_div.findData(params.polar_div))
        self.rate2phase = QComboBox()
        for sel in (1, 2, 4, 5, 10):
            self.rate2phase.addItem(f"{int(500 / sel)}M", sel)
        self.rate2phase.setCurrentIndex(
            self.rate2phase.findData(params.rate2phase_sel)
        )

        rows = [
            ("空间平均阶数", self.space_avg, "空间合并点数", self.space_merge),
            ("区域差分阶数", self.region_diff, "去趋势滤波带宽", self.detrend_bw),
            ("偏振分集", self.polar_div, "解调数据率", self.rate2phase),
        ]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            self.add_row(r, 0, l1, w1)
            self.add_row(r, 1, l2, w2)
        self.finish()

    def values(self) -> PhaseDemodParams:
        return PhaseDemodParams(
            space_avg=self.space_avg.value(),
            space_merge=self.space_merge.value(),
            region_diff=self.region_diff.value(),
            detrend_bw=self.detrend_bw.value(),
            polar_div=self.polar_div.currentData(),
            rate2phase_sel=self.rate2phase.currentData(),
        )


class ConfUserIpDialog(_ParamDialog):
    """修改板卡自身 IP（写入板卡存储，断电不丢失）。"""

    def __init__(self, octets: tuple[int, int, int, int], locked: bool, parent=None):
        super().__init__("修改板卡IP", locked, parent)
        self.octets = [_spin(0, 255, v) for v in octets]
        for i, sb in enumerate(self.octets):
            sb.setMinimumWidth(70)
            self.grid.addWidget(sb, 0, i)
        note = QLabel(
            "把板卡自己的 IP 永久改为上面的值：\n"
            "· 新 IP 在板卡重新上电后生效；\n"
            "· 改完后左下角连接地址要同步改成新 IP；\n"
            "· 建议把新 IP 写在标签上贴到板卡外壳。"
        )
        self.grid.addWidget(note, 1, 0, 1, 4)
        self.finish()

    def values(self) -> tuple[int, int, int, int]:
        return tuple(s.value() for s in self.octets)


class DigitalOutDialog(_ParamDialog):
    """数字输出：选择使能位与电平，确定后由主窗口下发。"""

    def __init__(self, bit_en: int, bit: int, locked: bool, parent=None) -> None:
        super().__init__("数字输出", locked, parent)
        self.do_bit_en = _spin(0, 255, bit_en)
        self.do_bit = _spin(0, 255, bit)
        self.add_row(0, 0, "使能位（DOBitEN）", self.do_bit_en)
        self.add_row(0, 1, "输出电平（DOBit）", self.do_bit)
        note = QLabel("点“确定”即向板卡发送一次设置命令")
        self.grid.addWidget(note, 2, 0, 1, 2)
        self.finish()

    def values(self) -> tuple[int, int]:
        return self.do_bit_en.value(), self.do_bit.value()
