"""Tiny plotting helpers shared by all windows."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QDialog, QVBoxLayout

_LABEL_STYLE = {"color": "#a0a0a0", "font-size": "11px"}


def set_labels(plot: pg.PlotWidget, bottom: str, left: str) -> None:
    """Set both axis labels (cheap enough to call on every redraw)."""
    plot.setLabel("bottom", bottom, **_LABEL_STYLE)
    plot.setLabel("left", left, **_LABEL_STYLE)


class _PopoutDialog(QDialog):
    """Holds a plot widget full-size; puts it back where it came from on close."""

    def __init__(self, plot, title, layout, index, stretch) -> None:
        super().__init__(plot.window())
        self._plot = plot
        self._layout = layout
        self._index = index
        self._stretch = stretch
        self.setWindowTitle(f"{title} — 放大查看（关闭窗口恢复）")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(plot)
        self.resize(1150, 720)

    def closeEvent(self, event) -> None:
        self._layout.insertWidget(
            min(self._index, self._layout.count()), self._plot, self._stretch
        )
        self._plot._popout = None
        super().closeEvent(event)


def make_zoomable(plot: pg.PlotWidget, title: str, layout, stretch: int) -> None:
    """Double-clicking the plot pops it out into its own resizable window.

    The very same widget is reparented, so live updates keep flowing; on
    close it is reinserted at its original layout position.
    """
    plot._popout = None
    plot.setToolTip("双击放大到独立窗口")

    def on_click(ev) -> None:
        if not ev.double() or plot._popout is not None:
            return
        index = layout.indexOf(plot)
        if index < 0:
            return
        dlg = _PopoutDialog(plot, title, layout, index, stretch)
        plot._popout = dlg
        dlg.show()

    plot.scene().sigMouseClicked.connect(on_click)
