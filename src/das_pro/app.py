"""Application entry point.

Usage:
    python -m das_pro.app                 # launch the GUI
    python -m das_pro.app --simulator     # also start an in-process board simulator

When --simulator is given, a fake ETH-5520 is started on 127.0.0.1 so the full
application can be exercised without hardware. The board-address fields are
pre-filled with the simulator's address; press START.
"""

import argparse
import os
import sys
import traceback

from PySide6.QtWidgets import QApplication

from .device import serve_simulator
from .gui import MainWindow
from .protocol.constants import DEFAULT_PORT


def _install_crash_log() -> None:
    """Windowed exes have no console; append unhandled errors to a file."""
    base = (
        os.path.dirname(sys.executable)
        if getattr(sys, "frozen", False)
        else os.getcwd()
    )
    log_path = os.path.join(base, "das_pro_error.log")

    def hook(exc_type, exc, tb):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                traceback.print_exception(exc_type, exc, tb, file=f)
        except OSError:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


def main() -> int:
    parser = argparse.ArgumentParser(description="DAS_pro host application")
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="start an in-process ETH-5520 simulator on 127.0.0.1",
    )
    parser.add_argument("--sim-port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    _install_crash_log()
    app = QApplication(sys.argv)

    if args.simulator:
        serve_simulator("127.0.0.1", args.sim_port)

    window = MainWindow()
    if args.simulator:
        for spin, octet in zip(window.ip_octets, (127, 0, 0, 1)):
            spin.setValue(octet)
        window.port = args.sim_port
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
