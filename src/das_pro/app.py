"""Application entry point.

Usage:
    python -m das_pro.app                 # launch the GUI
    python -m das_pro.app --simulator     # also start an in-process board simulator

When --simulator is given, a fake ETH-5520 is started on 127.0.0.1 so the full
application can be exercised without hardware. The board-address fields are
pre-filled with the simulator's address; press START.
"""

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .device import serve_simulator
from .gui import MainWindow
from .protocol.constants import DEFAULT_PORT


def main() -> int:
    parser = argparse.ArgumentParser(description="DAS_pro host application")
    parser.add_argument(
        "--simulator",
        action="store_true",
        help="start an in-process ETH-5520 simulator on 127.0.0.1",
    )
    parser.add_argument("--sim-port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

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
