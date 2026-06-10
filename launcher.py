"""PyInstaller entry point for the DAS_pro desktop application."""

import sys

from das_pro.app import main

if __name__ == "__main__":
    sys.exit(main())
