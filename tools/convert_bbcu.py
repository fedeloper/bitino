#!/usr/bin/env python3
"""Convert the BBCU (Zj-BinXia/BBCU: denoising, JPEG deblocking, BBCU-L / BBCU-M super-resolution; adapter torch_bbcu)
checkpoints to .bitino and validate them.

Thin wrapper around tools/convert_restoration.py (see its docstring for usage and outputs). Sources and checkpoints are
untrusted downloads: run with python3 -I."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_restoration import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("bbcu"))
