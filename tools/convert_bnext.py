#!/usr/bin/env python3
"""Convert a verified BNext (hpi-xnor/BNext) ImageNet checkpoint to .bitino and validate it.

Thin wrapper around tools/convert_torch_cnn.py (see its docstring for usage and outputs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_torch_cnn import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("bnext"))
