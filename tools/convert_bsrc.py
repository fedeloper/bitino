#!/usr/bin/env python3
"""Convert the BSRC-2023 binary super-resolution baselines (ilyazhara/BSRC-2023 @62bbcf2; adapter torch_bsrc)
checkpoints to .bitino and validate them.

Thin wrapper around tools/convert_restoration.py (see its docstring for usage and outputs). Sources and checkpoints are
untrusted downloads: run with python3 -I."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_restoration import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("bsrc"))
