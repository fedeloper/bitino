#!/usr/bin/env python3
"""Convert the IDa-Det BiRes-18 ImageNet backbone checkpoint (SteveTsui/IDa-Det bires18.pth; adapter torch_rbonn)
to .bitino and validate it. --repo is a clone of https://github.com/SteveTsui/IDa-Det (read only; mmcv/mmdet are
not built: the backbone source files are executed with the registry plumbing stubbed, see _load_idadet).

Thin wrapper around tools/convert_torch_cnn.py (see its docstring for usage and outputs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_torch_cnn import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("idadet"))
