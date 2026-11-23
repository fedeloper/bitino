#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert the SA-BNN ResNet-18 ImageNet checkpoint (anonymous1443/SA-BNN Drive release; adapter torch_sabnn)
to .bitino and validate it. --repo is the directory holding the Drive file birealnet.py.txt.

Thin wrapper around tools/convert_torch_cnn.py (see its docstring for usage and outputs). The Drive files are
untrusted downloads: run with python3 -I."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_torch_cnn import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("sabnn"))
