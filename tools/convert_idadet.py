#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
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
