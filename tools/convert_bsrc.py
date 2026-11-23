#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
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
