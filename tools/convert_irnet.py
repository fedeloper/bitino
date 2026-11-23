#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert an IR-Net-family checkpoint (ReCU, RBNN, SiMaN, BiPer; adapter torch_irnet) to .bitino and validate it.

  python3 tools/convert_irnet.py <recu|rbnn|siman|biper> --model-id ... --repo ... --weights ... \\
      --eval-dir ... --out-dir ...

Thin wrapper around tools/convert_torch_cnn.py (see its docstring for usage and outputs)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_torch_cnn import FAMILIES, main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2 or "irnet" not in FAMILIES.get(sys.argv[1], {}):
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv.pop(1)))
