#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert the AdaBin CIFAR-10 ResNet-20 (huawei-noah/Efficient-Computing BinaryNetworks/AdaBin; adapter torch_adabin)
to .bitino and validate it against the official PyTorch model. --repo is the Efficient-Computing clone.

Thin wrapper around tools/convert_s12_cifar.py (see its docstring for usage and outputs). Run with python3 -I."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_s12_cifar import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("adabin"))
