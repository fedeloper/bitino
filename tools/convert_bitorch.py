#!/usr/bin/env python3
# LEGACY CROSS-CHECK (QONNX step 6, docs/qonnx-standard.md): this IR-emitting converter is no longer the build path of
# the published files. They are built by tools/build_from_qonnx.py (checkpoint -> canonical QONNX file ->
# bitino.qonnx_import); this converter is kept unchanged to cross-check that route (tools/qonnx_legacy_crosscheck.py).
"""Convert a bitorch model-hub checkpoint (ResNetE-18, BinaryDenseNet-28, MeliusNet-22; adapter torch_bitorch)
to .bitino and validate it.

  python3 tools/convert_bitorch.py --model-id bitorch-resnete18-imagenet \\
      --repo .cache/repos/bitorch-pypi/bitorch-0.3.0 \\
      --weights .cache/checkpoints/bitorch-resnete18-imagenet/c7c515f083814fd181357e6acd12d5cb \\
      --eval-dir .cache/data/eval128_bitorch --out-dir artifacts/bitorch-resnete18-imagenet

--repo is the bitorch 0.3.0 source (the PyPI wheel extracted read-only; the hub checkpoints contain the
``current_dense_block`` keys of that release). The eval set is tools/eval_sets/bitorch_imagenet.py
(bitorch's ImageNet test transform, std (0.229, 0.224, 0.255)). Thin wrapper around
tools/convert_torch_cnn.py (see its docstring for outputs); only the "ge" variants are written, because
bitorch's sign maps 0 to +1, which is exactly Bitino's deployable binarization."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_torch_cnn import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("bitorch"))
