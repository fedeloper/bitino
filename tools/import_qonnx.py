#!/usr/bin/env python3
"""QONNX -> .bitino importer (docs/qonnx-frontend.md phase 1): thin wrapper around bitino.qonnx_import.main.

Runs in .cache/venv-qonnx (onnx, qonnx, onnxruntime), with python -I (the checkout's python/ is put on sys.path here):

  .cache/venv-qonnx/bin/python -I tools/import_qonnx.py model.onnx --model-id ID --out-dir DIR \
      --eval-dir .cache/data/eval128_mnist_finn --preprocess mnist_pixels --input-domain uint8 \
      [--bench build/host/bitino_bench] [--source-json export.json]

Same arguments as `bitino import-qonnx`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from bitino.qonnx_import import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
