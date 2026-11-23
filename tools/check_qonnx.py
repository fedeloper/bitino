#!/usr/bin/env python3
"""Checks ONNX files against the Bitino QONNX profile v1 (docs/qonnx-profile.md): thin wrapper around
bitino.qonnx_profile.main.

Runs in .cache/venv-qonnx (needs onnx), with python -I (the checkout's python/ is put on sys.path here):

  .cache/venv-qonnx/bin/python -I tools/check_qonnx.py model.onnx [more.onnx ...] [--json OUT|-] [--strict]

Same arguments as `bitino check-qonnx`. Exit code 0 = every file passes, 1 = violations, 2 = usage / unreadable.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from bitino.qonnx_profile import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
