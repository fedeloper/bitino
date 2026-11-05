#!/usr/bin/env python3
"""Convert the FDA-BNN CIFAR-10 ResNet-20 MindSpore checkpoint (fdabnn.ckpt; reader mindspore_ckpt, adapter
mindspore_fdabnn) to .bitino and validate it against the cited PyTorch port (float32/float64) and, with
--mindspore-logits, the official MindSpore model (tools/fdabnn_mindspore_reference.py). --repo holds quan.py/resnet.py.

Thin wrapper around tools/convert_s12_cifar.py (see its docstring for usage and outputs). Run with python3 -I."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_s12_cifar import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("fdabnn"))
