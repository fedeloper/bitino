#!/usr/bin/env python3
"""Convert the AdaBin CIFAR-10 ResNet-20 (huawei-noah/Efficient-Computing BinaryNetworks/AdaBin; adapter torch_adabin)
to .bitino and validate it against the official PyTorch model. --repo is the Efficient-Computing clone.

Thin wrapper around tools/convert_s12_cifar.py (see its docstring for usage and outputs). Run with python3 -I."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_s12_cifar import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main("adabin"))
