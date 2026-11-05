#!/usr/bin/env python3
"""CIFAR-10 test subset with the DARTS-style eval preprocessing used by BNAS (yonseivnl/bnas src/utils.py:122-137
``_data_transforms_cifar10``: ``ToTensor()`` then ``Normalize(CIFAR_MEAN, CIFAR_STD)`` with
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124], CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]; no resize/crop).

Same images, labels, selection rule and float32 arithmetic as tools/eval_sets/cifar10_normalized.py (first --n rows of
``test_batch``, torchvision ``k/255`` then ``sub_(mean).div_(std)`` in float32), written into its own directory with
the single preprocessing key ``cifar10_darts_norm``. Data are never committed.

  python3 tools/eval_sets/cifar10_darts.py --cifar-dir .cache/data/cifar10 --n 128   --out .cache/data/eval128_cifar10_darts
  python3 tools/eval_sets/cifar10_darts.py --cifar-dir .cache/data/cifar10 --n 10000 --out .cache/data/eval10000_cifar10_darts
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cifar10_normalized  # noqa: E402

VARIANTS = {
    "cifar10_darts_norm": ([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768],
                           "yonseivnl/bnas src/utils.py:122-137 _data_transforms_cifar10 valid_transform (DARTS CIFAR_MEAN/CIFAR_STD)"),
}

if __name__ == "__main__":
    cifar10_normalized.VARIANTS = VARIANTS
    sys.exit(cifar10_normalized.main())
