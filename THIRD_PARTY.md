# Third-party code and data

Bitino's runtime (`runtime/`) has **no third-party code and no dependencies** beyond the C++17 standard library
and POSIX threads. The items below are the only places where third-party code is ported, re-implemented or
loaded.

## Ported or re-implemented reference code (conversion tooling only)

| file | origin | licence | what |
|---|---|---|---|
| `tools/dorefa_alexnet_reference.py` | [tensorpack](https://github.com/tensorpack/tensorpack) `examples/DoReFa-Net/alexnet-dorefa.py`, `dorefa.py` | Apache-2.0 (Copyright (c) Tensorpack Contributors) | PyTorch port of the DoReFa AlexNet forward pass, float64 reference for one conversion |
| `tools/fdabnn_reference.py` | [MindSpore models](https://gitee.com/mindspore/models) `research/cv/FDA-BNN/src/resnet.py`, `src/quan.py` | Apache-2.0 (Copyright Huawei Technologies Co., Ltd) | PyTorch port of FDA-BNN ResNet-20, reference for one conversion |
| `tools/eval_sets/timm_bicubic_crop095.py` | [timm](https://github.com/huggingface/pytorch-image-models) eval transform | Apache-2.0 | re-implementation (with torchvision calls) of the timm 0.6.11 evaluation transform; no code copied |
| `python/bitino/adapters/torch_bnas.py` (`binop_targets`) | [yonseivnl/bnas](https://github.com/yonseivnl/bnas) `src/bin_utils.py` (no licence file) | none stated | a 20-line re-implementation of which modules the original code binarizes; no code copied |

## Loaded at conversion time, not redistributed

The converters import the **original authors' model code** from a read-only download of their repository at the
recorded commit (`registry/recipes.json` → `sources`), and the source frameworks from the reference environments
(`envs/`). None of that code is copied into this repository. Their licences are recorded per model in
`registry/models.json` (`license.code`).

## Build-time interfaces

* `tools/qemu_plugins/*.c` are original code compiled against QEMU's plugin API header `qemu-plugin.h`
  (QEMU, GPL-2.0-or-later). The plugins are developer tools for counting emulated instructions; they are not part of
  the runtime and no built plugin is distributed.
* `scripts/build_android.sh` uses the Android NDK toolchain; release binaries link the NDK's libc++ statically
  (Apache-2.0 with LLVM exceptions).

## Model weights and datasets

Not covered by Bitino's licence. Converted weights are redistributed only where the weights licence is explicitly
permissive, each with a `<model-id>.LICENSE.txt` (attribution, licence text, changes). Datasets (ImageNet, CIFAR,
GLUE, …) are never redistributed; the eval-set manifests in `registry/eval_sets*.json` hold preprocessing
definitions and hashes only.
