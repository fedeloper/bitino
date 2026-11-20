# Full ImageNet-1k validation accuracy (50,000 images)

Generated 2026-11-20 by `tools/accuracy_full_val.py summarize` from `.cache/full50k/results/`. Eval set `imagenet-val-50000` (`.cache/full50k_orig/manifest50k.json`, SHA-256 `8fc65aac0d6ecfbe…`). Host runtime `build/host/bitino_bench`, 1 thread per process. 95 % Wilson intervals; int8 Δ = paired difference int8_full − all on the same images, 95 % CI, exact McNemar p.

**Data caveat.** Full validation set at original resolution; each model's own evaluation preprocessing is applied.

| Model | Preprocessing | Published | all top-1 [95 % CI] | top-5 | Δ pub (pt) | int8_full top-1 | Δ int8 (pt) [95 % CI] | McNemar p | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ab-bnn-imagenet-reactnet18` ★ | torchvision_imagenet | 61.39 | 61.26 [60.8, 61.7] | 83.01 | -0.13 | 60.31 | -0.95 [-1.21, -0.69] | 1.1e-12 | 50,000 |
| `ab-bnn-imagenet-reactnet34` ★ | torchvision_imagenet | 65.19 | 64.93 [64.5, 65.3] | 85.90 | -0.26 | 64.41 | -0.52 [-0.76, -0.28] | 2.4e-05 | 50,000 |
| `ab-bnn-imagenet-reactneta` ★ | torchvision_imagenet | 66.89 | 66.68 [66.3, 67.1] | 86.99 | -0.21 | 66.21 | -0.47 [-0.70, -0.24] | 7.6e-05 | 50,000 |
| `bhvit-small-binary` | bhvit | 68.40 | 68.47 [68.1, 68.9] | 88.09 | +0.07 | not run | – | – | 50,000 |
| `bhvit-small-somefp` | bhvit | 70.10 | 70.13 [69.7, 70.5] | 89.26 | +0.03 | not run | – | – | 50,000 |
| `bhvit-tiny-binary` ★ | bhvit | 64.00 | 63.81 [63.4, 64.2] | 85.23 | -0.19 | 62.80 | -1.01 [-1.29, -0.72] | 3.8e-12 | 50,000 |
| `bhvit-tiny-somefp` | bhvit | 66.00 | 65.75 [65.3, 66.2] | 86.36 | -0.25 | not run | – | – | 50,000 |
| `bi-vit-deit-small` | bhvit | 40.90 | 16.82 [16.5, 17.2] | 35.51 | -24.08 | not run | – | – | 50,000 |
| `bi-vit-deit-tiny` | bhvit | 28.70 | 15.57 [15.3, 15.9] | 34.09 | -13.13 | not run | – | – | 50,000 |
| `biper-resnet18-imagenet` | torchvision_imagenet | 61.40 | 61.26 [60.8, 61.7] | 82.96 | -0.14 | not run | – | – | 50,000 |
| `biper-resnet34-imagenet` | torchvision_imagenet | 65.63 | 65.37 [65.0, 65.8] | 86.15 | -0.26 | not run | – | – | 50,000 |
| `bitorch-densenet28-imagenet` | bitorch_imagenet | 58.69 | 58.61 [58.2, 59.0] | 81.14 | -0.08 | not run | – | – | 50,000 |
| `bitorch-meliusnet22-imagenet` | bitorch_imagenet | 64.35 | 64.41 [64.0, 64.8] | 85.39 | +0.06 | not run | – | – | 50,000 |
| `bitorch-resnete18-imagenet` | bitorch_imagenet | 60.58 | 60.51 [60.1, 60.9] | 82.75 | -0.07 | not run | – | – | 50,000 |
| `bnas-d-imagenet` | torchvision_imagenet | 57.71 | 57.80 [57.4, 58.2] | 80.18 | +0.09 | not run | – | – | 50,000 |
| `bnas-e-imagenet` | torchvision_imagenet | 58.32 | 58.16 [57.7, 58.6] | 80.25 | -0.16 | not run | – | – | 50,000 |
| `bnas-h-imagenet` | torchvision_imagenet | 63.74 | 63.72 [63.3, 64.1] | 84.52 | -0.02 | not run | – | – | 50,000 |
| `bnext-large` | timm_bicubic_crop095 | 80.60 | 80.47 [80.1, 80.8] | 95.16 | -0.13 | not run | – | – | 50,000 |
| `bnext-middle` | timm_bicubic_crop095 | 78.30 | 78.13 [77.8, 78.5] | 94.05 | -0.17 | not run | – | – | 50,000 |
| `bnext-small` | timm_bicubic_crop095 | 76.10 | 76.00 [75.6, 76.4] | 92.89 | -0.10 | not run | – | – | 50,000 |
| `bnext-tiny` ★ | timm_bicubic_crop095 | 72.40 | 72.30 [71.9, 72.7] | 90.58 | -0.10 | 69.28 | -3.01 [-3.30, -2.73] | 3.2e-95 | 50,000 |
| `dabnn-bireal18` ★ | torchvision_imagenet | 56.40 | 55.92 [55.5, 56.4] | 78.91 | -0.48 | 55.65 | -0.28 [-0.56, +0.00] | 0.054 | 50,000 |
| `dabnn-bireal18-stem` | torchvision_imagenet | 56.40 | 56.09 [55.7, 56.5] | 79.27 | -0.31 | not run | – | – | 50,000 |
| `larq-zoo-binary-alexnet` | larq_zoo | 36.30 | 36.30 [35.9, 36.7] | 61.54 | +0.00 | not run | – | – | 50,000 |
| `larq-zoo-binary-densenet28` | larq_zoo | 60.91 | 60.90 [60.5, 61.3] | 82.79 | -0.01 | not run | – | – | 50,000 |
| `larq-zoo-binary-densenet37` | larq_zoo | 62.89 | 62.86 [62.4, 63.3] | 84.19 | -0.03 | no file | – | – | 50,000 |
| `larq-zoo-binary-densenet37-dilated` | larq_zoo | 64.34 | 64.35 [63.9, 64.8] | 85.25 | +0.01 | no file | – | – | 50,000 |
| `larq-zoo-binary-densenet45` | larq_zoo | 64.59 | 64.54 [64.1, 65.0] | 85.24 | -0.05 | no file | – | – | 50,000 |
| `larq-zoo-binary-resnet-e18` | larq_zoo | 58.32 | 58.31 [57.9, 58.7] | 80.87 | -0.01 | not run | – | – | 50,000 |
| `larq-zoo-birealnet` ★ | larq_zoo | 57.47 | 57.49 [57.1, 57.9] | 79.86 | +0.02 | 56.87 | -0.62 [-0.90, -0.33] | 2.7e-05 | 50,000 |
| `larq-zoo-dorefanet` | larq_zoo | 53.39 | 53.43 [53.0, 53.9] | 76.55 | +0.04 | not run | – | – | 50,000 |
| `larq-zoo-meliusnet22` | larq_zoo | 62.40 | 62.56 [62.1, 63.0] | 83.82 | +0.16 | no file | – | – | 50,000 |
| `larq-zoo-quicknet` ★ | larq_zoo | 63.30 | 63.42 [63.0, 63.8] | 84.60 | +0.12 | 61.99 | -1.43 [-1.71, -1.15] | 8.8e-24 | 50,000 |
| `larq-zoo-quicknet-large` ★ | larq_zoo | 66.90 | 66.67 [66.3, 67.1] | 86.98 | -0.23 | 65.04 | -1.62 [-1.91, -1.33] | 2.4e-28 | 50,000 |
| `larq-zoo-quicknet-small` ★ | larq_zoo | 59.40 | 59.54 [59.1, 60.0] | 81.71 | +0.14 | 56.87 | -2.67 [-2.97, -2.36] | 1.2e-65 | 50,000 |
| `larq-zoo-quicknet-xl-legacy` | larq_zoo | 67.00 | 67.13 [66.7, 67.5] | 87.44 | +0.13 | not run | – | – | 50,000 |
| `larq-zoo-real-to-binary-net` | larq_zoo | 65.01 | 65.04 [64.6, 65.5] | 85.71 | +0.03 | not run | – | – | 50,000 |
| `larq-zoo-xnornet` | larq_zoo | 44.96 | 45.00 [44.6, 45.4] | 69.17 | +0.04 | not run | – | – | 50,000 |
| `nncf-resnet18-binarization-dorefa` | torchvision_imagenet | 61.63 | 61.60 [61.2, 62.0] | 83.47 | -0.03 | not run | – | – | 50,000 |
| `nncf-resnet18-binarization-xnor` | torchvision_imagenet | 61.67 | 61.73 [61.3, 62.2] | 83.51 | +0.06 | not run | – | – | 50,000 |
| `omz-resnet18-xnor-binary-onnx-0001` | omz_bgr | 61.71 | 61.77 [61.3, 62.2] | 83.60 | +0.06 | not run | – | – | 50,000 |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | omz_bgr | 61.71 | 61.81 [61.4, 62.2] | 83.58 | +0.10 | not run | – | – | 50,000 |
| `omz-resnet50-binary-0001` ★ | omz_bgr | 70.69 | 70.57 [70.2, 71.0] | 89.22 | -0.12 | 70.26 | -0.31 [-0.55, -0.07] | 0.013 | 50,000 |
| `omz-resnet50-binary-0001-fp16` | omz_bgr | 70.69 | 70.82 [70.4, 71.2] | 89.29 | +0.13 | not run | – | – | 50,000 |
| `rbnn-resnet18-imagenet` | torchvision_imagenet | 59.90 | 59.55 [59.1, 60.0] | 81.75 | -0.35 | not run | – | – | 50,000 |
| `rbnn-resnet34-imagenet` | torchvision_imagenet | 63.10 | 62.86 [62.4, 63.3] | 84.32 | -0.24 | not run | – | – | 50,000 |
| `rbonn-reactnet-resnet18-imagenet` | torchvision_imagenet | 66.70 | 66.59 [66.2, 67.0] | 86.97 | -0.11 | not run | – | – | 50,000 |
| `recu-resnet18-imagenet` | torchvision_imagenet | 61.20 | 61.18 [60.8, 61.6] | 82.95 | -0.02 | not run | – | – | 50,000 |
| `recu-resnet34-imagenet` | torchvision_imagenet | 65.25 | 65.08 [64.7, 65.5] | 85.77 | -0.17 | not run | – | – | 50,000 |
| `sa-bnn-resnet18-imagenet` | torchvision_imagenet | 61.70 | 61.69 [61.3, 62.1] | 82.75 | -0.01 | not run | – | – | 50,000 |
| `siman-resnet18-imagenet` | torchvision_imagenet | 60.10 | 60.20 [59.8, 60.6] | 82.24 | +0.10 | not run | – | – | 50,000 |
| `siman-resnet34-imagenet` | torchvision_imagenet | 63.90 | 63.64 [63.2, 64.1] | 84.78 | -0.26 | no file | – | – | 50,000 |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | torchvision_imagenet | 63.40 | 63.39 [63.0, 63.8] | 84.67 | -0.01 | not run | – | – | 50,000 |
| `snn-subbit-reactnet18-w1a1-imagenet` | torchvision_imagenet | 65.70 | 65.80 [65.4, 66.2] | 86.34 | +0.10 | not run | – | – | 50,000 |
| `tensorpack-dorefa-alexnet-w1a1-g32` | tensorpack_bgr | 48.50 | 48.66 [48.2, 49.1] | 72.32 | +0.16 | not run | – | – | 50,000 |

★ headline network (paper Table tab:acc).

## Exact stage (`neon`) vs `all` on a slice

| Model | n | neon top-1 | all top-1 (same images) | top-1 agreement | Δ neon − all (pt) [95 % CI] | McNemar p | cpu s/img neon / all |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ab-bnn-imagenet-reactnet18` | 2,000 | 61.75 | 61.70 | 99.50 % | +0.05 [-0.21, +0.31] | 1 | 0.193 / 0.013 |
| `ab-bnn-imagenet-reactnet34` | 2,000 | 65.35 | 65.40 | 99.05 % | -0.05 [-0.38, +0.28] | 1 | 0.216 / 0.022 |
| `ab-bnn-imagenet-reactneta` | 2,000 | 66.65 | 66.80 | 99.10 % | -0.15 [-0.41, +0.11] | 0.45 | 0.107 / 0.026 |
| `bhvit-tiny-binary` | 2,000 | 63.20 | 63.00 | 94.55 % | +0.20 [-0.45, +0.85] | 0.65 | 0.263 / 0.034 |
| `bnext-tiny` | 2,000 | 71.95 | 72.60 | 95.20 % | -0.65 [-1.35, +0.05] | 0.092 | 0.142 / 0.033 |
| `dabnn-bireal18` | 2,000 | 56.90 | 57.00 | 99.15 % | -0.10 [-0.34, +0.14] | 0.69 | 0.171 / 0.013 |
| `larq-zoo-birealnet` | 2,000 | 57.65 | 57.55 | 98.85 % | +0.10 [-0.18, +0.38] | 0.73 | 0.172 / 0.013 |
| `larq-zoo-quicknet` | 2,000 | 64.45 | 64.55 | 98.05 % | -0.10 [-0.52, +0.32] | 0.81 | 0.044 / 0.011 |
| `larq-zoo-quicknet-large` | 50,000 | 66.71 | 66.67 | 95.62 % | +0.05 [-0.08, +0.17] | 0.47 | 0.055 / 0.018 |
| `larq-zoo-quicknet-small` | 2,000 | 60.60 | 60.40 | 98.50 % | +0.20 [-0.17, +0.57] | 0.42 | 0.025 / 0.007 |
| `omz-resnet50-binary-0001` | 2,000 | 71.45 | 71.30 | 96.55 % | +0.15 [-0.38, +0.68] | 0.71 | 0.414 / 0.037 |

## Stage `all`: this copy vs resized copy

Same file names and labels; `.cache/full50k` holds the other copy's results. Paired on the images both runs have.

| Model | Published | this copy | resized copy | n | Δ (other − this, pt) [95 % CI] | McNemar p | top-1 agreement | Δ pub (other) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ab-bnn-imagenet-reactnet18` | 61.39 | 61.26 | 59.55 | 50,000 | -1.71 [-1.99, -1.43] | 2.1e-33 | 77.4 % | -1.84 |
| `ab-bnn-imagenet-reactnet34` | 65.19 | 64.93 | 63.73 | 50,000 | -1.20 [-1.47, -0.92] | 5.9e-18 | 79.9 % | -1.46 |
| `ab-bnn-imagenet-reactneta` | 66.89 | 66.68 | 64.91 | 50,000 | -1.78 [-2.04, -1.51] | 1.2e-38 | 80.7 % | -1.98 |
| `bhvit-small-binary` | 68.40 | 68.47 | 66.02 | 50,000 | -2.45 [-2.74, -2.15] | 6.8e-60 | 78.4 % | -2.38 |
| `bhvit-small-somefp` | 70.10 | 70.13 | 67.71 | 50,000 | -2.42 [-2.69, -2.14] | 1.7e-66 | 81.1 % | -2.39 |
| `bhvit-tiny-binary` | 64.00 | 63.81 | 62.36 | 50,000 | -1.44 [-1.74, -1.14] | 3.3e-21 | 75.8 % | -1.64 |
| `bhvit-tiny-somefp` | 66.00 | 65.75 | 63.81 | 50,000 | -1.94 [-2.23, -1.66] | 5.4e-41 | 78.8 % | -2.19 |
| `bi-vit-deit-small` | 40.90 | 16.82 | 16.01 | 50,000 | -0.81 [-1.08, -0.55] | 2.7e-09 | 44.8 % | -24.89 |
| `bi-vit-deit-tiny` | 28.70 | 15.57 | 15.36 | 50,000 | -0.21 [-0.50, +0.08] | 0.17 | 35.6 % | -13.34 |
| `biper-resnet18-imagenet` | 61.40 | 61.26 | 59.34 | 50,000 | -1.91 [-2.22, -1.61] | 1.8e-34 | 73.1 % | -2.06 |
| `biper-resnet34-imagenet` | 65.63 | 65.37 | 63.81 | 50,000 | -1.56 [-1.86, -1.26] | 1.3e-24 | 76.3 % | -1.82 |
| `bitorch-densenet28-imagenet` | 58.69 | 58.61 | 57.15 | 50,000 | -1.46 [-1.75, -1.18] | 1.8e-23 | 75.7 % | -1.54 |
| `bitorch-meliusnet22-imagenet` | 64.35 | 64.41 | 61.96 | 50,000 | -2.45 [-2.74, -2.16] | 2.8e-62 | 77.7 % | -2.39 |
| `bitorch-resnete18-imagenet` | 60.58 | 60.51 | 58.29 | 50,000 | -2.22 [-2.52, -1.92] | 6.7e-48 | 74.3 % | -2.29 |
| `bnas-d-imagenet` | 57.71 | 57.80 | 55.16 | 50,000 | -2.64 [-2.96, -2.32] | 3e-58 | 69.0 % | -2.55 |
| `bnas-e-imagenet` | 58.32 | 58.16 | 55.60 | 50,000 | -2.56 [-2.88, -2.24] | 7.8e-55 | 69.6 % | -2.72 |
| `bnas-h-imagenet` | 63.74 | 63.72 | 61.60 | 50,000 | -2.13 [-2.44, -1.82] | 7.1e-42 | 74.3 % | -2.14 |
| `bnext-large` | 80.60 | 80.47 | 79.22 | 50,000 | -1.25 [-1.48, -1.02] | 7.1e-27 | 89.5 % | -1.38 |
| `bnext-middle` | 78.30 | 78.13 | 77.20 | 50,000 | -0.93 [-1.15, -0.71] | 8.3e-17 | 89.6 % | -1.10 |
| `bnext-small` | 76.10 | 76.00 | 74.97 | 50,000 | -1.03 [-1.26, -0.80] | 2.8e-18 | 87.9 % | -1.13 |
| `bnext-tiny` | 72.40 | 72.30 | 71.26 | 50,000 | -1.04 [-1.29, -0.78] | 2e-15 | 84.5 % | -1.14 |
| `dabnn-bireal18` | 56.40 | 55.92 | 53.27 | 50,000 | -2.66 [-2.98, -2.34] | 5.3e-60 | 68.3 % | -3.13 |
| `dabnn-bireal18-stem` | 56.40 | 56.09 | 52.78 | 50,000 | -3.31 [-3.64, -2.98] | 3.4e-86 | 66.9 % | -3.62 |
| `larq-zoo-binary-alexnet` | 36.30 | 36.30 | 35.60 | 50,000 | -0.70 [-1.01, -0.39] | 8.2e-06 | 57.9 % | -0.70 |
| `larq-zoo-binary-densenet28` | 60.91 | 60.90 | 59.47 | 50,000 | -1.43 [-1.72, -1.14] | 1.9e-22 | 76.0 % | -1.44 |
| `larq-zoo-binary-densenet37` | 62.89 | 62.86 | 61.10 | 50,000 | -1.76 [-2.05, -1.47] | 8.4e-33 | 76.6 % | -1.79 |
| `larq-zoo-binary-densenet37-dilated` | 64.34 | 64.35 | 62.60 | 50,000 | -1.75 [-2.02, -1.49] | 7.8e-38 | 80.4 % | -1.74 |
| `larq-zoo-binary-densenet45` | 64.59 | 64.54 | 62.76 | 50,000 | -1.78 [-2.06, -1.50] | 4.9e-35 | 78.4 % | -1.83 |
| `larq-zoo-binary-resnet-e18` | 58.32 | 58.31 | 56.37 | 50,000 | -1.95 [-2.26, -1.64] | 1.9e-34 | 71.2 % | -1.95 |
| `larq-zoo-birealnet` | 57.47 | 57.49 | 55.27 | 50,000 | -2.22 [-2.54, -1.90] | 4.3e-42 | 69.1 % | -2.20 |
| `larq-zoo-dorefanet` | 53.39 | 53.43 | 52.65 | 50,000 | -0.79 [-1.06, -0.51] | 2.4e-08 | 74.8 % | -0.74 |
| `larq-zoo-meliusnet22` | 62.40 | 62.56 | 60.69 | 50,000 | -1.86 [-2.16, -1.57] | 5.7e-35 | 75.8 % | -1.71 |
| `larq-zoo-quicknet` | 63.30 | 63.42 | 61.88 | 50,000 | -1.55 [-1.85, -1.25] | 1.3e-23 | 74.9 % | -1.42 |
| `larq-zoo-quicknet-large` | 66.90 | 66.67 | 66.06 | 50,000 | -0.61 [-0.89, -0.33] | 2.6e-05 | 79.0 % | -0.84 |
| `larq-zoo-quicknet-small` | 59.40 | 59.54 | 58.56 | 50,000 | -0.97 [-1.27, -0.67] | 2.3e-10 | 73.7 % | -0.84 |
| `larq-zoo-quicknet-xl-legacy` | 67.00 | 67.13 | 65.34 | 50,000 | -1.79 [-2.09, -1.49] | 1.3e-30 | 76.3 % | -1.66 |
| `larq-zoo-real-to-binary-net` | 65.01 | 65.04 | 62.76 | 50,000 | -2.28 [-2.57, -1.98] | 4e-51 | 76.4 % | -2.25 |
| `larq-zoo-xnornet` | 44.96 | 45.00 | 43.16 | 50,000 | -1.85 [-2.17, -1.52] | 3e-28 | 60.3 % | -1.80 |
| `nncf-resnet18-binarization-dorefa` | 61.63 | 61.60 | 59.94 | 50,000 | -1.66 [-1.96, -1.37] | 4.8e-28 | 75.0 % | -1.69 |
| `nncf-resnet18-binarization-xnor` | 61.67 | 61.73 | 60.36 | 50,000 | -1.37 [-1.66, -1.07] | 1e-19 | 75.6 % | -1.31 |
| `omz-resnet18-xnor-binary-onnx-0001` | 61.71 | 61.77 | 60.30 | 50,000 | -1.47 [-1.76, -1.17] | 2.8e-22 | 75.6 % | -1.41 |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | 61.71 | 61.81 | 60.35 | 50,000 | -1.46 [-1.75, -1.16] | 5.1e-22 | 75.5 % | -1.36 |
| `omz-resnet50-binary-0001` | 70.69 | 70.57 | 68.30 | 50,000 | -2.28 [-2.56, -1.99] | 3e-56 | 80.5 % | -2.39 |
| `omz-resnet50-binary-0001-fp16` | 70.69 | 70.82 | 68.33 | 50,000 | -2.48 [-2.76, -2.20] | 5.4e-68 | 80.7 % | -2.36 |
| `rbnn-resnet18-imagenet` | 59.90 | 59.55 | 59.58 | 50,000 | +0.04 [-0.25, +0.32] | 0.82 | 75.7 % | -0.32 |
| `rbnn-resnet34-imagenet` | 63.10 | 62.86 | 62.22 | 50,000 | -0.64 [-0.92, -0.36] | 7.9e-06 | 78.2 % | -0.88 |
| `rbonn-reactnet-resnet18-imagenet` | 66.70 | 66.59 | 64.79 | 50,000 | -1.80 [-2.08, -1.52] | 1.4e-36 | 79.4 % | -1.91 |
| `recu-resnet18-imagenet` | 61.20 | 61.18 | 59.59 | 50,000 | -1.59 [-1.89, -1.29] | 9e-25 | 74.1 % | -1.61 |
| `recu-resnet34-imagenet` | 65.25 | 65.08 | 63.53 | 50,000 | -1.55 [-1.84, -1.26] | 1.1e-25 | 77.1 % | -1.72 |
| `sa-bnn-resnet18-imagenet` | 61.70 | 61.69 | 59.78 | 50,000 | -1.90 [-2.19, -1.62] | 3.4e-38 | 75.9 % | -1.92 |
| `siman-resnet18-imagenet` | 60.10 | 60.20 | 59.80 | 50,000 | -0.40 [-0.69, -0.11] | 0.0075 | 75.3 % | -0.30 |
| `siman-resnet34-imagenet` | 63.90 | 63.64 | 63.11 | 50,000 | -0.53 [-0.82, -0.24] | 0.00039 | 77.3 % | -0.79 |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | 63.40 | 63.39 | 61.45 | 50,000 | -1.94 [-2.21, -1.67] | 8.4e-44 | 78.8 % | -1.95 |
| `snn-subbit-reactnet18-w1a1-imagenet` | 65.70 | 65.80 | 64.17 | 50,000 | -1.63 [-1.90, -1.36] | 4.3e-32 | 80.2 % | -1.53 |
| `tensorpack-dorefa-alexnet-w1a1-g32` | 48.50 | 48.66 | 48.19 | 50,000 | -0.47 [-0.78, -0.15] | 0.0037 | 64.9 % | -0.31 |

Mean Δ over 55 models: -1.59 pt.

## Rebuild

```
P="~/anaconda3/bin/python -I -B"; R=<directory holding imagenet/val>
$P tools/accuracy_full_val.py verify-prep          # 7 preprocessings == recorded eval5000 tensors, bit for bit
$P tools/accuracy_full_val.py manifest --val-dir $R/imagenet_resize/val      # registry/eval_sets/imagenet-val-50000.json
# resized copy (the registry manifest): float, exact slice, full-integer
scripts/capped.sh 24G nice -n 10 $P tools/accuracy_full_val.py run --workers 8 --val-dir $R/imagenet_resize/val \
    --passes float,neon,int8_headline,int8_rest,int8_fp16
# original JPEGs (diagnostic; stage all)
$P tools/accuracy_full_val.py manifest --copy original --val-dir $R/imagenet/val --out .cache/full50k_orig/manifest50k.json
scripts/capped.sh 7G nice -n 10 $P tools/accuracy_full_val.py run --workers 3 --passes float \
    --manifest .cache/full50k_orig/manifest50k.json --work .cache/full50k_orig \
    --out reports/accuracy_full50k/original_resolution --val-dir $R/imagenet/val
$P tools/accuracy_full_val.py summarize --other-work .cache/full50k_orig
$P tools/accuracy_full_val.py summarize --manifest .cache/full50k_orig/manifest50k.json --work .cache/full50k_orig \
    --out reports/accuracy_full50k/original_resolution --other-work .cache/full50k --other-label "resized copy"
```
