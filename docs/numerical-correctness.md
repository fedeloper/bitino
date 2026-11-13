# Numerical correctness

Correctness is checked at five levels. Every level has a tool and a stored JSON artifact.

| Level | What | Tool | Criterion |
|---|---|---|---|
| 1 kernel | optimized kernel vs reference kernel | `tests/cpp/test_*_kernels.cc` (host + device) | **bit-exact** for binary and INT8-vs-INT8-reference; documented bound for float |
| 2 operator | fused / optimized op vs unfused reference op | same + `tests/python` (numpy vs C++) | bit-exact or documented bound |
| 3 conversion | source framework vs converted IR (numpy executor) | `tools/convert_*.py` → `conversion_report.json` | strict float64 semantic check, see below |
| 4 end-to-end | source model vs C++ runtime on the eval set | `tools/validate_model.py` | runtime bit-exact with numpy executor; top-1 agreement vs fp64 reference |
| 5 device | same inputs on the Galaxy S22 | `tools/validate_model.py --outputs-file` | bit-exact with host |

## Why "max abs error" alone is the wrong metric for BNNs

A binary network applies `sign()` to real-valued pre-activations many times. If a value sits
within float rounding distance of 0, two correct float32 implementations can binarize it
differently. The flip then propagates and can move logits by several units. This happens in the
**source model itself**. For Larq QuickNet-Small, TF float32 vs TF float64 on 128 images:
max |Δlogit| = 3.2, cosine ≥ 0.9856, 1/128 top-1 change. Any float32 runtime has a
noise floor of this size, so a fixed absolute tolerance on logits is not a sound pass/fail rule.

## Level 3: the strict semantic check

The converter evaluates the source model in **float64** and the converted IR with the numpy
executor in **float64** (`executor.run(..., float_dtype=np.float64)`), on every eval sample.

* If the conversion is semantically exact, outputs agree to ~1e-6 or better. The only
  difference is that IR constants are float32, e.g. folded BatchNorm scale/shift.
* For any sample that deviates more, `tools/first_flip_analysis.py` (Keras) walks the binary
  convolutions in order. It finds the **first** layer whose binarized input differs and reports
  |pre-activation| at the flipped positions.
* A deviation is accepted as rounding only if the first flip happens at a pre-activation many
  orders of magnitude below the typical activation (in practice ≤ ~1e-5 vs typical ~0.5–1).
  A flip at a large |pre-activation| is a conversion bug.

Results so far (128-image subset):

| model | samples within 1e-6 (fp64) | deviating samples: first flip at \|pre\| | verdict |
|---|---|---|---|
| larq-zoo-quicknet-small | 126/128 | 6.6e-9, 1.0e-7 (1 flip each) | exact (rounding only) |
| larq-zoo-birealnet | 109/128 | ≤ 2.3e-6 (8 inspected, 1 flip each; typical \|pre\| ≈ 1) | exact (rounding only) |
| bhvit-tiny-binary | 100/128 within 2e-6 (no flip) | 28 samples, first flip at ≤ 4.3e-7 (≤ 7e-7 × typical; all 128 analyzed) | exact (rounding only); float64 constants: 128/128 ≤ 1.8e-14 |
| bhvit-small-binary | 26/128 within 3.2e-6 (no flip) | 102 samples, first flip at ≤ 1.3e-6 (≤ 3.9e-6 × typical; all 128 analyzed) | exact (rounding only); float64 constants: 128/128 ≤ 3.6e-14 |
| brevitas-bnn-pynq-{cnv,lfc,sfc,tfc}-1w1a | 128/128 each (max 1.1e-7) | none: 0 binarization flips vs the source | exact (only fp32 storage of the folded TensorNorm affine) |
| larq-zoo-quicknet-large | 120/128 | ≤ 1.3e-7 (8/8 samples, 1 flip each; typical ≥ 0.61) | exact (rounding only) |
| larq-zoo-binary-resnet-e18 | 0/128 | ≤ 2.2e-7 (2 samples); 126 samples without flip, max \|Δlogit\| 3.0e-6 | exact (rounding only) |
| larq-zoo-binary-densenet28 | 0/128 | ≤ 2.2e-6 (4 samples); 124 without flip, max \|Δlogit\| 6.5e-6 | exact (rounding only) |
| larq-zoo-binary-densenet37 | 0/128 | ≤ 2.0e-6 (13 samples); 115 without flip, max \|Δlogit\| 7.2e-6 | exact (rounding only) |
| larq-zoo-binary-densenet37-dilated | 0/128 | ≤ 1.5e-5 (37 samples); 91 without flip, max \|Δlogit\| 9.6e-6 | exact (rounding only) |
| larq-zoo-binary-densenet45 | 0/128 | ≤ 6.8e-6 (23 samples); 105 without flip, max \|Δlogit\| 1.2e-5 | exact (rounding only) |
| larq-zoo-meliusnet22 | 0/128 | ≤ 1.2e-6 (12 samples); 116 without flip, max \|Δlogit\| 7.4e-6 | exact (rounding only) |
| larq-zoo-binary-alexnet | 128/128 | no deviation | exact |
| larq-zoo-xnornet | 102/128 | no flip; max \|Δlogit\| 1.3e-6 | exact (rounding only) |
| larq-zoo-real-to-binary-net | 0/128 | ≤ 2.6e-7 (4 samples); 124 without flip, max \|Δlogit\| 3.8e-6 | exact (rounding only) |
| larq-zoo-dorefanet (2-bit act.) | 123/128 | no level change (k-bit "flip"); max \|Δlogit\| 1.1e-6 | exact (rounding only) |
| recu-resnet18-imagenet | 0/128 | 126 without flip (max \|Δlogit\| 1.0e-05); 2 with first flip(s) at ≤ 2.4e-08 | exact (rounding only); float64 constants: 128/128 ≤ 4.3e-14 |
| rbnn-resnet18-imagenet | 0/128 | 123 without flip (max \|Δlogit\| 1.1e-05); 5 with first flip(s) at ≤ 1.5e-08 | exact (rounding only); float64 constants: 128/128 ≤ 3.2e-14 |
| siman-resnet18-imagenet | 31/128 | 94 without flip (max \|Δlogit\| 1.5e-06); 3 with first flip(s) at ≤ 8.0e-09 | exact (rounding only); float64 constants: 128/128 ≤ 2.7e-14 |
| biper-resnet18-imagenet | 0/128 | 127 without flip (max \|Δlogit\| 1.0e-05); 1 with first flip(s) at ≤ 7.6e-09 | exact (rounding only); float64 constants: 128/128 ≤ 5.0e-14 |
| snn-subbit-reactnet18-w1a1-imagenet | 33/128 | 94 without flip (max \|Δlogit\| 2.6e-06); 1 with first flip(s) at ≤ 6.9e-09 | exact (rounding only); float64 constants: 128/128 ≤ 9.2e-14 |
| snn-subbit-reactnet18-w0.67a1-imagenet | 4/128 | 120 without flip (max \|Δlogit\| 4.1e-06); 4 with first flip(s) at ≤ 1.5e-07 | exact (rounding only); float64 constants: 128/128 ≤ 9.2e-14 |
| recu-resnet34-imagenet | 0/128 | 124 without flip (max \|Δlogit\| 1.3e-05); 4 with first flip(s) at ≤ 1.0e-08 | exact (rounding only); float64 constants: 128/128 ≤ 3.9e-14 |
| rbnn-resnet34-imagenet | 0/128 | 124 without flip (max \|Δlogit\| 9.7e-06); 4 with first flip(s) at ≤ 1.4e-08 | exact (rounding only); float64 constants: 128/128 ≤ 3.6e-14 |
| siman-resnet34-imagenet | 0/128 | 120 without flip (max \|Δlogit\| 2.0e-06); 8 with first flip(s) at ≤ 1.3e-08 | exact (rounding only); float64 constants: 128/128 ≤ 3.2e-14 |
| biper-resnet34-imagenet | 0/128 | 124 without flip (max \|Δlogit\| 1.0e-05); 4 with first flip(s) at ≤ 3.5e-08 | exact (rounding only); float64 constants: 128/128 ≤ 4.6e-14 |
| recu-vgg-small-cifar10 | 128/128 | no deviation | exact (rounding only); float64 constants: 128/128 ≤ 3.1e-13 |
| rbnn-vgg-small-cifar10 | 127/128 | 0 without flip (max \|Δlogit\| 0.0e+00); 1 with first flip(s) at ≤ 6.9e-09 | exact (rounding only); float64 constants: 128/128 ≤ 2.1e-13 |
| siman-vgg-small-cifar10 | 128/128 | no deviation | exact (rounding only); float64 constants: 128/128 ≤ 1.2e-13 |
| biper-vgg-small-cifar10 | 128/128 | no deviation | exact (rounding only); float64 constants: 128/128 ≤ 1.2e-13 |
| recu-resnet20-cifar10 | 91/128 | 37 without flip (max \|Δlogit\| 1.5e-06) | exact (rounding only); float64 constants: 128/128 ≤ 1.4e-14 |
| rbnn-resnet20-cifar10 | 0/128 | 128 without flip (max \|Δlogit\| 1.1e-05) | exact (rounding only); float64 constants: 128/128 ≤ 3.4e-14 |
| rbnn-resnet20-bireal-cifar10 | 0/128 | 128 without flip (max \|Δlogit\| 9.2e-06) | exact (rounding only); float64 constants: 128/128 ≤ 4.3e-14 |
| siman-resnet20-cifar10 | 126/128 | 1 without flip (max \|Δlogit\| 1.1e-06); 1 with first flip(s) at ≤ 3.6e-10 | exact (rounding only); float64 constants: 128/128 ≤ 7.1e-15 |
| biper-resnet20-cifar10 | 0/128 | 127 without flip (max \|Δlogit\| 3.9e-06); 1 with first flip(s) at ≤ 1.4e-10 | exact (rounding only); float64 constants: 128/128 ≤ 2.8e-14 |
| recu-resnet18-cifar10 | 0/128 | 128 without flip (max \|Δlogit\| 2.0e-06) | exact (rounding only); float64 constants: 128/128 ≤ 4.6e-14 |
| rbnn-resnet18-cifar10 | 0/128 | 128 without flip (max \|Δlogit\| 6.0e-06) | exact (rounding only); float64 constants: 128/128 ≤ 3.6e-14 |
| siman-resnet18-cifar10 | 128/128 | no deviation | exact (rounding only); float64 constants: 128/128 ≤ 1.8e-14 |
| biper-resnet18-cifar10 | 0/128 | 128 without flip (max \|Δlogit\| 9.1e-06) | exact (rounding only); float64 constants: 128/128 ≤ 3.9e-14 |
| shuaiz-bireal20-cifar10 | 128/128 | no deviation | exact (rounding only); float64 constants: 128/128 ≤ 1.1e-14 |
| finn-examples-bnn-pynq-cnv-w1a1 (FINN-ONNX) | 128/128 (max 7.5e-8) | none: 0 flips | exact; identical weights/thresholds/logits to brevitas-bnn-pynq-cnv-1w1a; float64 constants ≤ 4.4e-16 |
| finn-examples-bnn-pynq-tfc-w1a1 (FINN-ONNX) | 128/128 (3.4e-8) | none | exact; identical to brevitas-bnn-pynq-tfc-1w1a; ≤ 6.7e-16 |
| finn-examples-cnv-w1a1-gtsrb (QONNX) | 128/128 (3.1e-7) | none | exact; ≤ 1.8e-15 |
| bnn-pynq-cnv-w1a1-cifar10 (Theano) | 128/128 (7.6e-8) | fp64: none; fp32 source: 3 samples, flip at \|pre\| 1.9e-8 (source's round3 band) | exact; ≤ 6.7e-16 (fp64 reference uses the fp32-derived binary weights) |
| bnn-pynq-cnv-w1a1-gtsrb (Theano) | 128/128 (7.5e-8) | none | exact; ≤ 8.9e-16 |
| bnn-pynq-lfc-w1a1-mnist (Theano) | 128/128 (2.2e-7) | fp64: none; fp32 source: 2 samples at \|pre\| 1.7e-8 (round3 band) | exact; ≤ 8.9e-16 |
| dabnn-bireal18 | top-1 1.000 | 4 samples, first flip at ≤ 1.9e-7 | exact (rounding only); float64 constants: 128/128 ≤ 4.3e-14 |
| dabnn-bireal18-stem | top-1 1.000 / 0.992 fused | 3 / 5 samples at ≤ 7.7e-8 | exact (rounding only); ≤ 3.9e-14 |
| nncf-resnet18-binarization-xnor | 126/128 ≤ 1e-4 | 2 samples at \|x−t\| ≤ 5.6e-8 | exact (rounding only); ≤ 7.1e-14 |
| nncf-resnet18-binarization-dorefa | 122/128 ≤ 1e-4 | 6 samples at ≤ 3.7e-7 | exact (rounding only); ≤ 6.8e-14 (reference = NNCF with the CUDA kernel's per-tensor weight scale) |
| bhvit-small-somefp | 0/128 | 128 analyzed: 86 without flip (max \|Δlogit\| 3.6e-06), 42 with first flip at \|pre\| ≤ 4.6e-07 (≤ 1.5e-06 × typical; kinds {'bdense': 33, 'qkv_sign': 7, 'bconv': 2}) | exact (rounding only); float64 constants: 3.4e-14 |
| bibert-bert-base-mrpc | 128/128 | no sample deviates by > 1e-6 | exact (rounding only); float64 constants: n/a (no folded fp32 constants) |
| bibert-bert-base-mnli | 43/128 | 85 deviating samples, all with the first flip at sign(GELU) (FFN output.dense input): torch fp64 gelu is exactly -0.0 there (double erf saturates to -1 for very negative x) and the IR's glibc-erf GELU is \|x\| <= 4.6e-16, a rounding-scale flip (median typical \|pre\| 3.9e-03); max \|dlogit\| 0.90 | exact (rounding only); float64 constants: n/a (no folded fp32 constants) |
| bibert-bert-base-qqp | 106/128 | 22 deviating samples, all with the first flip at sign(GELU) (FFN output.dense input): torch fp64 gelu is exactly -0.0 there (double erf saturates to -1 for very negative x) and the IR's glibc-erf GELU is \|x\| <= 4.6e-16, a rounding-scale flip (median typical \|pre\| 3.0e-02); max \|dlogit\| 1.20 | exact (rounding only); float64 constants: n/a (no folded fp32 constants) |
| bibert-bert-base-mnli/gelu_fp32_zero_set | 128/128 | no sample deviates by > 1e-6 | exact (rounding only); float64 constants: n/a (no folded fp32 constants) |
| bibert-bert-base-qqp/gelu_fp32_zero_set | 128/128 | no sample deviates by > 1e-6 | exact (rounding only); float64 constants: n/a (no folded fp32 constants) |
| larq-zoo-quicknet-xl-legacy | 123/128 | 5 analyzed: 0 without flip (max \|Δlogit\| 0.0e+00), 5 with first flip(s) at \|pre\| ≤ 2.0e-08 | exact (rounding only); float64 constants: 6.0e-14 |
| aliasgharshinwari-finn-cnv-w1a1-cifar10 | 128/128 | no sample deviates by > 1e-6 | exact (rounding only); float64 constants: n/a (128/128 ≤ 1e-6, 0 flips) |
| ab-bnn-imagenet-reactneta | 7/128 | sign_zero IR: 121 analyzed: 0 without flip (max \|Δlogit\| 0.0e+00), 121 with first flip(s) at \|pre\| ≤ 3.5e-07 | exact (rounding only); float64 constants: 1.1e-13 |
| ab-bnn-cifar10-reactnet18 | 126/128 | sign_zero IR: 2 analyzed: 1 without flip (max \|Δlogit\| 1.1e-06), 1 with first flip(s) at \|pre\| ≤ 6.1e-08 | exact (rounding only); float64 constants: 6.4e-14 |
| ab-bnn-cifar10-reactneta | 127/128 | sign_zero IR: 1 analyzed: 0 without flip (max \|Δlogit\| 0.0e+00), 1 with first flip(s) at \|pre\| ≤ 3.8e-08 | exact (rounding only); float64 constants: 7.5e-14 |
| ab-bnn-cifar100-reactnet18 | 87/128 | sign_zero IR: 41 analyzed: 40 without flip (max \|Δlogit\| 1.4e-06), 1 with first flip(s) at \|pre\| ≤ 2.9e-10 | exact (rounding only); float64 constants: 1.2e-13 |
| ab-bnn-cifar100-reactneta | 122/128 | sign_zero IR: 6 analyzed: 5 without flip (max \|Δlogit\| 1.5e-06), 1 with first flip(s) at \|pre\| ≤ 7.8e-08 | exact (rounding only); float64 constants: 2.1e-13 |
| bitorch-resnete18-imagenet | 0/128 (rel. ≤1e-6: 79/128; logits ~115) | 128 analyzed: 79 without flip (max \|Δlogit\| 4.1e-05), 49 with first flip(s) at \|pre\| ≤ 7.6e-05; 2 top-1 changes; \|pre\| ≤ 0.98 × the max \|IR−source\| of the feeding tensor (≤ 1.2e-04 × typical) | exact (rounding only); float64 constants: 4.8e-13 |
| bitorch-densenet28-imagenet | 0/128 (rel. ≤1e-6: 84/128; logits ~105) | 128 analyzed: 84 without flip (max \|Δlogit\| 9.5e-05), 44 with first flip(s) at \|pre\| ≤ 1.7e-05; 2 top-1 changes; \|pre\| ≤ 0.74 × the max \|IR−source\| of the feeding tensor (≤ 1.5e-05 × typical) | exact (rounding only); float64 constants: 4.0e-13 |
| bitorch-meliusnet22-imagenet | 0/128 (rel. ≤1e-6: 75/128; logits ~61) | 128 analyzed: 75 without flip (max \|Δlogit\| 1.5e-05), 53 with first flip(s) at \|pre\| ≤ 5.1e-05; 3 top-1 changes; \|pre\| ≤ 0.93 × the max \|IR−source\| of the feeding tensor (≤ 7.1e-05 × typical) | exact (rounding only); float64 constants: 2.1e-13 |
| bi-vit-deit-tiny | 127/128 (≤ 1.1e-14) | 1 sample: no IR flip; the fp64 **reference** matmul leaves ±5.6e-17 at 396 exact-zero counts (torch.sign → ±1) | exact; float64 constants identical (reference = official fp64 with exact binary weights `mean|W|·sign W`) |
| bi-vit-deit-small | 128/128 (≤ 2.0e-14) | none | exact; float64 constants ≤ 2.0e-14 |
| gsb-deit-s-cifar100-stage2 | 121/128 | 6 samples, first flip at ≤ 1.5e-7 (≤ 1.8e-7 × typical); 1 sample only 8-bit head-input level changes | exact (rounding only); float64 constants: 128/128 ≤ 1.2e-14 (reference with the captured fp32 quantized weights, batch 1) |
| sa-bnn-resnet18-imagenet | 17/128 (125/128 ≤ 1e-4) | 111 analyzed: 108 without flip (max \|Δlogit\| 2.4e-06), 3 with a single first flip at \|pre\| ≤ 2.0e-07 (≤ 2.9e-7 × typical); 1 top-1 change; source fp32 vs fp64 itself: 5/128 top-1 changes (BN output binarized directly) | exact (rounding only); float64 constants: 128/128 ≤ 7.5e-14 |
| ida-det-bires18-imagenet-backbone | 7/128 (124/128 ≤ 1e-4) | 121 analyzed: 117 without flip (max \|Δlogit\| 3.5e-06), 4 with a single first flip at \|pre\| ≤ 6.3e-08 (≤ 1.2e-7 × typical); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 1.3e-13 (reference = official backbone code + assumed AvgPool/fc head) |
| larqface-binarydensenet28-msceleb (128-D embedding, LFW eval128) | 126/128 | 2 samples, single first flip at \|pre\| ≤ 5.7e-8 (≤ 1.8e-7 × typical) | exact (rounding only); float64 constants: 128/128 ≤ 1.2e-14 |
| larqface-binarydensenet37-msceleb | 128/128 (≤ 3.3e-7) | none | exact; float64 constants ≤ 1.3e-14 |
| larqface-binarydensenet45-msceleb | 126/128 | 2 samples, single first flip at \|pre\| ≤ 3.4e-8 (≤ 5.6e-7 × typical) | exact (rounding only); float64 constants ≤ 2.4e-14 |
| larqface-binaryfacenet-msceleb (DoReFa k=1) | 122/128 | 6 samples, first flip at a DoReFa threshold site, \|x − T\| ≤ 3.8e-7 (≤ 3.8e-7 × typical) | exact (rounding only); float64 constants ≤ 9.2e-15; reference = float64 model with the larq quantizer in float32 (pure float64 is a different function: cos_min 0.83) |
| larqface-quicknet-msceleb | 128/128 (≤ 2.7e-7) | none (fused IR: 124/128) | exact; float64 constants ≤ 1.5e-14 |
| brillsson-binarydensenet28 (1024 + 576 outputs up to ~60) | 45/128 | 83 analyzed: 82 without flip (max \|Δ\| 1.4e-6), 1 with a single flip at \|pre\| ≤ 5.9e-8 | exact (rounding only); float64 constants ≤ 6.4e-14 |
| brillsson-meliusnet22 | 123/128 | 5 analyzed: 2 without flip (max \|Δ\| 1.2e-6), 3 with a first flip at \|pre\| ≤ 2.6e-7 | exact (rounding only); float64 constants ≤ 9.2e-14 |
| brillsson-quicknet | 0/128 (outputs up to 63; rel. ≤ 3e-8 without flip) | 128 analyzed: 127 without flip (max \|Δ\| 1.8e-6), 1 with a single flip at \|pre\| ≤ 1.6e-7 | exact (rounding only); float64 constants ≤ 6.4e-14 |
| bbcu-denoise (image output, range [0, 1]) | 64/68 crops ≤ 1e-6 × range | 4 crops, first flip at \|pre\| ≤ 1.3e-06 (≤ 6.9e-08 × typical) | exact (rounding only); float64 constants: 68/68 ≤ 1.6e-15 |
| bbcu-deblocking (image output, range [0, 1]) | 63/68 crops ≤ 1e-6 × range | 5 crops, first flip at \|pre\| ≤ 9.8e-08 (≤ 6.1e-08 × typical) | exact (rounding only); float64 constants: 68/68 ≤ 4.4e-16 |
| bbcu-sr-l-x2 (image output, range [0, 1]) | 63/64 crops ≤ 1e-6 × range | 1 crops, first flip at \|pre\| ≤ 3.5e-09 (≤ 1.7e-09 × typical) | exact (rounding only); float64 constants: 64/64 ≤ 1.1e-15 |
| bbcu-sr-l-x4 (image output, range [0, 1]) | 63/64 crops ≤ 1e-6 × range | 1 crops, first flip at \|pre\| ≤ 5.6e-08 (≤ 2.5e-08 × typical) | exact (rounding only); float64 constants: 64/64 ≤ 1.2e-15 |
| bbcu-sr-m-x2 (image output, range [0, 1]) | 60/64 crops ≤ 1e-6 × range | 4 crops, first flip at \|pre\| ≤ 2.6e-08 (≤ 3.3e-08 × typical) | exact (rounding only); float64 constants: 64/64 ≤ 1.0e-15 |
| bbcu-sr-m-x4 (image output, range [0, 1]) | 62/64 crops ≤ 1e-6 × range | 2 crops, first flip at \|pre\| ≤ 3.4e-08 (≤ 1.5e-08 × typical) | exact (rounding only); float64 constants: 64/64 ≤ 1.1e-15 |
| bsrc2023-binary-sr-x2-v0 (image output, range [0, 255]) | 63/64 crops ≤ 1e-6 × range | 1 crops, first flip at \|pre\| ≤ 9.8e-08 (≤ 1.1e-08 × typical) | exact (rounding only); float64 constants: 64/64 ≤ 5.1e-13 |
| bsrc2023-binary-sr-x2-v1 (image output, range [0, 255]) | 63/64 crops ≤ 1e-6 × range | 1 crops, first flip at \|pre\| ≤ 5.3e-08 (≤ 1.3e-08 × typical) | exact (rounding only); float64 constants: 64/64 ≤ 5.1e-13 |
| bsrc2023-binary-sr-x4-v0 (image output, range [0, 255]) | 64/64 crops ≤ 1e-6 × range | none | exact (rounding only); float64 constants: 64/64 ≤ 1.3e-12 |
| bsrc2023-binary-sr-x4-v1 (image output, range [0, 255]) | 64/64 crops ≤ 1e-6 × range | none | exact (rounding only); float64 constants: 64/64 ≤ 2.2e-12 |
| adabin-resnet20-cifar10 | 17/128 (128/128 ≤ 1e-4, max 1.7e-6) | 111 analyzed: 111 without flip (max \|Δlogit\| 1.7e-06); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 3.9e-14 (reference = official model in float64 with the fp32 weight levels W± substituted; asymmetric-set decomposition incl. the constant border map) |
| fda-bnn-resnet20-cifar10-mindspore | 128/128 (max 5.6e-7) | none | exact (rounding only); float64 constants: 128/128 ≤ 1.2e-14 (reference = cited PyTorch port in float64; port fp32 vs official MindSpore fp32: max 7.6e-6, 128/128 top-1; 10,000 images: 2 top-1 differences, fp32 chaos) |
| tensorpack-dorefa-alexnet-w1a1-g32 | 0/128 (128/128 ≤ 1e-4, max 3.1e-6) | 128 analyzed: 0 binarization flips (max \|Δlogit\| 3.1e-06 = folded-constant rounding); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 2.2e-13 (reference = cited PyTorch port in float64; official TF fp32 vs port fp64: 2/128 top-1 changes = the fp32 noise floor, port fp32 vs fp64 1/128; eval5000: 64 vs 61 of 5000) |
| bnas-a-cifar10 | 127/128 (max 0.13) | 1 sample, first flip at \|pre\| 1.6e-08 (7.5e-08 × typical); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 1.4e-13 (reference = official first-release model in float64 after BinOp; 9,437,184 structural exact zeros at sign(), ge variant corrected exactly) |
| bnas-d-imagenet | 126/128 (max 1.3) | 2 samples, first flip at \|pre\| ≤ 2.0e-09 (≤ 1.9e-08 × typical); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 3.7e-14 (BN before sign as sign(s)·sign(x + b/s); source f32 vs f64: 1/128 top-1 changes) |
| bnas-e-imagenet | 124/128 (max 1.5) | 4 samples, first flip at \|pre\| ≤ 2.1e-08 (≤ 1.1e-07 × typical); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 4.4e-14 (source f32 vs f64: 2/128 top-1 changes) |
| bnas-h-imagenet | 120/128 (max 2.1) | 8 samples, first flip at \|pre\| ≤ 3.3e-08 (≤ 9.3e-08 × typical); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 5.0e-14 (source f32 vs f64: 9/128 top-1 changes, Bitino numpy f32 vs f64: 3/128) |
| bimatting-bimobilenet-backbone | 115/128 (max 2.4) | 13 analyzed: 8 without flip (max \|Δlogit\| 1.3e-06), 5 with a first flip at \|pre\| ≤ 6.5e-08 (≤ 1.9e-07 × typical); 0 top-1 changes | exact (rounding only); float64 constants: 128/128 ≤ 6.4e-14 (reference = official BiMobileNet.forward_single_frame + checkpoint head in float64; source f32 vs f64: 29/128 top-1 changes, Bitino numpy f32 (ge) vs f64: 0/128 thanks to the refolded thresholds) |
| burn-reacta-stage2 (1024-d embedding) | 120/128 (max 0.17, cos_min 0.99762) | 8 analyzed: 8 with a first flip at \|pre\| ≤ 7.7e-08 (≤ 8.4e-08 × typical) | exact (rounding only); float64 constants: 128/128 ≤ 3.2e-14 (source f32 vs f64: cos median 0.9966, min 0.9908; Bitino numpy f32 vs f64: median 1 − 1e-14, 13/128 below 0.99999) |
| s2bnn-reacta-distill-only (1024-d embedding) | 119/128 (max 0.35, cos_min 0.9637) | 9 analyzed: 9 with a first flip at \|pre\| ≤ 5.6e-08 (≤ 1.05e-07 × typical) | exact (rounding only); float64 constants: 128/128 ≤ 6.4e-15 (source f32 vs f64: cos median 0.970, min 0.937; Bitino numpy f32 vs f64: median 1 − 1e-14, 13/128 below 0.99999) |

The DenseNet-style models (and MeliusNet) binarize BatchNorm outputs of a growing concatenation
directly. Their logits deviate by up to ~1e-5 even without a flip, which is the size expected
from folded-BN constants stored in float32. In every flip sample of these models, the |pre| at
the first flip is below the max |IR − source| of the tensor that feeds the flipping layer
(ratio ≤ 0.93). For k-bit (DoReFa) inputs, `tools/first_flip_analysis.py` treats a different
integer level as a flip and uses the distance to the nearest level boundary as |pre|.
Full analysis: the development log (not published).
The S7 (FINN/QONNX/BNN-PYNQ) and S9 (daBNN, NNCF) rows are detailed in the development log (not published).
The S10 binary-DeiT rows (Bi-ViT ActBi affine codes as exact float32 thresholds + BMatMul counts; GSB group superposition,
per-sample max thresholds) are detailed in the development log (not published).
The S12 rows (AdaBin ResNet-20: adaptive binary sets {α(β−1), α(β+1)} with zero padding, decomposed into a sign-bit BConv2D, a one-channel count BConv2D + 1×1 Conv2D and a constant border map; FDA-BNN ResNet-20 from the MindSpore checkpoint, reference = cited PyTorch port cross-checked against the official MindSpore code; tensorpack DoReFa AlexNet W1A1, {0,1} activations as `gt(0.5)` with `pad_value −1`, reference = official tensorpack graph (fp32) + cited port (fp64)) are detailed in the development log (not published).
The S13 BNAS rows (official yonseivnl/bnas cell networks with the first-release BinOp, whose preprocess_res convs are binarized; BN before sign folded as sign(s)·sign(x + b/s); Zero ops / structurally zero channels) are detailed in the development log (not published).
The S11 rows (ReActNet-A with BatchNorm: BURN, S2-BNN, BiMatting backbone; BN folded into the BConv2D epilogue; binarization thresholds after a PReLU tail refolded as q + fl64(b_prev + b_next), which removes the sources' float32 ties on channels with tiny PReLU slopes and nearly cancelling biases) are detailed in the development log (not published).
The S1-remainder rows (SA-BNN ResNet-18: pre-sign BN, two binary convs per block; IDa-Det BiRes-18 backbone: BiConv with
bias, head assumed) are detailed in the development log (not published). The S8 image-to-image rows (BBCU denoise / deblock / SR, BSRC-2023 SR; image outputs, per-crop max |diff| relative to the output range, first flip from float32 vs float64 constants) are detailed in the development log (not published). Bi-ViT's released fp32 code scores
~1% because torch's fp32 erf-GELU returns exactly 0 for x ≤ −5.54 (torch.sign → 0); the IR's double GELU (cut-off −8.37)
matches the float64 reference.
The IR-Net-family / SNN / Bi-Real-20 rows (ImageNet `torchvision_imagenet`, CIFAR-10 `cifar10_*_norm`) compare against
the float64 source with the fp32-derived binary weights; see the development log (not published).
The S3–S5 rows (BHViT-Small-somefp, BiBERT MRPC/MNLI/QQP, legacy QuickNet-XL, the CNV retrain, AB-BNN ReActNet-A and CIFAR
models, bitorch ResNetE-18 / BinaryDenseNet-28 / MeliusNet-22) are detailed in the development log (not published).
bitorch models have unscaled activations (|x| up to ~50, logits ~10²), so their absolute deviations are larger at the same
relative size; their optimized-tier `--explain` uses `--flip-rule feed` (first runtime flip at |pre| ≤ the C++-vs-numpy
difference of the binarized tensor). BiBERT MNLI/QQP depend on the fp32 GELU underflow (sign(gelu) = 0 for x ≲ −5.546);
`tools/convert_bibert.py --gelu-fp32-zero-set` reproduces that zero set (BiBERT specifics below).

Bi-RealNet is more sensitive than QuickNet. It binarizes BN outputs directly, with no ReLU in
front of the sign, so pre-activations concentrate around 0.

BHViT is far more sensitive than the CNNs: a single flip at |pre| ~1e-7 typically cascades into
10^5 flips over ~80 binarization sites, and the source itself is fp32-chaotic (fp32 vs fp64 top-1
agreement 0.84 / 0.81 for Tiny / Small). Its flip sites include the Q/K/V signs and the 2-bit attention
quantizer (`tools/first_flip_bhvit.py`; a quantizer flip is a different `round(3P/alpha + zp)` level, and its
|pre| is the distance to the nearest `k + 0.5` boundary). For such models the default Level-4 top-1 rule
cannot pass even for the source's own fp32 run. Acceptance is therefore: (a) C++ bit-exact with the numpy
executor at the exact stages, (b) every IR-vs-fp64 deviation traced to a rounding-scale first flip, and
(c) `validate_model.py --explain` for inexact tiers (its per-sample tensor dumps run only the traced sample, so
the analysis is practical under qemu as well).

## Level 4/5 acceptance

`tools/validate_model.py` requires:

1. top-1 agreement with the float64 reference ≥ 0.98, and
2. top-1 disagreements ≤ (disagreements between the source fp32 and fp64) + max(1, 1% of samples).

It also reports agreement with the numpy executor running the *same* `.bitino` in float32.
For every validated model so far, the C++ runtime (all kernel tiers, 4 threads) is
**bit-identical** to the numpy executor on the host, and the S22 is bit-identical to the host.
That holds because:

* binary kernels accumulate in exact integers and apply the post-scale identically;
* float reference kernels accumulate in double; the numpy executor mirrors their exact summation order
  (Conv2D/Dense start from the bias and add in (ky, kx, c) / k order; pools, Softmax, LayerNorm, ReduceMean
  sum sequentially), so float32 results are bit-identical, not just within a bound;
* the build uses `-ffp-contract=off`, so no implicit FMA contraction changes rounding.

**Image-to-image models** (slice S8: BBCU denoising / JPEG deblocking / SR, BSRC-2023 SR) have no top-1.
`validate_model.py --output-kind image` reads the ground-truth crops and the metric from the eval set's manifest
(`targets`, `image_task`; built by `tools/eval_sets/restoration.py`, one center crop per test image) and reports per-sample
max |diff| and PSNR against the float64 / float32 references and the numpy executor, plus the task PSNR/SSIM against
ground truth with the source's own protocol (`bitino.validation.image_metrics`, cross-checked against the KAIR /
BasicSR / BSRC metric code: identical, BSRC ≤ 4e-7 dB). Exact stages pass by bit-exactness with the numpy executor
(`--accept-exact-numpy`); `--explain` traces every sample whose output differs from the numpy executor by more than
`--image-tol` (1e-3) × the output range to its first binarization flip (same rounding-scale rule); the default criterion
(device / INT8) is a task-PSNR drop vs the float64 reference within the source's float32-vs-float64 difference + 0.02 dB and
no sample below 30 dB PSNR against the float64 reference. Full-image PSNR/SSIM on the standard sets: `tools/restoration_eval.py`.

Optimized float kernels (`float_opt`) and INT8 kernels are *not* expected to be bit-exact with
the reference. Their outputs are validated with the same top-1 criteria, and their kernel-level
error bounds are documented in their tests.

Summation order matters for the `--explain` rule. Example (x86-64 host, AVX2):
`larq-zoo-binary-densenet37-dilated`, sample 112. The 7x7 stem conv (K = 147) feeds ReLU,
max-pool and a folded BatchNorm with scale 43.7 and shift −256.0 whose output is binarized. One
stem output is 5.851619 in the reference (double accumulation, one rounding) and 5.851622 with a
single sequential fp32 FMA chain (+6 ulp). BatchNorm then computes 256.0 − 256.0 by cancellation:
−1.37e−4 in the reference, exactly 0.0 in fp32, so one sign flips at |pre| = 1.7e−4 × typical,
just above the `--flip-rel 1e-4` threshold. The same output with two or more interleaved
accumulators, or blocked summation, rounds to the reference value. The AVX2 GEMM tiles therefore
keep at least two independent accumulator chains per output (3x16 and 6x8 tiles, k split between
chains), which halves the chain length at no measurable cost. The portable and NEON kernels keep
their summation order (after the refactoring that added the AVX2 backend, the portable path
reproduces the previous outputs bit for bit on all 31 models without grouped convolutions, and the
NEON path on the 8 models checked under qemu).

### Transformer / BHViT kernels and plan-time rewrites (2026-10-30)

* **Bit-exact by construction** (same IEEE operations, same order per element, verified bit for bit in
  `tests/cpp/test_transformer_opt.cc` on x86 and emulated ARM): the plan-time rewrites (threshold folding,
  `FusedElementwise`, BConv2D epilogues; end to end, fused and `--no-fuse` outputs are byte-identical on the
  eval sets), `layernorm.float_opt` (double formula, several rows per vector, each lane summing its row in
  index order), `reduce_mean.float_opt`, `fake_quant.float_opt` (vector IEEE division + round-half-even),
  `add_broadcast`, `transpose`, `concat`, `gather`, `mul`, `channel_gate` (sigmoid still in double),
  hardswish / sign / sign_zero, the BMatMul bit-plane kernels, and INT8 MatMul vs `int8_reference`.
  NEON `clamp` was changed to the exact `std::min(std::max(v, lo), hi)` selection: FMAX turned
  `relu6(−0)` and FakeQuant's clamp of `−0` into `+0`.
* GELU on NEON (`gelu_neon`, 2026-11-02) evaluates exactly the per-vector formula (`vgelu`) lane by
  lane but skips work whose result the selects would discard: the small-`|x|` Taylor branch for vectors
  without such lanes, and, in blocks of 256 with ≥ 5/16 lanes at or below the cutoff (`y = x·0`), the whole
  formula for those lanes (live lanes are compacted with TBL, evaluated, and expanded back). It is
  bit-identical to the plain path (`activation_ext_gelu_fast_path_bit_identical`, also for all 2^32 inputs
  with `BITINO_EXHAUSTIVE_GELU=1`).
* **Bounded**: GELU `≤ 6e-7·|ref| + 2^-52·|x|` and **sign-exact** (the reference is exactly `−0` for
  `x ≤ −8.37438965`, where `1 + erf(x/√2)` rounds to 0 in double; the kernel reproduces that cutoff, and
  `Phi > 0` elsewhere), sigmoid / tanh `≤ 4e-7·|ref| + 1e-37`, Softmax `≤ 2e-6·|ref| + 1e-12`, MatMul
  float GEMM `≤ 1e-4·(1+|ref|)`.
* Consequence for BiBERT: its GELUs feed only `sign_zero` binarizations and LayerNorm is exact, so at stage
  `optimized` the encoder is bit-identical to the numpy executor; only the pooler `tanh` and the classifier
  differ (max |Δlogit| vs numpy 9.5e-7). (Since 2026-11-12 the fused stages select `layernorm.float_fast` on
  AArch64, see below; the exact kernel can still be forced.)

### Float32 fast kernels of the fused stages (2026-11-12)

`runtime/kernels/float_opt/transformer_fast.h`, tier `float_opt`, priority 45: preferred over the exact
`float_opt` kernels above (40) by every stage that allows `float_opt` (`optimized`, `optimized_int8`, `int8_full`,
`all`), never by the bit-exact stage `neon` (reference LayerNorm / Softmax / ReduceMean / Activation). On x86-64
hosts with AVX2 the `*_avx2` kernels (50) still win, so host outputs are unchanged. Bounds against the double
reference, measured in `tests/cpp/test_transformer_opt.cc` (host x86-64 portable path / emulated arm64 NEON path):

| kernel | arithmetic | bound (asserted) | measured |
|---|---|---|---|
| `layernorm.float_fast` | float sums in 16 lanes, corrected two-pass mean / variance, per-row scalars in float (4 rows per vector), `y = ((x − mh) − ml)·r·g + b` (one FMA) | `|y − ref| ≤ 4·2⁻²³·((|t| + 1)·|g| + |b|)`, t = exact normalized value | 1.56 / 1.51 of 2⁻²³·(…); without γ/β and |t| ≥ 0.25: ≤ 3 ulp (max abs 7.2e-7) |
| `reduce_mean.float_fast` (last axis) | len 4 / 8: FADDP trees, else 16-lane float sums; × 1/len (exact for powers of two) or one division | `≤ (len/16 + 6)·2⁻²³·mean|x|` | 0.19 of the bound (max abs 1.9e-6 at |x| ~ 50) |
| `softmax.float_fast` | exp by the shift trick + Cephes polynomial, lower clamp only, y = e·(1/Σ) | `≤ 2e-6·|ref| + 1e-12` | 0.13 / 0.12 of the bound; ≤ 3 ulp where ref > 1e-30 |
| `activation_gelu.float_fast` | x·Φ(x), Φ from erfc for every lane (no small-|x| branch), cut-offs of the exact kernel | sign-exact; `≤ 1e-6·|ref| + 2⁻⁵²·|x|` | 0.46 of the bound; max relative 4.6e-7 for x > −5.5 (below that the double reference cancels) |

LayerNorm's error is relative to `(|t| + 1)·|g|`, not to |y|: the mean comes from float sums of the deviations
(absolute error ~2⁻²³·sd), so a value near the mean is only accurate to ~2⁻²³ in normalized units (the
double kernel is exact there). ReduceMean over other axes keeps `reduce_mean.float_opt` (double).
The binary kernels' fused chain epilogues (`EltwiseProgram::store_unit16`, `docs/operators.md`) are bit-exact.

End to end (`tools/validate_model.py`, static arm64 build under qemu, 4 threads, 128 samples, 2026-11-12): stage `all`
passes for Bi-ViT DeiT-Tiny / -Small (top-1 agreement vs fp64 0.953 / 0.938, all disagreements traced to
rounding-scale first flips by `--explain`; the same counts as before), BHViT-Tiny / -Small (0.813 / 0.836, `--explain`,
as before), BiBERT SST-2 (1.000, max |Δlogit| vs numpy 1.2e-6), ReActNet-18 (0.984), QuickNet (0.992, final Softmax
only), BNext-Tiny (0.953) and the Brevitas CNV (exact vs numpy); stage `neon` stays bit-exact with the numpy executor
(Brevitas CNV, QuickNet, ReActNet-18 re-run) and byte-identical to the previous emulated references for all nine
models. The emulated references changed only for the models with LayerNorm / Softmax / ReduceMean / GELU at the
`float_opt` stages (Bi-ViT, BHViT, QuickNet's Softmax); BiBERT's outputs are unchanged (its LayerNorm outputs only
reach the logits through `sign_zero` binarizations), and the CNNs without them (ReActNet-18, BNext-Tiny, Brevitas)
are byte-identical, which also checks the in-register chain epilogues end to end.

## 32-bit ARM (armeabi-v7a)

AArch32 Advanced SIMD always uses the "standard FPSCR value": round to nearest, default NaN and
flush-to-zero, so a vector float op with a denormal operand or result sees / produces +-0 (AArch64 NEON and
the scalar VFP unit are IEEE). Measured on the Redmi 9A's Cortex-A53 (`a32_neon_flushes_denormals`). What
the AArch32 builds of the NEON kernels do about it:

* Integer work (popcounts, bit counts, the dot products) is unaffected.
* Binarization is exact: compares against 0 work on the bit patterns (`vcgezq_f32` etc. in
  `kernels/neon_a32.h`, e.g. -1e-40 >= 0 is false as in IEEE), compares against thresholds use the float
  compare only when no threshold is a denormal and no threshold that is +-0 meets a denormal input (a flushed
  denormal keeps its order against a normal threshold), otherwise the scalar code packs that row.
* Clang also auto-vectorizes the *scalar* float compare loops into AArch32 NEON (it did so in the old
  portable armeabi-v7a build: `pack_row_f32`, the ternary packing, `Activation sign` of the portable
  float_opt), so those loops carry `BF_A32_IEEE_LOOP` (`#pragma clang loop vectorize(disable)` on 32-bit ARM).
* float_opt selections against 0 (`relu`, `sign`, `sign_zero`, prelu / leaky_relu selects, clamp bounds 0)
  are exact; GELU and hardswish of |x| < 2^-124 are fixed up to the reference value (x/2 resp. the scalar
  formula), so `sign(gelu(x))` always equals the reference's.
* Float arithmetic whose result (or an operand) is a denormal is flushed: a BConv2D epilogue
  `dot * mul + add` cannot produce one unless `|mul|` or `|add|` is below ~2^-100, but a fused elementwise
  epilogue reading a denormal activation (e.g. `x * gate`, `0 + r` with a denormal r) gives 0 where numpy gives
  the denormal. Activations of trained networks are not denormal in practice; the tests that inject 1e-40
  replace it by a tiny normal value on armeabi-v7a (`with_specials`, test_transformer_opt.cc). The old
  portable armeabi-v7a build was not exact there either (6 bitino_tests failed on the Redmi with the
  auto-vectorized compares; all pass with the NEON build).
* Semantics otherwise follow the portable path, not the AArch64 one: `madd` is VMLA (rounded product, then
  add), `fma_` is VFMA (VFPv4, runtime check) or `std::fma`, division is exact (VFP, or a multiply by the
  exact reciprocal of a power-of-two divisor), the GEMM tile accumulates in the same k order
  (`float_conv2d_neon_gemm_tile_order_exact`). Stage `neon` stays bit-identical to the numpy executor on the
  device eval sets (validate_model.py `--accept-exact-numpy`); stage `all` is judged by the usual tolerances.

* int8_full on armeabi-v7a (2026-11-12): the `int8_full_neon` kernels run there through `kernels/neon_a32.h`.
  Integer kernels (LUTs, AffineSum, Add, Mul, pooling, means, the SMLAL GEMM `i8conv2d.neon_a32`) are exact on both
  ISAs. The fp32 steps end in an int8 rounding, which absorbs a flushed denormal (a denormal and +-0 both round to
  code 0): I8Quantize (`x / s` by IEEE VFP division, or an exact power-of-two reciprocal), I8LayerNorm (`u * r` with an
  integer `|u| >= 1` cannot be denormal; `v * G` denormal only for `|G| < 2^-100`, and then `w + B` equals the IEEE sum
  whenever its rounding can be nonzero), I8Softmax, the int8 outputs of the binary layers. fp32 graph outputs
  (`output: f32` of I8BDense / I8BConv2D / I8BMatMul / I8Softmax) stay on the IEEE portable / reference kernels on
  armeabi-v7a, so int8_full outputs are byte-identical to the AArch64 ones (checked under qemu-arm and on the Redmi 9A).
* AArch32 int8 kernels (2026-11-12, later): the VMULL.S8 GEMM tiles are integer (exact for weights in [-127, 127];
  I8MatMul repacks b codes of -128 as -127 and restores the products). I8Quantize and the int8_neon quantizer use one
  NEON multiply by `1/s` and the IEEE division (VFP) for groups holding a value within 2^-12 of a half-integer (the
  multiply is within 3 ulp of `fl(x / s)`, below 2^-13 for the reachable range) or a denormal input (flushed by the
  multiply). The int8_neon fp32 epilogue `float(acc) * (sx * sw[o]) + bias[o]` runs in NEON only when every
  `sx * sw[o]` is 0 or at least 2^-100 and every bias is 0 or normal (then no operand or result is a denormal; checked
  per run), else in scalar VFP; max |x| is taken on the bit patterns. I8BConv2DRes runs the NEON epilogue when every
  value of its chain (`fp32(acc) * post_mul + post_add`, pre stage, `+ R(res)`, PReLU, `+ out_bias`) is a multiple of
  2^G with G >= -126 (least-significant-bit exponents of the constants added along products, minimised along sums;
  IEEE rounding keeps the grid), so every nonzero intermediate is normal; bit thresholds that are denormal
  (`nextafter(0, +inf)` of a `gt` compare with 0) are replaced by the grid values giving the same compares
  (`2^-126` resp. 0). Otherwise the IEEE portable kernel. qemu-arm does not flush NEON arithmetic (it did flush the
  compare in the one failing test case), so these properties were checked on the Redmi 9A.
* I8BConv2DRes and the l8 output of I8Conv2D on armeabi-v7a (2026-11-13): their fp32 epilogues (`v = fp32(acc) *
  post_mul + post_add`, the pre stage, the residual `R`, `u = v + R`, PReLU, bias, the l8 code and the bit compares of
  `u` / `w`) run on AArch32 NEON when a plan-time proof shows that **no value of the chain can be denormal**
  (`int8_full::FtzBound`, `res_epilogue_normal`, `gemm_f32_epilogue_normal`): FTZ only changes denormal operands and
  results, so the NEON values are the IEEE ones. Each value carries a lower bound `L` of its nonzero magnitudes and a
  power of two `G` it is a multiple of: an integer has `L = G = 1`; `x * c` (constants over channels) gives
  `L' = 2^floor(log2(L min|c|))` (RN is monotone) and `G' = L' 2^-23` (every float `>= L'` is a multiple of it);
  `x + y` gives `G' = min(G_x, G_y)` and `L' = G'` (a nonzero multiple of `G'`; RN keeps the sum a multiple of `G'`);
  PReLU the minimum of both branches; a nonzero l8 residual is `>= 2^(e-16)`, a multiple of `2^(e-39)`. The kernel is
  taken only if every `L >= 2^-126` and every constant (thresholds after the `gt` adjustment included) is 0 or normal.
  All 379 I8BConv2DRes nodes of the zoo pass; the others would stay on the portable kernel. Unit test
  `int8_full_ftz_bound` checks the bound against brute force (a chain it accepts never reaches a denormal) and its
  rejections.

## Binarization semantics that must be preserved

| Source | sign(0) | padding of binarized inputs | IR attributes |
|---|---|---|---|
| larq (`ste_sign`, `approx_sign`, `SteSign`) | +1 (`tf.sign(tf.sign(x)+0.1)`) | `pad_values` (QuickNet: +1, default 0) | `input_binarize=ge`, `pad_value` |
| PyTorch `torch.sign` (ReActNet-style, AB-BNN) | 0 | zero padding (contributes 0) | `sign_zero` (exact, reference-only) or `ge` (deployable; exact-zero count reported) |
| Bi-Real Caffe | −1 (`x>0?1:-1`) | zero | `gt`, `pad_value=0` |
| BHViT (`torch.sign`, LSQ-style 2-bit attention) | 0 for activations; weights `sign(W − mean W)` computed in fp64 (no zeros) | zero (`padding='same'` on the binarized input) | `BDense/BConv2D threshold=−move, ge` (deployable) or `sign_zero`; Q/K/V `BMatMul sign_zero` / `Activation sign_zero`; probs `FakeQuant` (half-to-even) |
| BNext (`HardSign` STE, `HardBinaryConv` at temperature ≈ 7e−24) / RBONN (`BinaryActivation` STE, `RBOConv`) | 0 (the STE `(sign − r) + r` returns `torch.sign` exactly in fp32 and fp64); weights `sign(W)`, no zeros | zero | `BConv2D sign_zero` or `ge`; BNext has no weight scale, RBONN `post_mul = 0.02·|scale_factor|` (`docs/operators_torch_cnn.md`) |
| IR-Net family (ReCU, RBNN, SiMaN, BiPer; `BinarizeConv2d`), Bi-Real-20 | 0 (`torch.sign`; RBNN/SiMaN binarize `a − mean_CHW(a)` per sample) | zero | `BConv2D sign_zero` or `ge`, `post_mul = alpha`; centering = `Reshape → ReduceMean → ChannelAffine(−1) → AddBroadcast`; weights = the fp32 source's own `bw` (ReCU clamp, RBNN rotation, BiPer `sign(sin(freq·w))`, SiMaN `2B−1`), checked against the `F.conv2d` argument (`operators_torch_cnn.md`) |
| SNN ReActNet-18 (`HardBinaryConv`, 0.67-bit pattern decode) | 0 | zero | as RBONN, `post_mul = mean\|W\|`; sub-bit kernels decoded to ±1 offline |
| SA-BNN ResNet-18 (`BinaryActivation` of a BN output, `HardBinaryConv`) / IDa-Det BiRes-18 (`BiConv` with bias) | 0 | zero | SA-BNN: `ChannelAffine` (pre-sign BN) → `BConv2D` (`sign_zero` or `ge`, `post_mul = mean\|W\|`); IDa-Det: as RBONN plus `post_add = conv bias` |
| insightface_larq_keras BinaryFaceNet (`DoReFa(k_bit=1, mode="weights")` on inputs and kernels) | −1 (`round(0.5)` = 0, half to even); float32: **every** `0 < x ≤ 2^-24·max\|tanh x\|` → −1 (whole near-dead channels; 1.69M values in 128 LFW images); float64: band ~2^-53 (a different function: pure-fp64 vs fp32-rule source cos_min 0.83) | zero (`pad_values` 0, after quantization) | `BConv2D gt` on `x − 2^-24·tanh(max\|x\|)` (per sample, existing ops; `operators_larq.md`); kernels ±1 from the live quantizer; the float64 reference runs the larq quantizer in float32 on float32(x) |
| BRILLsson / insightface_larq_keras QuickNet & BinaryDenseNet (larq `SteSign`) | +1 | QuickNet `pad_values` 1, BinaryDenseNet / MeliusNet 0 | as the Larq Zoo models (`ge`); BRILLsson `GlobalMaxPool2D` = full-window `MaxPool2D` at the fixed 98×64 log-mel input (STFT/mel on the host) |
| BiBERT (`torch.sign`, ZMeanBinaryQuantizer) | 0 for weights *and* activations; attention probs `score > 0 → 1 else 0` | none (linear layers) | `BDense input_binarize=sign_zero` (+ `w_zero` for exact-zero weights), `BMatMul sign_zero / gt0`; exact in `reference` and `bdense.ternary_bitplane` |
| bitorch model hub (`QConv2d`: `Sign` = `torch.sign` with 0 → +1; `ProgressiveSign` weights = `Sign` in eval) | +1 (also for −0.0) | −1, applied to the *binarized* input (`F.pad(sign(x), value=−1)`) | `BConv2D ge, pad_value −1`, no scale: exact on the deployable path, no `sign_zero` variant |
| Brevitas BNN-PYNQ (`binary_sign` = `torch.ge(x,0)*2-1`) | +1 (also for −0.0) | none (all convs VALID) | `ge`; BN+sign → integer thresholds `s·(acc−T)` (exact), see `operators_brevitas.md` |
| FINN-ONNX `MultiThreshold` BIPOLAR / QONNX `BipolarQuant` | +1 (`x >= thr`) | none (VALID) | as Brevitas (`finn_bnn` emitter); int8 input quantizer verified on all 256 pixel values |
| BNN-PYNQ Theano (`2·round3(hard_sigmoid(x))−1`; weights `round(hard_sigmoid(W))` half-to-even) | +1, and also for x in (−~3e-8, 0) in fp32 (round3 band) | none (VALID) | `ge` + integer thresholds; pool-before-BN → thresholds (+ input-channel polarity) in the next conv; weights = Theano's fp32 quantized weights |
| daBNN (onnx2bnn implicit binarization) | +1 for +0, −1 for −0 (aarch64 sign-bit packing) | −1 | `BConv2D ge, pad_value −1` (−0.0 never occurs on the eval set) |
| Bi-ViT (`BiLinearBiReal` torch.sign; `ActBi` codes `(sign(fl(x/α)+r) − r)·α`) | 0 (structural: integer counts with no bias are often exactly 0) | none (linear layers) | qkv/fc1 `BDense ge` (deployable) or `sign_zero`; proj/fc2 always `sign_zero`; ActBi r≠0 → operand `sign(α)·x − T_lo` (exact float32 tie float), `BMatMul sign_zero`, codes expanded into integer counts |
| GSB (`sign(x+move)`, unsigned fc2 `round((x+move)/α) ≥ 1`, masks vs `0.7/0.9·max`) | 0 (torch.sign) | none | `BDense ge|sign_zero, threshold −move`; fc2 `BDense gt` with exact float32 threshold; masks `BMatMul gt0` on `x − c` (c from MaxPool2D max) |
| NNCF binarization ({0,s} activations, `x > s·sigmoid(t_c)`; weights `w>0`) | activation tie → 0; weight sign(0) = −1 | 0 of the {0,s} activations (= −1 in ±1 form) | `BConv2D gt, threshold t_c, pad_value −1, post_mul α·s/2, post_add α·s·ΣW/2` |
| BBCU (denoise / deblock / SR; `BinaryActivation` = torch.sign through the STE, `HardBinaryConv` mean\|W\|) | 0 | zero (`padding=1` on the binarized input) | `ChannelAffine(move0)` → `BConv2D sign_zero` (exact) or `ge` (deployable), `pad_value 0`, `post_mul = mean\|W\|` (+ `post_add` = conv bias, deblocking) |
| BSRC-2023 (`sign(sign(x)+0.1)` input binarizers, STE-with-clip × L1Scaling weights; v1 LAB = argmax of two depthwise outputs) | +1 (weights too: `W ≥ 0 → +1`); LAB tie → −1 | zero | `BConv2D ge` (exact), LAB → `Conv2D(groups C)` + 1×1 (−1, +1) difference + `BConv2D gt` (exact) |
| AdaBin (`BinaryActivation` α(sign((x−β)/α)+β), α > 0; weights sign((w−mean)/std)·std+mean per channel) | 0 (`torch.sign` through the STE; tie x = β gives αβ, the midpoint of the set) | **zero**, which is not in the binary set | `BConv2D threshold β, sign_zero` (exact) or `ge`, `pad_value 0`, levels as `p + q·s`, `c + d·s_w`: main BConv2D + count BConv2D(1 channel) + 1×1 Conv2D + constant border map (`operators_torch_cnn.md`) |
| FDA-BNN (MindSpore `Sign`, `sign(W)·mean\|W\|`) | 0 | zero | `BConv2D sign_zero` or `ge`, `post_mul = mean\|W\|` |
| tensorpack DoReFa W1A1 (`round(clip(x,0,1))`, half to even; `fw` with 0 → +1) | x = 0.5 → 0 (tf.round half to even) | zero of the {0,1} activations (= −1 in ±1 form) | `BConv2D/BDense gt, threshold 0.5, pad_value −1, post_mul E/2, post_add E/2·ΣW (+ bias)` (exact, no variant) |
| BNAS (`BinActive` = `torch.sign` of a BN output; XNOR weights `sign(w')·mean\|w'\|`, first-release BinOp) | 0 (structural zeros: BNAS-A's all-zero cell states reach `cells.7.preprocess0`, 72 of 144 channels, on every image) | **zero** (padding of the binarized input) | `ChannelAffine(bias b/s)` (BN sign folded into the weight bits) → `BConv2D sign_zero` (exact) or `ge` + exact data-independent correction map for the structurally zero channels; quirk convs `Activation(sign_zero)` + `Conv2D` in both variants |
| ReActNet-A with BN (BURN `BinaryConv2d` input.sign(); S2-BNN / BiMatting `BinaryActivation` STE = torch.sign exactly, checked by hooks) | 0: float32 ties `fl(fl(q + b_prev) + b_next) = 0` on channels with tiny PReLU slopes and nearly cancelling biases (BiMatting 8,735 / BURN 761 / S2-BNN 1,957 per 128 images in fp32, 0 in fp64) | zero | `ChannelAffine(q, fl64(b_prev + b_next))` → `BConv2D sign_zero` (exact) or `ge`, `pad_value 0`, `post_mul = mean\|W\|·s_bn`, `post_add = t_bn`; numpy fp32 then has 9 / 8 / 12 zeros per 128 images |

BiBERT specifics: weights are binarized at run time as `alpha_row·sign(W − mean_row W)`. The adapter
evaluates exactly those torch fp32 expressions, so the IR carries the binary weights the fp32 source
uses. Redoing the centering in float64 changes 43 of 110M weights (36 fp32 exact zeros disappear,
7 signs change), so the primary float64 reference keeps the fp32-derived binary weights and evaluates
all activations in float64 (`tools/convert_bibert.py`); plain `model.double()` is reported as well.
GELU is evaluated in double: an fp32 `erf` underflows for `x ≲ −5.5`, making `sign(GELU(x)) = 0`
in the fp32 source where the exact value is −1.
For SST-2 and MRPC the exact GELU is harmless (it even agrees better with the claims), but the MNLI and QQP checkpoints
depend on the fp32 zero set: on the 128 MNLI pairs 10.8% of all FFN values fall below the threshold, and the fp32 model
with only GELU in double drops from 62.5% to 45.3% (= the float64 model). `tools/convert_bibert.py --gelu-fp32-zero-set`
therefore builds a variant that forces gelu = 0 for x < t = −5.546368598937988 (the smallest float32 above which the CPU fp32
gelu is never 0, from an exhaustive scan of [−12, −3]) with existing ops (ChannelAffine → Activation sign_zero →
ChannelAffine → Mul) and checks it against a float64 reference with the same zero set (128/128 ≤ 6.7e-15 for MNLI and QQP).
The exact-GELU conversions also show the float64 analogue: torch's double `erf` saturates, so the float64 source has
gelu = −0.0 where glibc's GELU gives −4.6e−16, which is the (rounding-scale) first flip of every deviating MNLI/QQP sample.

IR-Net / SNN-0.67 specifics: the binary weights are a function of the parameters. Re-deriving them in float64
(`model.double()`) flips a few weights (ReCU-R18: 5 of 11.0M; SNN-0.67: 401 pattern choices), and a weight flip
changes a channel at every pixel of every sample, so the naive float64 model is a *different* network (ReCU-R18:
15/128 top-1 changes). As for BiBERT, the primary float64 reference keeps the fp32-derived binary weights
(substituted at the source's `F.conv2d` call) and evaluates everything else in float64 (`tools/convert_torch_cnn.py`).

BNext specifics: the models are strongly fp32-chaotic. PyTorch fp32 vs fp64 changes 7/128 top-1 decisions
for BNext-Tiny, and one BNext-Small sample's fp32 logits have cosine −0.18 against fp64. Storing the folded
constants (BN, `1 − scale`) as float32 moves a few binarization inputs that sit within ~1e−7 of zero, and a
single such flip can scramble the logits. With the constants kept in float64, the IR matches the fp64
source to ~1e−13 on all samples (`tools/convert_torch_cnn.py`, `float64_constants`). The SE gate sigmoid is
evaluated in double (`ChannelGate`), so the C++ reference and numpy agree bit for bit.

## GELU in front of sign binarizations (model variants, 2026-11-02)

In the Bi-ViT MLP (`fc1 → nn.GELU → clip(−10, 10) → fc2` with `torch.sign`) and the BiBERT FFN, a GELU feeds only
`sign_zero` binarizations. Mathematically `GELU(x) = x·Φ(x)` with `Φ > 0`, so `sign(GELU(x)) = sign(x)` for every
real `x` (and `GELU(x) = 0` iff `x = 0`); a clip with `min < 0 < max` keeps the sign too. The GELU (a
transcendental, the most expensive elementwise op of both models) could therefore be dropped and the binarization
applied to its input. The catch is a floating-point artifact of *evaluating* GELU: `0.5·(1 + erf(x/√2))` cancels.
For `x ≤ −5.5463` the float32 `1 + erf` rounds to 0 (torch's fp32 GELU), for `x ≤ −8.37438965` the double one does
(glibc and bionic libm; the IR's GELU is double, `kGeluCutoff`), so the evaluated GELU is exactly `(−)0` there and
`torch.sign` / `sign_zero` give 0 where the exact value is −1 (`ge` would give +1). Without the GELU the
binarization sees `x` itself: the mathematically exact sign.

`passes.drop_gelu_before_sign` (`python/bitino/passes.py`) recognizes `Activation(gelu)` (optionally followed by
sign-preserving clips, or multiplied by a provably non-negative mask) whose only uses are sign binarizations
(`BConv2D` / `BDense` `ge` / `gt` / `sign_zero` without a threshold, `BMatMul`, `Activation sign / sign_zero`) and
rewires the uses to the GELU input, keeping each binarization mode (exact zeros of `x` still give 0). It is exposed
as `--drop-gelu-before-sign` in `tools/convert_bivit.py` / `tools/convert_bibert.py` (the variant's references are
computed with the source GELU replaced by the identity) and, without a framework, as
`tools/derive_variant.py --pass drop_gelu_before_sign` on an existing `.bitino` (variants in
`artifacts/<model>/no_gelu/`). GSB's GELU feeds a thresholded `gt` binarization and BHViT's an `Add`: not
rewritten.

**The trained models depend on the zero set** (paired study, `tools/accuracy_eval.py`, exact McNemar test, stage
`neon` both sides, the development log (not published); decision rule as for INT8: the variant becomes
`model.bitino` only if p ≥ 0.05 or the difference is not negative):

| model | eval set (n) | runner | top-1 with GELU | top-1 without | Δ pt | McNemar p | agreement | decision |
|---|---|---|---|---|---|---|---|---|
| bi-vit-deit-tiny | ImageNet subset (5000) | host x86 | 14.60% | 5.46% | −9.14 | 2e−64 | 3.9% | variant `no_gelu`; default = exact `gelu_zero_mask` |
| bi-vit-deit-small | ImageNet subset (5000) | host x86 | 14.90% | 1.74% | −13.16 | 3e−154 | 1.0% | variant `no_gelu`; default = exact `gelu_zero_mask` |
| bibert-bert-base-sst2 | SST-2 dev (872) | emulated arm64 | 87.96% | 86.12% | −1.83 | 0.048 | 93.3% | variant `no_gelu`; default = exact `gelu_zero_mask` |
| bibert-bert-base-mrpc | MRPC dev (408) | emulated arm64 | 70.34% | 71.32% | +0.98 | 0.64 | 90.2% | **`no_gelu` adopted** as `model.bitino` |
| bibert-bert-base-mnli (fp32 zero-set model) | MNLI-m dev | host x86 / emulated | identical outputs | | 0 | 1 | 100% | **`no_gelu` adopted** (exact, see below) |
| bibert-bert-base-qqp (fp32 zero-set model) | QQP dev | host x86 / emulated | identical outputs | | 0 | 1 | 100% | **`no_gelu` adopted** (exact, see below) |

The emulated MRPC / SST-2 logits are byte-identical to the host x86 ones (both configurations), as are the emulated Bi-ViT-Tiny logits on the first 1000 images (14.4% → 5.2%, p = 4e−13; the development log (not published)). MNLI / QQP: byte-identical
outputs of the GELU and the no-GELU model on the first 2000 dev pairs (host, stage `neon`) and on the 128-pair eval
sets at stages `neon` / `optimized` / `optimized_int8` (host and emulated; the regenerated emulated references equal the
previous ones byte for byte). Without the zero-set mask (the exact-GELU MNLI conversion, `artifacts/bibert-bert-base-mnli/exact_gelu/`)
dropping the GELU costs 47.7% → 35.9% on the 128 pairs. ARM instructions per inference (stage `optimized`, 1 thread,
`tools/insn_count.py`): Bi-ViT-Tiny −17.0% without GELU / −11.8% with the exact mask, Bi-ViT-Small −15.5% / −11.2%,
BiBERT SST-2 −12.8% / −9.7%, MRPC −13.5%, MNLI −12.5%.

Bi-ViT's `fc1` outputs are integer counts times `mean|W|`: on Bi-ViT-Tiny 88% are negative and 47% are ≤ −8.374
(numpy executor, 4 images), so the double GELU silences half of all `fc2` inputs; with the exact sign they become −1
and the network collapses, the same mechanism (one cut-off
further) that makes the released fp32 code score ~1% (§ Level 3). BiBERT MNLI / QQP already run the fp32 zero-set
variant (`gelu = 0` for `x < −5.5464`, a `Mul` by a {0, ½, 1} mask): the mask covers every input where the double
GELU is zero, so dropping the GELU there is exact, the outputs are byte-identical and the variant became the default.

**Exact alternative: keep the zero set, drop the GELU.** `drop_gelu_before_sign(keep_zero_below=t)`
(`tools/derive_variant.py --keep-zero-below float64-gelu`, variants in `artifacts/<model>/gelu_zero_mask/`) replaces
the GELU by `x · keep`, `keep = (sign_zero(x − t) + 1)/2` (the ops of the BiBERT zero-set variant, which fuse into
one pass), with `t = −8.37438869` the first float at which the double GELU is non-zero. Every sign binarization then
sees the same value class as from the GELU (NaN and ±inf included), so the model is bit-identical to the GELU one
(checked: host and emulated outputs byte-identical at `neon` / `optimized` / `optimized_int8`). Being bit-identical
(difference 0), it passes the decision rule and is the `model.bitino` of Bi-ViT-Tiny / -Small and BiBERT SST-2; the
GELU models are kept in `artifacts/<model>/with_gelu/` (`registry/bitino_status.json` `alternate_variants`).

## Eval set limitations

The 128-image subset covers 100 of 1000 ImageNet classes. It is a **smoke and regression test**,
not an estimate of ImageNet accuracy. Accuracy on it is reported only to show agreement between
implementations.
