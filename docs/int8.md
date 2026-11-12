# INT8 execution of Conv2D and Dense

(The full-integer stage `int8_full`, in which every non-binary operator runs on int8 tensors, is described in
§ [Full-integer execution: `int8_full`](#full-integer-execution-int8_full) at the end of this file.)

Bitino can run the float `Conv2D` and `Dense` operators in INT8 using
dynamic-range quantization. The results approximate the float ops; they are
not equal to them. Three kernel tiers implement this:

| tier | kernels | ops | requires |
|---|---|---|---|
| `int8_reference` | `conv2d.int8_reference`, `dense.int8_reference` | Conv2D (any `groups`), Dense | nothing (portable C++) |
| `int8_neon` | `conv2d.int8_neon_sdot`, `dense.int8_neon_sdot` | Conv2D (`groups == 1`), Dense | AArch64 `dotprod` |
| `int8_neon` | `conv2d.int8_neon_i8mm`, `dense.int8_neon_i8mm` | Conv2D (`groups == 1`), Dense | AArch64 `dotprod` + `i8mm` |
| `int8_avx2` | `conv2d.int8_avx2`, `dense.int8_avx2` | Conv2D (`groups == 1`), Dense | x86-64 `avx2` (VPMADDWD) |
| `int8_avx2` | `conv2d.int8_avx2_vnni`, `dense.int8_avx2_vnni` | Conv2D (`groups == 1`), Dense | x86-64 `avx2` + `avxvnni` (VPDPBUSD) |

`int8_reference` is the oracle: every `int8_neon` and `int8_avx2` kernel must
produce bit-identical output to it. Bench stages `int8_neon` and
`optimized_int8` allow both platform tiers, so the same stage name selects the
NEON kernels on the phone and the AVX2 kernels on an x86-64 host.

## Selecting the tiers (opt-in)

The int8 tiers are lossy, so the selector treats them as opt-in. An empty
`KernelPolicy::allowed_tiers` ("all tiers", also bench stage `all`) never picks
them. They are used only when listed explicitly or when a kernel is forced:

```
bitino_microbench --op conv ... --tiers reference,int8_neon      # int8 where possible, float otherwise
bitino_microbench --op conv ... --force conv2d.int8_neon_sdot    # one specific kernel
bitino_bench --model m.bitino --stage int8_neon
```

When the platform int8 tiers are allowed they take precedence over float kernels:
`*.int8_neon_sdot` / `*.int8_avx2` have priority 140, `*.int8_neon_i8mm` /
`*.int8_avx2_vnni` 150. `int8_reference` has priority 30: above the float
reference (0) but below every `float_opt` kernel (40-55), so in mixed stages
such as `optimized_int8` the scalar oracle never displaces an optimized float
kernel. In particular depthwise and grouped Conv2D, which the platform int8
kernels reject, stay in float (`float_opt`) there.
The selector checks each kernel's `required_features` against `cpu_info()`, which
`BITINO_DISABLE_FEATURES` can mask, and then calls its `supports()`:

* `BITINO_DISABLE_FEATURES=i8mm` selects the SDOT kernels.
* `BITINO_DISABLE_FEATURES=dotprod` selects neither NEON kernel. The op then
  falls back to `int8_reference` if that tier is allowed, and to float otherwise.
* x86-64: `BITINO_DISABLE_FEATURES=avxvnni` selects the VPMADDWD kernels;
  `=avx2` selects no `int8_avx2` kernel (same fallback).
* Grouped or depthwise Conv2D is rejected by the `int8_neon` and `int8_avx2`
  kernels and falls back the same way.
* A reduction length `K > 133144` is rejected by every int8 kernel (see the
  overflow note below).

## Contract

The single source of truth is the header comment of
`runtime/kernels/int8_quant.h`. The rules are restated here.

**prepare(): weights.** Weights are quantized per output channel, symmetric.
`k` runs over `kh*kw*kic` for Conv2D (OHWI) and over `K` for Dense (`[O,K]`).

```
m_o        = max_k |W[o][k]|
scale_w[o] = m_o / 127           (1 if that rounds to 0: an all-zero row, or
                                  subnormal m_o < ~127*2^-150)
qw[o][k]   = clamp(rne(W[o][k] / scale_w[o]), -127, 127)
```

**run(): activations.** Activations are quantized per tensor, symmetric, over
every element of the input (all batch entries and all rows).

```
m        = max |x|
scale_x  = m / 127               (1 if that rounds to 0, as for scale_w)
qx       = clamp(rne(x / scale_x), -127, 127)
qpad     = clamp(rne(pad_value / scale_x), -127, 127)   (Conv2D padding taps)
acc      = sum_k qx * qw         exact int32
y        = act(float(acc) * (scale_x * scale_w[o]) + bias[o])
```

* `rne` is round to nearest, ties to even: `std::nearbyint` in the default
  rounding mode, which is the same as AArch64 `FCVTNS` / `vcvtnq_s32_f32`.
* Every `x / scale` is an IEEE fp32 division, never a multiply by a precomputed
  reciprocal. `x * (1/s)` can differ from `x / s` by one ulp, and that is enough
  to move a value across a `.5` rounding boundary.
* `pad_value` does not take part in `m`. If `|pad_value| > m`, `qpad` saturates.
* The epilogue is evaluated in fp32 in exactly the order shown, with one
  rounding per operation and no FMA (the build uses `-ffp-contract=off`, and the
  NEON code uses explicit `vmulq`/`vaddq`).
* An absent bias counts as `+0.0f` and is still added. `act` is
  `apply_activation()`. The NEON code reproduces it exactly, including `-0.0f`.
* **Overflow.** `|acc| <= K*127*127`, so `K <= 133144 = floor((2^31-1)/16129)`
  keeps the int32 accumulation exact. Every partial sum obeys the same bound.
* Inputs must be finite.

All int8 kernels follow these rules, so for a given input their outputs are
identical regardless of tiling, K padding (zeros contribute nothing) or thread
count. The max reduction is exact and independent of order, so a parallel
reduction gives the same `scale_x`.

## Accuracy bound vs the float reference

The bound is tested by `int8_reference_within_analytic_bound_of_float` in
`tests/cpp/test_int8_kernels.cc`.

Every quantized operand can be written as `x = sx*(qx + dx)` and
`w = sw*(qw + dw)`, with `|dx|, |dw| <= 1/2 + 1e-5`:

* rounding to nearest contributes `1/2`;
* the fp32 division `x/s` contributes at most `u*|x/s| <= 127*2^-24 < 1e-5`;
* clamping never triggers for `|x| <= max|x|`.

With `ex = sx*dx` and `ew = sw*dw`:

```
x*w - sx*sw*qx*qw = x*ew + ex*w - ex*ew
|x*w - sx*sw*qx*qw| <= |x||ew| + |w||ex| + |ex||ew|
```

Sum this over the K taps of an output. Padding taps use `x = pad_value`, and
need `|pad_value| <= max|x|`:

```
Q = sum_k ( |x_k|*sw/2' + |w_k|*sx/2' + sx*sw/4' )      (primes: the 1e-5 slack)
```

Two floating-point terms remain:

* the int8 epilogue rounds three times on `D = sx*sw*acc` and once on the bias
  add, which gives at most `4.02u|D| + u|b|`, with `u = 2^-24`;
* the float reference rounds its double sum once, which gives at most
  `u*(S + |b|)`, with `S = sum_k |x_k*w_k|`.

Since `|D| <= S + Q`, the total is

```
|y_int8 - y_float| <= Q + 8u*(S + Q + |b|)
```

relu and relu6 are 1-Lipschitz, so the bound still holds after the activation.
Measured worst `err/bound` on the test shapes is 0.49 on the S22 and 0.52 on
the x86 host (inputs differ because std::normal_distribution differs between
standard libraries). The bound is therefore
reasonably tight and it is never exceeded.

## End-to-end accuracy (5,000 ImageNet validation images)

Paired study: exact binary path (stage `neon`) vs `optimized_int8` on the same images, McNemar
exact test (`tools/accuracy_eval.py`, JSONs in the development log (not published)). The 5,000 images cover only
the ImageNet-100 classes, so the numbers compare implementations; they do not estimate
ImageNet accuracy. The INT8 side ran under qemu-aarch64 with the same kernels as the device.

| model | exact top-1 | INT8, depthwise in INT8 | INT8, final config (depthwise stays float) |
|---|---|---|---|
| QuickNet | 61.12% | 59.46% (−1.66 pt, p=5e-4) | 61.10% (−0.02 pt, p=1.0) |
| QuickNet-Small | 56.42% | 55.92% (−0.50 pt, p=0.30) | 56.66% (+0.24 pt, p=0.61) |
| Bi-RealNet | 54.80% | 54.22% (−0.58 pt, p=0.23) | 54.22% (−0.58 pt, p=0.23) |

The final configuration (`int8_reference` priority 30, below `float_opt`) keeps depthwise
convolutions in float. That removes QuickNet's significant loss. Bi-RealNet has no depthwise
layers, so both configurations give identical predictions. In every model, INT8 changes 19–25% of the top-1
predictions while leaving accuracy statistically unchanged. Treat INT8 as a different model of
equal accuracy, not as a bit-faithful approximation.

## Which layer costs accuracy: IR-Net CIFAR-10 stems

The paired sweep over every converted model (`scripts/int8_accuracy_all.sh`, report section "Final configuration per
model") finds INT8 significantly worse (McNemar p < 0.05) only for six CIFAR-10 models: four of the IR-Net family and the S12 FDA-BNN and AdaBin ResNet-20. In these models the
only non-binary layers are the first convolution (3 input channels, on the normalized image) and the 10-class classifier.
`tools/int8_layer_ablation.py` forces one op type back to its float kernel (`bitino_bench --force`) on the full
10,000-image test set:

| model | exact | INT8 all | INT8 except Conv2D (stem float) | INT8 except Dense (classifier float) |
|---|---|---|---|---|
| ReCU ResNet-18 | 93.24% | 92.66% (p=0.0038) | 93.23% (p=1) | 92.67% (p=0.0046) |
| ReCU VGG-Small | 92.25% | 91.77% (p=0.012) | 92.24% (p=1) | 91.79% (p=0.016) |
| BiPer VGG-Small | 92.45% | 91.94% (p=0.011) | 92.45% (p=1) | 91.93% (p=0.0095) |
| RBNN ResNet-20-Bi-Real | 87.81% | 87.18% (p=0.014) | 87.75% (p=0.15) | 87.20% (p=0.017) |
| SiMaN VGG-Small (control) | 92.47% | 92.40% (p=0.72) | 92.43% (p=0.22) | 92.41% (p=0.77) |
| FDA-BNN ResNet-20 (S12) | 86.61% | 85.64% (p=0.00079) | 86.61% (p=1) | 85.63% (p=0.00066) |
| AdaBin ResNet-20 (S12) | 87.99% | 87.38% (p=0.023) | 88.03% (p=0.45) | 87.40% (p=0.028) |

The whole loss is the INT8 stem: per-tensor symmetric quantization of a 3-channel normalized input. The two S12 CIFAR-10 models (FDA-BNN, AdaBin) show the same pattern; for AdaBin "except Conv2D" also keeps the 1×1 Conv2D of its border-correction term in float, so its attribution is to the float convolutions as a group. The INT8 classifier is
harmless. For these four models the final configuration is `optimized` (float stem and classifier); because the stem is
their only non-binary convolution, the latency cost of not using INT8 is negligible. A general "first layer stays float"
policy is a candidate follow-up (it would change `optimized_int8` outputs of every model, so it is deferred until after the
device run).

One result goes the other way: the BNN-PYNQ Theano CNV (CIFAR-10) is significantly *better* in INT8, 79.79% vs 79.09%
(+0.70 pt, p=0.013). Its only non-binary layer is also the stem convolution, and forcing it back to float restores the exact
accuracy (79.09%, p=1; the development log (not published)). The p-values are per model:
with about 85 paired tests, roughly 4 are expected below 0.05 by chance, so a single positive result is not evidence that
INT8 helps. The IR-Net losses are treated differently because they share one sign across the family (most IR-Net CIFAR-10
deltas are negative) and the ablation attributes all four to the same layer. The rule keeps `optimized_int8` for the CNV.

## Implementation (`int8_neon`)

The code is split by ISA level. Everything outside the micro-kernel TUs uses the
baseline `-march=armv8.2-a+dotprod` and contains no SDOT or SMMLA instructions.

| file | `-march` | contents |
|---|---|---|
| `int8_neon/int8_neon_kernels.cc` | baseline | Kernel classes, quantization, im2col, weight packing, threading, registration |
| `int8_neon/int8_gemm_sdot.cc` | `+dotprod` | SDOT 8x8 micro-kernel |
| `int8_neon/int8_gemm_i8mm.cc` | `+dotprod+i8mm` | SMMLA 8x8 micro-kernel |

The micro-kernel TUs include nothing with external linkage, so no inline
function can be merged across `-march` boundaries.
`int8_gemm_epilogue.h` is `static`: each micro-kernel TU gets its own copy.

### GEMM

The op is computed as a GEMM: `y[p][o] = epilogue(sum_k A[p][k] W[o][k])`, with
P rows (output pixels or Dense rows) and O columns (output channels). Both
operands are packed in tiles of 8 rows. K is split into chunks of C bytes and
zero-padded to `kc*C`:

```
A [ceil(P/8)][kc][8][C]        W [ceil(O/8)][kc][8][C]
```

* **SDOT** (C = 4). Each chunk of a tile is 32 bytes, i.e. two registers, with 4
  rows times 4 k. `vdotq_laneq_s32(acc[r][h], w_h, x_{r/4}, r%4)` accumulates 4
  output channels of pixel r per instruction. There are 16 accumulators.
* **SMMLA** (C = 8). Each chunk is 64 bytes: four registers of 2 rows times 8 k.
  `vmmlaq_s32(acc[i][j], a_i, b_j)` accumulates a 2x2 block (32 MACs). There are
  16 accumulators. The epilogue re-pairs the 64-bit halves into rows with
  `vzip1q_s64` and `vzip2q_s64`.
* **Short tiles.** Both micro-kernels are instantiated for the number of valid
  rows in the tile (SDOT R = 1..8, SMMLA 1..4 row pairs). Dense with one row and
  the last pixel tile therefore skip the padding rows. Tiles of at most 2 rows
  alternate two accumulator sets to hide dot-product latency.

### Steps of one run()

1. **Quantize the input.** A NEON max-abs reduction, then a quantize pass:
   `vdivq_f32`, `vcvtnq_s32_f32`, saturating narrows and a clamp at -127. About
   3 Gelem/s on one core, 6x faster than scalar. Both passes are split over the
   pool only for tensors of at least 1M elements: the pass is memory bound and a
   pool dispatch costs 20-100 µs on the S22. Max-abs partials go into a scratch
   buffer allocated in `prepare()`.
2. **Pack A.** Conv2D runs im2col straight into the tiled layout. When
   `ic % C == 0`, whole chunks are copied; otherwise bytes are scattered.
   Padding taps read a row filled with `qpad`. Dense packs its rows. Padding
   rows and K padding are zeroed once in `prepare()` and never written again.
3. **GEMM.** The work items are (output-channel chunk, pixel tile) pairs, with
   the pixel tile varying fastest. A channel chunk holds at most
   `kWeightChunkBytes` of packed weights, so consecutive items reuse cached
   weights. Each item computes 8x8 tiles and applies the epilogue
   (`vcvtq_f32_s32`, `vmulq`, `vaddq`, activation, store). When the weights fit
   one chunk, each item packs its own pixel tile right before computing it,
   which saves a pool dispatch. A layer whose estimated single-thread time is
   below 80 µs runs inline on the calling thread. The estimate in ns is
   `1.25*outputs + {0.012 SDOT, 0.006 SMMLA}*MACs + 20*im2col pixels`.

The ledger lists the tuning decisions: the development log (not published).

## Implementation (`int8_avx2`, x86-64)

Same split by ISA level as `int8_neon`, and the same contract; only the
micro-kernels and the operand encoding differ.

| file | flags | contents |
|---|---|---|
| `int8_avx2/int8_avx2_kernels.cc` | baseline x86-64 | Kernel classes, weight quantization and packing, im2col, threading, registration |
| `int8_avx2/int8_gemm_avx2.cc` | `-mavx2` | max-abs, activation quantizer, VPMADDWD micro-kernel |
| `int8_avx2/int8_gemm_avxvnni.cc` | `-mavx2 -mavxvnni` | VPDPBUSD micro-kernel |

The ISA translation units include only `int8_avx2.h`, `int8_avx2_tile.h`
(internal linkage) and `<immintrin.h>`; `tools/check_isa_isolation.py` checks
the linked binaries (see `docs/architecture.md`). They are built without
`-mfma`, so the epilogue's multiply and add are rounded separately.

**Exactness of the two dot products.**

* *VNNI* (`VPDPBUSD`, u8 x s8). Activations are stored as `u = qx + 128`
  (`0..254`), weights as `qw` (s8). Each instruction adds
  `sum_t u[4c+t]*qw[4c+t]` (4 products of at most 255*127) to an int32 lane
  **without saturation**: the accumulation wraps modulo 2^32. So the
  accumulator holds `sum_k qx*qw + 128*sum_k qw` modulo 2^32. The kernel
  subtracts `corr[o] = 128*sum_k qw[o][k]` (computed in int64 in `prepare()`,
  stored modulo 2^32) with a wrapping int32 subtraction. The result is
  `sum_k qx*qw` modulo 2^32, and since `|sum_k qx*qw| <= K*16129 < 2^31` for
  `K <= 133144`, it is the exact value. `VPDPBUSDS` (saturating) is not used.
  K padding: weights are zero there, so whatever the activation bytes hold
  contributes nothing.
* *AVX2* (`VPMADDWD`). Both operands are int16 (sign-extended int8), two `k`
  per 4-byte chunk; each instruction gives `a0*w0 + a1*w1` exactly
  (`|.| <= 2*16129`) and the int32 accumulation is exact as for the reference.
  `VPMADDUBSW` is not used: its int16 pair sums saturate (`2*255*127 > 32767`).

**Quantizer.** `q = clamp(rne(x / scale), -127, 127)` with `VDIVPS` (IEEE
division) and `VCVTPS2DQ` under the default MXCSR mode (round to nearest even,
as `std::nearbyint`), clamping in float **before** the conversion (equal to
clamping the rounded value for every finite `x`, and it keeps the conversion
in range), then packed to u8 (`q + 128`) or int16. Tails go through a
zero-padded 8-lane buffer, so there is no scalar code path with different
rounding. Max-abs is a lane-wise maximum of `|x|` (exact, order independent).

**GEMM.** A is the quantized input in row-major form (Dense: the quantized
rows; Conv2D: im2col rows, padding taps = `qpad`), row stride `4*kc` bytes,
K padding zero. W is packed in panels of 16 output channels:
`[ceil(O/16)][kc][16][4 bytes]`. The micro-tile is 6 rows x 16 columns (12
int32 accumulators; per chunk 2 W loads, 6 four-byte broadcasts, 12 dot
instructions); short tiles use 2-4 accumulator sets over k. The epilogue is
`VCVTDQ2PS`, `VMULPS` by `scale_x*scale_w[o]`, `VADDPS` bias, then
relu (`max(v, 0)`) or relu6 (`min(6, max(0, v))`): the operand order of
`MAXPS`/`MINPS` reproduces `apply_activation()` exactly, including `-0.0f`
and NaN. Work items are (panel chunk, block of 24 rows); when the weights fit
one 256 KB chunk, each item builds its own im2col / quantized rows right
before its GEMM (Dense quantizes rows inside the items). Layers estimated
below 30 µs run inline.

Measured on the host (i9-14900K P-core at 3.2 GHz, 1 thread,
`bitino_microbench --tiers reference,binary_scalar,int8_avx2`): 1x1 conv
7x7x256->512 at 318 GOPS (VNNI) and 1x1 conv 28x28x64->128 at 203 GOPS
(VNNI) / 106 GOPS (VPMADDWD), against 94-99 GOPS for the AVX2 fp32 kernels;
Bi-Real's 7x7 stem takes 1.24 ms (VNNI) vs 3.05 ms in fp32. See
the development log (not published).

## MatMul (activation × activation)

The float `MatMul` operator (BHViT's attention product `P·sign(V)`; BMatMul stays binary) has INT8 kernels
in the same opt-in tiers, `runtime/kernels/int8_reference/int8_matmul.cc`:

| kernel | tier | requires | priority |
|---|---|---|---|
| `matmul.int8_reference` | int8_reference | — | 30 |
| `matmul.int8_avx2` / `matmul.int8_avx2_vnni` | int8_avx2 | avx2 / + avxvnni | 140 / 150 |
| `matmul.int8_neon_sdot` / `matmul.int8_neon_i8mm` | int8_neon | dotprod / + i8mm | 140 / 150 |

Contract (dynamic, per-tensor, symmetric on **both** operands):

```
sa = int8_scale(max |a|) over all of a     qa = clamp(rne(a / sa), -127, 127)
sb = int8_scale(max |b|) over all of b     qb = clamp(rne(b / sb), -127, 127)
acc[bt][i][j] = sum_k qa[bt][i][k] * qb[bt][k][j]      exact int32, K <= 133144
c = (sa * sb) * scale                                   fp32, in that order
y = float(acc) * c + 0.0f                               the Conv2D/Dense epilogue with a zero bias
```

`transpose_b` only changes where `qb[bt][k][j]` is read. The NEON and AVX2 kernels reuse the Conv2D/Dense
GEMM micro-kernels: per batch entry, A = the quantized rows of `a` (AVX2: the whole tensor quantized in one
pass, rows back to back; the micro-kernel's K padding then reads up to 3 bytes of the next row, which only
meet zero weights), W = the quantized columns of `b` packed at run time (AVX2: `pack_w_kn`, a vector 4×16
byte / 2×16 int16 transpose with the VNNI correction `128·Σ_k qb` computed on the fly; NEON: transpose to
`[N][K]` then the tiled layout). Every kernel is bit-exact with `matmul.int8_reference`
(`int8_matmul_bit_exact_vs_int8_reference`: 10 shapes × 2 kernels × 1/4 threads, K tails 1..300,
transposed and not, values on rounding ties), and within the analytic bound of the float reference
(the bound of the Conv2D section with `sw → sb`, per-tensor): worst measured err/bound 0.81 on x86, 0.72 on
emulated ARM. `allow_int8: 0` keeps a node in float; masking `avxvnni` / `i8mm` selects the VPMADDWD / SDOT
kernels, masking `avx2` / `dotprod` the float kernel (`int8_matmul_feature_masking`).

On the host (i9-14900K, 1 thread) BHViT-Tiny's attention-product shape `[64,4,53,53] × [64,4,53,64]`
takes 1.3 ms with VNNI against 2.1 ms with the AVX2 float GEMM and 72 ms with the double reference.
Accuracy at `optimized_int8` (paired against the exact stage, `tools/accuracy_eval.py`): BHViT-Tiny 61.70% →
62.36% on 5,000 ImageNet-100 images (+0.66 pt, p = 0.13), BiBERT 87.96% → 87.84% on the SST-2 dev set (p = 1.0),
BNext-Tiny 70.72% → 70.52% (p = 0.63); details in the development log (not published).

## Testing

`tests/cpp/test_int8_kernels.cc` covers the following:

* **Hand-computed values.** These include ties-to-even rounding, an all-zero
  weight row and a quantized `pad_value`, checked on every kernel the CPU
  supports.
* **Bit-exactness.** SDOT and I8MM (AArch64), or VPMADDWD and VNNI (x86-64),
  are compared against `int8_reference` on 23 Conv2D shapes and 16 Dense shapes
  at 1-4 threads. The shapes cover tails in K,
  O and P, strides, dilation, asymmetric padding, a saturating `pad_value`,
  all-zero input, `ic` = 1, 3, 4 and 12, batch 2, every tile height
  R = 1..8, layers large enough to split over the pool (both GEMM paths) and a
  1.1M-element input (parallel quantization). On the S22 and on the x86 host
  that is 312 comparisons each, and a kernel that the CPU supports is never
  skipped. A Dense layer at `K = 133144` with every product at +-127*127 checks
  the VNNI wrap-around correction at the int32 limit.
* **Quantizer.** `int8_avx2_quantizer_bit_exact_vs_scalar` compares the AVX2
  quantizer (u8 and int16 outputs) and max-abs with `int8_quantize()` on
  20,776 adversarial values: exact `.5` ties of both parities and their float
  neighbours, `+-127*s` and beyond, `+-0`, huge values, all tail lengths.
* **Accuracy** against the float reference, within the bound above.
* **Gating.** The test checks that the int8 tiers are opt-in and that grouped
  Conv2D falls back. It re-executes the test binary with
  `BITINO_DISABLE_FEATURES=i8mm` (SDOT expected) and with `=dotprod` (no NEON
  kernel expected); on x86-64 with `=avxvnni` (VPMADDWD expected) and `=avx2`
  (no `int8_avx2` kernel expected).

`bitino_int8_quantbench` times the activation quantization alone, scalar
against NEON, and checks that the two are bit-exact.

## Full-integer execution: `int8_full`

Stage `optimized_int8` (above) quantizes only the float `Conv2D` / `Dense` / `MatMul` of a model, dynamically,
and every other non-binary operator stays fp32. Stage **`int8_full`** runs a separate, statically quantized model,
`artifacts/<model>/model_int8_full.bitino`, in which **every non-binary operator reads and writes int8 tensors**:
the graph input is quantized once (`I8Quantize`), and only the graph output is dequantized, inside its producing
operator. `optimized_int8` is unchanged, so the two stages can be compared.

```
python3 tools/int8_full_suite.py quantize --models bi-vit-deit-tiny        # calibration + model_int8_full.bitino
bitino_bench --model artifacts/bi-vit-deit-tiny/model_int8_full.bitino --stage int8_full
python3 tools/int8_full_suite.py accuracy --models bi-vit-deit-tiny --emulated build/android-arm64-static
```

### Quantization scheme

TFLite-style static quantization:

* **Activations**: per-tensor asymmetric int8, `x = s·(q − z)`. The range is calibrated, widened to contain 0
  (TFLite zero-point nudging), `s = (hi − lo)/255`, `z = round(−128 − lo/s)`. Default range: the (0.01 %, 99.99 %)
  percentiles of the tensor over the calibration samples (`--percentile`, 2048-bin histograms over the min/max
  range; min/max without `--percentile`).
* **Weights** of `Conv2D` / `Dense`: per-output-channel symmetric int8 (`max|w|/127`), int32 bias at `s_x·s_w[o]`,
  int32 accumulators. Binary operators keep their bit weights.
* **Tensors that are only ever binarized** (every consumer is a `BDense` / `BConv2D` / `BMatMul` operand, also
  through Reshape / Transpose / Concat / Gather) get a *binarize-aware* range: the grid only has to resolve values
  near the consumers' thresholds, so `[min t − µ, max t + µ]` with a small margin µ (2^-8 of the threshold
  span, 2^-12 of the largest |t|, 2^-20 of the data range). Values beyond it saturate to codes whose dequantized
  value is still on the same side of every threshold, so their binarization is exact; e.g. a LayerNorm output
  feeding binary layers keeps its sign up to `s/2 ≈ 2^-21·max|x|`. Zero points of such grids may lie outside
  [−128, 127] (they are int32 attributes; `|z| < 2^20` by construction).
* **Graph outputs** are produced in fp32 by the last operator (`output: f32` of I8Dense / I8Conv2D / I8MatMul /
  I8BDense / I8BMatMul / I8Softmax / I8Lut): logits are dequantized int32 accumulators, not rounded to int8 first
  (an int8 rounding of 1000 logits would only add ties).

**Fixed-point requantization** (`q8::requant`, `int8_ops.requant`): a real multiplier m > 0 is stored as
`(M, shift)`, `M ∈ [2^30, 2^31)`, `m ≈ M·2^shift/2^31`, and

```
requant(x) = rshr(mulhi(sat32(x << max(shift, 0)), M), max(−shift, 0))
mulhi(x, M) = (x·M + 2^30) >> 31          AArch64 SQRDMULH
rshr(x, n)  = (x + 2^(n−1)) >> n          AArch64 SRSHL by −n
```

Both roundings are **round half up** (towards +∞), `>>` an arithmetic shift; this differs from gemmlowp's
`RoundingDivideByPOT` (half away from zero) only on exact negative ties. `M` comes from `frexp` with
round-half-away-from-zero, offline (the model stores the integers; kernels never recompute them).
fp32 epilogues (binary layers, LayerNorm, Softmax) are one IEEE operation at a time in the written order and are
converted with **round half to even** (`nearbyint` = FCVTNS), clamped in float first.

### Operators (`docs/operators.md` § Full-integer operators)

| operator | replaces | arithmetic |
|---|---|---|
| `I8Quantize` / `I8Dequantize` | graph input / output | `clamp(rne(x/s) + z)` (IEEE division) / `fp32(q−z)·s` |
| `I8Conv2D`, `I8Dense`, `I8MatMul` | Conv2D, Dense, MatMul | `Σ (x−z_x)·w` in int32 (MatMul: both zero points), `y = clamp(z_y + requant(acc + bias))`; f32 output: `fp32(acc)·scale[o] + bias` |
| `I8Add` | Add, AddBroadcast (b broadcast) | TFLite: inputs `<< 20`, rescaled to `2·max(s_a, s_b)`, summed, requantized |
| `I8AffineSum` | trees of Add / ChannelAffine | `y = z_y + requant(Σ_i (q_i−z_i)·K[i][c] + B[c])`, int32 accumulation, one rounding per tree |
| `I8Mul` | Mul | `z_y + requant((a−z_a)(b−z_b))` |
| `I8Lut` | chains of ChannelAffine / Activation (GELU, tanh, sigmoid, PReLU, RPReLU, hardtanh, sign, …) / FakeQuant | 256-entry table per tensor or per channel (any axis), computed in float64 from the composed function |
| `I8LayerNorm` | LayerNorm | integer statistics `S1 = Σt`, `S2 = Σt²` (t = q − z), per row `r = fp32(s/√(s²(D·S2 − S1²) + ε·D²))` in double, per element `rne(fp32(fp32(fp32(D·t − S1)·r)·G[c]) + B[c])` with `G = γ/s_y`, `B = β/s_y + z_y` |
| `I8Softmax` | Softmax | `e = E[max_q − q]` from a 256-entry int32 table `round(2^b·exp(−s_x·d))` (b ≤ 22, `N·2^b < 2^31`), integer row sum S, `y = clamp(rne(fp32(e)·fp32(inv_scale/S)) + z_y)` |
| `I8MaxPool2D`, `I8AvgPool2D`, `I8GlobalAvgPool`, `I8ReduceMean` | pooling | max; average with TFLite rounding division (same parameters); global / axis mean requantized |
| `I8Concat`, `I8Transpose`, `I8Gather`, `Reshape`, `I8Pad` | data movement | copies; Concat requantizes inputs with other parameters; Gather from int8 or bit tables (bit tables: per-row `(q₊, q₋)` codes) |
| `I8BDense`, `I8BConv2D`, `I8BMatMul` | BDense, BConv2D, BMatMul | int8 input binarized by integer thresholds, exact integer accumulation, fp32 epilogue rounded to int8 |

**LayerNorm** uses integer statistics (exact, order independent) and a float per-row factor; the per-element
part is three fp32 operations. **Softmax** is LUT based like TFLite's int8 softmax; the output grid is calibrated
(or binarize-aware) instead of TFLite's fixed 1/256.

### Binary operators with int8 inputs

Binarizing `x ≥ t` of an int8 operand means binarizing its dequantized value `deq(q) = fp32(fp32(q − z)·s)`.
`deq` is non-decreasing in q (s > 0), so `{q : deq(q) ≥ t}` is an upper set of the 256 codes and

```
deq(q) ≥ t  ⇔  q ≥ T_ge(t),   T_ge(t) = min{q ∈ [−128,127] : deq(q) ≥ t}   (128 if empty)
deq(q) > t  ⇔  q ≥ T_gt(t),   T_gt(t) = min{q : deq(q) > t}
```

`ge`: +1 iff `q ≥ T_ge`; `gt`: `q ≥ T_gt`; `sign_zero`: +1 iff `q ≥ T_gt`, −1 iff `q < T_ge`, 0 otherwise (a tie
`deq(q) = t` exists exactly when `T_ge < T_gt`); `gt0`: `{0,1} = (q ≥ T_gt)`. The thresholds are computed by
definition (scan of the 256 codes, `q8::qthr_ge/gt`, `int8_ops.qthr_ge/gt`); the analytic `⌈t/s + z⌉` is within
one code of `T_ge` (tested). Tests: every code of 160,000 random and adversarial `(s, z, t)` (ties `t = deq(q)`,
float neighbours, thresholds outside the grid, zero points up to ±500,000), C++ and Python.
The epilogue folds the output quantization: `v = fp32(fp32(float(acc)·post_mul'[o]) + post_add'[o])` with
`post_mul' = post_mul/s_y`, `post_add' = post_add/s_y + z_y`, then `clamp(rne(v), qmin, qmax)` (a fused relu /
relu6 becomes qmin / qmax). `I8BMatMul` evaluates `acc = c_ab·Σ_k A·B + c_rs·Σ_k A + c_cs·Σ_k B + c0` with
integer coefficients and a per-column `mul` / `add`, which absorbs Bi-ViT's affine-code products (below).
The NEON kernels run the binary kernels on that exact binarization (`bdense.neon`, `bdense.ternary_neon`,
`bconv2d.neon_opt`, `bmatmul.neon`; reachable for int8 tensors only through their exported factory functions,
their registry rules still require f32 / bit inputs):

* `ge` / `gt` (I8BDense, I8BConv2D): the wrapper packs the codes into bits (`q < T`, compared in int8 lanes when
  every `T <= 127`, else in int16 lanes) and the binary kernel reads the bit tensor as its input.
* `sign_zero` I8BDense: `bdense.ternary_neon` reads the codes itself (int32 `[T_ge, T_gt]` thresholds as input 4)
  and packs the planes `neg = q < T_ge`, `nz = neg | q >= T_gt`.
* I8BMatMul: `bmatmul.neon` reads both int8 operands and packs the planes `neg = q < T_neg`, `pos = q >= T_pos`
  (`T = 128` by flag) for every mode; row-major `b` columns are built 8 rows at a time (`OR` of `mask & 1 << row`
  gives one byte per column) and zipped into 32-bit column words. Row / column sums for `c_rs` / `c_cs` are taken
  from the codes.
* int8 output: the binary kernels compute the same fp32 value `v` as on the float path and store
  `clamp(rne(v), qmin, qmax)` (FCVTNS, saturating narrows, clamp; equal to `q8::round_f32` for finite `v`)
  directly, four rows of a 4-channel tile per store group; f32 outputs are written directly too. Before
  2026-11-03 the wrappers wrote {−1, 0, +1} float copies, ran the kernels into a float buffer and rounded it in a
  separate pass; outputs are byte-identical to that.

### Graph rewrites of the quantizer

All exact in real arithmetic (listed per model in the development log (not published)):

* **R1** `sign(Softmax(x))` as a BMatMul operand is the constant +1 (a softmax is > 0; the calibration data never
  underflowed to 0). Bi-ViT DeiT-Tiny has 6 such heads (r = 0 attention quantizer); their q·k products, LayerNorms
  and softmaxes become dead code (42 nodes removed).
* **R2** `Add(Dense(x1), Dense(x2)) [· s]` → `Dense(Concat(x1, x2))` (DeiT class + distillation heads).
* **R3** a ChannelAffine feeding only binarizations becomes the binarization threshold (and a sign flip for
  scale −1, which swaps ge/gt and negates the product).
* **R4** Bi-ViT's affine-code products (an AddBroadcast tree of BMatMul counts, row / column sums with constant
  ones operands and a constant, scaled by a ChannelAffine) → one `I8BMatMul` with integer coefficients.
* **R5** a ChannelAffine after a BDense / BConv2D → folded into post_mul / post_add.
* **R3b** a BConv2D / BDense whose input is a strictly increasing per-channel chain (ChannelAffine with positive
  scale, PReLU / leaky ReLU with positive slope, e.g. RPReLU) of a tensor x reads x itself, with the thresholds
  mapped through the inverse chain (`f(x) ≥ t ⇔ x ≥ f⁻¹(t)`); the chain stays for its other consumers.
* **R6** chains of unary operators → one `I8Lut`.
* **R7** trees of Add / ChannelAffine → one `I8AffineSum`.
* **ChannelGate** → `I8Lut` (sigmoid of the gate, output grid [0, 1]) + Reshape + `I8Mul` with broadcasting.

**Mixed tensors** (read by binarizations *and* by other operators, typically the residual stream before a
binary layer) get a second int8 version on a binarize-aware grid: the producer is emitted twice (same inputs,
the second time with the binarize-aware output parameters; pass-through producers are duplicated on the split
version of their input), and the binarizing consumers read that version. It is kept only where its grid is at
least 2× finer than the tensor's own. A product compared only with 0 (`sign_zero(a·b)`) only needs the signs of
its operands, which are then treated as binarized with threshold 0, and its grid is the product of the input grids
(BiBERT MNLI / QQP multiply GELU by a keep mask before the sign; without this rule MNLI fell to 36.9 %).

LUT entries of a tensor that is binarized (only, or also by some consumers) are chosen so that the dequantized
entry lies on the same side of each consumer threshold as the float function value (searching ±4 codes when
the tensor is only binarized, ±1 when other consumers read it). Without this, GELU's tiny negative tail rounds
onto the zero point and `sign_zero` turns it into 0: Bi-ViT DeiT-Tiny fell to 2.1 % top-1 (exact 14.4 %).

### Calibration data

Calibration runs the float32 numpy executor, which is bit-identical to the exact stage, on samples that are
disjoint from every evaluation set (each manifest records the check, `split: train`, `purpose: calibration`):

| family | calibration set | samples used |
|---|---|---|
| ImageNet models (Bi-ViT, BHViT, CNNs) | `.cache/data/calib512_imagenet100train`: 512 ImageNet-100 **train** images (class i % 100, file i // 100), preprocessings `bhvit`, `larq_zoo`, `torchvision_imagenet`, `timm_bicubic_crop095`, `bitorch_imagenet`, `omz_bgr`, `tensorpack_bgr`; no file name or JPEG SHA-256 in eval128 / eval5000 | 256 (transformers), 128 (CNNs) |
| BiBERT SST-2 / MRPC / MNLI / QQP | `.cache/data/sst2_calib512`, `.cache/data/glue_calib512/<task>`: first 512 GLUE **train** examples, BiBERT tokenization, no text in the dev sets | 256 |
| CIFAR-10 / CIFAR-100 models | `.cache/data/calib512_cifar10train`, `calib512_cifar100train`: first 512 **train** images | 128 |
| MNIST / GTSRB models | none (only the test sets are local) | not covered |

The 5,000 ImageNet-100 validation images are all in eval5000, so ImageNet calibration comes from the train split.
The model's `manifest.quantization` records the calibration set, its tensor SHA-256, the sample ids, the range
rule, the rewrites, the int8 operator counts and the parameters of every tensor.

### Format

`model_int8_full.bitino` is a regular `.bitino` (format version 1). New in the IR: the `I8*` operators, int8
activation tensors (`Reshape` accepts i8), and constants of dtype i8 / i32 for weights, multipliers, shifts,
biases and tables. Quantization parameters are node attributes (`x_zero`, `x_scale`, `y_zero`, `qmin`, `qmax`,
`mult`, `shift`, `zeros`, ...) or constant inputs; nothing is inferred at load time. The runtime loader needed no
change; the session exempts the `int8_full_*` tiers from the opt-in rule of the lossy `int8_*` tiers (they are the
only kernels of the I8 operators and exact for them).

### Kernels and tiers

| tier | contents |
|---|---|
| `int8_full_reference` | every I8 operator, portable C++, bit-identical to `python/bitino/int8_ops.py` (the oracle) |
| `int8_full_neon` | AArch64: Quantize, LUT (TBL/TBX; per-channel tables on the last axis as transposed 16×16 blocks), Add (incl. scalar b), AffineSum, LayerNorm, Softmax, Transpose, ReduceMean (last axis incl. rows of 2/4/8, middle axis), GlobalAvgPool, Pad, Concat, Mul, pooling, depthwise convolution, an int32 SDOT GEMM for I8Conv2D / I8Dense (any K with a provably exact int32 accumulator, grouped convolutions) and I8MatMul, and the binary wrappers |
| `int8_full_opt` | portable wrappers of the binary operators around the best float-path binary kernel of the CPU (x86-64 hosts: AVX2) |

Since 2026-11-12 the `int8_full_neon` kernels also build for 32-bit ARM (armeabi-v7a, `kernels/neon_a32.h`) except the
SDOT / SMMLA GEMMs, which have an AArch32 SMLAL GEMM instead; see "int8_full latency" at the end of this section.

Stage `int8_full` = `optimized` + these three tiers; `int8_full_reference` = reference only. Every NEON and
portable kernel is bit-identical to the reference (`int8_full_neon_bit_exact_vs_reference`: random shapes and
tails, all modes, 1 and 4 threads; the selected kernel is checked), and the reference is bit-identical to the
numpy oracle on whole models (Bi-ViT DeiT-Tiny, BHViT-Tiny, and `tests/python/test_int8_full.py`).

## Measured results

Per-model accuracy (exact vs int8_full, eval set, n, McNemar p) and emulated instruction counts are published as data in `results/models/<model>.json`; the website renders them.
