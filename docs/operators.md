# Operator contracts

The reference kernels (`runtime/kernels/reference/`) and the numpy executor
(`python/bitino/executor.py`) implement exactly these contracts. Every optimized
kernel is differentially tested against the reference kernel.

Operators added for the Larq Zoo models beyond QuickNet/Bi-RealNet (`QBConv2D`, `QBDense`,
`ChannelGate`) are specified in `operators_larq.md`.

Common conventions: NHWC activations, OHWI filters, explicit pads `[top, bottom, left, right]`,
`strides [sh, sw]`, `dilations [dh, dw]`. A fused `activation` is one of `none | relu | relu6`.

## Conv2D (float)

Inputs: `x f32[N,H,W,C]`, `w f32[O,KH,KW,C/groups]`, optional `bias f32[O]`.
Attributes: `strides, dilations, pads, groups, pad_value (float, default 0), activation`.
Output: `act(bias[o] + Σ x_pad · w)`, where out-of-bounds positions read `pad_value`.
`OH = (H + pt + pb − dh·(KH−1) − 1) / sh + 1`. The reference accumulates in double.

### Kernels of the float operators

Selected by tier, CPU features and `supports()` (highest priority wins, see
`docs/architecture.md`). `_avx2` kernels exist on x86-64 builds only and need `avx2` + `fma`;
`int8_*` tiers are opt-in and lossy (`docs/int8.md`).

| op | kernel | tier | prio | accepts | numerics vs reference |
|---|---|---|---|---|---|
| Conv2D | `conv2d.reference` | reference | 0 | all | oracle (double accumulation) |
| Conv2D | `conv2d.float_opt` / `_avx2` | float_opt | 40 / 50 | `groups == 1` | fp32 GEMM, bound `1e-4·(1+|ref|)` |
| Conv2D | `conv2d_grouped.float_opt` / `_avx2` | float_opt | 40 / 50 | `groups > 1` (one GEMM per group) | as above |
| Conv2D | `conv2d_depthwise.float_opt` / `_avx2` | float_opt | 45 / 55 | `groups == C`, multiplier 1 | as above |
| Conv2D | `conv2d.int8_reference` | int8_reference | 30 | all, `K ≤ 133144` | INT8 bound (`docs/int8.md`) |
| Conv2D | `conv2d.int8_neon_sdot` / `_i8mm` | int8_neon | 140 / 150 | `groups == 1` | bit-exact with `int8_reference` |
| Conv2D | `conv2d.int8_avx2` / `_vnni` | int8_avx2 | 140 / 150 | `groups == 1` | bit-exact with `int8_reference` |
| Dense | `dense.reference` | reference | 0 | all | oracle |
| Dense | `dense.float_opt` / `_avx2` | float_opt | 40 / 50 | f32 | fp32 GEMM bound |
| Dense | `dense.int8_reference`, `_neon_*`, `_avx2*` | int8_* | 30 / 140 / 150 | `K ≤ 133144` | as Conv2D |
| Add, ChannelAffine, MaxPool2D | `*.float_opt` / `_avx2` | float_opt | 40 / 50 | all | bit-exact |
| Activation | `activation.float_opt` / `_avx2` | float_opt | 40 / 50 | relu, relu6, hardtanh, prelu, leaky_relu | bit-exact |
| Activation | `activation_ext.float_opt` / `_avx2` | float_opt | 40 / 50 | gelu, sigmoid, tanh, hardswish, sign, sign_zero | hardswish/sign/sign_zero bit-exact; gelu/sigmoid/tanh vector math, bounds below |
| AvgPool2D, GlobalAvgPool | `*.float_opt` / `_avx2` | float_opt | 40 / 50 | all | float sums, test bound |
| Mul, ChannelGate | `mul.float_opt`, `channel_gate.float_opt` / `_avx2` | float_opt | 40 / 50 | all | bit-exact (gate sigmoid in double, as the reference) |
| Concat, Transpose, Gather | `concat.float_opt`, `transpose.float_opt`, `gather.float_opt` | float_opt | 40 | all (Gather: f32 tables) | bit-exact (data movement) |
| Reshape | `reshape.reference` | reference | 0 | all | zero-copy: the session aliases the output to the input buffer (only a graph output is copied) |
| FusedElementwise | `fused_elementwise.reference`, `.float_opt` / `_avx2` | reference / float_opt | 0 / 40 / 50 | internal op (plan-time fusion) | bit-exact with the unfused nodes |
| LayerNorm, ReduceMean, FakeQuant, AddBroadcast | `*.float_opt` / `_avx2` | float_opt | 40 / 50 | all | bit-exact (LayerNorm/ReduceMean: the reference's double sums, rows per lane) |
| Softmax | `softmax.float_opt` / `_avx2` | float_opt | 40 / 50 | all | float exp and sums, `≤ 2e-6·|ref|` |
| MatMul | `matmul.float_opt` / `_avx2` | float_opt | 40 / 50 | f32 | fp32 GEMM bound |
| MatMul | `matmul.int8_reference`, `_neon_sdot` / `_i8mm`, `_avx2` / `_vnni` | int8_* | 30 / 140 / 150 | `K ≤ 133144` | INT8 (`docs/int8.md` § MatMul), bit-exact with `int8_reference` |

GELU/sigmoid/tanh bounds (`tests/cpp/test_transformer_opt.cc`, grid of 2.2M / 0.55M floats per backend):
GELU `|y − ref| ≤ 6e-7·|ref| + 2^-52·|x|` (measured worst 4.4e-7 relative, the `2^-52·|x|` term covers
the double reference's own cancellation for `x < −5.5`) and `sign(y) = sign(ref)` for every input, including
the reference's exact `−0` for `x ≤ −8.37438965` (BiBERT binarizes GELU outputs with `sign_zero`, so BiBERT's
encoder stays bit-exact); sigmoid and tanh `≤ 4e-7·|ref| + 1e-37` (measured 0.47 / 0.30 of that).
Formulas in `runtime/kernels/float_opt/transformer_leaf.h`.

### Plan-time rewrites (session)

At stages that allow `float_opt` (and with `SessionOptions::fuse`, `bitino_bench --no-fuse` to disable)
the session rewrites its private copy of the graph (`runtime/src/graph_rewrite.cc`); every rewrite performs
the same IEEE operations in the same order per element, so outputs are bit-identical to the unfused graph:

* **Threshold folding.** `ChannelAffine(x, scale=none, bias b)` whose output only feeds the f32 input of
  `BConv2D`/`BDense` (`ge`, `gt`, `sign_zero`, no threshold) is removed; the consumers get `threshold = −b`.
  `fl(x + b)` has the sign of `x + b` (and is zero exactly when it is) for finite operands.
  With a scale (`ChannelAffine(x, s, b)` before `ge` / `gt` consumers: DenseNet / MeliusNet batch norms) the
  per-channel threshold `t` with `fl(fl(x·s) + b) ≥ 0` (`gt`: `> 0`) ⇔ `x ≥ t` is found by bisection over the
  ordered floats (the expression is monotone in `x` for `s ≠ 0`); a channel with `s < 0` is binarized negated
  (`x ≥ next(u)`, `u` the largest `x` giving +1) and its weight bits are flipped in a copy of each consumer's
  weights, so every product is unchanged. `BConv2D` consumers with negated channels need `groups == 1` and
  `pad_value == 0` (a padded tap cannot be negated); channels with `s == 0`, non-finite parameters or a
  constant binarization keep the node. Exact for finite inputs (NaN inputs of negated channels would differ).
* **Elementwise fusion.** Maximal trees of `ChannelAffine`, `Add` (+relu/relu6), `Mul`, `Activation`
  (relu, relu6, hardtanh, prelu, leaky_relu, sign, sign_zero) and `ChannelGate` whose inner tensors have one
  consumer become one internal `FusedElementwise` node (`runtime/ops/fused_elementwise.h`): a stack program,
  one pass over memory instead of one per node. The `float_opt` kernels compile it at prepare time into
  register micro-ops (`FeMicro`, `runtime/kernels/float_opt/float_opt_args.h`) and run the whole program on one
  chunk of 32 floats of a row at a time (16 when only that divides the row, e.g. 16 or 48 channels) (`leaf_fe_rows`, `float_opt_leaf.h`; NEON / portable 4-lane and AVX2):
  the stack top and a binary op's operand stay in vector registers, a load feeding a binary op is fused into it
  in either operand position (operand order kept), deeper stack values are spilled to a small local array, and
  `bias -> prelu -> bias` (RPReLU) is one micro-op. The program ends with a sentinel micro-op and the full-width
  chunks of a row (or of a flat multi-row run) are evaluated by direct threading (`FeChunk::run_chunks`: each
  micro-op's handler jumps to the next one's; the single-vector and scalar tails keep the switch). Each micro-op
  is the instruction's exact float operation, vector and scalar tails alike, so the chunking cannot change a bit. Trees larger than 8 stack
  slots / 24 inputs are split at their largest subtree. Two data-movement producers become *loads* of the
  program instead of materialized tensors: a row `Gather` with an f32 table (optionally between `Reshape`s,
  BHViT's `torch.roll`; rows of `D` channels with `C % D == 0`, out-of-range ids give zero rows, as the
  operator) and a `Concat` of one tensor with itself along the last axis (BHViT's repeat shortcut). The
  compiled evaluator reads such rows in place when the piece width is a multiple of the chunk (16-float pieces
  run with half chunks), else it first copies the row to a per-thread row buffer.
* **Fused binary-layer epilogues.** When one tree operand is the f32 output of a `BConv2D`, or of a `BDense`
  with a binary input (`ge` / `gt`, no `w_zero`), used only by the tree and every other operand already exists
  when that node runs, the tree (even a single node, e.g. BNext's PReLU; BHViT's MLP chain norm -> + repeated
  block input -> RPReLU after the intermediate `BDense`) becomes the node's `epilogue` attribute
  (`runtime/kernels/float_opt/eltwise_program.h`; the program's operands are appended after the node's 5
  (`BConv2D`) or 6 (`BDense`) fixed inputs). `bconv2d.avx2` / `bconv2d.neon_opt` run it on each pixel tile right
  after computing it, `bdense.neon` on each row tile (~32 KB of output, computed and packed by one task),
  `bdense.avx2` on each row tile (when channels are split across threads, after the tasks), `binary_scalar` /
  `reference` after the whole product; the layer output is never written back separately. When the program is a
  chain on the layer's value (affine, prelu, RPReLU, adds of row or tile operands; e.g. ReActNet's + shortcut ->
  RPReLU, QuickNet's BN + shortcut, BNext's PReLU, BHViT's MLP chain) and the channel count is a multiple of 16,
  `bconv2d.neon_opt` (two-phase path) and `bdense.neon` (`tile2`) evaluate it on each 16-channel unit in registers,
  right after converting the unit's dot products (`EltwiseProgram::store_unit16`): no store and reload of the
  tile, per-channel constants loaded once per unit, the same float operations in program order (bit-identical). `packed_naive`, the
  old `bconv2d.neon` and the ternary (`sign_zero`) BDense kernels reject such nodes; hosting in an op is off when
  a kernel is forced for it (`SessionOptions::host_epilogues`, `bitino_bench --no-epilogues` keeps such trees
  as separate fused nodes).

* **int8_full rewrites** (`runtime/src/graph_rewrite_i8.cc`, with `SessionOptions::fuse`, at every stage; I8* graphs
  only, exact by construction, details in `docs/int8.md` § "int8_full latency"): `collapse_unary` (an `I8Mul` /
  `I8Add` / `I8AffineSum` whose operands are all `x` or `I8Lut(x)` of one tensor becomes one `I8Lut` of `x`, its table
  the operator's integer formula on all 256 codes), `compose_luts` (`I8Lut` of an `I8Lut` -> one table),
  `fold_lut_thresholds` (a binary layer binarizing an `I8Lut` output reads the LUT input with integer thresholds when
  every channel's set `{x : L[x] >= T}` is an upper set; when every channel's set `{x : L[x] < T}` is an interval
  `[lo, hi]` instead, e.g. V-shaped PReLU tables, it reads the bit tensor of the internal `I8PackBits(x, [lo; hi])`,
  bit -1 iff `lo[c] <= x <= hi[c]`), and twin merging (nodes reading the same inputs: `I8Lut` -> internal `I8LutN`,
  `I8Add` with equal input parameters -> `I8AddN`, `I8Conv2D` with byte-identical weights / bias -> `I8Conv2DN`,
  `I8LayerNorm` with the same statistics attributes -> `I8LayerNormN`: one pass, one shared computation, one output
  stage per twin). `I8BDense` also accepts a bit-tensor input (as `I8BConv2D`), used by `I8PackBits`.

* **Bit-packed outputs.** Last rewrite (after fusion, so epilogue-fused outputs qualify): a `BConv2D`, or a
  `BDense` with a binary (`ge` / `gt`) input and no `w_zero`, whose f32 output (possibly the output of its hosted
  epilogue) has exactly one use, input 0 of a `BConv2D` / `BDense` that binarizes it with `ge` / `gt`, writes that
  binarization itself: `output = bit`, `output_binarize` = the consumer's mode, the consumer's threshold (e.g. a
  folded bias) moved to the producer as `output_threshold_input` (it must exist when the producer runs); the
  consumer reads the bits (no threshold, no packing). Exact by construction: the stored bit is `pack_row_f32` of
  the very float the consumer would have read. `sign_zero` consumers are skipped (three values do not fit one bit),
  as are outputs with several uses (residual CNNs: the float also feeds the shortcut) and graph outputs; off when a
  `BConv2D` / `BDense` kernel is forced (`SessionOptions::bit_outputs`, `bitino_bench --no-bit-outputs`). Kernels
  (`kernels/bit_output.h`): without an epilogue the bit depends only on the integer dot product `d`, and
  `act(fl(fl(d·post_mul) + post_add)) ≥ t` is monotone in `d` (rounding and relu / relu6 are monotone), so it is
  `d < lim[o]` or `d > lim[o]` per channel (LCE's fused thresholds, but derived exactly: `IntThresholds` evaluates
  the kernels' float expression for every reachable `d ∈ [−K, K]` once per parameter set and checks the single
  transition; NaN cases such as `inf·0` that break it keep the float compare). `bconv2d.neon_opt` /
  `bconv2d.avx2` / `bdense.neon` / `bdense.avx2` then binarize each 4- / 8-channel block with integer compares: no
  float epilogue, no float store, no packing in the consumer. With an epilogue the float tile goes to a
  kernel-private buffer, the epilogue runs on it and the finished rows are packed right away (`pack_row_f32_neon` /
  AVX2 `pack_row`) while in cache. `binary_scalar` / `reference` / `packed_naive` / `bconv2d.neon` implement the
  same contract with float rows (bit-exact, `tests/cpp/test_bit_outputs.cc`). The ternary (`sign_zero`) BDense
  kernels reject bit output. On the 114 converted models the rewrite fires in 25 (stage
  `optimized`, `config.plan_rewrites.bit_outputs`): the FINN / Brevitas / BNN-PYNQ MLPs and CNVs (2-3 each),
  Larq Zoo Binary AlexNet (4) and XNOR-Net (3), the IR-Net-family / FDA-BNN / NNCF / SA-BNN / OMZ ResNets
  (3-9: the first binary convolution of a basic block, whose float output only feeds the block's second one,
  through BN / PReLU / clip epilogues), VGG-small (2), DoReFa AlexNet (2) and
  BNAS-A (34); none in the residual CNNs with float shortcuts after every convolution (QuickNet, ReActNet,
  Bi-Real, BNext), nor in Bi-ViT / BiBERT (`sign_zero` consumers) or BHViT. ARM instructions per inference
  (`tools/insn_count.py`, stage `optimized`, 1 thread, vs master): BNN-PYNQ CNV −4.1%, BNAS-A −3.9%, ReCU-ResNet-18
  CIFAR −3.3%, Brevitas LFC −1.6%, Larq Binary AlexNet −0.8%, XNOR-Net −0.7%. The same change specializes the
  `bconv2d.neon_opt` tile at compile time (f32 store / integer-threshold bits / float-compare bits), which removes
  the output-mode branch from the f32 path: −1.9% to −2.7% on QuickNet-Small, ReActNet-18, BNext-Tiny, Bi-Real-18
  (no bit outputs there), −1.2% on BHViT-Tiny; Bi-ViT and BiBERT unchanged (±0.01%).
* **Shared binarizations.** With bit outputs on: an f32 tensor that two or more `BDense` nodes binarize identically
  (input 0, same `ge` / `gt` mode and threshold input, no `w_zero`) is packed once by an internal `PackBits` node
  (`runtime/ops/op_defs_fused.cc`; kernels `pack_bits.reference` = `pack_row_f32`, `pack_bits.float_opt` = the NEON
  packing of `bdense.neon` on AArch64) placed before the first of them; they read its bits, without threshold.
  Exact: each consumer reads the bits it would have packed. The tensor stays materialized for other uses. Bi-ViT's
  `norm1` output feeds the q / k / v `BDense` of every head with `ge` and no threshold: one packing instead of 9
  (DeiT-Tiny) / 18 (DeiT-Small) per block (`config.plan_rewrites.shared_binarizations` = 12). Off when a `BDense`
  kernel is forced.

Counts on the stage-`optimized` plans: BHViT-Tiny 1,166 nodes absorbed into 92 fused nodes + 51 epilogues
(44 of them in `BDense`; 192 Gathers and 31 Concats among the absorbed nodes), 268 remaining Reshapes aliased;
BHViT-Small 143 fused nodes + 77 epilogues; BNext-Tiny 31 thresholds folded, 31 epilogues; ReActNet-18 16
thresholds, 13 epilogues; QuickNet-Small 16 epilogues (`bitino_bench` JSON: `config.plan_rewrites`).

## BConv2D (binary weights × binary activations)

Inputs: `x f32[N,H,W,C]` or `x bit[N,H,W,C]`; `w bit[O,KH,KW,C]`; optional `post_mul f32[O]`,
`post_add f32[O]`, `threshold f32[C]` (f32 input only). Attributes: `strides, dilations, pads`,
`groups` (filter `w bit[O,KH,KW,C/groups]`; output channel `o` sees input channels
`[g·C/groups, (g+1)·C/groups)` of group `g = o / (O/groups)`), `pad_value ∈ {−1, 0, +1}`,
`input_binarize`, `activation`, `output ∈ {f32, bit}`.

1. Binarize the input: `b(x) = +1 if x ≥ t else −1` (`ge`, larq `sign`, sign(0)=+1);
   `+1 if x > t else −1` (`gt`); `sign(x − t) ∈ {−1, 0, +1}` (`sign_zero`, `torch.sign`
   semantics, where exact zeros contribute 0; **reference kernels only**). `t = threshold[c]` or 0.
2. Integer accumulation: `acc = Σ_{taps, c} b_pad · w`. An out-of-bounds tap contributes
   `pad_value · w` (pad 0 = zero padding *after* binarization, as in PyTorch/larq default).
3. `v = act(acc · post_mul[o] + post_add[o])` in fp32, in that order (then the fused `epilogue`, if the
   plan-time rewrite hosted one, see below).
4. `output = f32`: store `v`. `output = bit`: store the next layer's binarization of `v`: bit 1 (−1) iff
   `!(v ≥ t[o])` (`output_binarize = ge`, default) or `!(v > t[o])` (`gt`), with `t` the `f32[O]` input whose
   index is the attribute `output_threshold_input` (an input after the fixed ones and any epilogue operands),
   or 0 without it. This is exactly `pack_row_f32` of the float output (NaN → 1, −0 like +0), so a consumer
   reading the bits computes the same binary values as from the float tensor (`kernels/bit_output.h`).
   Models do not use it; the plan-time rewrite sets it (§ Plan-time rewrites).

Bit-packed kernels use `a·b = K − 2·popcount(a ⊕ b)`. Out-of-bounds taps are filled
with +1 bits and corrected exactly by `(pad_value − 1) · Σ_{oob taps} S[o][tap]`, with
`S[o][tap] = Σ_c w[o,tap,c]`. Results are **bit-exact** with the reference. That's enforced
by tests on the host and on the device, and the build uses `-ffp-contract=off`.

Grouped convolution in the packed kernels (`BConvGroupLayout`, `runtime/kernels/bconv_plan.h`):
the patch row of a group concatenates, per tap, that group's `C/groups` input bits, in
32-bit units read straight from the packed input when `C/groups % 32 == 0` (or `groups = 1`),
in 16-bit units when `C/groups % 32 == 16` (taps packed densely: 16 channels × 3×3 = 144 bits
instead of 9 words), and otherwise after repacking the input per group into
`[pixel][group][⌈(C/groups)/32⌉]` words with zero padding bits. Weights use the same layout, so
the formula above holds with `K = KH·KW·C/groups`.

### BConv2D / BDense kernels

| kernel | tier | CPU features | groups | notes |
|---|---|---|---|---|
| `bconv2d.reference` / `bdense.reference` | reference | — | any | literal loops, also `sign_zero` |
| `bconv2d.packed_naive` | packed_naive | — | any | 32-bit XOR/popcount, no blocking |
| `bconv2d.binary_scalar` / `bdense.binary_scalar` | binary_scalar | — | any | 64-bit popcounts, 2 px × 4 oc |
| `bconv2d.neon` | neon | NEON | 1 | first NEON kernel (kept for comparison) |
| `bconv2d.neon_opt` / `bdense.neon` | neon | NEON | any | 4×4 / 8×4 inline-asm micro-kernel, ≤ 64 taps |
| `bconv2d.avx2` / `bdense.avx2` | binary_avx2 | AVX2 + POPCNT | any | VPSHUFB popcount, 8 oc × 4 px, ≤ 64 taps |
| `bdense.ternary_bitplane` | binary_scalar | — | — | `sign_zero` BDense (+ `w_zero`), two bit-planes |
| `bdense.ternary_neon` / `bdense.ternary_avx2` | neon / binary_avx2 | NEON / AVX2 + POPCNT | — | same contract, SIMD |
| `qbconv2d.avx2` / `qbdense.avx2` | binary_avx2 | AVX2 + POPCNT | 1 | DoReFa `act_bits ≤ 4` as bit-planes (`operators_larq.md`) |
| `qbconv2d.neon` / `qbdense.neon` | neon | NEON | 1 | DoReFa `act_bits ≤ 8`: each (pixel, plane) is a row of the `bconv2d.neon_opt` micro-kernel with AND instead of XOR |

Every ±1 packed kernel rejects `sign_zero` (and BDense `w_zero`); the ternary kernels cover
`sign_zero` BDense.

### BMatMul kernels

| kernel | tier | CPU features | notes |
|---|---|---|---|
| `bmatmul.reference` | reference | — | int8 values, int32 dots |
| `bmatmul.bitplane` | binary_scalar | — | every operand as `{nz, neg}` bit-planes (all four modes, incl. `gt0` = `{0,1}`), 64-bit popcounts |
| `bmatmul.neon` | neon | NEON | same planes as 32-bit words; `b` packed once per batch entry into blocks of 8 columns (`b[K,N]` straight from rows with SLI, no transpose), register-blocked 4 rows × 8 columns tiles with uint8 VCNT counts; rows of `a` whose `nz` plane is full (no exact zeros) and/or full column blocks of `b` use 4 / 3 instead of 7 vector ops per word (same integers); work split over (batch, row range), so single-head attention matmuls also use several threads |
| `bmatmul.avx2` | binary_avx2 | AVX2 + POPCNT | AVX2 plane packing (`binary_avx2_simd.cc`), POPCNT dots |

`a·b = popcount(m) − 2·popcount(m & (neg_a ⊕ neg_b))`, `m = nz_a & nz_b`, exact for every mode pair;
`y = float(acc)·float(scale)` as the reference, so all are bit-exact (1584 comparisons per CPU in
`bmatmul_kernels_bit_exact`, 16 mode pairs, transposed and not, K = 1..2100, NaN inputs, and inputs
without zeros / with zero rows or columns for the NEON full-plane tiles). With a full side,
`popcount(m)` is the other side's `popcount(nz)` (k if both are full) and `m & (neg_a ⊕ neg_b)` needs only
the other side's `nz` (none if both are full: `neg ⊆ nz`, padding bits are 0).

## Dense / BDense

`Dense`: `x f32[..., K]`, `w f32[O,K]`, optional bias, `activation`.
`BDense`: `x f32|bit [..., K]`, `w bit[O,K]`, `post_mul`, `post_add`, `threshold`,
`input_binarize`, `activation`, `output ∈ {f32, bit}` (+ `output_binarize`, `output_threshold_input`).
Same arithmetic as BConv2D with one tap, including step 4 (bit output: the next layer's `ge` / `gt`
binarization of the final float row). Bit input cannot use a threshold or `sign_zero`.
Optional input 5 `w_zero bit[O,K]`: a set bit (the "−1" encoding) marks an **exact zero
weight**, so `w ∈ {−1, 0, +1}` (e.g. `torch.sign` of a per-row centered weight, BiBERT). It
requires `input_binarize = sign_zero` (OpDef check). Every packed ±1 kernel rejects `sign_zero`,
so only ternary-aware kernels can run such a node: `bdense.reference` and
`bdense.ternary_bitplane` (tier `binary_scalar`, see below).

### Ternary bit-plane encoding (`bdense.ternary_bitplane`)

A `{−1, 0, +1}` operand is stored as two bit-planes in 64-bit words: `nz` (value ≠ 0) and
`neg` (value < 0, same polarity as the `bit` dtype). For ternary `x` and weights with an
optional zero mask, `m = nz_x & nz_w` and `x·w = popcount(m) − 2·popcount(m & (neg_x ⊕ neg_w))`,
which is exact; without `w_zero`, `nz_w` is all ones. The post-scale is applied exactly as in the
reference, so results are bit-identical (`tests/cpp/test_transformer_ops.cc`).

## Elementwise, pooling, shape

| Op | Contract |
|---|---|
| `ChannelAffine(x, scale?, bias?)` | `x·scale[c] + bias[c]` over the last dim (folded BatchNorm) |
| `Activation(x, slope?)` | `type`: relu, relu6, hardtanh(min,max), prelu(slope[c]), leaky_relu(alpha), sigmoid, tanh, hardswish, gelu, sign (`x ≥ 0 → +1`). `gelu = x·0.5·(1 + erf(x/√2))` and `tanh` are evaluated in **double** with libm and rounded once (reference + numpy), so no fp32 `erf` underflow: `gelu(x) < 0` for all `x ∈ (−8.3, 0)` (an fp32 torch GELU returns 0 for `x ≲ −5.5`) |
| `Add(a, b)` | same shapes, optional fused activation |
| `Concat(xs…)` | `axis` (default −1) |
| `MaxPool2D` | `kernel, strides, pads`; out-of-bounds taps are ignored |
| `AvgPool2D` | `kernel, strides, pads, count_include_pad` (0: divide by in-bounds count, TF "same"; 1: by kernel area) |
| `GlobalAvgPool` | mean over H, W; `keepdims` 0 → `[N, C]` |
| `Reshape` | `shape` with at most one −1 |
| `Softmax` | last axis, max-subtracted, double accumulation in reference |
| `Pad` | `pads` (before/after per dim), constant `value` |

## Transformer operators

Added for BiBERT (`runtime/ops/transformer_op_defs.cc`, `runtime/kernels/reference/transformer_reference.cc`,
numpy mirror in `python/bitino/executor.py`). Generic: no BERT-specific attributes. Activations
are row-major with the feature axis last (`[batch, tokens, features]`, heads as `[B, H, L, Dh]`).
The float reference kernels and the numpy executor perform the **same IEEE operations in the
same order**, so the numpy f32 executor and the C++ reference are bit-identical
(`tests/python/test_transformer_ops.py`).

| Op | Inputs | Attributes | Contract |
|---|---|---|---|
| `Gather` | `table f32[V,D] \| bit[V,D]`, `indices i32[...]`, `scale f32[V]?` (bit only), `zero_mask bit[V,D]?` (bit only) | — | `y[..., :] = table[idx]` → `f32[..., D]`. Bit table: `±1 · scale[idx]` (exact fp32 product), `0` where `zero_mask` bit is set. **Out-of-range ids (<0 or ≥V) give a zero row** (no throw in the hot path). The table may be a runtime tensor (e.g. selecting the CLS token). |
| `LayerNorm` | `x f32[..., D]`, `gamma f32[D]?`, `beta f32[D]?` | `eps` (default 1e-5) | Last axis, biased variance. Reference in double: `mean = (Σ x_i)/D` and `var = (Σ (x_i−mean)²)/D` summed sequentially in index order, `y = float((((x−mean)/sqrt(var+eps))·gamma)+beta)`. |
| `MatMul` | `a f32[..., M, K]`, `b f32[..., K, N]` (or `[..., N, K]`) | `transpose_b` (0/1), `scale` (1.0) | Both operands may be runtime tensors; equal batch dims, no broadcasting. Reference: double accumulation in `k` order from 0, `y = float(acc·scale)`. The numpy mirror uses float64 BLAS (not order-identical; documented bound). |
| `BMatMul` | `a f32[..., M, K]`, `b f32[..., K, N]` (or `[..., N, K]`) | `a_binarize`, `b_binarize` ∈ {`ge`, `gt`, `sign_zero`, `gt0`} (default `sign_zero`), `transpose_b`, `scale` | Binarize both operands, **exact int32** accumulation, `y = float(acc) · float(scale)`. `gt0` is the Heaviside `{0,1}` map `x > 0 → 1 else 0` (BiBERT's ZMeanBinaryQuantizer = `sign` with −1→0, so `0 → 0`). `sign_zero` gives `{−1,0,+1}`. `K < 2^24`. |
| `AddBroadcast` | `a f32[S]`, `b f32[...]` | — | `a + b` with `b` right-aligned and broadcast (each dim equal or 1); output shape `S` (e.g. additive attention mask `[B,1,1,L]`). A separate op so optimized same-shape `Add` kernels never see broadcasting. |
| `Transpose` | `x f32[...]` (rank ≤ 8) | `perm` | `y = x.transpose(perm)` (e.g. heads `[0,2,1,3]`). |

Optimized kernels of these operators (all `float_opt` unless noted): `layernorm.float_opt[_avx2]` evaluates
the reference's double-precision formula with several rows per vector (4 per `__m256d`, 2 per
`float64x2_t`; NEON runs 4 such chains = 8 rows side by side to hide the FADD latency of the sequential
sums, and widens gamma / beta to double once per run), each lane summing its row in index order, so it is
**bit-exact**; `matmul.float_opt[_avx2]`
is a float GEMM on the Conv2D/Dense micro-kernels with B packed per run (bound `1e-4·(1+|ref|)`);
`add_broadcast`, `transpose`, `gather` (f32 table) are bit-exact; `BMatMul` has bit-plane kernels (above);
`MatMul` has opt-in INT8 kernels (`docs/int8.md` § MatMul).

Graph inputs may be `i32` (token ids) as well as `f32`. `bitino_bench --inputs FILE` reads one
record per sample = the raw bytes of **every graph input in graph input order** (single-input
models: unchanged). Timing-only runs leave `i32` inputs at 0 (a valid id).

### BiBERT attention as IR (one head group)

`Q,K,V = BDense(sign_zero)` → `Reshape [1,L,H,Dh]` → `Transpose [0,2,1,3]`;
`S = BMatMul(Q, K, transpose_b=1, sign_zero × sign_zero, scale=1/√Dh)`;
`S' = AddBroadcast(S, mask)` with `mask = ChannelAffine(attention_mask, 10000, −10000)` reshaped to `[1,1,1,L]`
(equals `(1−m)·(−10000)` exactly for `m ∈ {0,1}`); `C = BMatMul(S', V, gt0 × sign_zero)` (softmax removed, as in the source);
`Transpose [0,2,1,3]` → `Reshape [1,L,H·Dh]` → `BDense(sign_zero)`. All attention arithmetic is exact
(scores are multiples of 1/8, context values are integers).

## BHViT operators

`FakeQuant`, `ReduceMean` and the `Activation` type `sign_zero` were added for BHViT, together with the
mapping of every BHViT construct onto IR operators: see `docs/operators_bhvit.md`.

## PyTorch binary-CNN operators

`Mul(a, b)` (same shapes, one fp32 product per element) was added for BNext's ELM attention product.
The squeeze-and-excitation gate reuses `ChannelGate`, and the info-recoupling mix reuses `ChannelAffine` + `Add`. The
mapping of every BNext / RBONN construct onto IR operators is in `docs/operators_torch_cnn.md`.

## Image-feature operators

`ImageGrayU8` (OpenCV fixed-point gray of `trunc(x·255)`), `FftLogMagnitude` (`mul·log(|fftshift(FFT2)| + eps)/div`)
and `LocalBinaryPattern` (scikit-image LBP, `default` / `uniform`) compute hand-crafted input channels inside the
model, so preprocessing runs in the runtime (the "Faster Than Lies" deepfake detectors). They are defined by a fixed
double-precision algorithm (mixed-radix Stockham FFT, polynomial twiddles and log, no libm) that the reference
kernels, the optimized kernels (tier `neon` on AArch64, `binary_scalar` elsewhere) and the numpy executor share bit for
bit: `docs/operators_image_features.md`. `concat.small_channels` (same tiers, priority 45) concatenates last-axis inputs
of at most 8 channels each in one pass over the pixels.

## Brevitas / FINN BNN-PYNQ

No new operator. The Q1.7 8-bit input of Brevitas CNV uses the generic `FakeQuant` above (scale 1/128,
zero point 0, `qmin −128`, `qmax 127`). The Brevitas → Bitino layer mapping (BN+sign → exact integer
thresholds) is in [operators_brevitas.md](operators_brevitas.md).

## Full-integer operators (`I8*`, stage `int8_full`)

Produced by `tools/quantize_int8_full.py` (`python/bitino/quantize_int8.py`) from a float model; scheme,
rewrites and calibration in `docs/int8.md` § Full-integer execution. Oracle: `python/bitino/int8_ops.py`;
C++ OpDefs `runtime/ops/op_defs_int8_full.cc`, kernels `runtime/kernels/int8_full/`. Activations are int8 with
`x = s·(q − z)`; `requant` is the fixed-point multiply of `docs/int8.md` (SQRDMULH + SRSHL, half up); `rne` is
round half to even after clamping in float. "f32 out" = `output: f32` (dequantized result, graph outputs only).

| Op | Inputs | Attributes | Contract |
|---|---|---|---|
| `I8Quantize` | `x f32` | `scale, zero` | `clamp(rne(fp32(x / scale)) + zero, −128, 127)` |
| `I8Dequantize` | `q i8` | `scale, zero` | `fp32(fp32(q − zero) · scale)` |
| `I8Lut` | `q i8`, `table i8\|f32 [T,256]` | `axis` (−1), `output` | `table[row][q + 128]`, row = index along `axis` (T = that size) or 0 (T = 1) |
| `I8Add` | `a i8`, `b i8` (broadcast to a) | `a_zero, b_zero, left_shift (20), a_mult/a_shift, b_mult/b_shift, out_mult/out_shift, y_zero, qmin, qmax` | `x_a = requant((a−z_a) << ls)`, same for b, `y = clamp(z_y + requant(x_a + x_b))` (TFLite Add) |
| `I8AffineSum` | `x_0..x_{N−1} i8`, `K i32 [N,1\|C]`, `B i32 [1\|C]` | `zeros [N], mult, shift, y_zero, qmin, qmax` | `clamp(z_y + requant(Σ_i (x_i − z_i)·K[i][c] + B[c]))`, int32 accumulation |
| `I8Mul` | `a i8`, `b i8` (broadcast) | `a_zero, b_zero, mult, shift, y_zero, qmin, qmax` | `clamp(z_y + requant((a − z_a)(b − z_b)))` |
| `I8Conv2D` | `x i8 NHWC`, `w i8 OHWI`, `bias i32[O]?`, `mult i32[O]`, `shift i32[O]` (f32 / l8 out: `bias f32[O]?`, `scale f32[O]`) | conv attrs, `x_zero, pad_q, y_zero, qmin, qmax, output` (`i8`, `f32`, `l8`) | `acc = Σ (x − x_zero)·w` (padding taps read `pad_q`), `clamp(z_y + requant(sat32(acc + bias), mult[o], shift[o]))`; f32: `fp32(acc)·scale[o] + bias[o]`; l8: `l8_encode` of the f32 value |
| `I8Dense` | `x i8 [...,K]`, `w i8 [O,K]`, as Conv2D | `x_zero, y_zero, qmin, qmax, output` | as I8Conv2D, `K ≤ 133144` |
| `I8MatMul` | `a i8`, `b i8` | `a_zero, b_zero, transpose_b, mult, shift, y_zero` / `scale` (f32 out) | `acc = Σ (a − z_a)(b − z_b)`; `clamp(z_y + requant(sat32(acc)))` or `fp32(acc)·scale` |
| `I8LayerNorm` | `x i8 [...,D]`, `G f32[D]`, `B f32[D]` | `x_zero, x_scale, eps, qmin, qmax` | `t = q − z`, `S1 = Σt`, `S2 = Σt²` (int64), `r = fp32(xs / √(xs²·(D·S2 − S1²) + (ε·D)·D))` in double, `y = rne(fp32(fp32(fp32(D·t − S1)·r)·G) + B)` |
| `I8Softmax` | `x i8 [...,N]`, `E i32[256]` | `inv_scale, y_zero, qmin, qmax, output` | `e = E[max − q]`, `S = Σe`, `r = fp32(inv_scale / S)` (double division; f32 out: `1/S`), `y = clamp(rne(fp32(e)·r) + z_y)` |
| `I8Concat` | `x_i i8` | `axis, y_zero, zeros, mults, shifts` | input i copied if `mults[i] = 0`, else `clamp(z_y + requant(q − zeros[i]))` |
| `I8Transpose` | `x i8` | `perm` | permutation |
| `I8Gather` | `table i8 [V,D]` or `bit [V,D]` + `qpm i8 [V,2]` (+ `zero_mask bit`), `ids i32` | `fill_q` | rows; bit tables: `+1 → qpm[v][0]`, `−1 → qpm[v][1]`, masked → `fill_q`; out-of-range ids → `fill_q` |
| `I8MaxPool2D` / `I8AvgPool2D` | `x i8 NHWC` | pool attrs, `x_zero`, `count_include_pad` | max / `clamp(z + rdiv(Σ (q − z), count))`, rdiv half away from zero; output parameters = input's |
| `I8GlobalAvgPool` / `I8ReduceMean` | `x i8` | `x_zero, y_zero, mult, shift` (`axis, keepdims`) | `clamp(z_y + requant(Σ (q − x_zero)))` |
| `I8Pad` | `x i8` | `pads, pad_q` | constant `pad_q` |
| `I8BDense` | `x i8`, `w bit[O,K]`, `post_mul f32[O]`, `post_add f32[O]`, `threshold f32[K]?`, `w_zero bit?` | `input_binarize, x_scale, x_zero, qmin, qmax, output` | binarize `deq(q)` against the threshold with the integer thresholds `T_ge`, `T_gt` (`docs/int8.md`), exact accumulation, `v = fp32(fp32(acc)·post_mul) + post_add`, `clamp(rne(v))` |
| `I8BConv2D` | as BConv2D with `x i8` (or `x bit`, taken as ±1) | as BConv2D + `x_scale, x_zero` (or `x_code: l8, x_exp`), `qmin, qmax, output` | same, `pad_value` in the binary domain |
| `I8BConv2DRes` | `x i8\|bit`, `w bit`, `post_mul f32[O]`, `post_add f32[O]`, `threshold f32[C]?`, `res i8 [N,OH,OW,O]`, `slope f32[O]`, `out_bias f32[O]?`, `bit_thr f32[O]?`, (9: `out_lut`, Session only), `pre_slope f32[O]?`, `pre_mul f32[O]?`, `pre_add f32[O]?`, `gate i8 [N,1,1,O]?`; outputs `y i8` (l8), `bits bit` (iff `bit_thr`) | conv attrs, `input_binarize`, x attrs as I8BConv2D, `r_code (l8\|linear), r_exp \| r_scale, r_zero, bit_binarize, bit_src (sum\|out), y_exp`, `g_scale, g_zero` (with `gate`) | `v` as I8BConv2D, pre stage (each step only with its input): `v = v ≥ 0 ? v : v·pre_slope`, `v = v·pre_mul`, `v = v + pre_add`, `v = v·G` with `G = (gate − g_zero)·g_scale` per sample and channel; `u = v + R(res)`, `w = u ≥ 0 ? u : u·slope`, `w += out_bias`, `y = l8_encode(w)`, bit = `!(s ≥ bit_thr)` (ge) / `!(s > bit_thr)` with `s = u` (bit_src sum, default) or `s = w` (bit_src out); l8 code and R in `docs/int8.md` § log2 residual stream |
| `I8BMatMul` | `a i8`, `b i8`, `mul f32[1\|N]`, `add f32[1\|N]?` | `a/b_binarize, a/b_scale, a/b_zero, a/b_threshold, transpose_b, c_ab, c_rs, c_cs, c0, qmin, qmax, output` | `acc = c_ab·Σ_k A·B + c_rs·Σ_k A + c_cs·Σ_k B + c0` over the binarized operands, `v = fp32(fp32(acc)·mul[j]) + add[j]`, `clamp(rne(v))` |

`Reshape` accepts i8 tensors (zero-copy, as for f32).

## Adding an operator

1. Add an `OpDef` (validation + shape inference) in `runtime/ops/op_defs.cc`.
2. Add a reference kernel in `runtime/kernels/reference/`.
3. Mirror the semantics in `python/bitino/executor.py` and document them here.
4. Add tests: reference vs hand computation, and numpy vs C++ (`tests/python`).
5. Optional: optimized kernels in their own directory and tier, with `supports()` and
   `required_features`, differentially tested against the reference.
