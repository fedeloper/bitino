# Architecture

Bitino separates five concerns that are often entangled in BNN code. Each has its own
code location and its own tests.

| Concern | Where | Owns |
|---|---|---|
| Model definition | source framework (larq_zoo, PyTorch repos) + `python/bitino/adapters/` | reading a verified checkpoint into the IR |
| Graph IR + format | `python/bitino/ir.py`, `format.py`, `runtime/src/model.cc` | tensors, nodes, `.bitino` (see `model-format.md`) |
| Operator semantics | `runtime/ops/op_defs.cc`, `docs/operators.md`, `python/bitino/executor.py` | attribute validation, shape inference, exact math |
| Kernels | `runtime/kernels/<tier>/` | fast implementations of one op on one class of CPU |
| Instrumentation | `runtime/tools/bitino_bench.cc`, `bitino_microbench.cc`, `tools/*.py` | timing, memory, device fingerprint, reports |

```
 checkpoint (.h5 / .pth / safetensors)        research/ + registry/models.json
        │  verified SHA-256                      (provenance, licence, precision)
        ▼
 source framework model ──adapter──▶ IR Graph ──passes──▶ .bitino (+manifest)
        │                              │                      │
  reference outputs              numpy executor          C++ runtime ── host / Android
        └────────── Level-3/4 comparison ──────────────────────┘
```

## Runtime

* `Model::load`: maps the file (`MAP_PRIVATE`; `LoadOptions::map_file`, otherwise a heap copy), parses the header
  and JSON, verifies the data SHA-256, and exposes constants as aligned zero-copy views. After preparing its kernels a
  `Session` drops the pages of constants that every reader prepacked (`Kernel::released_inputs`,
  `SessionOptions::release_constants`): a dropped page is read again from the file if anything touches it, so the
  resident memory holds the packed weights only (S22: ReActNet-18 log2 12.0 → 10.1 MB RSS, BNext-Small 48.5 → 36.9 MB).
  A mapped model file must not be rewritten while a session uses it.
* `Session(model, options)`:
  1. shape inference through each `OpDef`, on the session's own copy of the nodes;
  1b. plan-time rewrites when `float_opt` is allowed and `SessionOptions::fuse` / `reuse_memory` are
     on (`src/graph_rewrite.h`): bias-only `ChannelAffine` folded into the threshold of the binary op it
     feeds, trees of elementwise nodes fused into `FusedElementwise` nodes, and such trees moved into the
     epilogue of the `BConv2D` / `BDense` producing one of their operands, and binary layers whose (epilogue-fused)
     float output only feeds the `ge` / `gt` binarization of the next binary layer switched to bit-packed output
     (the next layer reads the bits). All are bit-exact (same float operations in the same order per element,
     the stored bit is the binarization the consumer would have computed); the `Model` is never modified
     (`docs/operators.md` § Plan-time rewrites);
  1c. int8_full rewrites with `SessionOptions::fuse` (`src/graph_rewrite_i8.h`, any stage, I8* nodes only): element-wise
     ops of one tensor collapsed to a LUT, LUT chains composed, LUTs read only by binarizations folded into integer
     thresholds, the out_lut absorption into `I8BConv2DRes`, and twin nodes (same inputs) merged into internal
     multi-output nodes (`I8LutN`, `I8AddN`, `I8Conv2DN`); exact by construction (`docs/int8.md` § "int8_full latency");
  2. memory planning (graph inputs and outputs get dedicated buffers; intermediates share
     buffers by liveness; a node never aliases its own inputs, except that a `Reshape` output is a
     view of its input's buffer, which then lives until the last use of any view, unless the
     output is a graph output);
  3. kernel selection, where the highest priority kernel wins among those whose tier is
     allowed by `KernelPolicy`, whose `required_features` are present, and whose `supports()`
     accepts the node; `KernelPolicy::force` pins a kernel by name;
  4. `Kernel::prepare` (weight prepacking, scratch allocation). None of this is timed as inference.
* `Session::run` executes the steps in order. Kernels parallelize internally with the
  session's `ThreadPool`.

### Threading model

`SessionOptions::num_threads = N` means N CPU workers: the calling thread plus N−1 pool
threads. N = 1 never creates threads. `parallel_for` hands out contiguous chunks
(about 8 per worker, to balance big.LITTLE cores) claimed with a tagged compare-and-swap; the
caller returns when every item is finished and never waits for a worker that claimed nothing.
During `Session::run` the pool is hot by default (`spin_run`: waiting threads spin instead of
blocking, with smaller chunks and a lower serial threshold). Optional affinity
pins worker *i* to `affinity[i % len]` with `sched_setaffinity`, which needs no root.
Pinning failures are counted in reports, not fatal, and retried once per run (the S22 pauses idle
cores). A `Session` must not be run concurrently.
Share the `Model` and create one session per thread instead. Details and the phone measurements
behind the defaults: `docs/threads.md`.

### Kernel tiers

| Tier | Purpose | Benchmark stage |
|---|---|---|
| `reference` | oracle, literal loops | 1–2 |
| `packed_naive` | bit packing + XOR/popcount, no blocking | 3 |
| `binary_scalar` | portable optimized bit-packed kernels | 4 |
| `neon` | AArch64 NEON binary kernels | 5 |
| `binary_avx2` | x86-64 AVX2 binary kernels (VPSHUFB popcount; `bconv2d.avx2`, `bdense.avx2`, `bdense.ternary_avx2`, `qbconv2d.avx2`, `qbdense.avx2`) | 5 (stage `neon` allows it, so stage names match across targets) |
| `int8_reference`, `int8_neon`, `int8_avx2` | dynamic-range INT8 for float Conv2D/Dense (SDOT / I8MM; x86-64 VPMADDWD / AVX-VNNI) | 6–7 |
| `float_opt` | optimized fp32 kernels for full-precision layers: 4-lane NEON / portable kernels, plus AVX2+FMA kernels (`*.float_opt_avx2`) on x86-64; also the inexact float32 `*.float_fast` LayerNorm / Softmax / ReduceMean / GELU kernels (priority 45 over the exact double ones, `docs/numerical-correctness.md`), so only stages that allow `float_opt` use them | 8 |

Stages are tier sets (`bitino_bench --stage`), so every stage runs the same binary and the
same model file. The platform-SIMD stages are named after the phone (`neon`, `int8_neon`) and
select the x86-64 tiers (`binary_avx2`, `int8_avx2`) on a host build.

A tier may hold several backends of the same kernel, distinguished by `required_features` and
priority: `float_opt` registers `conv2d.float_opt` (NEON or portable, priority 40) and, on x86-64,
`conv2d.float_opt_avx2` (`avx2|fma`, priority 50; depthwise 45 / 55). The selector takes the AVX2
kernel when the CPU reports the features, the portable one otherwise or under
`BITINO_DISABLE_FEATURES=avx2`. The arithmetic shared by the backends lives in
`float_opt_leaf.h`, written once against a vector policy (4-lane `Simd4`, 8-lane `Simd8`); the
GEMM micro-kernels differ (8x8 FMLA-by-lane on NEON, 3x16 / 6x8 broadcast-FMA on AVX2).

### CPU feature detection

`cpu_info()` reads `getauxval(AT_HWCAP/AT_HWCAP2)` on AArch64 Linux and Android
(asimddp, i8mm, fp16, bf16, sve) and `__builtin_cpu_supports` on x86 (popcnt, avx2, fma, bmi2,
avxvnni). Code that needs instructions beyond the build baseline (`armv8.2-a+dotprod`; plain
x86-64 on the host) lives in separate translation units with their own flags (`-march=...+i8mm`,
`-mavx2 -mfma`, `-mavx2 -mavxvnni`, set per file in `CMakeLists.txt`). It is only selected when
the feature is reported. For testing, `BITINO_DISABLE_FEATURES=i8mm,dotprod,avx2,avxvnni,...`
masks features.

Those translation units contain only leaf routines on raw pointers (micro-kernels, quantizers,
the float_opt arithmetic); kernel classes, packing, scheduling and registration stay in baseline
code. They define nothing with external linkage except their entry points and call no inline
function with external linkage: such a function, emitted with extension instructions in that TU,
could be picked by the linker for code that runs on every CPU. `tools/check_isa_isolation.py`
verifies this on a linked x86-64 binary (every function containing VEX/ymm code must be a local or
strong symbol of the AVX translation units, and those must define no weak symbols); it passes for
the Release and the `-O0` builds.

The AVX2 binary kernels follow the same rule: their SIMD code is in `runtime/kernels/binary_avx2/binary_avx2_simd.cc` (`-mavx2 -mpopcnt`).

The transformer / BHViT float kernels, the fused elementwise interpreter, the bit-plane BMatMul and the
INT8 MatMul added no new ISA translation unit: their AVX2 leaves live in the existing ones
(`float_opt_avx2.cc`: vector math `transformer_leaf.h`, double-precision LayerNorm/ReduceMean helpers,
mul/sign leaves, the compiled fused-elementwise evaluator `leaf_fe_rows`; `binary_avx2_simd.cc`: BMatMul plane packing and POPCNT dots; `int8_gemm_avx2.cc`: the
run-time W packer), so `tools/check_isa_isolation.py` keeps its list. The NEON code uses baseline ASIMD
(including `float64x2_t`) in baseline translation units.

## Extension points

* **New model family**: write an adapter producing IR (see `adapters/keras_larq.py`). Record
  per-layer precision and add a conversion tool or CLI entry plus a registry entry.
* **New operator**: see `docs/operators.md` § Adding an operator.
* **New kernel**: implement `Kernel`, register a `KernelDef` in your tier's
  `register_*_kernels`, gate it with `required_features` and `supports()`, and add
  differential tests.
* **New backend / CPU class**: add a tier and feature bits. The selector, benchmarks and tests
  stay unchanged.

## Ownership and lifetimes

* `Model` owns the file buffer. Constant `Tensor`s share ownership through `shared_ptr`.
* `Session` holds a `shared_ptr<const Model>` and owns all activation buffers, kernels and
  scratch.
* Kernels own their prepacked weights. They never allocate in `run()`.

## Errors

Load and plan-time problems throw `bitino::Error` with the node or tensor name and the
reason, e.g. `no kernel for node 'conv3' (op BConv2D) under the current policy. Rejected:
bconv2d.neon (missing CPU features)`. CLI tools return non-zero exit codes. Kernel hot paths
never throw.
