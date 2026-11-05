# `.bitino` format (version 1)

A single file: fixed 64-byte header, a JSON graph description, and a data section holding
constant tensors. Writer: `python/bitino/format.py`. Reader: `runtime/src/model.cc`
(C++) and `format.read` (Python). Both are tested against each other
(`tests/python/test_ir_and_format.py`).

## Header (little-endian)

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic `BFMODEL1` (fixed for format v1: it predates the project's current name and is kept so that every existing v1 file stays readable) |
| 8 | 4 | `format_version` = 1 |
| 12 | 4 | header size = 64 |
| 16 | 8 | JSON offset (always 64) |
| 24 | 8 | JSON size |
| 32 | 8 | data offset (64-byte aligned) |
| 40 | 8 | data size |
| 48 | 16 | reserved, zero |

## JSON section

```json
{
  "format_version": 1,
  "data_sha256": "<sha256 of the whole data section>",
  "manifest": { ... provenance, see below ... },
  "tensors": [{"name": "...", "dtype": "f32|i32|i8|u8|bit", "shape": [...],
               "offset": 0, "nbytes": 0, "layout": "OHWI|OHWI_bit|..."}],
  "inputs":  [{"name": "input", "dtype": "f32", "shape": [1, 224, 224, 3]}],
  "nodes":   [{"op": "BConv2D", "name": "...", "inputs": ["x", "w", null, ...],
               "outputs": ["..."], "attrs": {...}}],
  "outputs": ["logits", "probs"]
}
```

* `tensors` lists constants only; activation tensors are created by node outputs and their
  shapes are inferred by the runtime (`OpDef::infer`).
* Nodes are topologically sorted; an input of `null` is an omitted optional input.
* Each constant is 64-byte aligned inside the data section, so the runtime uses them as
  zero-copy views.

## Tensor encodings

* `f32`, `i32`, `i8`, `u8`: dense row-major arrays of `shape`.
* `bit`: binary tensor of logical `shape`, bit-packed along the **last** dimension into
  little-endian `uint32` words. Element `c` is bit `c % 32` of word `c / 32`.
  **Bit 1 encodes −1, bit 0 encodes +1.** Bits beyond the logical length are zero.
  Storage = `prod(shape[:-1]) * ceil(shape[-1] / 32) * 4` bytes.
* Activations are NHWC. Convolution filters are OHWI (`[out, kh, kw, in/groups]`). Dense weights are `[out, in]`.

## Manifest (provenance)

Written by the converters and copied verbatim into every benchmark report:

| Key | Meaning |
|---|---|
| `model_id` | registry ID (`registry/models.json`) |
| `variant` | e.g. `fused` / `unfused` |
| `source.checkpoint_sha256` | SHA-256 of the original checkpoint file used |
| `source.framework`, `source.class` | source framework and model constructor |
| `conversion.*` | tool, adapter, passes applied (with counts), library versions, date |
| `input` | input name, shape, layout, preprocessing definition |
| `layer_precision` | per-layer weight/activation precision as found in the source model |

## Integrity

`Model::load` verifies `data_sha256` by default. `bitino_bench --no-verify-hash` disables
this; the report then records `hash_verified: false`. Version mismatches,
truncated files, misaligned tensors and size mismatches are rejected with explicit errors.
