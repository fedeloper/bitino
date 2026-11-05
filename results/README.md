# results/ (schema `bitino.results.v1`)

Published measurements, as data. `tools/build_site.py` renders them into the README tables, `docs/MODEL_ZOO.md`
and the website; nothing in those is typed by hand.

## `models/<model-id>.json`

| key | meaning |
|---|---|
| `files` | shipped converted files (`model.bitino`, `model_int8_full*.bitino`): size and `data_sha256` |
| `accuracy.headline` | conversion check: `metric`, `eval_set`, `n`, `reference` (source framework, float32), `runtime` (Bitino exact stage), `delta_vs_reference` (`pt` or `dB`), `author_claim` |
| `accuracy.studies[]` | paired studies on larger sets: `a` / `b` = `file:stage`, `a_top1`, `b_top1`, `delta_pt`, `ci95_pt`, `mcnemar_p`, `agreement`, `n`, `date` |
| `accuracy.int8_full[]` | per shipped int8 file: `eval_set`, `n`, `top1`, `exact_top1`, `delta_vs_exact_pt`, `mcnemar_p`, `note` |
| `instructions_M` | exact ARM instructions per inference (millions) under qemu, 1 thread, per stage, with `method` and `date`: a cost measure, **not a timing** |
| `speed.s22[]` | measured Galaxy S22 runs (same records as `speed_s22.json`) |
| `speed.x86[]` | measured x86-64 host runs (same records as `speed_x86.json`) |

## `speed_s22.json`, `speed_x86.json`

`device` / `host` (hardware and conditions), `method`, `runs[]`, `next_suite` (slot for the next full run).

A run: `model_id`, `date`, `timestamp_utc`, `plan` / `suite`, `config`, `model_file`, `stage`, `threads`,
`affinity`, `median_ms`, `median_range_ms` (min / max over repeats), `p90_ms`, `repeats`, `iterations`, `warmup`,
`validity`, `caps_seen`, `notes`, `model_data_sha256`, `runtime`.

`validity`:

* `clean`: every repeat ran with every cpufreq policy at its maximum frequency;
* `throttled`: at least one repeat ran under a thermal frequency cap (listed in `caps_seen`);
* `unknown-caps`: the run predates the frequency-cap check;
* `superseded-runtime`: measured on an older runtime than the current one (kept until re-measured).

Rules: timings come only from real hardware. Emulated (qemu) and CI timings are never recorded here.

Adding runs: `python3 tools/ingest_speed.py --s22 <dir written by tools/benchmark.py>` or
`--x86 <results.json of tools/bench_host_all.py>`, then `python3 tools/build_site.py --readme-only`.
