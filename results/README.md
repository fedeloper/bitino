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
| `speed.<device>[]` | measured runs on that phone (`s22`, `oneplus-open`, ...; same records as `speed_<device>.json`) |
| `speed.x86[]` | measured x86-64 host runs (same records as `speed_x86.json`) |

## `devices.json`

`devices.<id>`: `kind` (`phone`), `name`, `short` (column header), `soc`, `android`, `abi` (`arm64-v8a`, or
`armeabi-v7a` for 32-bit Android), `kernels` (`neon`, or `portable-fallback` for builds without the ARM kernels),
`isa`, `cores[]` (`name`, `count`, `max_ghz`, `cpus`), `policies` (cpufreq policy id → core name, used in
`caps_seen`), `fastest_cpu` (the CPU 1-thread runs pin to), `product_models` (Android `ro.product.model` values a
run's recorded device must match), `tags` (benchmark directory suffixes), `notes`, `runs` (count).

## `speed_<device>.json`, `speed_x86.json`

`device_id`, `device` / `host` (hardware and conditions), `method`, `runs[]`, `next_suite` (slot for the next full run,
when one is pending).

A run: `model_id`, `device`, `date`, `timestamp_utc`, `plan` / `suite`, `config`, `model_file`, `stage`, `threads`,
`affinity`, `median_ms`, `median_range_ms` (min / max over repeats), `p90_ms`, `repeats`, `iterations`, `warmup`,
`validity`, `caps_seen` (frequency ceilings `scaling_max/cpuinfo_max` per policy, fastest first), `notes`, `abi`,
`kernels`, `model_data_sha256`, `runtime`.

`validity`:

* `clean`: every repeat ran with every readable cpufreq policy at its maximum frequency;
* `vendor-policy`: a frequency ceiling below the maximum while Android reported no thermal event (thermal status 0):
  the phone's own power policy (e.g. screen off), steady, listed in `caps_seen`;
* `thermal-throttled`: at least one repeat ran under a thermal frequency cap (listed in `caps_seen`); `throttled`
  in older files means the same;
* `unknown-caps`: the run predates the frequency-cap check;
* `superseded-runtime`: measured on an older runtime than the current one (kept until re-measured).

Rules: timings come only from real hardware. Emulated (qemu) and CI timings are never recorded here.

Adding runs: `python3 tools/ingest_speed.py --device <id> <dirs written by tools/benchmark.py>` (`--s22 <dirs>` =
`--device s22`; a new phone: `--add-device`, then complete its `devices.json` entry) or
`--x86 <results.json of tools/bench_host_all.py>`, then `python3 tools/build_site.py --readme-only`.
