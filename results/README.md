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

## `community/` (submissions from other people's phones)

Measured with `scripts/bench_my_phone.sh` on phones the maintainers do not own, and labelled `community` everywhere
(guide: `docs/benchmark-your-phone.md`). Kept apart from the maintainers' runs above; the README tables do not mix
them.

A submission (`bitino-bench-<phone model>-<date>.json`, schema `bitino.community_bench.v1`): `tool` (script version,
release), `created_utc`, `device` (`model`, `manufacturer`, `brand`, `device`, `soc_manufacturer`, `soc_model`,
`board_platform`, `hardware`, `android_release`, `api_level`, `security_patch`, `abi`, `kernel`, `cpu_features`,
`emulator_markers`, `cpus[]` (part numbers), `cpufreq_policies` (CPUs, `cpuinfo_max_khz`, ceiling, governor),
`fastest_cpu`, `t4_affinity`), `binary` (`source` release / local, `sha256`, `runtime_version`), `models[]` (file
SHA-256), `plan`, `conditions` (clock ceilings, thermal status and battery at the start and the end),
`output_checks[]` (`output_sha256` vs `expected_sha256` on the `pattern-v1` input at the file's exact stage),
`summary[]` and `runs[]` (one per repeat: configuration, `throttle` block as in `tools/benchmark.py`, and the raw
`bitino_bench` report with the build fingerprint and the model manifest removed). No serial number or other
identifier.

Adding one: `python3 tools/ingest_community.py <file.json> --issue <n>` (`--check` validates without writing), then
`python3 tools/build_site.py --readme-only`. It refuses a file (and writes nothing) unless:

* the schema and the script version are known;
* the device block names a phone model, SoC or board platform, Android release and an arm64-v8a / armeabi-v7a ABI;
* it is a physical phone: no emulator marker, cpufreq policies present, every raw report recorded on the same
  `ro.product.model` (read by the runtime on the phone);
* no identifying field (serial, IMEI, Android ID, accounts, build fingerprint, host name, ...) carries a value;
* every timed file is a published release file (its `data_sha256` is in `registry/release_models.json`) and the
  runtime verified its hash;
* each raw report matches its run (stage, threads, affinity, iterations) and its median matches its samples;
* the numbers are plausible (0.05 ms < median < 60 s; 4 threads at most 4.6× faster than 1; int8_full / float within
  0.05–20);
* every output check with a reference matched.

Files:

* `community/speed_<device-id>.json` (device id `community-<ro.product.model>`, e.g. `community-cph2551`): `device`
  (as in `devices.json`, plus `source: "community"`), `method`, `submissions[]` (`id` = first 12 hex digits of the
  file's SHA-256, `issue`, `created_utc`, Android version and patch, runtime, conditions), `runs[]`.
  A run has the keys of a maintainers' run (`model_id`, `date`, `config` `t1_fast` / `t4_fast`, `model_file`,
  `stage`, `threads`, `affinity`, `median_ms`, `median_range_ms`, `p90_ms`, `repeats`, `iterations`, `warmup`,
  `peak_rss_kb`, `validity`, `caps_seen`, `notes`, `abi`, `kernels`, `model_data_sha256`, `runtime`) plus
  `source: "community"`, `submission`, `issue`, `android`, `binary_sha256` and `output_check` (`match`,
  `unchecked`, `not run`). `validity` follows the rules above (worst repeat; repeats whose pinned thread stayed
  unpinned are dropped). `notes` flag repeats more than 1.5× apart and medians more than 2× away from the
  maintainers' own runs of the same phone model. Re-ingesting a file replaces its runs; other submissions are kept.
* `community/index.json`: per device `name`, `soc`, `abi`, `submissions`, `runs`, `latest`, and `headline_ms`
  (median over submissions of ReActNet-18 (A&B BNN, ImageNet), float and int8_full files, 1 and 4 threads).
