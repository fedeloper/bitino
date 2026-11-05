#!/usr/bin/env bash
# bitino_bench (static Android arm64 build) under qemu-aarch64 — a drop-in --bench for
# tools/validate_model.py. Correctness only; timings from this wrapper are meaningless.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/scripts/run_emulated.sh" "${BITINO_STATIC_BUILD:-$ROOT/build/android-arm64-static}/bitino_bench" "$@"
