#!/usr/bin/env bash
# Runs a statically linked Android arm64 Bitino binary under qemu-aarch64 user mode.
# CORRECTNESS ONLY: emulated timings are meaningless and must never be reported.
#
#   BITINO_STATIC=ON scripts/build_android.sh build/android-arm64-static
#   scripts/run_emulated.sh build/android-arm64-static/bitino_tests
#   QEMU_CPU=cortex-a76 scripts/run_emulated.sh ...   # no i8mm: exercises the SDOT/feature fallbacks
#
# qemu-aarch64 is taken from $QEMU_AARCH64, PATH, or the project-local copy extracted from
# the Ubuntu qemu-user package into .cache/qemu (dpkg-deb -x, no root needed).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QEMU="${QEMU_AARCH64:-$(command -v qemu-aarch64 || echo "$ROOT/.cache/qemu/root/usr/bin/qemu-aarch64")}"
[[ -x "$QEMU" ]] || { echo "qemu-aarch64 not found (see header of this script)" >&2; exit 1; }
# -cpu max: all ARMv8/9 features QEMU implements (NEON, dotprod, i8mm, bf16, fp16; also SVE).
# Tests that re-exec themselves (feature-masking children) go through qemu too.
export BITINO_TEST_LAUNCHER="$QEMU|-cpu|${QEMU_CPU:-max}"
exec "$QEMU" -cpu "${QEMU_CPU:-max}" "$@"
