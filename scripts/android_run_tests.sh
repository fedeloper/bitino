#!/usr/bin/env bash
# Pushes and runs the C++ test binary on the connected device (no root needed).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
OUT="${1:-$ROOT/build/android-arm64}"
DEV_DIR=/data/local/tmp/bitino
"$ADB" shell mkdir -p "$DEV_DIR"
"$ADB" push "$OUT/bitino_tests" "$DEV_DIR/" >/dev/null
# Serialize with other device users (benchmarks) when a lock file is configured.
LOCK="${BITINO_DEVICE_LOCK:-}"
if [[ -n "$LOCK" ]]; then flock "$LOCK" "$ADB" shell "$DEV_DIR/bitino_tests"; else "$ADB" shell "$DEV_DIR/bitino_tests"; fi
