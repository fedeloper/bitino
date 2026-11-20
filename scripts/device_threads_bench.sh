#!/usr/bin/env bash
# Thread scaling on a phone: every model's float file (stage all) and full-integer file (stage int8_full) at 1, 2, 4
# and 8 threads, each run pinned to the N fastest cores of the phone (CORES, fastest first). Batch 1. Every run is
# pinned through bitino_bench --affinity and its affinity_errors are recorded (tools/benchmark.py).
#
#   ANDROID_SERIAL=e3698ea7 CORES=7,3,4,5,6,0,1,2 scripts/device_threads_bench.sh                     # OnePlus Open
#   ANDROID_SERIAL=ZY22HWKBP4 CORES=7,6,5,4,3,2,1,0 scripts/device_threads_bench.sh                   # moto e13
#   ANDROID_SERIAL=75AUS4CUDYN75LEQ CORES=0,1,2,3,4,5,6,7 BIN=build/android-armv7/bitino_bench \
#       BITINO_DEVICE_DIR=/data/local/tmp/bitino32 scripts/device_threads_bench.sh                   # Redmi 9A (32-bit)
#   MODELS="larq-zoo-quicknet bnext-tiny" THREADS="1 4" ...                                           # subsets
#
# Output: reports/benchmarks/<m>/threads_<TS>_<tag>/ (configs t<N>_fast and t<N>_fast_int8_full); log in
# reports/device_threads_<TS>_<tag>/bench.log. Resume after an interruption with START_AT=<model>.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ADB="${ADB:-$HOME/Android/Sdk/platform-tools/adb}"
[[ -n "${ANDROID_SERIAL:-}" ]] || { echo "ERROR: set ANDROID_SERIAL" >&2; exit 1; }
[[ -n "${CORES:-}" ]] || { echo "ERROR: set CORES (fastest first)" >&2; exit 1; }
export ANDROID_SERIAL
export BITINO_DEVICE_LOCK="${BITINO_DEVICE_LOCK:-/tmp/bitino-device-$ANDROID_SERIAL.lock}"
[[ -n "${BITINO_DEVICE_DIR:-}" ]] && export BITINO_DEVICE_DIR
"$ADB" get-state >/dev/null 2>&1 || { echo "ERROR: $ANDROID_SERIAL not reachable" >&2; exit 1; }
BIN="${BIN:-build/android-arm64/bitino_bench}"
THREADS="${THREADS:-1 2 4 8}"
TS="$(date +%Y%m%d_%H%M)"
DEVICE_TAG="${DEVICE_TAG:-$("$ADB" shell getprop ro.product.model | tr -d '\r' | tr 'A-Z _' 'a-z--')}"
OUT="reports/device_threads_${TS}_$DEVICE_TAG"
mkdir -p "$OUT"
LOG="$OUT/bench.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
[[ -x "$BIN" ]] || { log "ERROR: $BIN missing (scripts/build_android.sh)"; exit 1; }
MODELS="${MODELS:-$(python3 -c '
import json, os
st = json.load(open("registry/bitino_status.json"))
print(" ".join(sorted(m for m, v in st.items()
                      if v.get("conversion_status") == "converted" and os.path.isfile(f"artifacts/{m}/model.bitino"))))')}"
COOLDOWN="${COOLDOWN:-10}"; REPEATS="${REPEATS:-3}"; ITERS="${ITERS:-50}"; WARMUP="${WARMUP:-10}"
log "device: $("$ADB" shell getprop ro.product.model) serial $ANDROID_SERIAL tag $DEVICE_TAG; binary $BIN (commit $(git rev-parse --short HEAD)); cores $CORES; threads $THREADS"
started=""; [[ -z "${START_AT:-}" ]] && started=1
for m in $MODELS; do
  [[ -z "$started" && "$m" == "${START_AT:-}" ]] && started=1
  [[ -n "$started" ]] || continue
  [[ "$("$ADB" get-state 2>/dev/null)" == device ]] || { log "ERROR: device lost before $m; resume with START_AT=$m"; exit 2; }
  dir="reports/benchmarks/$m/threads_${TS}_$DEVICE_TAG"
  mkdir -p "$dir"
  python3 - "$m" "$CORES" "$THREADS" > "$dir/configs.json" <<'EOF'
import json, os, sys
m, cores, threads = sys.argv[1], sys.argv[2].split(","), [int(t) for t in sys.argv[3].split()]
c = []
for f, stage, suffix in (("model.bitino", "all", ""), ("model_int8_full.bitino", "int8_full", "_int8_full")):
    p = os.path.abspath(f"artifacts/{m}/{f}")
    if not os.path.isfile(p):
        continue
    for t in threads:
        if t <= len(cores):
            c.append(dict(name=f"t{t}_fast{suffix}", binary="rel", model=p, stage=stage, threads=t,
                          affinity=",".join(cores[:t])))
print(json.dumps(c))
EOF
  log "benchmark $m -> $dir"
  python3 tools/benchmark.py --plan custom --configs "$dir/configs.json" --model-dir "artifacts/$m" --out "$dir" \
      --rel-binary "$BIN" --warmup "$WARMUP" --iterations "$ITERS" --repeats "$REPEATS" --cooldown "$COOLDOWN" \
      --cap-max "${CAP_MAX:-0}" >>"$LOG" 2>&1 || log "WARNING: $m had failures"
  python3 tools/report_benchmarks.py "$dir" --baseline t1_fast > /dev/null 2>>"$LOG" && log "summary: $dir/summary.md"
done
log "done"
