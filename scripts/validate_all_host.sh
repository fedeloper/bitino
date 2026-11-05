#!/usr/bin/env bash
# End-to-end validation of every converted model with the native host runtime (x86-64: AVX2 tiers).
#   scripts/validate_all_host.sh                 # all artifacts/*/model.bitino, stages neon + optimized
#   MODELS="larq-zoo-quicknet" JOBS=2 LABEL=host_x86 scripts/validate_all_host.sh
# neon must be bit-exact with the numpy executor (--accept-exact-numpy); optimized must pass --explain.
# Writes artifacts/<model>/validation_<LABEL>_<stage>.json and prints one line per run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/eval_sets.sh
BENCH="${BENCH:-build/host/bitino_bench}"; LABEL="${LABEL:-host_x86}"; JOBS="${JOBS:-4}"; THREADS="${THREADS:-4}"
MODELS="${MODELS:-$(for d in artifacts/*/; do [[ -f "$d/model.bitino" ]] && basename "$d"; done | tr '\n' ' ')}"
one() {
  m="$1"; st="$2"
  read -r edir pp extra <<<"$(eval_of "$m")"
  x="--accept-exact-numpy"; [[ "$st" == optimized ]] && x="--explain"
  r="$(python3 tools/validate_model.py --artifacts "artifacts/$m" --eval-dir "$edir" --preprocess "$pp" $extra \
       --bench "$BENCH" --stage "$st" --threads "$THREADS" --label "${LABEL}_$st" $x \
       --json "artifacts/$m/validation_${LABEL}_$st.json" 2>&1 | tail -1)"
  echo "$m $st: $r"
}
export -f one eval_of; export BENCH LABEL THREADS
for m in $MODELS; do for st in neon optimized; do echo "$m $st"; done; done | xargs -P "$JOBS" -L1 bash -c 'one "$0" "$1"'
