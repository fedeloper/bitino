#!/usr/bin/env bash
# Expected device outputs: runs the static Android arm64 build under qemu for every model and stage and
# stores artifacts/<model>/emulated/<model>_<stage>.f32. scripts/device_final_suite.sh then checks that
# the phone produces bit-identical outputs (--expect-identical). Deterministic kernels make this exact
# for neon; for float_opt / INT8 it holds as long as the device's libm (exp, erf, tanh in the float
# reference ops) rounds like the NDK's static libm; differences there are reported, not hidden.
# Each output gets a <output>.commit stamp. Correctness only: timings under qemu are meaningless.
#   scripts/emulated_reference_outputs.sh                 # all models, stages neon optimized optimized_int8 int8_full int8_full_log2
#   MODELS="larq-zoo-quicknet" STAGES="neon" JOBS=4 scripts/emulated_reference_outputs.sh
#   VERIFY=1 scripts/emulated_reference_outputs.sh        # after a runtime change that must not change outputs:
#       compare with the stored reference and only re-stamp it when byte-identical (IDENTICAL); a differing output
#       is kept next to it as <output>.new and the reference is left alone (DIFF); a missing one is stored (NEW)
# Stage int8_full runs artifacts/<model>/model_int8_full.bitino, stage int8_full_log2 model_int8_full_log2.bitino
# (bench stage int8_full; models without the file are skipped).
# BITINO_STATIC_BUILD selects the build (default build/android-arm64-static, see scripts/bench_emulated.sh).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/eval_sets.sh
STAGES="${STAGES:-neon optimized optimized_int8 int8_full int8_full_log2}"; JOBS="${JOBS:-6}"; THREADS="${THREADS:-4}"
MODELS="${MODELS:-$(for d in artifacts/*/; do [[ -f "$d/model.bitino" ]] && basename "$d"; done | tr '\n' ' ')}"
one() {
  m="$1"; st="$2"
  read -r edir pp extra <<<"$(eval_of "$m")"
  [[ -n "$edir" && -f "$edir/manifest.json" ]] || { echo "$m $st: SKIP (no eval set)"; return; }
  efile="$(python3 -c "import json;print(json.load(open('$edir/manifest.json'))['preprocessing']['$pp']['file'])")"
  model="artifacts/$m/model.bitino"; rst=$st; [[ "$st" == int8_full* ]] && { model="artifacts/$m/model_$st.bitino"; rst=int8_full; }
  [[ -f "$model" ]] || { echo "$m $st: SKIP (no $(basename "$model"))"; return; }
  mkdir -p "artifacts/$m/emulated"
  out="artifacts/$m/emulated/${m}_${st}.f32"
  if scripts/bench_emulated.sh --model "$model" --inputs "$edir/$efile" --outputs "$out.tmp" \
       --stage "$rst" --threads "$THREADS" --warmup 0 --iterations 1 >/dev/null 2>&1; then
    if [[ -n "${VERIFY:-}" && -f "$out" ]]; then
      if cmp -s "$out.tmp" "$out"; then rm -f "$out.tmp" "$out.new"; echo "$REV" > "$out.commit"; echo "$m $st: IDENTICAL"
      else mv "$out.tmp" "$out.new"; echo "$m $st: DIFF (kept $out.new, reference unchanged)"; fi
    else
      [[ -f "$out" ]] && state=ok || state=NEW
      mv "$out.tmp" "$out"; echo "$REV" > "$out.commit"; echo "$m $st: $state"
    fi
  else
    rm -f "$out.tmp"; echo "$m $st: FAILED"
  fi
}
# Stamp each output with the commit of the runtime it came from (device_final_suite.sh only demands bit identity
# when the runtime sources are unchanged since then). Uncommitted runtime changes make the stamp "dirty".
REV="$(git rev-parse HEAD)"; git diff --quiet HEAD -- runtime CMakeLists.txt || REV="dirty"
export -f one eval_of; export THREADS REV VERIFY
for m in $MODELS; do for st in $STAGES; do echo "$m $st"; done; done | xargs -P "$JOBS" -L1 bash -c 'one "$0" "$1"'
