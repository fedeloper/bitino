#!/usr/bin/env bash
# Head-to-head latency benchmark: Bitino vs Larq Compute Engine vs plain TFLite (fp32, int8), 1 thread, batch 1,
# on the connected phones, same Larq Zoo models and weights. Method and caveats: docs/comparison-lce-tflite.md.
#
#   scripts/h2h_lce_tflite.sh [download|build|models|run|report|all]       (default: all)
#
# env:
#   H2H_WORK   work dir for downloads and converted models (default: $TMPDIR/h2h); downloads go to $H2H_WORK/dl
#   PHONES     space-separated serial:tag:cpu:abi (default: OnePlus Open cpu7, Moto e13 cpu7; the Redmi 9A entry is
#              75AUS4CUDYN75LEQ:redmi9a:0:armv7: LCE has no Android armeabi-v7a binary, so there the official Linux
#              AArch32 LCE binary runs through a Debian armhf glibc loader pushed to /data/local/tmp/h2h/glibc)
#   MODELS     comma-separated model ids (default: the 11 Larq Zoo models below)
#   REPEATS    rounds per model (default 5); WARMUP / ITERATIONS (default 10 / 50)
#   OUT        results dir (default: reports/h2h/<date>)
#   SKIP_BUILD=1  reuse build/android-arm64 and build/android-armv7
# The 2026-11-12 results (reports/h2h/2026-11-12) ran OnePlus with REPEATS=5; Moto with REPEATS=5 for the QuickNets,
# Bi-Real and ResNet-E18 and 3 for the rest; Redmi (PHONES=75AUS4CUDYN75LEQ:redmi9a:0:armv7) with REPEATS=3 on
# MODELS=quicknet-small,quicknet,quicknet-large,birealnet,binary-resnet-e18, after its correctness suite had ended.
# The 2026-11-19 results (reports/h2h/2026-11-19) ran PHONES="e3698ea7:oneplus-a715:4:arm64 e3698ea7:oneplus-a510:0:arm64"
# with REPEATS=5 (OnePlus Open Cortex-A715 and Cortex-A510), binaries and models copied back from the phone after a
# SHA-256 check against the pinned hashes and the 2026-11-12 build reports.
# Control (TFLite fp32 model under LCE's binary): tools/h2h_bench.py run ... --runtimes tflite_fp32,lcebin_fp32.
# Heavy host steps run under scripts/capped.sh; timing runs hold /tmp/bitino-device-<serial>.lock (tools/h2h_bench.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STEP="${1:-all}"
WORK="${H2H_WORK:-${TMPDIR:-/tmp}/h2h}"
DL="$WORK/dl"
MDIR="$WORK/models"
OUT="${OUT:-reports/h2h/$(date +%Y-%m-%d)}"
PHONES="${PHONES:-e3698ea7:cph2551:7:arm64 ZY22HWKBP4:moto-e13:7:arm64}"
MODELS="${MODELS:-larq-zoo-quicknet-small,larq-zoo-quicknet,larq-zoo-quicknet-large,larq-zoo-birealnet,larq-zoo-binary-resnet-e18,larq-zoo-xnornet,larq-zoo-binary-alexnet,larq-zoo-binary-densenet28,larq-zoo-binary-densenet37,larq-zoo-binary-densenet37-dilated,larq-zoo-binary-densenet45}"
LCE_PY="${LCE_PY:-$HOME/anaconda3/envs/lce/bin/python}"   # conda env lce: TF 2.16.1, tf_keras, larq 0.14.0, LCE 0.16.0
NDK="${ANDROID_NDK:-$HOME/Android/Sdk/android-ndk-r27c}"
CAP="scripts/capped.sh 8G nice -n 10"

# Official prebuilt benchmark binaries (URL, SHA-256 at download time 2026-11-12). The TFLite ones are the
# "nightly/latest" objects (Last-Modified Fri, 07 Feb 2025 07:59:51 GMT); a changed hash means the object moved.
LCE_URL=https://github.com/larq/compute-engine/releases/download/v0.16.0
TFL_URL=https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest
declare -A SHA=(
  [lce_benchmark_model_android_arm64]=4068e0c40b3ef51d167be1ab2677304c3bf3fcb8ff2651ef81f8295e025d1c90
  [lce_benchmark_model_aarch32]=8670b8a96b9a8617c635f1e47b24b54734e802b925f7e167fdba25d557158877
  [android_aarch64_benchmark_model]=5d45116eb86e3a23ea03318e93bfc8efb4dd0149d4692c127e7b0343faba1374
  [android_arm_benchmark_model]=3f3c0ed3436d1e506d78bf11ed5342c93acf64cff3145f1237ebd8a9047de068
  [libc6_2.36-9+deb12u14_armhf.deb]=758c68b92654747025b48476a45c315bf16df1321224abdd9be672bfd120be45
  [libgcc-s1_12.2.0-14+deb12u1_armhf.deb]=f58562bf01efd6112b914182147ee149e0d4c90869046e31844da8803669eeb3
  [libstdc++6_12.2.0-14+deb12u1_armhf.deb]=66d53ddeb10a37f22e455611c2c515f9d444e523b55d873ff48bde47267963db
)
DEBS=(pool/main/g/glibc/libc6_2.36-9+deb12u14_armhf.deb pool/main/g/gcc-12/libgcc-s1_12.2.0-14+deb12u1_armhf.deb
      pool/main/g/gcc-12/libstdc++6_12.2.0-14+deb12u1_armhf.deb)

download() {
  mkdir -p "$DL"
  for f in lce_benchmark_model_android_arm64 lce_benchmark_model_aarch32; do
    [[ -f "$DL/$f" ]] || curl -sSL --fail -o "$DL/$f" "$LCE_URL/$f"
  done
  for f in android_aarch64_benchmark_model android_arm_benchmark_model; do
    [[ -f "$DL/$f" ]] || curl -sSL --fail -D "$DL/$f.headers" -o "$DL/$f" "$TFL_URL/$f"
  done
  # Debian bookworm armhf runtime for the Linux AArch32 LCE binary (Redmi 9A: no Android armeabi-v7a LCE build exists)
  for deb in "${DEBS[@]}"; do
    f=$(basename "$deb")
    [[ -f "$DL/$f" ]] || curl -sSL --fail -o "$DL/$f" "https://deb.debian.org/debian/$deb"
  done
  for f in "${!SHA[@]}"; do
    got=$(sha256sum "$DL/$f" | cut -d' ' -f1)
    [[ "$got" == "${SHA[$f]}" ]] && echo "ok  $f $got" || { echo "ERROR: $f sha256 $got != pinned ${SHA[$f]}" >&2; [[ $f == android_* ]] || exit 1; }
  done
  rm -rf "$WORK/glibc_x" "$WORK/glibc"; mkdir -p "$WORK/glibc_x" "$WORK/glibc"
  for deb in "${DEBS[@]}"; do dpkg-deb -x "$DL/$(basename "$deb")" "$WORK/glibc_x"; done
  cp -L "$WORK/glibc_x/lib/ld-linux-armhf.so.3" "$WORK"/glibc_x/lib/arm-linux-gnueabihf/{libc.so.6,libm.so.6,libgcc_s.so.1} \
    "$WORK/glibc_x/usr/lib/arm-linux-gnueabihf/libstdc++.so.6" "$WORK/glibc/"
}

build() {
  if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    $CAP scripts/build_android.sh
    $CAP env ANDROID_ABI=armeabi-v7a scripts/build_android.sh "$ROOT/build/android-armv7"
  fi
  local T="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
  mkdir -p build/h2h
  "$T/aarch64-linux-android30-clang" -O2 -Wall -Wextra -o build/h2h/h2h_run_arm64 tools/h2h_run.c
  "$T/armv7a-linux-androideabi30-clang" -O2 -Wall -Wextra -o build/h2h/h2h_run_armv7 tools/h2h_run.c
}

models() {
  $CAP env TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES= "$LCE_PY" tools/h2h_models.py build \
    --out "$MDIR" --models "$MODELS"
}

run() {
  for p in $PHONES; do
    IFS=: read -r serial tag cpu abi <<< "$p"
    rts="bitino_all,bitino_neon,lce,tflite_fp32,tflite_int8"
    python3 tools/h2h_bench.py run --serial "$serial" --tag "$tag" --cpu "$cpu" --abi "$abi" --models-dir "$MDIR" \
      --bins-dir "$DL" --glibc-dir "$WORK/glibc" --models "$MODELS" --runtimes "$rts" --repeats "${REPEATS:-5}" --warmup "${WARMUP:-10}" \
      --iterations "${ITERATIONS:-50}" --throttle-max 600 --out "$OUT/$tag" 2>&1 | tee -a "$OUT/$tag.log"
    # second pass: re-measure the rounds that were throttled (the old records move to raw/throttled/)
    python3 tools/h2h_bench.py run --serial "$serial" --tag "$tag" --cpu "$cpu" --abi "$abi" --models-dir "$MDIR" \
      --bins-dir "$DL" --glibc-dir "$WORK/glibc" --models "$MODELS" --runtimes "$rts" --repeats "${REPEATS:-5}" \
      --warmup "${WARMUP:-10}" --iterations "${ITERATIONS:-50}" --throttle-max 600 --redo-throttled \
      --out "$OUT/$tag" 2>&1 | tee -a "$OUT/$tag.log"
  done
}

report() {
  python3 tools/h2h_bench.py report --in "$OUT" --models-dir "$MDIR" --md "$OUT/summary.md"
  mkdir -p "$OUT/models"
  for m in ${MODELS//,/ }; do [[ -f "$MDIR/$m/build_report.json" ]] && cp "$MDIR/$m/build_report.json" "$OUT/models/$m.json"; done
}

mkdir -p "$OUT"
case "$STEP" in
  download) download ;; build) build ;; models) models ;; run) run ;; report) report ;;
  all) download; build; models; run; report ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
