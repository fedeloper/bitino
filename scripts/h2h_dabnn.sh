#!/usr/bin/env bash
# Head-to-head latency benchmark: Bitino vs daBNN (JDAI-CV/dabnn, ACM MM 2019) on daBNN's own Bi-Real Net-18 models,
# 1 thread, batch 1, same core, same launcher. Method, flags, versions and caveats: docs/comparison-dabnn.md.
#
#   scripts/h2h_dabnn.sh [src|build|convert|run|eval|report|sens|all]  (default: all; sens is not part of all)
#
# env:
#   DABNN_WORK  work dir (default: .cache/dabnn-h2h): pinned submodule tarballs in dl-*/, source copy in src/,
#               build trees build-arm64/, build-armv7/, build-onnx2bnn/, converted models in dab/
#   PHONES      space-separated serial:tag:cpu:abi (default: OnePlus Open cpu7, Moto e13 cpu7, Redmi 9A cpu0)
#   REPEATS / WARMUP / ITERATIONS   rounds per model, untimed runs, timed runs (default 5 / 10 / 50)
#   OUT         results dir (default: reports/h2h_dabnn/<date>)
#   SKIP_BUILD=1  reuse build/android-arm64, build/android-armv7 (Bitino) and the daBNN builds
# daBNN is built out of tree from a copy of .cache/repos/dabnn (master d93aa95, read only) with its submodules
# at the commits pinned by that tree (GitHub tarballs, SHA-256 below). The only additions to the copy are
# tools/dabnn_h2h.cc (our timing/output driver, compiled as one more target of daBNN's binaries/ dir, linked
# to the unmodified libdabnn) and -Werror removed from daBNN's treat_warnings_as_errors and from google/benchmark's
# release flags (clang 18 warnings; no code-generation flag changes). Heavy host steps run under scripts/capped.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STEP="${1:-all}"
MAIN="${BITINO_MAIN:-~/BNN arm developments}"
WORK="${DABNN_WORK:-$MAIN/.cache/dabnn-h2h}"
UPSTREAM="$MAIN/.cache/repos/dabnn"
OUT="${OUT:-reports/h2h_dabnn/$(date +%Y-%m-%d)}"
PHONES="${PHONES:-e3698ea7:cph2551:7:arm64 ZY22HWKBP4:moto-e13:7:arm64 75AUS4CUDYN75LEQ:redmi9a:0:armv7}"
NDK="${ANDROID_NDK:-$HOME/Android/Sdk/android-ndk-r27c}"
API="${ANDROID_API:-30}"
CAP="scripts/capped.sh 8G nice -n 10"

# Submodule commits pinned by dabnn d93aa95 (git ls-tree HEAD third_party/), GitHub codeload tarballs.
declare -A PIN=(
  [benchmark]=f0901417c89d123474e6b91365029cfe32cf89dc   # google/benchmark
  [flatbuffers]=9e7e8cbe9f675123dd41b7c62868acad39188cae # google/flatbuffers
  [glog]=de82428e520f0a1a71089043140c4bfc3b44e554        # google/glog
  [onnx]=9f70c118910f7174f65dac84c90d46d608d9cfd2        # daquexian/onnx (onnx2bnn only)
  [protobuf]=6973c3a5041636c1d8dc5f7f6c8c1f3c15bc63d6    # protocolbuffers/protobuf (onnx2bnn only)
)
declare -A REPO=([benchmark]=google/benchmark [flatbuffers]=google/flatbuffers [glog]=google/glog
                 [onnx]=daquexian/onnx [protobuf]=protocolbuffers/protobuf)
declare -A SHA=(
  [benchmark]=9bf69a4a67fad34f910e948967ddea01a85155d6f8ee2bcb57d8b8f3e73723e9
  [flatbuffers]=4e506107bf2a3b2ad73fa12133e93ce07f8704233f7f2fb24ce179d77082bab3
  [glog]=8fb090dbb2327191966a1dd84ba22414c3f4ec67a260f73528e6766174d5ac1a
  [onnx]=73b920cc9b3b24a0f6310a7d175b8c7a455128114088c37e313e3709852c47cb
  [protobuf]=a16022446d4da4485f22210ef54cfe6142f880774be5827758f734122f535328
)

src() {
  for n in "${!PIN[@]}"; do
    d="$WORK/dl-$n"; f="$d/$n-${PIN[$n]}.tar.gz"
    if [[ ! -f "$f" ]]; then mkdir -p "$d"; curl -sSL --fail -o "$f" "https://codeload.github.com/${REPO[$n]}/tar.gz/${PIN[$n]}"; fi
    got=$(sha256sum "$f" | cut -d' ' -f1)
    [[ "$got" == "${SHA[$n]}" ]] || { echo "ERROR: $f sha256 $got != ${SHA[$n]}" >&2; exit 1; }
  done
  rm -rf "$WORK/src"; mkdir -p "$WORK/src"
  (cd "$UPSTREAM" && tar cf - --exclude=.git .) | tar xf - -C "$WORK/src"
  for n in "${!PIN[@]}"; do
    rm -rf "$WORK/src/third_party/$n"; mkdir -p "$WORK/src/third_party/$n"
    tar xzf "$WORK/dl-$n/$n-${PIN[$n]}.tar.gz" -C "$WORK/src/third_party/$n" --strip-components=1
  done
  sed -i 's/"-Wall" "-Wextra" "-Werror"/"-Wall" "-Wextra"/' "$WORK/src/cmake/common.cmake"
  # google/benchmark f0901417 adds -Werror to release builds; clang 18 flags an unused variable in complexity.cc
  sed -i '/add_cxx_compiler_flag(-Werror /d' "$WORK/src/third_party/benchmark/CMakeLists.txt"
  cp tools/dabnn_h2h.cc "$WORK/src/binaries/dabnn_h2h.cpp"
  printf '\nadd_executable(dabnn_h2h dabnn_h2h.cpp)\ntarget_link_libraries(dabnn_h2h dabnn)\n' >> "$WORK/src/binaries/CMakeLists.txt"
}

build_android() {  # $1 = ABI, $2 = build dir
  # glog: no execinfo backtrace() below API 33; static libunwind of the NDK's clang runtime (glog's find_library
  # otherwise picks the host libunwind.so)
  local arch; arch=$([[ "$1" == arm64-v8a ]] && echo aarch64 || echo arm)
  cmake -S "$WORK/src" -B "$2" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$1" -DANDROID_PLATFORM="android-$API" -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
    -DBNN_BUILD_TEST=OFF -DBNN_BUILD_BENCHMARK=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DHAVE_EXECINFO_H=OFF \
    -DUNWIND_LIBRARY="$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux/$arch/libunwind.a"
  cmake --build "$2" --target dabnn_h2h benchmark_single_model dabnn_benchmark
}

build() {
  if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    $CAP scripts/build_android.sh
    $CAP env ANDROID_ABI=armeabi-v7a scripts/build_android.sh "$ROOT/build/android-armv7"
    export -f build_android; export WORK NDK API
    $CAP bash -c 'build_android arm64-v8a "$WORK/build-arm64"'
    $CAP bash -c 'build_android armeabi-v7a "$WORK/build-armv7"'
    # onnx2bnn for the host with daBNN's packed protobuf (BNN_SYSTEM_PROTOBUF=OFF)
    # (host GCC 15: daBNN's common/StrKeyMap.h uses std::out_of_range without <stdexcept>, hence the forced include)
    cmake -S "$WORK/src" -B "$WORK/build-onnx2bnn" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBNN_SYSTEM_PROTOBUF=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPYTHON_EXECUTABLE=/usr/bin/python3 "-DCMAKE_CXX_FLAGS=-include stdexcept"
    scripts/capped.sh 12G nice -n 10 cmake --build "$WORK/build-onnx2bnn" --target onnx2bnn-bin -j 12
  fi
  local T="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
  mkdir -p build/h2h
  "$T/aarch64-linux-android30-clang" -O2 -Wall -Wextra -o build/h2h/h2h_run_arm64 tools/h2h_run.c
  "$T/armv7a-linux-androideabi30-clang" -O2 -Wall -Wextra -o build/h2h/h2h_run_armv7 tools/h2h_run.c
}

convert() {  # daBNN's documented conversion (ci/download_and_convert_models.sh): onnx2bnn --aggressive
  mkdir -p "$WORK/dab"
  "$WORK/build-onnx2bnn/tools/onnx2bnn/onnx2bnn" "$MAIN/.cache/checkpoints/dabnn-bireal18/birealnet18.onnx" \
    "$WORK/dab/dabnn-bireal18.dab" --aggressive --verbose
  "$WORK/build-onnx2bnn/tools/onnx2bnn/onnx2bnn" "$MAIN/.cache/checkpoints/dabnn-bireal18-stem/birealnet18stem.onnx" \
    "$WORK/dab/dabnn-bireal18-stem.dab" --aggressive --verbose
  sha256sum "$WORK"/dab/*.dab "$MAIN"/.cache/checkpoints/dabnn-bireal18*/*.dab
}

run() {
  for p in $PHONES; do
    IFS=: read -r serial tag cpu abi <<< "$p"
    for pass in "" --redo-throttled; do
      python3 tools/h2h_dabnn.py run --serial "$serial" --tag "$tag" --cpu "$cpu" --abi "$abi" --work "$WORK" \
        --repeats "${REPEATS:-5}" --warmup "${WARMUP:-10}" --iterations "${ITERATIONS:-50}" --throttle-max 600 $pass \
        --out "$OUT/$tag-cpu$cpu" 2>&1 | tee -a "$OUT/$tag-cpu$cpu.log"
    done
  done
}

eval_() {  # 128-image eval set on the device (correctness, untimed)
  for p in $PHONES; do
    IFS=: read -r serial tag cpu abi <<< "$p"
    python3 tools/h2h_dabnn.py eval --serial "$serial" --tag "$tag" --cpu "$cpu" --abi "$abi" --work "$WORK" --out "$OUT/eval-$tag"
  done
}

report() {
  python3 tools/h2h_dabnn.py report --in "$OUT" --work "$WORK" --md "$OUT/summary.md"
}

sens() {  # sensitivity check: daBNN (driver + libdabnn) rebuilt with Bitino's -march=armv8.2-a+dotprod, arm64 phones
  local arch=aarch64
  cmake -S "$WORK/src" -B "$WORK/build-arm64-v82" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM="android-$API" -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
    -DBNN_BUILD_TEST=OFF -DBNN_BUILD_BENCHMARK=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DHAVE_EXECINFO_H=OFF \
    -DUNWIND_LIBRARY="$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux/$arch/libunwind.a" \
    "-DCMAKE_CXX_FLAGS=-march=armv8.2-a+dotprod"
  $CAP cmake --build "$WORK/build-arm64-v82" --target dabnn_h2h
  mkdir -p "$OUT/sensitivity-march"
  for p in ${SENS_PHONES:-e3698ea7:cph2551:7:arm64 ZY22HWKBP4:moto-e13:7:arm64}; do
    IFS=: read -r serial tag cpu abi <<< "$p"
    python3 tools/h2h_dabnn.py run --serial "$serial" --tag "$tag" --cpu "$cpu" --abi "$abi" --work "$WORK" \
      --runtimes dabnn,dabnn_v82,bitino_all --repeats 3 --out "$OUT/sensitivity-march/$tag-cpu$cpu" 2>&1 \
      | tee -a "$OUT/sensitivity-march/$tag-cpu$cpu.log"
  done
  python3 tools/h2h_dabnn.py report --in "$OUT/sensitivity-march" --work "$WORK" --md "$OUT/sensitivity-march/summary.md"
}

mkdir -p "$OUT"
case "$STEP" in
  src) src ;; build) build ;; convert) convert ;; run) run ;; eval) eval_ ;; report) report ;; sens) sens ;;
  all) src; build; convert; run; eval_; report ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
