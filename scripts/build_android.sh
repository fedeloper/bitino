#!/usr/bin/env bash
# Cross-compiles Bitino for Android with the NDK (arm64-v8a; ANDROID_ABI=armeabi-v7a builds the 32-bit binaries,
# whose NEON kernels are the AArch32 builds of the AArch64 ones, see runtime/kernels/neon_a32.h).
# Usage: scripts/build_android.sh [build-dir]   (env: ANDROID_NDK, ANDROID_API, BUILD_TYPE, ANDROID_ABI)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NDK="${ANDROID_NDK:-$HOME/Android/Sdk/android-ndk-r27c}"
API="${ANDROID_API:-30}"
OUT="${1:-$ROOT/build/android-arm64}"
[[ -f "$NDK/build/cmake/android.toolchain.cmake" ]] || { echo "NDK not found at $NDK (set ANDROID_NDK)" >&2; exit 1; }
cmake -S "$ROOT" -B "$OUT" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI:-arm64-v8a}" -DANDROID_PLATFORM="android-$API" -DANDROID_STL=c++_static \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" -DBITINO_STATIC="${BITINO_STATIC:-OFF}" \
  -DBITINO_BUILD_TESTS=ON
cmake --build "$OUT"
echo "built: $OUT/bitino_bench $OUT/bitino_tests"
