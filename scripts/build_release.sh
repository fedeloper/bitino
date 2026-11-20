#!/usr/bin/env bash
# Builds the release binaries of bitino_bench from this tree, with SHA-256 sums and build info.
#
#   scripts/build_release.sh [OUT_DIR]          # default dist/bin; TARGETS="linux-x86_64 linux-aarch64 android-arm64 android-armv7"
#   TARGETS=android-arm64 ANDROID_NDK=~/Android/Sdk/android-ndk-r27c scripts/build_release.sh
#
# linux-x86_64   static (glibc), portable x86-64 baseline; AVX2 / AVX-VNNI kernels are selected at run time.
# linux-aarch64  static. On an aarch64 host: native build with clang++ (glibc).
#                Else (fallback): a static Android NDK build (bionic libc, -static), which runs on any aarch64 Linux
#                kernel; BUILDINFO.txt records which way was used.
# android-arm64  NDK build for Android API 30+ (PIE, static libc++), plus an SDK archive (libbitino.a + headers).
# android-armv7  the same for 32-bit armeabi-v7a phones (NEON; the AArch32 builds of the kernels).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/dist/bin}" && cd "${1:-$ROOT/dist/bin}" && pwd)"
TARGETS="${TARGETS:-linux-x86_64 linux-aarch64 android-arm64 android-armv7}"
JOBS="${JOBS:-6}"
NDK="${ANDROID_NDK:-$HOME/Android/Sdk/android-ndk-r27c}"
B="$ROOT/build/release"
REV="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
INFO="$OUT/BUILDINFO.txt"
: > "$INFO.tmp"

configure_build() {   # <build dir> <cmake args...>
  local dir="$1"; shift
  rm -rf "$dir"   # clean configure, no stale cache
  mkdir -p "$dir"
  if ! { cmake -S "$ROOT" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBITINO_BUILD_TESTS=OFF "$@" &&
         cmake --build "$dir" --target bitino_bench -j "$JOBS"; } > "$dir.log" 2>&1; then
    tail -40 "$dir.log" >&2; echo "build failed: $dir (log $dir.log)" >&2; return 1
  fi
}

info() { echo "$*" >> "$INFO.tmp"; }

for t in $TARGETS; do
  case "$t" in
    linux-x86_64)
      [[ "$(uname -m)" == x86_64 ]] || { echo "skip $t: not an x86-64 host" >&2; continue; }
      configure_build "$B/$t" -DBITINO_STATIC=ON
      cp "$B/$t/bitino_bench" "$OUT/bitino_bench-$t"; strip "$OUT/bitino_bench-$t"
      info "$t: static, $(c++ --version | head -1), commit $REV"
      ;;
    linux-aarch64)
      if [[ "$(uname -m)" == aarch64 ]]; then
        # the NEON kernels' inline asm needs clang (GCC allows at most 30 asm operands)
        command -v clang++ >/dev/null || { echo "skip $t: clang++ required on aarch64 (apt install clang)" >&2; continue; }
        configure_build "$B/$t" -DCMAKE_CXX_COMPILER=clang++ -DBITINO_STATIC=ON
        cp "$B/$t/bitino_bench" "$OUT/bitino_bench-$t"; strip "$OUT/bitino_bench-$t"
        info "$t: static (glibc), native, $(clang++ --version | head -1), commit $REV"
      else
        [[ -f "$NDK/build/cmake/android.toolchain.cmake" ]] || { echo "skip $t: no aarch64 toolchain and no NDK" >&2; continue; }
        configure_build "$B/$t" -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
          -DANDROID_PLATFORM=android-30 -DANDROID_STL=c++_static -DBITINO_STATIC=ON
        cp "$B/$t/bitino_bench" "$OUT/bitino_bench-$t"
        "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip "$OUT/bitino_bench-$t"
        info "$t: static NDK build (bionic libc, -static; no aarch64 glibc toolchain on the build host). Runs on aarch64 Linux kernels; checked under qemu-aarch64 user mode. $(basename "$NDK"), commit $REV"
      fi
      ;;
    android-arm64|android-armv7)
      [[ -f "$NDK/build/cmake/android.toolchain.cmake" ]] || { echo "skip $t: NDK not found at $NDK" >&2; continue; }
      if [[ "$t" == android-arm64 ]]; then abi=arm64-v8a; else abi=armeabi-v7a; fi
      configure_build "$B/$t" -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=$abi \
        -DANDROID_PLATFORM=android-30 -DANDROID_STL=c++_static -DBITINO_BUILD_JNI=OFF
      cmake --build "$B/$t" --target bitino -j "$JOBS" >/dev/null
      cp "$B/$t/bitino_bench" "$OUT/bitino_bench-$t"
      "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip "$OUT/bitino_bench-$t"
      sdk="$B/$t/sdk/bitino-$t"; rm -rf "$sdk"; mkdir -p "$sdk/lib" "$sdk/include"
      cp "$B/$t/libbitino.a" "$sdk/lib/"; cp -r "$ROOT/runtime/include/bitino" "$sdk/include/"
      "$NDK"/toolchains/llvm/prebuilt/*/bin/llvm-strip --strip-debug "$sdk/lib/libbitino.a"   # no build paths in the archive
      cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$sdk/"
      tar -C "$B/$t/sdk" --owner=0 --group=0 --mtime='2026-01-01' -czf "$OUT/bitino-sdk-$t.tar.gz" "bitino-$t"
      info "$t: Android API 30+ ($abi), PIE, static libc++, $(basename "$NDK"), commit $REV; SDK: libbitino.a + headers"
      ;;
    *) echo "unknown target $t" >&2; exit 2 ;;
  esac
done
mv "$INFO.tmp" "$INFO"
(cd "$OUT" && rm -f SHA256SUMS && shopt -s nullglob && files=(bitino_bench-* bitino-sdk-*) && sha256sum "${files[@]}" > SHA256SUMS)
cat "$INFO"; cat "$OUT/SHA256SUMS"
