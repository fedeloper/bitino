#!/usr/bin/env bash
# Builds qemu-aarch64 (user mode) with TCG plugin support plus tools/qemu_plugins/bf_symcount.so,
# used by tools/insn_count.py for exact guest instruction counts. The distro qemu-user package is
# built without plugins, so QEMU is built from the signed release tarball.
#
#   scripts/build_qemu_plugins.sh            # -> $QEMU_PLUGIN_ROOT/{qemu-aarch64,libbf_symcount.so}
#
# Needs glib >= 2.66 headers. Without root, an isolated conda env provides them:
#   conda create -n qemu-build -c conda-forge glib meson ninja pkg-config python=3.12
# QEMU's configure rejects paths with spaces, so the build lives in $QEMU_PLUGIN_ROOT
# (default ~/.cache/bitino-qemu), not inside this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${QEMU_VERSION:-10.2.1}"
OUT="${QEMU_PLUGIN_ROOT:-$HOME/.cache/bitino-qemu}"
ENV="${QEMU_BUILD_ENV:-$HOME/anaconda3/envs/qemu-build}"
export PATH="$ENV/bin:$PATH" PKG_CONFIG_PATH="$ENV/lib/pkgconfig"
mkdir -p "$OUT"
cd "$OUT"
if [[ ! -d "qemu-$VER" ]]; then
  curl -sSfLO "https://download.qemu.org/qemu-$VER.tar.xz"
  curl -sSfLO "https://download.qemu.org/qemu-$VER.tar.xz.sig"
  # Release signing key (Michael Roth), fingerprint CEAC C9E1 5534 EBAB B82D 3FA0 3353 C9CE F108 B584.
  gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys CEACC9E15534EBABB82D3FA03353C9CEF108B584 >/dev/null 2>&1 || true
  gpg --verify "qemu-$VER.tar.xz.sig" "qemu-$VER.tar.xz"
  tar xf "qemu-$VER.tar.xz"
fi
mkdir -p "qemu-$VER/build"
cd "qemu-$VER/build"
[[ -f build.ninja ]] || ../configure --target-list=aarch64-linux-user --enable-plugins --disable-docs --disable-werror \
    --python="$ENV/bin/python3" --extra-ldflags="-L$ENV/lib -Wl,-rpath-link,$ENV/lib -Wl,-rpath,$ENV/lib"
ninja qemu-aarch64
cp qemu-aarch64 "$OUT/qemu-aarch64"
gcc -O2 -shared -fPIC -Wall -I"../include/qemu" $(pkg-config --cflags glib-2.0) \
    "$ROOT/tools/qemu_plugins/bf_symcount.c" -o "$OUT/libbf_symcount.so" $(pkg-config --libs glib-2.0)
echo "built $OUT/qemu-aarch64 and $OUT/libbf_symcount.so"
