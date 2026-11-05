#!/usr/bin/env python3
"""Steady-state emulated instructions per source line (qemu-aarch64 + tools/qemu_plugins/bf_pccount.c).

Same method as tools/insn_count.py ((warmup 1 + 2 iterations) - (warmup 1 + 1 iteration), 1 thread), but per
instruction address, mapped to function and source line with the NDK's llvm-addr2line (the static build carries
debug info). Instruction counts, not cycles.

  gcc -O2 -shared -fPIC -I<qemu>/include/qemu $(pkg-config --cflags glib-2.0) tools/qemu_plugins/bf_pccount.c \\
      -o ~/.cache/bitino-qemu/libbf_pccount.so $(pkg-config --libs glib-2.0)
  python3 tools/pc_profile.py --model artifacts/larq-zoo-quicknet-small/model_int8_full.bitino --stage int8_full
"""
from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QROOT = os.environ.get("QEMU_PLUGIN_ROOT", os.path.expanduser("~/.cache/bitino-qemu"))
NDK = os.environ.get("ANDROID_NDK", os.path.expanduser("~/Android/Sdk/android-ndk-r27c"))
A2L = os.path.join(NDK, "toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-addr2line")


def run(bench, model, stage, iters, cpu, plugin, tmp, extra):
    out = os.path.join(tmp, f"pcs_{iters}.tsv")
    cmd = [os.path.join(QROOT, "qemu-aarch64"), "-cpu", cpu, "-plugin", f"{plugin},out={out}", bench, "--model", model,
           "--stage", stage, "--threads", "1", "--warmup", "1", "--iterations", str(iters), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"bitino_bench failed: {r.stderr[-500:]}")
    c = collections.Counter()
    for line in open(out):
        a, n = line.split()
        c[int(a, 16)] += int(n)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", default="int8_full")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build/android-arm64-static/bitino_bench"))
    ap.add_argument("--cpu", default="max")
    ap.add_argument("--plugin", default=os.path.join(QROOT, "libbf_pccount.so"))
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--func", help="only functions whose name contains this")
    ap.add_argument("--by", choices=["line", "func", "pc"], default="line")
    ap.add_argument("--extra", nargs="*", default=[], help="extra bitino_bench arguments")
    ap.add_argument("--json", help="also write every aggregated key with its count to this file")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        c1 = run(a.bench, a.model, a.stage, 1, a.cpu, a.plugin, tmp, a.extra)
        c2 = run(a.bench, a.model, a.stage, 2, a.cpu, a.plugin, tmp, a.extra)
    d = {k: c2.get(k, 0) - c1.get(k, 0) for k in set(c1) | set(c2)}
    d = {k: v for k, v in d.items() if v > 0}
    addrs = sorted(d)
    r = subprocess.run([A2L, "-f", "-C", "-e", a.bench], input="\n".join(hex(x) for x in addrs), capture_output=True, text=True)
    lines = r.stdout.splitlines()
    func, loc = {}, {}
    for i, x in enumerate(addrs):
        func[x] = lines[2 * i][:140]
        l = lines[2 * i + 1].split(" (discriminator")[0]
        loc[x] = os.path.relpath(l, ROOT) if l.startswith("/") else l
    total = sum(d.values())
    agg = collections.Counter()
    for x, v in d.items():
        if a.func and a.func not in func[x]:
            continue
        key = {"line": (func[x], loc[x]), "func": (func[x], ""), "pc": (func[x], f"{loc[x]} {x:#x}")}[a.by]
        agg[key] += v
    sel = sum(agg.values())
    if a.json:
        import json
        json.dump({"model": a.model, "stage": a.stage, "total": total,
                   "rows": [[f, l, v] for (f, l), v in agg.most_common()]}, open(a.json, "w"))
    print(f"total {total / 1e6:.3f} M instructions/inference; selected {sel / 1e6:.3f} M")
    for (f, l), v in agg.most_common(a.top):
        print(f"{v / 1e6:9.3f} M {100 * v / total:5.1f}%  {l:48s} {f}")


if __name__ == "__main__":
    main()
