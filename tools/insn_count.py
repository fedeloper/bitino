#!/usr/bin/env python3
"""Exact ARM instruction counts per inference (and per kernel) under qemu-aarch64.

QEMU has no timing model, so emulated *cycles* do not exist; what it can measure exactly is the
number of guest instructions executed. One steady-state inference is isolated by running
bitino_bench twice (warmup 1 + 1 iteration, warmup 1 + 2 iterations) and subtracting the
per-symbol counts, which cancels model loading, hashing, input generation and session setup.
Counts are deterministic at 1 thread and independent of host load. They are not latency:
instructions differ in cost (a NEON FMLA vs a scalar add), and cache misses and stalls are not
modelled. Combined with device latency they give IPC (instructions / (ms * GHz)).

  scripts/build_qemu_plugins.sh                       # once: plugin-enabled qemu + bf_symcount.so
  python3 tools/insn_count.py --models larq-zoo-quicknet-small --out reports/insn_counts
  python3 tools/insn_count.py --all --jobs 8 --out reports/insn_counts
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QROOT = os.environ.get("QEMU_PLUGIN_ROOT", os.path.expanduser("~/.cache/bitino-qemu"))
NDK = os.environ.get("ANDROID_NDK", os.path.expanduser("~/Android/Sdk/android-ndk-r27c"))
CXXFILT = os.path.join(NDK, "toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-cxxfilt")

# Helpers that are not methods of a kernel class, attributed to the kernel that calls them.
HELPER_OWNER = {"micro_kernel": "Conv2DGemm"}


def run_counts(bench: str, model: str, stage: str, iters: int, tmp: str, cpu: str) -> Counter:
    out = os.path.join(tmp, f"counts_{iters}.tsv")
    cmd = [os.path.join(QROOT, "qemu-aarch64"), "-cpu", cpu, "-plugin", f"{QROOT}/libbf_symcount.so,out={out}",
           bench, "--model", model, "--stage", stage, "--threads", "1", "--warmup", "1", "--iterations", str(iters)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"bitino_bench failed ({r.returncode}): {r.stderr[-500:]}")
    c = Counter()
    for line in open(out):
        n, name = line.rstrip("\n").split("\t", 1)
        c[name] += int(n)
    return c


def demangle(names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    r = subprocess.run([CXXFILT], input="\n".join(names), capture_output=True, text=True)
    out = r.stdout.splitlines() if r.returncode == 0 else names
    return dict(zip(names, out))


def group_of(d: str) -> str:
    m = re.search(r"(\w+)::run\(bitino::KernelContext", d)
    if m:
        return m.group(1)
    m = re.search(r"bitino::\(anonymous namespace\)::(\w+)::", d)
    if m:
        return m.group(1)
    m = re.search(r"bitino::(?:\(anonymous namespace\)::)?(\w+)", d)
    if m:
        return HELPER_OWNER.get(m.group(1), m.group(1))
    return "libc/other: " + re.split(r"[(<]", d)[0][:40]


def count_model(model_dir: str, stage: str, bench: str, cpu: str, model_file: str = "model.bitino") -> dict:
    model = os.path.join(model_dir, model_file)
    with tempfile.TemporaryDirectory() as tmp:
        c1 = run_counts(bench, model, stage, 1, tmp, cpu)
        c2 = run_counts(bench, model, stage, 2, tmp, cpu)
    diff = {k: c2.get(k, 0) - c1.get(k, 0) for k in set(c1) | set(c2) if k != "#total"}
    diff = {k: v for k, v in diff.items() if v}
    neg = sum(v for v in diff.values() if v < 0)
    dm = demangle(sorted(diff))
    groups: Counter = Counter()
    for k, v in diff.items():
        groups[group_of(dm.get(k, k))] += v
    total = c2["#total"] - c1["#total"]
    top_syms = sorted(diff.items(), key=lambda kv: -kv[1])[:15]
    return {
        "schema": "bitino.insn_count.v1",
        "model": os.path.basename(os.path.normpath(model_dir)),
        "stage": stage, "model_file": model_file, "threads": 1, "qemu_cpu": cpu,
        "instructions_per_inference": total,
        "nonsteady_residual": neg,  # negative per-symbol diffs (should be ~0: run-to-run differences)
        "per_kernel": dict(groups.most_common()),
        "top_symbols": [{"symbol": dm.get(k, k), "instructions": v} for k, v in top_syms],
        # every symbol with a nonzero steady-state count (for cross-model aggregation by kernel symbol)
        "symbols": {dm.get(k, k): v for k, v in sorted(diff.items(), key=lambda kv: -kv[1])},
        "method": "qemu-aarch64 TCG plugin bf_symcount; (warmup1+2 iters) - (warmup1+1 iter); exact guest "
                  "instruction counts, NOT cycles (QEMU has no timing model)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="every artifacts/*/<model-file>")
    ap.add_argument("--stage", default="optimized")
    ap.add_argument("--cpu", default="max", help="qemu -cpu model (max: NEON, dotprod, i8mm)")
    ap.add_argument("--bench", default=os.path.join(ROOT, "build/android-arm64-static/bitino_bench"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model-file", default="model.bitino",
                    help="model file inside artifacts/<model>/ (e.g. model_int8_full.bitino for stage int8_full)")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports/insn_counts"))
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"),
                    help="directory holding artifacts/<model>/ (read only)")
    a = ap.parse_args()
    models = list(a.models)
    if a.all:
        models += sorted(d for d in os.listdir(a.artifacts)
                         if os.path.isfile(os.path.join(a.artifacts, d, a.model_file)))
    if not models:
        ap.error("no models (--models ... or --all)")
    os.makedirs(a.out, exist_ok=True)
    results = {}
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(count_model, os.path.join(a.artifacts, m), a.stage, a.bench, a.cpu, a.model_file): m for m in models}
        for f in cf.as_completed(futs):
            m = futs[f]
            try:
                r = f.result()
            except Exception as e:  # keep going; report the failure
                print(f"{m}: FAILED {e}", file=sys.stderr)
                continue
            results[m] = r
            json.dump(r, open(os.path.join(a.out, f"{m.replace(os.sep, '__')}_{a.stage}.json"), "w"), indent=1)
            print(f"{m}: {r['instructions_per_inference'] / 1e6:.2f} M instructions/inference", flush=True)
    lines = [f"# ARM instruction counts per inference (qemu-aarch64, stage `{a.stage}`, 1 thread)", "",
             "Exact executed guest instructions for one steady-state inference. **Not cycles and not latency**: "
             "QEMU has no timing model. Divide by device cycles (latency x clock) for IPC.", "",
             "| model | M instructions | top kernels (share of instructions) |", "|---|---|---|"]
    for m in sorted(results, key=lambda m: results[m]["instructions_per_inference"]):
        r = results[m]
        t = r["instructions_per_inference"]
        top = ", ".join(f"{k} {100 * v / t:.0f}%" for k, v in list(r["per_kernel"].items())[:3])
        lines.append(f"| {m} | {t / 1e6:,.2f} | {top} |")
    md = os.path.join(a.out, f"summary_{a.stage}.md")
    if a.all or not os.path.exists(md):
        open(md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
