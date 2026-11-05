#!/usr/bin/env python3
"""Checks that AVX/AVX2/FMA/AVX-VNNI instructions in an x86-64 Bitino binary come only from the
translation units built with -mavx2 (-mfma / -mavxvnni), so the default host binary runs on any x86-64.

The ISA-specific TUs (float_opt_avx2.cc, int8_gemm_avx2.cc, int8_gemm_avxvnni.cc, binary_avx2_simd.cc) contain only leaf routines
that the selector reaches after checking cpu_info(). An inline function with external linkage instantiated
in such a TU could be merged by the linker with the baseline copy and leak VEX-encoded instructions into
code every CPU runs; this tool would flag it.

  python3 tools/check_isa_isolation.py build/host/bitino_tests --objects build/host/CMakeFiles/bitino.dir

Every function of the binary is disassembled; a function is VEX code if any instruction uses a ymm
register or a VEX mnemonic (v...). Allowed: functions whose symbol is a local or strong global text symbol
of one of the ISA objects; weak / COMDAT symbols in those objects are errors by themselves. Exit code 1
lists the offending functions.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

ISA_SOURCES = ("float_opt_avx2.cc.o", "int8_gemm_avx2.cc.o", "int8_gemm_avxvnni.cc.o", "binary_avx2_simd.cc.o")
FUNC = re.compile(r"^[0-9a-f]+ <(.+)>:$")
VEX = re.compile(r"\t(v[a-z0-9]+)\b|%ymm")


def defined_symbols(obj: str, kinds: str) -> set[str]:
    out = subprocess.run(["nm", "--defined-only", obj], capture_output=True, text=True, check=True).stdout
    return {line.split()[-1] for line in out.splitlines() if len(line.split()) >= 3 and line.split()[1] in kinds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--objects", required=True, help="CMake object dir of the bitino library (searched recursively)")
    a = ap.parse_args()
    isa_objs = [p for p in glob.glob(os.path.join(a.objects, "**", "*.o"), recursive=True) if os.path.basename(p) in ISA_SOURCES]
    if len(isa_objs) != len(ISA_SOURCES):
        print(f"expected {len(ISA_SOURCES)} ISA objects, found {isa_objs}", file=sys.stderr)
        return 2
    allowed: set[str] = set()
    weak: set[str] = set()
    for o in isa_objs:
        allowed |= defined_symbols(o, "tT")   # local and strong global text symbols
        weak |= defined_symbols(o, "wWvVui")  # COMDAT / weak: shared with other TUs, must not exist
    for w in sorted(weak):
        print(f"  WEAK symbol in an ISA object (linker may pick its AVX copy): {w}")
    # Local (anonymous-namespace) symbols of the ISA objects keep their mangled names in the binary.
    dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", a.binary], capture_output=True, text=True, check=True).stdout
    func, vex_funcs, n_funcs = None, {}, 0
    for line in dis.splitlines():
        m = FUNC.match(line)
        if m:
            func = m.group(1)
            n_funcs += 1
            continue
        if func and VEX.search(line):
            vex_funcs.setdefault(func, line.strip())
    bad = {f: l for f, l in vex_funcs.items() if f.split("@")[0] not in allowed and not f.startswith(".")}
    print(f"{a.binary}: {n_funcs} functions, {len(vex_funcs)} contain VEX/AVX code, "
          f"{len(vex_funcs) - len(bad)} of them in the ISA translation units, {len(bad)} elsewhere")
    for f, l in sorted(bad.items()):
        demangled = subprocess.run(["c++filt", f], capture_output=True, text=True).stdout.strip()
        print(f"  LEAK {demangled}: {l}")
    return 1 if bad or weak else 0


if __name__ == "__main__":
    sys.exit(main())
