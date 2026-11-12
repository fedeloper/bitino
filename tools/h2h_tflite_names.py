#!/usr/bin/env python3
"""Renames the graph input tensors of a .tflite in place so the benchmark tools can address them.

LCE 0.16.0's lce_benchmark_model splits --input_layer_value_files items on ':' and rejects the converter's
default names ("serving_default_input_1:0"). This replaces every ':' in the graph input tensor names by
'_' byte for byte (same length, so no flatbuffer offset moves; each name must occur exactly once in the file).
Nothing else changes: the same graph, weights and kernels run. Prints the old -> new names and both SHA-256.

  python3 tools/h2h_tflite_names.py model.tflite [...]      (no TensorFlow needed)
"""
from __future__ import annotations

import hashlib
import re
import sys


def fix(path: str) -> dict:
    buf = bytearray(open(path, "rb").read())
    before = hashlib.sha256(buf).hexdigest()
    names = sorted(set(re.findall(rb"serving_default_[A-Za-z0-9_]+:\d+", bytes(buf))))
    done = []
    for n in names:
        if buf.count(n) != 1:
            raise RuntimeError(f"{path}: name {n!r} occurs {buf.count(n)} times; not renaming")
        i = buf.find(n)
        new = n.replace(b":", b"_")
        buf[i:i + len(n)] = new
        done.append((n.decode(), new.decode()))
    open(path, "wb").write(buf)
    return {"file": path, "renamed": done, "sha256_converter": before, "sha256": hashlib.sha256(buf).hexdigest()}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(fix(p))
