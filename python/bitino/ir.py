"""Graph IR shared by adapters, the numpy executor and the .bitino writer.

Conventions (identical to the C++ runtime, see docs/model-format.md):
  * activations are NHWC float32;
  * conv filters are OHWI;
  * binary tensors (dtype "bit") are packed along the last dimension into
    little-endian uint32 words, element c -> word c // 32, bit c % 32;
    bit 1 encodes -1, bit 0 encodes +1, padding bits are 0.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

DTYPES = {"f32": np.float32, "i32": np.int32, "i8": np.int8, "u8": np.uint8}


def pack_bits(values: np.ndarray) -> np.ndarray:
    """Packs a {-1,+1}-valued array along its last axis. Returns uint32 [..., ceil(C/32)]."""
    v = np.asarray(values)
    if not np.all((v == 1) | (v == -1)):
        raise ValueError("pack_bits expects strictly binary {-1,+1} values")
    c = v.shape[-1]
    words = (c + 31) // 32
    neg = (v < 0).astype(np.uint64)
    pad = words * 32 - c
    if pad:
        neg = np.concatenate([neg, np.zeros(v.shape[:-1] + (pad,), np.uint64)], axis=-1)
    neg = neg.reshape(v.shape[:-1] + (words, 32))
    weights = (np.uint64(1) << np.arange(32, dtype=np.uint64))
    return (neg * weights).sum(axis=-1).astype(np.uint32)


def unpack_bits(words: np.ndarray, c: int) -> np.ndarray:
    """Inverse of pack_bits: returns int8 {-1,+1} with last dim c."""
    w = np.asarray(words, dtype=np.uint32)
    bits = (w[..., None] >> np.arange(32, dtype=np.uint32)) & 1
    bits = bits.reshape(w.shape[:-1] + (w.shape[-1] * 32,))[..., :c]
    return np.where(bits == 1, -1, 1).astype(np.int8)


@dataclasses.dataclass
class Constant:
    name: str
    dtype: str          # "f32" | "i32" | "i8" | "u8" | "bit"
    shape: tuple        # logical shape (for "bit": unpacked shape)
    data: np.ndarray    # f32/i32/...: array of `shape`; bit: uint32 packed words
    layout: str = ""


@dataclasses.dataclass
class Node:
    op: str
    inputs: list
    outputs: list
    attrs: dict
    name: str


class Graph:
    def __init__(self) -> None:
        self.constants: dict[str, Constant] = {}
        self.inputs: list[dict] = []
        self.nodes: list[Node] = []
        self.outputs: list[str] = []
        self.manifest: dict[str, Any] = {}
        self._counter = 0

    # ------------------------------------------------------------ building
    def unique(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def add_input(self, name: str, shape, dtype: str = "f32") -> str:
        self.inputs.append({"name": name, "dtype": dtype, "shape": [int(s) for s in shape]})
        return name

    def add_constant(self, name: str, array: np.ndarray, dtype: str = "f32", layout: str = "") -> str:
        if name in self.constants:
            raise ValueError(f"duplicate constant {name}")
        if dtype == "bit":
            packed = pack_bits(array)
            self.constants[name] = Constant(name, "bit", tuple(array.shape), packed, layout)
        else:
            arr = np.ascontiguousarray(np.asarray(array, dtype=DTYPES[dtype]))
            self.constants[name] = Constant(name, dtype, tuple(arr.shape), arr, layout)
        return name

    def add_node(self, op: str, inputs, attrs: dict | None = None, name: str | None = None, n_outputs: int = 1):
        name = name or self.unique(op.lower())
        outs = [f"{name}:out{i}" if n_outputs > 1 else f"{name}:out" for i in range(n_outputs)]
        self.nodes.append(Node(op, list(inputs), outs, dict(attrs or {}), name))
        return outs[0] if n_outputs == 1 else outs

    def set_outputs(self, *names: str) -> None:
        self.outputs = list(names)

    # ------------------------------------------------------------ queries
    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in self.nodes:
            counts[n.op] = counts.get(n.op, 0) + 1
        return counts

    def constant_bytes(self) -> int:
        return int(sum(c.data.nbytes for c in self.constants.values()))
