""".bitino serialization (must stay byte-compatible with runtime/src/model.cc)."""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from .ir import Constant, Graph, Node

MAGIC = b"BFMODEL1"
FORMAT_VERSION = 1
ALIGN = 64


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def serialize(graph: Graph) -> bytes:
    data = bytearray()
    tensors = []
    for c in graph.constants.values():
        raw = np.ascontiguousarray(c.data).tobytes()
        off = _align(len(data))
        data.extend(b"\0" * (off - len(data)))
        data.extend(raw)
        tensors.append({"name": c.name, "dtype": c.dtype, "shape": [int(s) for s in c.shape],
                        "offset": off, "nbytes": len(raw), "layout": c.layout})
    data.extend(b"\0" * (_align(len(data)) - len(data)))
    doc = {
        "format_version": FORMAT_VERSION,
        "manifest": graph.manifest,
        "data_sha256": hashlib.sha256(bytes(data)).hexdigest(),
        "tensors": tensors,
        "inputs": graph.inputs,
        "nodes": [{"op": n.op, "name": n.name, "inputs": [i if i else None for i in n.inputs],
                   "outputs": n.outputs, "attrs": n.attrs} for n in graph.nodes],
        "outputs": graph.outputs,
    }
    js = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    json_off = 64
    data_off = _align(json_off + len(js))
    header = MAGIC + struct.pack("<IIQQQQ", FORMAT_VERSION, 64, json_off, len(js), data_off, len(data))
    header += b"\0" * (64 - len(header))
    out = bytearray(header)
    out.extend(js)
    out.extend(b"\0" * (data_off - len(out)))
    out.extend(data)
    return bytes(out)


def write(graph: Graph, path: str) -> str:
    """Writes the model and returns the SHA-256 of the whole file."""
    blob = serialize(graph)
    with open(path, "wb") as f:
        f.write(blob)
    return hashlib.sha256(blob).hexdigest()


def read(path: str, verify: bool = True) -> Graph:
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != MAGIC:
        raise ValueError(f"{path}: not a .bitino file")
    version, _hdr, json_off, json_size, data_off, data_size = struct.unpack("<IIQQQQ", blob[8:48])
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported format version {version}")
    doc = json.loads(blob[json_off:json_off + json_size])
    data = blob[data_off:data_off + data_size]
    if verify and hashlib.sha256(data).hexdigest() != doc["data_sha256"]:
        raise ValueError(f"{path}: data SHA-256 mismatch")
    g = Graph()
    g.manifest = doc.get("manifest", {})
    for t in doc["tensors"]:
        raw = data[t["offset"]:t["offset"] + t["nbytes"]]
        if t["dtype"] == "bit":
            shape = tuple(t["shape"])
            words = (shape[-1] + 31) // 32
            arr = np.frombuffer(raw, np.uint32).reshape(shape[:-1] + (words,))
        else:
            arr = np.frombuffer(raw, {"f32": np.float32, "i32": np.int32, "i8": np.int8, "u8": np.uint8}[t["dtype"]]).reshape(t["shape"])
        g.constants[t["name"]] = Constant(t["name"], t["dtype"], tuple(t["shape"]), arr, t.get("layout", ""))
    g.inputs = doc["inputs"]
    g.nodes = [Node(n["op"], [i or "" for i in n["inputs"]], n["outputs"], n.get("attrs", {}), n["name"]) for n in doc["nodes"]]
    g.outputs = doc["outputs"]
    return g
