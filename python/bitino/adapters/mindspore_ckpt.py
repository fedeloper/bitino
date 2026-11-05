"""Safe reader for MindSpore ``.ckpt`` files (protobuf), without MindSpore or the protobuf package.

MindSpore writes ``mindspore/ccsrc/utils/checkpoint.proto``::

    message Checkpoint {
      message Value { required string tag = 1; required TensorProto tensor = 2; optional MapTensorProto maptensor = 3; }
      repeated Value value = 1;
    }
    message TensorProto { repeated int64 dims = 1; required string tensor_type = 2; required bytes tensor_content = 3; }

The reader decodes only the protobuf wire format (varints and length-delimited fields); it never
evaluates anything from the file. Tensor bytes are copied into numpy arrays of the declared dtype and
shape, and every size is checked against the declared shape. Unknown fields are rejected, so a file
with a different layout fails loudly instead of being misread.
"""
from __future__ import annotations

import numpy as np

_DTYPES = {"Float32": np.float32, "Float64": np.float64, "Float16": np.float16, "Int64": np.int64,
           "Int32": np.int32, "Int16": np.int16, "Int8": np.int8, "UInt8": np.uint8, "Bool": np.bool_}


def _varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift, val = 0, 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _fields(buf: bytes):
    """Yields (field_number, wire_type, value); value is an int (varint) or bytes (length-delimited)."""
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wt = key >> 3, key & 7
        if wt == 0:
            v, pos = _varint(buf, pos)
        elif wt == 2:
            n, pos = _varint(buf, pos)
            if pos + n > len(buf):
                raise ValueError("truncated length-delimited field")
            v, pos = buf[pos:pos + n], pos + n
        else:
            raise ValueError(f"unsupported wire type {wt} (field {field})")
        yield field, wt, v


def _tensor(buf: bytes) -> np.ndarray:
    dims: list[int] = []
    dtype, content = None, None
    for field, wt, v in _fields(buf):
        if field == 1:
            if wt == 0:
                dims.append(int(v))
            else:   # packed repeated int64
                p = 0
                while p < len(v):
                    d, p = _varint(v, p)
                    dims.append(int(d))
        elif field == 2 and wt == 2:
            dtype = v.decode("ascii")
        elif field == 3 and wt == 2:
            content = v
        else:
            raise ValueError(f"unexpected TensorProto field {field}")
    if dtype not in _DTYPES or content is None:
        raise ValueError(f"unsupported tensor type {dtype!r}")
    dt = np.dtype(_DTYPES[dtype]).newbyteorder("<")
    if dims == [0] and len(content) == dt.itemsize:   # MindSpore stores a scalar Parameter with dims [0]
        dims = []
    n = int(np.prod(dims)) if dims else 1
    if len(content) != n * dt.itemsize:
        raise ValueError(f"tensor content {len(content)} B does not match shape {dims} x {dt}")
    return np.frombuffer(content, dt).astype(dt.newbyteorder("=")).reshape(dims).copy()


def read(path: str) -> dict[str, np.ndarray]:
    """Returns {parameter name: numpy array} in file order."""
    with open(path, "rb") as f:
        data = f.read()
    out: dict[str, np.ndarray] = {}
    for field, wt, v in _fields(data):
        if field != 1 or wt != 2:
            raise ValueError(f"unexpected Checkpoint field {field}")
        tag, tensor = None, None
        for f2, wt2, v2 in _fields(v):
            if f2 == 1 and wt2 == 2:
                tag = v2.decode("utf-8")
            elif f2 == 2 and wt2 == 2:
                tensor = _tensor(v2)
            else:
                raise ValueError(f"unexpected Checkpoint.Value field {f2} (map tensors are not supported)")
        if tag is None or tensor is None:
            raise ValueError("Checkpoint.Value without tag or tensor")
        if tag in out:
            raise ValueError(f"duplicate parameter {tag}")
        out[tag] = tensor
    return out
