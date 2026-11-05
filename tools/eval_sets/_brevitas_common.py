"""Shared helpers for the CIFAR-10 / MNIST evaluation subsets (Brevitas BNN-PYNQ models).

Only numpy and the standard library are used. The CIFAR-10 "python version" batches are
pickles; they are read with a restricted unpickler that only allows the numpy array
reconstruction globals, so no arbitrary code can run.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import pickle

import numpy as np

_ALLOWED_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("_codecs", "encode"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):  # noqa: D401
        if (module, name) not in _ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global {module}.{name} not allowed")
        return super().find_class(module, name)


def safe_unpickle(raw: bytes):
    return _SafeUnpickler(io.BytesIO(raw), encoding="bytes").load()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_idx(raw: bytes) -> np.ndarray:
    """IDX (MNIST) format: big-endian magic (0x00 0x00 dtype ndim), dims, data."""
    if raw[:2] != b"\x00\x00" or raw[2] != 0x08:
        raise ValueError("expected unsigned-byte IDX file")
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    data = np.frombuffer(raw, np.uint8, offset=4 + 4 * ndim)
    assert data.size == int(np.prod(dims)), "IDX size mismatch"
    return data.reshape(dims)


def read_maybe_gz(path: str) -> bytes:
    with open(path, "rb") as f:
        raw = f.read()
    return gzip.decompress(raw) if path.endswith(".gz") else raw


def to_tensor_f32(u8: np.ndarray) -> np.ndarray:
    """torchvision.transforms.ToTensor() value mapping: float32(k) / 255 (IEEE division)."""
    return u8.astype(np.float32) / np.float32(255.0)
