"""Bitino: binary neural networks on CPUs.

    import bitino
    m = bitino.load("ab-bnn-imagenet-reactnet18")   # zoo id or .bitino path
    y = m.predict(x)                                  # numpy in, numpy out
    m.benchmark(iterations=50, threads=1)             # median ms on this machine
    bitino.cpu_info()

The runtime API (load, Model, cpu_info, version) needs the native library shipped in the binary wheels (or built
from a source checkout, see bitino/_native.py); the zoo, conversion and validation tooling is pure Python.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["load", "Model", "TensorSpec", "cpu_info", "version", "STAGES", "__version__"]

_RUNTIME = {"load", "Model", "TensorSpec", "cpu_info", "version", "STAGES"}


def __getattr__(name: str):
    # Lazy: `bitino list` and the converters import this package without loading numpy or the native library.
    if name in _RUNTIME:
        from . import runtime
        return getattr(runtime, name)
    if name == "images":
        import importlib
        return importlib.import_module(".images", __name__)
    raise AttributeError(f"module 'bitino' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _RUNTIME | {"images"})
