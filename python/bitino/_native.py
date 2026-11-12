"""ctypes binding of the runtime's C API (runtime/include/bitino/bitino_c.h) in libbitino_c.

The shared library is a plain C ABI with no Python dependency, so one wheel per platform serves every Python 3
version (wheel tag py3-none-<platform>). It is looked up, in order, at:
  1. $BITINO_LIB (a path to the library file);
  2. next to this module (wheel / regular install);
  3. bitino/<lib> in every sys.path entry (editable installs, where the compiled files live in site-packages);
  4. build/*/<lib> of a source checkout (cmake -DBITINO_BUILD_SHARED=ON).
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

C_API_VERSION = 2
MAX_RANK = 8

if sys.platform == "win32":
    LIB_NAME = "libbitino_c.dll"
elif sys.platform == "darwin":
    LIB_NAME = "libbitino_c.dylib"
else:
    LIB_NAME = "libbitino_c.so"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))


class NativeLibraryNotFound(ImportError):
    pass


class TensorInfo(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("dtype", ctypes.c_int32), ("rank", ctypes.c_int32),
                ("shape", ctypes.c_int64 * MAX_RANK), ("bytes", ctypes.c_uint64)]


def candidates() -> list[str]:
    out = []
    env = os.environ.get("BITINO_LIB")
    if env:
        out.append(env)
    out.append(os.path.join(_HERE, LIB_NAME))
    for p in sys.path:
        if p and os.path.isdir(p):
            out.append(os.path.join(p, "bitino", LIB_NAME))
    builds = sorted(glob.glob(os.path.join(_SRC_ROOT, "build", "*", LIB_NAME)), key=os.path.getmtime, reverse=True)
    out += builds
    seen, uniq = set(), []
    for c in out:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _declare(lib: ctypes.CDLL) -> None:
    c = ctypes
    vp, i, sz = c.c_void_p, c.c_int, c.c_size_t
    sigs = {
        "bf_version": (c.c_char_p, []),
        "bf_c_api_version": (i, []),
        "bf_cpu_features": (c.c_char_p, []),
        "bf_last_error": (c.c_char_p, []),
        "bf_model_load": (i, [c.c_char_p, i, c.POINTER(vp)]),
        "bf_model_free": (None, [vp]),
        "bf_model_manifest": (c.c_char_p, [vp]),
        "bf_session_create_ex": (i, [vp, i, c.c_char_p, c.POINTER(i), i, c.POINTER(vp)]),
        "bf_session_free": (None, [vp]),
        "bf_session_num_inputs": (i, [vp]),
        "bf_session_num_outputs": (i, [vp]),
        "bf_session_input_info": (i, [vp, i, c.POINTER(TensorInfo)]),
        "bf_session_output_info": (i, [vp, i, c.POINTER(TensorInfo)]),
        "bf_session_input_data": (vp, [vp, i]),
        "bf_session_output_data": (vp, [vp, i]),
        "bf_session_set_input": (i, [vp, i, vp, sz]),
        "bf_session_invoke": (i, [vp]),
        "bf_session_benchmark": (i, [vp, i, i, c.POINTER(c.c_double)]),
        "bf_session_describe": (c.c_char_p, [vp]),
    }
    for name, (res, args) in sigs.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args


_lib: ctypes.CDLL | None = None
_lib_path: str | None = None


def lib() -> ctypes.CDLL:
    """The loaded library (loaded on first use)."""
    global _lib, _lib_path
    if _lib is not None:
        return _lib
    tried = []
    for path in candidates():
        if not os.path.isfile(path):
            continue
        try:
            cand = ctypes.CDLL(path)
            ver = getattr(cand, "bf_c_api_version", None)
            if ver is None:
                tried.append(f"{path}: C API version 1 (rebuild it)")
                continue
            ver.restype = ctypes.c_int
            if ver() < C_API_VERSION:
                tried.append(f"{path}: C API version {ver()} < {C_API_VERSION} (rebuild it)")
                continue
            _declare(cand)
        except OSError as e:
            tried.append(f"{path}: {e}")
            continue
        _lib, _lib_path = cand, path
        return _lib
    raise NativeLibraryNotFound(
        f"{LIB_NAME} not found or not loadable. Install a binary wheel (pip install bitino), or build it from a "
        f"source checkout (cmake -S . -B build/host -DBITINO_BUILD_SHARED=ON && cmake --build build/host "
        f"--target bitino_c) or set BITINO_LIB." + ("".join("\n  " + t for t in tried) if tried else ""))


def lib_path() -> str | None:
    return _lib_path


def available() -> bool:
    try:
        lib()
        return True
    except NativeLibraryNotFound:
        return False


class NativeError(RuntimeError):
    pass


def check(rc: int) -> None:
    if rc != 0:
        raise NativeError(lib().bf_last_error().decode("utf-8", "replace"))
