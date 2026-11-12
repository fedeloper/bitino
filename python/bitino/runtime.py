"""Python API of the native runtime (ctypes over libbitino_c, see _native.py).

    import bitino
    m = bitino.load("ab-bnn-imagenet-reactnet18")    # zoo id or path to a .bitino file
    y = m.predict(x)                                   # numpy in (model layout, NHWC for images), numpy out
    m.benchmark(iterations=50, threads=1)              # median latency in ms on this machine
    m.inputs, m.outputs, bitino.cpu_info()

Inputs are copied once into the session's own buffers and outputs once out of them (no other copies; an input that
is already a C-contiguous array of the model's dtype is not converted). One Model holds one native session; calls on
the same Model are serialised by a lock (create one Model per thread for parallel inference).
"""
from __future__ import annotations

import ctypes
import json
import os
import statistics
import threading
from dataclasses import dataclass

import numpy as np

from . import _native, zoo

# bf_tensor_info.dtype -> numpy dtype (BF_DTYPE_BIT tensors are exposed as their packed uint32 words)
_DTYPES = {0: np.dtype(np.float32), 1: np.dtype(np.int32), 2: np.dtype(np.int8), 3: np.dtype(np.uint8),
           4: np.dtype(np.uint32)}
STAGES = ("all", "optimized", "optimized_int8", "neon", "int8_neon", "binary_scalar", "reference", "packed_naive",
          "int8_reference", "int8_full", "int8_full_reference")


@dataclass(frozen=True)
class TensorSpec:
    """A graph input or output: name, shape (fixed; dim 0 is the batch the model was converted with), numpy dtype."""
    name: str
    shape: tuple
    dtype: np.dtype
    packed_bits: bool = False   # bit tensors: shape is the packed uint32 word layout

    def __repr__(self) -> str:
        return f"TensorSpec(name={self.name!r}, shape={self.shape}, dtype={self.dtype.name})"


def version() -> str:
    return _native.lib().bf_version().decode()


def cpu_info() -> dict:
    """What the runtime detected on this CPU (it picks kernels from this at run time)."""
    s = _native.lib().bf_cpu_features().decode()
    arch, _, feats = s.partition(":")
    return {"arch": arch, "features": feats.split(), "runtime_version": version(), "library": _native.lib_path()}


def _spec(info: _native.TensorInfo) -> TensorSpec:
    shape = tuple(int(info.shape[i]) for i in range(info.rank))
    bit = info.dtype == 4
    if bit and shape:
        shape = shape[:-1] + ((shape[-1] + 31) // 32,)
    return TensorSpec(info.name.decode(), shape, _DTYPES[info.dtype], bit)


class _Session:
    """One native session (bf_session) with numpy views on its input buffers."""

    def __init__(self, model_handle, threads: int, stage: str, affinity):
        lib = _native.lib()
        if int(threads) < 1:
            raise ValueError("threads must be >= 1")
        aff = list(affinity or [])
        arr = (ctypes.c_int * len(aff))(*aff) if aff else None
        h = ctypes.c_void_p()
        _native.check(lib.bf_session_create_ex(model_handle, int(threads), (stage or "all").encode(), arr, len(aff),
                                               ctypes.byref(h)))
        self.h, self.lib = h, lib
        self.threads, self.stage, self.affinity = int(threads), stage or "all", tuple(aff)
        info = _native.TensorInfo()
        self.inputs, self.outputs, self.in_views, self.out_bytes = [], [], [], []
        for i in range(lib.bf_session_num_inputs(h)):
            _native.check(lib.bf_session_input_info(h, i, ctypes.byref(info)))
            spec = _spec(info)
            self.inputs.append(spec)
            buf = (ctypes.c_char * int(info.bytes)).from_address(lib.bf_session_input_data(h, i))
            self.in_views.append(np.frombuffer(buf, dtype=spec.dtype).reshape(spec.shape))
        for i in range(lib.bf_session_num_outputs(h)):
            _native.check(lib.bf_session_output_info(h, i, ctypes.byref(info)))
            self.outputs.append(_spec(info))
            self.out_bytes.append(int(info.bytes))

    def output_view(self, i: int) -> np.ndarray:
        spec = self.outputs[i]
        ptr = self.lib.bf_session_output_data(self.h, i)
        buf = (ctypes.c_char * self.out_bytes[i]).from_address(ptr)
        return np.frombuffer(buf, dtype=spec.dtype).reshape(spec.shape)

    def invoke(self) -> None:
        _native.check(self.lib.bf_session_invoke(self.h))

    def fill_synthetic(self) -> None:
        """The deterministic timing input of bitino_bench: sin(0.37 k) for float inputs, zeros otherwise."""
        for v, spec in zip(self.in_views, self.inputs):
            if spec.dtype == np.float32:
                v.reshape(-1)[:] = np.sin(0.37 * np.arange(v.size, dtype=np.float64)).astype(np.float32)
            else:
                v[...] = 0

    def time(self, warmup: int, iterations: int) -> list[float]:
        out = (ctypes.c_double * iterations)()
        _native.check(self.lib.bf_session_benchmark(self.h, int(warmup), int(iterations), out))
        return list(out)

    def describe(self) -> dict:
        return json.loads(self.lib.bf_session_describe(self.h).decode())

    def free(self) -> None:
        if self.h:
            self.in_views = []
            self.lib.bf_session_free(self.h)
            self.h = None


class Model:
    """A loaded model with one inference session (kernels selected and weights prepacked for this CPU)."""

    def __init__(self, path: str, *, stage: str = "all", threads: int = 1, affinity=None, verify_hash: bool = True,
                 model_id: str | None = None, preprocess: str | None = None):
        lib = _native.lib()
        h = ctypes.c_void_p()
        _native.check(lib.bf_model_load(os.fsencode(os.path.abspath(path)), int(bool(verify_hash)), ctypes.byref(h)))
        self._model, self._lib = h, lib
        self._lock = threading.Lock()
        self._session = None
        try:
            self._session = _Session(h, threads, stage, affinity)
        except Exception:
            lib.bf_model_free(h)
            self._model = None
            raise
        self.path = os.path.abspath(path)
        self.model_id = model_id
        self.preprocess = preprocess
        self.manifest = json.loads(lib.bf_model_manifest(h).decode() or "{}")

    # ---------------------------------------------------------------------------------------------- properties
    @property
    def inputs(self) -> list[TensorSpec]:
        return list(self._sess().inputs)

    @property
    def outputs(self) -> list[TensorSpec]:
        return list(self._sess().outputs)

    @property
    def stage(self) -> str:
        return self._sess().stage

    @property
    def threads(self) -> int:
        return self._sess().threads

    def describe(self) -> dict:
        """Kernels chosen per op, plan rewrites, prepare time, memory: what the runtime does with this model here."""
        with self._lock:
            return self._sess().describe()

    def __repr__(self) -> str:
        name = self.model_id or os.path.basename(self.path)
        if self._session is None:
            return f"<bitino.Model {name} (closed)>"
        ins = ", ".join(f"{s.name} {list(s.shape)} {s.dtype.name}" for s in self._session.inputs)
        return f"<bitino.Model {name}: {ins}; stage {self.stage}, {self.threads} thread(s)>"

    # ---------------------------------------------------------------------------------------------- inference
    def _sess(self) -> _Session:
        if self._session is None:
            raise ValueError("model is closed")
        return self._session

    def _feeds(self, xs: tuple, named: dict) -> list[np.ndarray]:
        specs = self._sess().inputs
        if len(xs) == 1 and isinstance(xs[0], dict) and not named:
            named, xs = dict(xs[0]), ()
        if named:
            if xs:
                raise TypeError("pass inputs either positionally or by name, not both")
            unknown = set(named) - {s.name for s in specs}
            if unknown:
                raise KeyError(f"unknown input(s) {sorted(unknown)}; the model's inputs: {[s.name for s in specs]}")
            missing = [s.name for s in specs if s.name not in named]
            if missing:
                raise TypeError(f"missing input(s) {missing}")
            return [named[s.name] for s in specs]
        if len(xs) != len(specs):
            raise TypeError(f"the model has {len(specs)} input(s) ({[s.name for s in specs]}), got {len(xs)}")
        return list(xs)

    def predict(self, *xs, **named):
        """Runs the model. Inputs: one array per graph input (positionally, by name, or as a dict), each either the
        input's exact shape, the shape without the batch dimension, or N x the per-sample shape (N a multiple of the
        model's batch: run sample by sample). Arrays of another dtype are converted (float64 -> float32, ...).

        Returns the output array (one output) or a list of arrays, each with a leading dimension of N samples
        (times the output's own batch dimension)."""
        s = self._sess()
        feeds = self._feeds(xs, named)
        prepared, chunks = [], None
        for spec, x in zip(s.inputs, feeds):
            if spec.packed_bits:
                raise TypeError(f"input {spec.name} is a bit tensor; not supported by predict()")
            x = np.asarray(x)
            shp, b = spec.shape, spec.shape[0] if spec.shape else 1
            if x.shape == shp:
                k = 1
            elif len(shp) >= 1 and x.ndim == len(shp) - 1 and x.shape == shp[1:] and b == 1:
                x, k = x[None], 1
            elif len(shp) >= 1 and x.ndim == len(shp) and x.shape[1:] == shp[1:] and x.shape[0] % b == 0 and x.shape[0]:
                k = x.shape[0] // b
            else:
                raise ValueError(f"input {spec.name!r}: expected shape {shp}" +
                                 (f" or N x {shp[1:]}" if len(shp) > 1 else "") + f", got {x.shape}")
            if chunks is not None and k != chunks:
                raise ValueError(f"inputs have different numbers of samples ({chunks} vs {k})")
            chunks = k
            prepared.append(np.ascontiguousarray(x, dtype=spec.dtype))
        outs = [np.empty((chunks * (o.shape[0] if o.shape else 1),) + tuple(o.shape[1:]), dtype=o.dtype)
                for o in s.outputs]
        with self._lock:
            for j in range(chunks):
                for v, x, spec in zip(s.in_views, prepared, s.inputs):
                    b = spec.shape[0] if spec.shape else 1
                    np.copyto(v, x[j * b:(j + 1) * b].reshape(v.shape))
                s.invoke()
                for i, o in enumerate(s.outputs):
                    ob = o.shape[0] if o.shape else 1
                    outs[i][j * ob:(j + 1) * ob] = s.output_view(i).reshape((ob,) + tuple(o.shape[1:]))
        return outs[0] if len(outs) == 1 else outs

    __call__ = predict

    def predict_image(self, image, preprocess: str | None = None):
        """Decodes and preprocesses an image (path, file object or PIL image; needs Pillow) the way this zoo model
        was validated (self.preprocess, or `preprocess`: a key of bitino.images.SPECS), then predict()."""
        from . import images
        key = preprocess or self.preprocess
        if not key:
            raise ValueError("unknown preprocessing for this model: pass preprocess=<key> "
                             f"({', '.join(images.supported())}) or a preprocessed array to predict()")
        if hasattr(image, "convert") and hasattr(image, "size"):
            return self.predict(images.preprocess(image, key))
        return self.predict(images.load_image(image, key))

    def benchmark(self, iterations: int = 50, warmup: int = 10, threads: int | None = None, stage: str | None = None,
                  affinity=None, inputs=None, full: bool = False):
        """Latency of one inference on this machine: median over `iterations` timed runs (after `warmup`), in ms.

        Times the native run only (as bitino_bench does), on its deterministic synthetic input unless `inputs`
        (a tuple or dict as for predict(), one sample) is given. threads / stage / affinity other than the model's
        build a temporary session. full=True returns a dict with all statistics and the samples."""
        if iterations < 1 or warmup < 0:
            raise ValueError("iterations >= 1 and warmup >= 0")
        own = self._sess()
        want_threads = own.threads if threads is None else int(threads)
        want_stage = own.stage if stage is None else stage
        want_aff = own.affinity if affinity is None else tuple(affinity)
        with self._lock:
            tmp = None
            if (want_threads, want_stage, want_aff) != (own.threads, own.stage, own.affinity):
                tmp = _Session(self._model, want_threads, want_stage, want_aff)
            s = tmp or own
            try:
                if inputs is None:
                    s.fill_synthetic()
                else:
                    feeds = self._feeds(inputs if isinstance(inputs, tuple) else (inputs,), {})
                    for v, x in zip(s.in_views, feeds):
                        np.copyto(v, np.asarray(x, dtype=v.dtype).reshape(v.shape))
                samples = s.time(warmup, iterations)
                desc = s.describe() if full else None
            finally:
                if tmp is not None:
                    tmp.free()
        med = statistics.median(samples)
        if not full:
            return med
        srt = sorted(samples)
        return {"median_ms": med, "mean_ms": statistics.fmean(samples), "min_ms": srt[0], "max_ms": srt[-1],
                "p90_ms": srt[min(len(srt) - 1, int(round(0.9 * (len(srt) - 1))))], "samples_ms": samples,
                "iterations": iterations, "warmup": warmup, "threads": want_threads, "stage": want_stage,
                "affinity": list(want_aff), "cpu": cpu_info(), "session": desc,
                "note": "measured on this machine; not comparable with phone numbers unless this machine is that phone"}

    # ---------------------------------------------------------------------------------------------- lifetime
    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.free()
                self._session = None
            if self._model:
                self._lib.bf_model_free(self._model)
                self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def resolve(model: str, variant: str = "float", download: bool = True) -> tuple[str, str | None]:
    """(path, zoo id or None) for a model file path or a zoo id; see load()."""
    if os.path.isfile(model):
        return model, None
    if zoo.registry_entry(model) is None and model not in zoo.recipes():
        if model.endswith(zoo.MODEL_EXT) or os.sep in model or "/" in model:
            raise FileNotFoundError(f"{model}: no such file")
        raise KeyError(f"{model!r} is neither a file nor a zoo model id (see `bitino list --converted`)")
    p = zoo.fetched_file(model, variant)
    if os.path.isfile(p):
        return p, model
    ent, why = zoo.release_entry(model)
    if ent and download:
        from . import fetch
        fetch.fetch_prebuilt(model)
        if os.path.isfile(p):
            return p, model
        raise FileNotFoundError(f"{model}: the release has no {variant} file ({os.path.basename(p)})")
    if ent:
        raise FileNotFoundError(f"{model} is not downloaded yet ({p}); call with download=True or run "
                                f"`bitino fetch {model} --prebuilt`")
    if download and variant == "float" and zoo.source_checkout() and model in zoo.recipes():
        # Not redistributable: run its fetch recipe (original weights, SHA-256 check, converter in the recipe's
        # reference environment; FetchError says how to point at that environment if it is not configured).
        from . import fetch
        fetch.fetch(model)
        if os.path.isfile(p):
            return p, model
    raise PermissionError(
        f"{model}: converted files are redistributed only for models whose weights licence is explicitly permissive; "
        f"this one is not ({why}). Convert it from the authors' original checkpoint with its fetch recipe: "
        f"`bitino fetch {model}` in a source checkout ({zoo.REPO_URL}); bitino.load('{model}') then finds "
        f"{p}.")


def load(model: str, *, stage: str = "all", threads: int = 1, affinity=None, variant: str = "float",
         verify_hash: bool = True, download: bool = True) -> Model:
    """Loads a model and prepares an inference session.

    model:    path to a .bitino file, or a zoo model id. A zoo id is looked up in BITINO_HOME (default
              ~/.cache/bitino/models/<id>/, where `bitino fetch` puts converted files); if it is not there and
              its weights licence is explicitly permissive (the rule of `bitino list --release`), the converted
              file is downloaded from the GitHub release with its SHA-256 checked. Other models are converted from
              their original source by their fetch recipe: run here when this is a source checkout (as
              `bitino fetch <id>` does), otherwise PermissionError says how.
    download: False: never download or convert, only use files already in BITINO_HOME.
    stage:    kernel tiers: "all" (default: fastest exact kernels, float fusion), "neon" (bit-exact with the numpy
              executor: SIMD binary kernels, NEON on ARM / AVX2 on x86-64), "reference", "int8_full" ... (STAGES).
    threads:  worker threads of the session (>= 1).
    affinity: optional CPU ids; the calling thread is pinned to affinity[0] and worker i to affinity[i % n]
              (Linux only; the pin applies to the Python thread that creates and runs the session).
    variant:  zoo files: "float" (exact model), "int8" (model_int8_full), "int8_log2".
    """
    path, mid = resolve(model, variant, download)
    return Model(path, stage=stage, threads=threads, affinity=affinity, verify_hash=verify_hash, model_id=mid,
                 preprocess=zoo.preprocess_key(mid) if mid else None)
