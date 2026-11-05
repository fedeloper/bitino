#!/usr/bin/env python3
"""Face-verification eval set from an insightface verification .bin (lfw.bin, agedb_30.bin, ...).

The .bin is a pickle of (bins, issame_list): bins = 2 * n_pairs JPEG/PNG byte strings (as numpy uint8 arrays),
issame_list = n_pairs booleans; image 2k and 2k+1 form pair k. It is read with a RESTRICTED unpickler that only
resolves numpy's array reconstruction globals (numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype,
numpy.core.multiarray.scalar); any other global raises, so no code from the file can run.

Preprocessing = insightface_larq_keras evals.py eval_callback (the authors' verification code):
  _imread = (tf.cast(tf.image.decode_image(xx, channels=3), "float32") - 127.5) * 0.0078125
and, for flip=True, tf.image.flip_left_right of the same tensor (embeddings of image and flip are summed).
Keys written: `insightface` and `insightface_flip` (float32 NHWC 112x112x3). Needs TensorFlow (Larq reference env):

  .cache/venv-larq/bin/python -I tools/eval_sets/face_verification_bin.py --bin <dir>/lfw.bin \
      --sha256 2cdf0242...  --name lfw --n 128 --out .cache/data/eval128_lfw

Images are NOT redistributed (LFW / AgeDB terms; research use); the .bin source is recorded in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle

import numpy as np

_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "scalar")}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")
        import numpy.core.multiarray as ma
        return {"_reconstruct": ma._reconstruct, "ndarray": np.ndarray, "dtype": np.dtype, "scalar": ma.scalar}[name]


def read_bin(path: str):
    with open(path, "rb") as f:
        obj = _RestrictedUnpickler(f, encoding="bytes").load()
    bins, issame = obj
    bins = [bytes(np.asarray(b, np.uint8).tobytes()) if not isinstance(b, bytes) else b for b in bins]
    issame = [bool(v) for v in issame]
    assert len(bins) == 2 * len(issame), (len(bins), len(issame))
    return bins, issame


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--name", required=True, help="e.g. lfw, agedb_30")
    ap.add_argument("--source-url", default="")
    ap.add_argument("--n", type=int, default=0, help="first n images (even; 0 = all)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    actual = sha256_file(args.bin)
    if actual != args.sha256:
        raise SystemExit(f"sha256 mismatch: {actual}")
    bins, issame = read_bin(args.bin)
    n = args.n or len(bins)
    assert n % 2 == 0 and n <= len(bins)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    os.makedirs(args.out, exist_ok=True)
    files = {"insightface": open(os.path.join(args.out, "insightface.f32"), "wb"),
             "insightface_flip": open(os.path.join(args.out, "insightface_flip.f32"), "wb")}
    hashes = {k: hashlib.sha256() for k in files}
    samples = []
    for i in range(n):
        img = tf.image.decode_image(bins[i], channels=3)
        assert tuple(img.shape) == (112, 112, 3), img.shape
        x = (tf.cast(img, "float32") - 127.5) * 0.0078125
        for k, t in (("insightface", x), ("insightface_flip", tf.image.flip_left_right(x))):
            b = np.ascontiguousarray(t.numpy(), np.float32).tobytes()
            files[k].write(b)
            hashes[k].update(b)
        samples.append({"file": f"{os.path.basename(args.bin)}[bins][{i}]",
                        "image_sha256": hashlib.sha256(bins[i]).hexdigest(),
                        "label": int(issame[i // 2]), "pair": i // 2})
    for f in files.values():
        f.close()
    tf_def = (f"tensorflow {tf.__version__}: (tf.cast(tf.image.decode_image(bytes, channels=3), 'float32') - 127.5) * 0.0078125, "
              "NHWC 112x112x3 (insightface_larq_keras evals.py eval_callback._imread)")
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"{args.name}-insightface-bin-{'first-' if n < len(bins) else ''}{n}",
        "task": "face-verification",
        "source": f"insightface verification set {os.path.basename(args.bin)} ({len(issame)} pairs, 112x112 aligned crops); "
                  f"downloaded from {args.source_url or 'see research notes'}; not redistributed (research use only).",
        "source_files": {os.path.basename(args.bin): {"sha256": actual, "bytes": os.path.getsize(args.bin)}},
        "selection_rule": f"first {n} images of the .bin (pairs 0..{n // 2 - 1}); image 2k and 2k+1 form pair k",
        "pairs": {"n_pairs": n // 2, "issame": [bool(v) for v in issame[: n // 2]]},
        "label_meaning": "label = issame of the image's pair (not a class; top-1 metrics on this set are meaningless)",
        "limitations": "embedding model eval set; per-image validation compares embeddings, verification accuracy needs all pairs",
        "samples": samples,
        "preprocessing": {k: {"shape": [n, 112, 112, 3], "dtype": "float32", "layout": "NHWC", "file": f"{k}.f32",
                              "sha256": hashes[k].hexdigest(),
                              "definition": tf_def + (" then tf.image.flip_left_right" if k.endswith("flip") else "")}
                          for k in files},
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{args.out}: {n} images, {n // 2} pairs ({sum(issame[: n // 2])} same)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
