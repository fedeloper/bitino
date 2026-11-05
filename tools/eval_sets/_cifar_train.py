"""CIFAR-10 / CIFAR-100 TRAIN-split calibration sets for the CIFAR eval-set scripts (--split train).

The eval-set scripts (cifar10_normalized.py / cifar10_darts.py, cifar10_brevitas.py, finn_bnn_pynq.py cifar10,
cifar100_normalized.py, cifar100_deit.py) keep their test-split code path and preprocessing; with --split train
only the source rows change:

  selection: the first --n rows of the training set in torchvision order (CIFAR10(train=True) =
             data_batch_1..5 concatenated; CIFAR100(train=True) = cifar-100-python/train), skipping any row whose
             pixel SHA-256 (sha256 of the uint8 CHW bytes, the eval manifests' image_sha256 convention) occurs in a
             --disjoint-from eval manifest. Source files are md5-checked against torchvision's values and read with
             the restricted unpickler of _brevitas_common; CIFAR-100 `train` may be read straight from
             cifar-100-python.tar.gz (in memory, nothing extracted).
  manifest:  split "train", purpose "calibration", a "disjointness" record; --add-to-manifest adds the
             preprocessing entries to the manifest already in --out (same samples) so that every preprocessing
             of one calibration set lives in one directory.

Without --split train the scripts' output is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import tarfile

import numpy as np

from _brevitas_common import safe_unpickle, sha256_bytes

C10_TRAIN_MD5 = {  # torchvision.datasets.CIFAR10.train_list
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
}
C100_TRAIN_MD5 = "16019d7e3df5f24257cddd939b257f8d"  # torchvision.datasets.CIFAR100.train_list
C100_TGZ_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def add_args(ap) -> None:
    ap.add_argument("--split", choices=["test", "train"], default="test",
                    help="train: calibration set from the training split (needs --train-dir, --disjoint-from)")
    ap.add_argument("--train-dir", help="(--split train) directory with cifar-10-batches-py/data_batch_* or "
                                        "cifar-100-python/train (or cifar-100-python.tar.gz)")
    ap.add_argument("--disjoint-from", action="append", default=[], metavar="MANIFEST",
                    help="(--split train) eval manifest whose images must not be selected (repeatable)")
    ap.add_argument("--add-to-manifest", action="store_true",
                    help="(--split train) add the preprocessing entries to the existing manifest in --out")


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _load(dataset: str, train_dir: str):
    """All training rows (NCHW uint8, labels, row names, source_files) in torchvision order."""
    if dataset == "cifar10":
        datas, labels, names, src = [], [], [], {}
        for b, want in C10_TRAIN_MD5.items():
            raw = open(os.path.join(train_dir, "cifar-10-batches-py", b), "rb").read()
            if _md5(raw) != want:
                raise SystemExit(f"ERROR: {b} md5 {_md5(raw)} != torchvision's {want}")
            d = safe_unpickle(raw)
            x = np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32)
            datas.append(x)
            labels += [int(v) for v in d[b"labels"]]
            names += [f"{b}[{j}]" for j in range(len(x))]
            src[b] = {"md5": want, "sha256": sha256_bytes(raw)}
        return np.concatenate(datas), labels, names, src
    path = os.path.join(train_dir, "cifar-100-python", "train")
    if os.path.exists(path):
        raw = open(path, "rb").read()
        src = {}
    else:
        tgz = os.path.join(train_dir, "cifar-100-python.tar.gz")
        tb = open(tgz, "rb").read()
        if _md5(tb) != C100_TGZ_MD5:
            raise SystemExit(f"ERROR: {tgz} md5 {_md5(tb)} != {C100_TGZ_MD5}")
        with tarfile.open(tgz, "r:gz") as t:
            raw = t.extractfile("cifar-100-python/train").read()
        src = {"cifar-100-python.tar.gz": {"md5": C100_TGZ_MD5, "member": "cifar-100-python/train"}}
    if _md5(raw) != C100_TRAIN_MD5:
        raise SystemExit(f"ERROR: CIFAR-100 train md5 {_md5(raw)} != torchvision's {C100_TRAIN_MD5}")
    src["train"] = {"md5": C100_TRAIN_MD5, "sha256": sha256_bytes(raw)}
    d = safe_unpickle(raw)
    x = np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32)
    return x, [int(v) for v in d[b"fine_labels"]], [f"train[{i}]" for i in range(len(x))], src


def load_selection(dataset: str, args):
    """Returns (data NCHW uint8 [n], labels, names, calib manifest fields) for --split train."""
    if not args.train_dir or not args.disjoint_from:
        raise SystemExit("ERROR: --split train needs --train-dir and at least one --disjoint-from")
    data, labels, names, src = _load(dataset, args.train_dir)
    evals = []
    for mp in args.disjoint_from:
        ev = json.load(open(mp))["samples"]
        evals.append((mp, {s["image_sha256"] for s in ev}, len(ev)))
    banned = set().union(*(e[1] for e in evals))
    idx, skipped = [], []
    for i in range(len(data)):
        if len(idx) == args.n:
            break
        (skipped if sha256_bytes(data[i].tobytes()) in banned else idx).append(i)
    if len(idx) < args.n:
        raise SystemExit(f"ERROR: only {len(idx)} disjoint training rows")
    sel = np.ascontiguousarray(data[idx])
    shas = [sha256_bytes(x.tobytes()) for x in sel]
    checked = []
    for mp, hs, cnt in evals:
        ov = sum(h in hs for h in shas)
        assert ov == 0, (mp, ov)
        checked.append({"manifest": os.path.relpath(os.path.abspath(mp), os.getcwd()),
                        "manifest_sha256": hashlib.sha256(open(mp, "rb").read()).hexdigest(),
                        "eval_samples": cnt, "overlap_image_sha256": ov})
    tv = "CIFAR10(train=True) order = data_batch_1..5" if dataset == "cifar10" else "CIFAR100(train=True) order, fine labels"
    title = "CIFAR-10" if dataset == "cifar10" else "CIFAR-100"
    calib = {
        "name": f"{dataset}-train-calib-{args.n}",
        "split": "train",
        "purpose": "calibration",
        "source": f"{title} python version (https://www.cs.toronto.edu/~kriz/cifar.html), training split; images not "
                  "redistributed. Terms: no formal licence; cite Krizhevsky 2009 (TR, U. Toronto).",
        "source_files": src,
        "selection_rule": f"first {args.n} training rows in torchvision {tv}, skipping rows whose pixel SHA-256 occurs "
                          "in a disjointness manifest",
        "disjointness": {"rule": "no selected image's pixel SHA-256 (uint8 CHW) occurs in any listed eval manifest",
                         "train_rows_scanned": len(idx) + len(skipped), "train_rows_skipped_overlap": len(skipped),
                         "skipped_rows": [names[i] for i in skipped], "unique_image_sha256": len(set(shas)),
                         "checked": checked, "passed": True},
        "limitations": f"Calibration set for static INT8 quantization (activation statistics), drawn from the {title} "
                       "TRAIN split and disjoint from the evaluation sets listed under disjointness; labels are recorded "
                       "for completeness only. Never report accuracy on it.",
    }
    return sel, [labels[i] for i in idx], [names[i] for i in idx], calib


def write_manifest(args, manifest: dict, calib: dict | None) -> None:
    """Default (calib None): plain json.dump, as the scripts always did. Train: calibration fields, optional merge."""
    path = os.path.join(args.out, "manifest.json")
    if calib is None:
        json.dump(manifest, open(path, "w"), indent=1)
        return
    manifest.update(calib)
    manifest.pop("derived_from", None)
    if args.add_to_manifest:
        old = json.load(open(path))
        key = lambda s: (s["file"], s["image_sha256"], s["label"])  # noqa: E731
        if old.get("split") != "train" or [key(s) for s in old["samples"]] != [key(s) for s in manifest["samples"]]:
            raise SystemExit(f"ERROR: {path} lists different samples; refusing to mix sets")
        old["preprocessing"].update(manifest["preprocessing"])
        manifest = old
    json.dump(manifest, open(path, "w"), indent=1)
