#!/usr/bin/env python3
"""Builds the GLUE SST-2 dev evaluation tensors for BiBERT-style models.

Tokenization uses the BiBERT repository's own code path (quant_task_glue.py):
  Sst2Processor.get_dev_examples -> convert_examples_to_features(max_seq_length=64)
  with transformer.BertTokenizer.from_pretrained(<checkpoint dir>, do_lower_case=True)
(quant_task_glue.py:253-337 default_params / tokenizer). Run in an env where their
utils_glue imports (scipy, sklearn) work, e.g. ~/anaconda3/envs/bibert/bin/python.

  python tools/build_sst2_eval.py --zip .cache/data/sst2/download/SST-2.zip \
      --repo .cache/repos/BiBERT --ckpt-dir .cache/checkpoints/bibert-bert-base-sst2 \
      --out .cache/data/sst2/eval128 --n 128      (and --n 0 --out .../dev872 for the full set)

Writes input_ids.i32 / token_type_ids.i32 / attention_mask.f32 ([n, L] each), labels.i32,
inputs.bin (bitino_bench multi-input records: ids | token_type | mask per sample) and
manifest.json (bitino.eval_set.v1; sentence text is not stored, only its SHA-256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import types
import zipfile

import numpy as np

URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--repo", required=True, help="BiBERT repository checkout")
    ap.add_argument("--ckpt-dir", required=True, help="directory with vocab.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128, help="first n dev examples (0 = all 872)")
    ap.add_argument("--max-seq-length", type=int, default=64, help="quant_task_glue.py default_params['sst-2']")
    args = ap.parse_args()

    for name in ("matplotlib", "matplotlib.pyplot", "seaborn"):  # unused plotting imports in their modules
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, os.path.abspath(args.repo))
    from transformer import BertTokenizer  # noqa: E402
    from utils_glue import Sst2Processor, convert_examples_to_features  # noqa: E402

    zip_sha = sha256_file(args.zip)
    with zipfile.ZipFile(args.zip) as z:
        dev = z.read("SST-2/dev.tsv")
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "dev.tsv"), "wb") as f:
            f.write(dev)
        examples = Sst2Processor().get_dev_examples(td)
    total = len(examples)
    if args.n:
        examples = examples[: args.n]
    tok = BertTokenizer.from_pretrained(args.ckpt_dir, do_lower_case=True)
    feats = convert_examples_to_features(examples, Sst2Processor().get_labels(), args.max_seq_length, tok, "classification")
    n, L = len(feats), args.max_seq_length
    ids = np.array([f.input_ids for f in feats], np.int32)
    tt = np.array([f.segment_ids for f in feats], np.int32)
    mask = np.array([f.input_mask for f in feats], np.float32)
    labels = np.array([f.label_id for f in feats], np.int32)
    os.makedirs(args.out, exist_ok=True)
    ids.tofile(os.path.join(args.out, "input_ids.i32"))
    tt.tofile(os.path.join(args.out, "token_type_ids.i32"))
    mask.tofile(os.path.join(args.out, "attention_mask.f32"))
    labels.tofile(os.path.join(args.out, "labels.i32"))
    with open(os.path.join(args.out, "inputs.bin"), "wb") as f:
        for i in range(n):
            f.write(ids[i].tobytes() + tt[i].tobytes() + mask[i].tobytes())
    files = {k: sha256_file(os.path.join(args.out, k)) for k in
             ("input_ids.i32", "token_type_ids.i32", "attention_mask.f32", "labels.i32", "inputs.bin")}
    name = f"sst2-dev-{n}"
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": name,
        "source": {"url": URL, "zip_sha256": zip_sha, "member": "SST-2/dev.tsv", "dev_tsv_sha256": sha256_bytes(dev),
                   "dev_examples_total": total},
        "selection_rule": f"first {n} examples of GLUE SST-2 dev.tsv in file order" if args.n else "all dev examples",
        "tokenization": {"code": "BiBERT quant_task_glue.py path: utils_glue.Sst2Processor + convert_examples_to_features",
                         "tokenizer": "transformer.BertTokenizer.from_pretrained(ckpt_dir, do_lower_case=True)",
                         "vocab_sha256": sha256_file(os.path.join(args.ckpt_dir, "vocab.txt")),
                         "max_seq_length": L, "format": "[CLS] tokens [SEP] + zero padding; token_type 0; mask 1/0"},
        "preprocessing": {"bibert": {
            "shape": [n, L], "file": "inputs.bin", "sha256": files["inputs.bin"],
            "record": [{"name": "input_ids", "dtype": "i32", "shape": [1, L]},
                       {"name": "token_type_ids", "dtype": "i32", "shape": [1, L]},
                       {"name": "attention_mask", "dtype": "f32", "shape": [1, L]}],
            "definition": "per-sample record = input_ids i32[L] | token_type_ids i32[L] | attention_mask f32[L] (bitino_bench multi-input order)"}},
        "files": files,
        "limitations": ("Smoke/regression subset (first 128 dev sentences); accuracy on it is not the SST-2 dev accuracy. "
                        "The full 872-sentence dev accuracy is computed separately on the host.") if args.n else
                       "Full GLUE SST-2 dev set (872).",
        "label_counts": {str(k): int((labels == k).sum()) for k in (0, 1)},
        "samples": [{"guid": e.guid, "sentence_sha256": sha256_bytes(e.text_a.encode("utf-8")), "label": int(f.label_id),
                     "seq_length": int(f.seq_length)} for e, f in zip(examples, feats)],
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{name}: n={n} L={L} labels={manifest['label_counts']} truncated={int(sum(f.seq_length >= L for f in feats))} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
