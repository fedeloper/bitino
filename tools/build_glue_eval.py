#!/usr/bin/env python3
"""Builds GLUE dev evaluation tensors (MRPC, MNLI-matched, QQP; SST-2 too) for BiBERT-style models.

Same tensors and manifest layout as tools/build_sst2_eval.py, generalised to sentence-pair tasks.
Tokenization uses the BiBERT repository's own code path (quant_task_glue.py):
  <Task>Processor.get_dev_examples -> convert_examples_to_features(max_seq_length=default_params[task])
  with transformer.BertTokenizer.from_pretrained(<dir with vocab.txt>, do_lower_case=True)
(quant_task_glue.py:230-337). Run in an env where their utils_glue imports work, e.g.
~/anaconda3/envs/bibert/bin/python.

Sources (downloaded to .cache/data/glue/download, never redistributed):
  MNLI   https://dl.fbaipublicfiles.com/glue/data/MNLI.zip       member MNLI/dev_matched.tsv
  QQP    https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip  member QQP/dev.tsv
  MRPC   GLUE's download_glue_data.py recipe: dev.tsv = header + the rows of SentEval's
         msr_paraphrase_train.txt whose (#1 ID, #2 ID) is listed in mrpc_dev_ids.tsv, in file order
         (https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv,
          https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt).
         Cross-checked sentence by sentence against the HF nyu-mll/glue mrpc validation parquet if given.

  python tools/build_glue_eval.py --task mrpc --download-dir .cache/data/glue/download \
      --repo .cache/repos/BiBERT --vocab-dir .cache/checkpoints/bibert-bert-base-mrpc \
      --out .cache/data/glue/mrpc/eval128 --n 128          (--n 0 --out .../dev408 for the full set)

Writes input_ids.i32 / token_type_ids.i32 / attention_mask.f32 ([n, L] each), labels.i32,
inputs.bin (bitino_bench multi-input records: ids | token_type | mask per sample) and
manifest.json (bitino.eval_set.v1; sentence text is not stored, only its SHA-256).
Preprocessing key: "bibert".

Calibration sets (--split train): the first --n examples of the task's GLUE TRAIN split in file order
(SST-2/train.tsv, MNLI/train.tsv, QQP/train.tsv; MRPC train.tsv rebuilt with the same format_mrpc
recipe = the msr_paraphrase_train.txt rows whose id pair is NOT a dev id pair), skipping any train
example whose text (text_a TAB text_b, exact or lower-cased/whitespace-normalised) occurs in the dev
split used for evaluation (for MNLI also dev_mismatched). Same tokenization and record layout as the
dev sets; the manifest adds split "train", purpose "calibration" and a "disjointness" record. Without
--split the output is unchanged.

  python tools/build_glue_eval.py --task mnli --split train --download-dir .cache/data/glue/download \
      --repo .cache/repos/BiBERT --vocab-dir .cache/checkpoints/bibert-bert-base-mnli \
      --out .cache/data/glue_calib512/mnli --n 512
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import zipfile

import numpy as np

TASKS = {
    # task: (processor class, max_seq_length (quant_task_glue.py default_params), zip, member, url)
    "sst-2": ("Sst2Processor", 64, "SST-2.zip", "SST-2/dev.tsv", "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"),
    "mnli": ("MnliProcessor", 128, "MNLI.zip", "MNLI/dev_matched.tsv", "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip"),
    "qqp": ("QqpProcessor", 128, "QQP-clean.zip", "QQP/dev.tsv", "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip"),
    "mrpc": ("MrpcProcessor", 128, None, None, "https://dl.fbaipublicfiles.com/glue/data/mrpc_dev_ids.tsv + "
             "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"),
}
DEV_FILE = {"mnli": "dev_matched.tsv"}
TRAIN_MEMBER = {"sst-2": "SST-2/train.tsv", "mnli": "MNLI/train.tsv", "qqp": "QQP/train.tsv"}
EXTRA_DEV_MEMBERS = {"mnli": ["MNLI/dev_mismatched.tsv"]}  # also kept out of the calibration set


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def mrpc_dev_tsv(download_dir: str) -> tuple[bytes, dict]:
    """download_glue_data.py format_mrpc(): header + train-file rows whose id pair is a dev id pair."""
    ids_path = os.path.join(download_dir, "mrpc_dev_ids.tsv")
    train_path = os.path.join(download_dir, "msr_paraphrase_train.txt")
    dev_ids = []
    with io.open(ids_path, encoding="utf-8") as fh:
        for row in fh:
            dev_ids.append(row.strip().split("\t"))
    out = io.StringIO()
    with io.open(train_path, encoding="utf-8") as fh:
        header = fh.readline()
        out.write(header)
        for row in fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] in dev_ids:
                out.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
    src = {"mrpc_dev_ids_sha256": sha256_file(ids_path), "msr_paraphrase_train_sha256": sha256_file(train_path),
           "recipe": "GLUE download_glue_data.py format_mrpc (dev split)"}
    return out.getvalue().encode("utf-8"), src


def mrpc_train_tsv(download_dir: str) -> tuple[bytes, dict]:
    """download_glue_data.py format_mrpc(): header + train-file rows whose id pair is NOT a dev id pair."""
    ids_path = os.path.join(download_dir, "mrpc_dev_ids.tsv")
    train_path = os.path.join(download_dir, "msr_paraphrase_train.txt")
    with io.open(ids_path, encoding="utf-8") as fh:
        dev_ids = [row.strip().split("\t") for row in fh]
    out = io.StringIO()
    with io.open(train_path, encoding="utf-8") as fh:
        out.write(fh.readline())
        for row in fh:
            label, id1, id2, s1, s2 = row.strip().split("\t")
            if [id1, id2] not in dev_ids:
                out.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
    src = {"mrpc_dev_ids_sha256": sha256_file(ids_path), "msr_paraphrase_train_sha256": sha256_file(train_path),
           "recipe": "GLUE download_glue_data.py format_mrpc (train split)"}
    return out.getvalue().encode("utf-8"), src


def text_key(e) -> str:
    return e.text_a + "\t" + (e.text_b or "")


def norm_key(e) -> str:
    return " ".join(text_key(e).lower().split())


def read_examples(processor, tsv: bytes, fname: str, train: bool) -> list:
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, fname), "wb") as f:
            f.write(tsv)
        return processor.get_train_examples(td) if train else processor.get_dev_examples(td)


def select_calibration(processor, task: str, args, dev_examples: list, dev_sha: str) -> tuple[list, dict, dict]:
    """First args.n train examples (file order) whose text is in none of the dev splits."""
    if task == "mrpc":
        train, tsrc = mrpc_train_tsv(args.download_dir)
    else:
        zp = os.path.join(args.download_dir, TASKS[task][2])
        with zipfile.ZipFile(zp) as z:
            train = z.read(TRAIN_MEMBER[task])
        tsrc = {"zip": TASKS[task][2], "zip_sha256": sha256_file(zp), "member": TRAIN_MEMBER[task]}
    tsrc["train_tsv_sha256"] = sha256_bytes(train)
    train_examples = read_examples(processor, train, "train.tsv", train=True)
    tsrc["train_examples_total"] = len(train_examples)
    devs = [{"split": TASKS[task][3] or "MRPC dev (format_mrpc)", "tsv_sha256": dev_sha, "examples": dev_examples}]
    if EXTRA_DEV_MEMBERS.get(task):
        zp = os.path.join(args.download_dir, TASKS[task][2])
        with zipfile.ZipFile(zp) as z:
            for m in EXTRA_DEV_MEMBERS[task]:
                raw = z.read(m)
                with tempfile.TemporaryDirectory() as td:  # same csv reader / example layout as the dev path
                    with open(os.path.join(td, "x.tsv"), "wb") as f:
                        f.write(raw)
                    ex = processor._create_examples(processor._read_tsv(os.path.join(td, "x.tsv")), "dev_extra")
                devs.append({"split": m, "tsv_sha256": sha256_bytes(raw), "examples": ex})
    exact = set().union(*({text_key(e) for e in d["examples"]} for d in devs))
    normed = set().union(*({norm_key(e) for e in d["examples"]} for d in devs))
    picked, skipped = [], []
    for e in train_examples:
        if len(picked) == args.n:
            break
        if text_key(e) in exact or norm_key(e) in normed:
            skipped.append(e.guid)
        else:
            picked.append(e)
    if len(picked) < args.n:
        raise SystemExit(f"ERROR: only {len(picked)} disjoint train examples")
    sents = lambda ex: {s for e in ex for s in (e.text_a, e.text_b) if s}  # noqa: E731
    checked = []
    for d in devs:  # re-verify the final selection against every dev split
        d_exact, d_norm = {text_key(x) for x in d["examples"]}, {norm_key(x) for x in d["examples"]}
        ov_exact = sum(text_key(e) in d_exact for e in picked)
        ov_norm = sum(norm_key(e) in d_norm for e in picked)
        assert ov_exact == 0 and ov_norm == 0, (d["split"], ov_exact, ov_norm)
        checked.append({"dev_split": d["split"], "dev_tsv_sha256": d["tsv_sha256"], "dev_examples": len(d["examples"]),
                        "overlap_exact_text": ov_exact, "overlap_normalised_text": ov_norm,
                        "info_single_sentence_overlap": len(sents(picked) & sents(d["examples"]))})
    disjoint = {"rule": "no selected example's text (text_a TAB text_b), exact or lower-cased/whitespace-normalised, "
                        "occurs in the listed dev splits; overlapping train rows are skipped",
                "train_rows_scanned": len(picked) + len(skipped), "train_rows_skipped_overlap": len(skipped),
                "skipped_guids": skipped, "checked": checked, "passed": True}
    return picked, tsrc, disjoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--download-dir", required=True)
    ap.add_argument("--repo", required=True, help="BiBERT repository checkout")
    ap.add_argument("--vocab-dir", required=True, help="directory with vocab.txt (BERT-base uncased)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128, help="first n dev examples (0 = all)")
    ap.add_argument("--split", choices=["dev", "train"], default="dev",
                    help="train: calibration set (first --n disjoint train examples); default dev (eval set)")
    ap.add_argument("--hf-parquet", help="(MRPC) HF nyu-mll/glue validation parquet to cross-check the rebuilt dev.tsv "
                                       "(read with pandas; if this env has none, a sibling <parquet>.json with the "
                                       "sentence1/sentence2/label columns, written by --export-parquet-json in an env with pandas or "
                                       "assembled from the datasets-server.huggingface.co /rows API of the same split)")
    ap.add_argument("--export-parquet-json", action="store_true", help="only write <hf-parquet>.json and exit")
    args = ap.parse_args()

    if args.export_parquet_json:
        import pandas as pd

        df = pd.read_parquet(args.hf_parquet)
        json.dump({c: df[c].tolist() for c in ("sentence1", "sentence2", "label")}, open(args.hf_parquet + ".json", "w"))
        return 0
    proc_name, L, zip_name, member, url = TASKS[args.task]
    for name in ("matplotlib", "matplotlib.pyplot", "seaborn"):  # unused plotting imports in their modules
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, os.path.abspath(args.repo))
    import utils_glue  # noqa: E402
    from transformer import BertTokenizer  # noqa: E402

    processor = getattr(utils_glue, proc_name)()
    if args.task == "mrpc":
        dev, src = mrpc_dev_tsv(args.download_dir)
    else:
        zp = os.path.join(args.download_dir, zip_name)
        with zipfile.ZipFile(zp) as z:
            dev = z.read(member)
        src = {"zip": zip_name, "zip_sha256": sha256_file(zp), "member": member}
    src.update({"url": url, "dev_tsv_sha256": sha256_bytes(dev)})
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, DEV_FILE.get(args.task, "dev.tsv")), "wb") as f:
            f.write(dev)
        examples = processor.get_dev_examples(td)
    total = len(examples)
    src["dev_examples_total"] = total

    if args.task == "mrpc" and args.hf_parquet:
        try:
            import pandas as pd

            df = pd.read_parquet(args.hf_parquet)
        except ImportError:
            df = json.load(open(args.hf_parquet + ".json"))
        assert len(df["label"]) == total, (len(df["label"]), total)
        ok = all(e.text_a == a and e.text_b == b and int(e.label) == int(lab)
                 for e, a, b, lab in zip(examples, df["sentence1"], df["sentence2"], df["label"]))
        src["hf_parquet_crosscheck"] = {"sha256": sha256_file(args.hf_parquet), "rows": int(len(df["label"])), "identical_text_and_labels": bool(ok)}
        assert ok, "rebuilt MRPC dev differs from the HF validation split"

    calib = args.split == "train"
    if calib:
        if not args.n:
            ap.error("--split train needs --n > 0")
        examples, train_src, disjoint = select_calibration(processor, args.task, args, examples, src["dev_tsv_sha256"])
    elif args.n:
        examples = examples[: args.n]
    tok = BertTokenizer.from_pretrained(args.vocab_dir, do_lower_case=True)
    label_list = processor.get_labels()
    feats = utils_glue.convert_examples_to_features(examples, label_list, L, tok, "classification")
    n = len(feats)
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
    split = "dev-matched" if args.task == "mnli" else "dev"
    name = f"{args.task.replace('-', '')}-{split}-{n}"
    pair = args.task != "sst-2"
    if calib:
        name = f"{args.task.replace('-', '')}-train-calib-{n}"
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": name,
        "task": args.task,
        "source": src,
        "selection_rule": f"first {n} examples of GLUE {args.task.upper()} {split} in file order" if args.n else f"all {split} examples",
        "tokenization": {"code": f"BiBERT quant_task_glue.py path: utils_glue.{proc_name} + convert_examples_to_features",
                         "tokenizer": "transformer.BertTokenizer.from_pretrained(vocab_dir, do_lower_case=True)",
                         "vocab_sha256": sha256_file(os.path.join(args.vocab_dir, "vocab.txt")),
                         "max_seq_length": L,
                         "format": ("[CLS] a [SEP] b [SEP] + zero padding (_truncate_seq_pair); token_type 0 for a, 1 for b; mask 1/0"
                                    if pair else "[CLS] tokens [SEP] + zero padding; token_type 0; mask 1/0")},
        "label_list": label_list,
        "preprocessing": {"bibert": {
            "shape": [n, L], "file": "inputs.bin", "sha256": files["inputs.bin"],
            "record": [{"name": "input_ids", "dtype": "i32", "shape": [1, L]},
                       {"name": "token_type_ids", "dtype": "i32", "shape": [1, L]},
                       {"name": "attention_mask", "dtype": "f32", "shape": [1, L]}],
            "definition": "per-sample record = input_ids i32[L] | token_type_ids i32[L] | attention_mask f32[L] (bitino_bench multi-input order)"}},
        "files": files,
        "limitations": (f"Smoke/regression subset (first {n} {split} examples); accuracy on it is not the {args.task.upper()} {split} accuracy."
                        if args.n else f"Full GLUE {args.task.upper()} {split} set ({n})."),
        "label_counts": {str(k): int((labels == k).sum()) for k in range(len(label_list))},
        "samples": [{"guid": e.guid, "text_sha256": sha256_bytes((e.text_a + "\t" + (e.text_b or "")).encode("utf-8")),
                     "label": int(f.label_id), "seq_length": int(f.seq_length)} for e, f in zip(examples, feats)],
    }
    if calib:
        head = {k: manifest.pop(k) for k in ("schema", "name", "task")}
        manifest.pop("selection_rule")
        manifest = {**head, "split": "train", "purpose": "calibration",
                    "source": {**train_src, "url": url, "dev": src},
                    "selection_rule": f"first {n} examples of GLUE {args.task.upper()} train in file order, skipping "
                                      "examples whose text occurs in the dev split(s) (see disjointness)",
                    "disjointness": disjoint, **{k: v for k, v in manifest.items() if k != "source"}}
        manifest["limitations"] = ("Calibration set for static INT8 quantization (activation statistics), drawn from the GLUE "
                                   f"{args.task.upper()} TRAIN split and disjoint by text from the dev split(s); labels are "
                                   "recorded for completeness only. Never report accuracy on it.")
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{name}: n={n} L={L} labels={manifest['label_counts']} truncated={int(sum(f.seq_length >= L for f in feats))} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
