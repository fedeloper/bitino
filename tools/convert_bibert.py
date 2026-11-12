#!/usr/bin/env python3
"""Convert the official BiBERT checkpoint to .bitino and produce reference outputs.

Run in an env with torch (the official model code needs no transformers package):

  ~/anaconda3/envs/bibert/bin/python tools/convert_bibert.py --model-id bibert-bert-base-sst2 \
      --repo .cache/repos/BiBERT --ckpt-dir .cache/checkpoints/bibert-bert-base-sst2 \
      --eval-dir .cache/data/sst2/eval128 --full-dev-dir .cache/data/sst2/dev872 \
      --out-dir artifacts/bibert-bert-base-sst2
  (MRPC / MNLI-m / QQP: --model-id bibert-bert-base-<task> with the eval sets of tools/build_glue_eval.py,
   e.g. --eval-dir .cache/data/glue/mrpc/eval128 --full-dev-dir .cache/data/glue/mrpc/dev408)

Produces in --out-dir:
  model.bitino                       converted graph
  reference_logits.f32                source model, fp32 (official forward, batch 1)
  reference_logits_f64.f64            source model in float64 with the binary weights the fp32 model
                                      uses (primary semantic reference, see below)
  reference_logits_f64_naive.f64      source model .double() as is (weight binarization redone in fp64)
  numpy_logits.f32                    numpy executor (float32) of model.bitino
  conversion_report.json, first_flip.json

Why two fp64 references: the source binarizes weights at run time as sign(W - mean_row W).
Re-evaluating that centering in float64 changes 43 of 110M binary weights (36 weights are
exactly 0 in fp32 and not in fp64, 7 change sign), so model.double() is a (very slightly)
different binary network. The primary fp64 reference keeps the fp32-derived binary weights
(alpha_row * sign, exactly what the fp32 model uses) and evaluates everything else in fp64.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import platform
import sys
import time
import types

# BLAS oversubscription on many-core hosts makes the small per-sample matmuls ~8x slower.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bitino import executor, format as bfformat, passes  # noqa: E402
from bitino.adapters import torch_bibert  # noqa: E402
from bitino.validation.metrics import compare_outputs  # noqa: E402
from first_flip_bibert import capture_torch, first_flip  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval(d: str):
    em = json.load(open(os.path.join(d, "manifest.json")))
    n, L = em["preprocessing"]["bibert"]["shape"]
    ids = np.fromfile(os.path.join(d, "input_ids.i32"), np.int32).reshape(n, L)
    tt = np.fromfile(os.path.join(d, "token_type_ids.i32"), np.int32).reshape(n, L)
    mask = np.fromfile(os.path.join(d, "attention_mask.f32"), np.float32).reshape(n, L)
    labels = np.fromfile(os.path.join(d, "labels.i32"), np.int32)
    return em, ids, tt, mask, labels


def torch_logits(model, ids, tt, mask, dtype=None, batch=1):
    """Official forward (quant_task_glue.do_eval: model(input_ids, segment_ids, input_mask)), batch 1
    (batch > 1 only for full-dev accuracy), long tensors as produced by their TensorDataset."""
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            a = torch.from_numpy(ids[i:i + batch].astype(np.int64))
            b = torch.from_numpy(tt[i:i + batch].astype(np.int64))
            c = torch.from_numpy(mask[i:i + batch].astype(np.int64))
            out.append(model(a, b, c)[0].double().numpy())
    return np.concatenate(out)


def f1_binary(pred, labels) -> float:
    """GLUE MRPC/QQP F1 of the positive class (sklearn f1_score default, as in utils_glue.acc_and_f1)."""
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def fp64_fixed_binary(model32):
    """float64 copy of the source model whose quantized layers use the binary weights computed by
    the fp32 model (alpha_row * sign(W - mean_row W) in fp32, utils_quant.py:221-224); activation
    binarization (utils_quant.py:230-237) and everything else unchanged, in float64."""
    import torch
    from torch import nn

    m64 = copy.deepcopy(model32).double()
    mods32 = dict(model32.named_modules())
    for name, mod in m64.named_modules():
        kind = type(mod).__name__
        if kind not in ("QuantizeLinear", "QuantizeEmbedding"):
            continue
        w32 = mods32[name].weight.detach()
        sf = torch.mean(abs(w32), dim=1, keepdim=True)
        wb = (sf * torch.sign(w32 - torch.mean(w32, dim=-1, keepdim=True))).double()  # exact fp32 products
        mod.register_buffer("_bf_wb", wb)
        if kind == "QuantizeLinear":
            def fwd(self, input, type=None):
                binary_input_no_grad = torch.sign(input)
                cliped_input = torch.clamp(input, -1.0, 1.0)
                ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
                out = nn.functional.linear(ba, self._bf_wb)
                if self.bias is not None:
                    out += self.bias.view(1, -1).expand_as(out)
                return out
        else:
            def fwd(self, input, type=None):
                return nn.functional.embedding(input, self._bf_wb, self.padding_idx, self.max_norm, self.norm_type,
                                               self.scale_grad_by_freq, self.sparse)
        mod.forward = types.MethodType(fwd, mod)
    return m64


def fp32_gelu_zero_threshold(gelu) -> dict:
    """Zero set of the source's float32 gelu (modeling_quant.gelu on CPU) for negative inputs: evaluated on every
    float32 value in [-12, -3] (16.8M values) plus a sweep down to -3e38. Returns t such that the fp32 gelu is
    non-zero on [t, -3] and reports the non-zero values below t (a narrow band, float32 erf rounding)."""
    import torch

    lo, hi = np.float32(-12.0).view(np.int32), np.float32(-3.0).view(np.int32)   # negative floats: larger bits = more negative
    bits = np.arange(hi, lo + 1, dtype=np.int64).astype(np.int32)                 # -3 ... -12
    x = torch.from_numpy(bits.view(np.float32).copy())
    with torch.no_grad():
        z = (gelu(x) == 0).numpy()
    xs = x.numpy()
    assert z[-1] and not z[0], "zero set not inside [-12, -3]"
    first = int(np.argmax(z))                       # first (closest to 0) zero
    gaps = xs[first:][~z[first:]]                   # non-zero inputs below the first zero (erf rounding is not monotone)
    far = torch.tensor([-12.0, -20.0, -100.0, -1e4, -1e10, -1e30, -3e38], dtype=torch.float32)
    with torch.no_grad():
        assert bool((gelu(far) == 0).all()), "fp32 gelu not 0 for very negative inputs"
    t = float(xs[first - 1])                        # smallest non-zero (most negative) input
    return {"t": t, "t_hex": float(t).hex(), "largest_zero_input": float(xs[first]), "n_values_checked": int(len(xs)),
            "nonzero_below_t": {"count": int(len(gaps)), "range": [float(gaps.min()), float(gaps.max())] if len(gaps) else None},
            "rule": "variant zero set: gelu(x) = 0 for all x < t (t = smallest float32 above which the CPU fp32 gelu is never 0). "
                    "The CPU fp32 gelu itself is non-zero again on 'nonzero_below_t' (float32 erf rounding is not monotone "
                    "there); the variant treats that band as zero"}


def gelu_underflow_stats(model, ids, tt, mask):
    """How often the fp32 source GELU returns exactly 0 for a negative input (fp32 erf underflow,
    x <~ -5.5): there sign(GELU) = 0 in the fp32 source but -1 in fp64 / the IR."""
    import torch

    tot = {"ffn1_values": 0, "fp32_gelu_zero_for_negative_x": 0, "x_exactly_zero": 0, "min_x": 0.0}
    hooks = []
    for name, mod in model.named_modules():
        if name.endswith("intermediate.dense"):
            def hook(m, inp, out):
                x = out.detach()
                g = x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
                tot["ffn1_values"] += x.numel()
                tot["fp32_gelu_zero_for_negative_x"] += int(((g == 0) & (x < 0)).sum())
                tot["x_exactly_zero"] += int((x == 0).sum())
                tot["min_x"] = min(tot["min_x"], float(x.min()))
            hooks.append(mod.register_forward_hook(hook))
    torch_logits(model, ids, tt, mask)
    for h in hooks:
        h.remove()
    return tot


def zero_stats(graph, env) -> dict:
    """Exact zeros reaching sign_zero binarizations in the IR (the ternary cases) for one sample."""
    z = {"bdense_input_zeros": 0, "attention_score_zeros_unmasked": 0, "attention_q_k_v_zeros": 0}
    for n in graph.nodes:
        if n.op == "BDense":
            z["bdense_input_zeros"] += int(np.sum(env[n.inputs[0]] == 0))
        elif n.op == "AddBroadcast":
            s = env[n.outputs[0]]
            z["attention_score_zeros_unmasked"] += int(np.sum(s == 0))
        elif n.op == "BMatMul" and n.attrs.get("a_binarize") == "sign_zero":
            z["attention_q_k_v_zeros"] += int(np.sum(env[n.inputs[0]] == 0) + np.sum(env[n.inputs[1]] == 0))
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="bibert-bert-base-sst2")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry", "models.json"))
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--full-dev-dir", help="full GLUE dev tensors (reference fp32 accuracy only)")
    ap.add_argument("--full-dev-batch", type=int, default=1, help="batch size of the full-dev fp32 reference (accuracy only)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sem-tol", type=float, default=1e-6)
    ap.add_argument("--gelu-fp32-zero-set", action="store_true",
                    help="variant: the IR forces gelu(x) = 0 for x below the fp32 underflow threshold (the zero set of the "
                         "fp32 source), and the float64 references use gelu64(x) with the same zero set")
    ap.add_argument("--drop-gelu-before-sign", action="store_true",
                    help="variant: the FFN GELU, used only through sign_zero binarizations, is removed "
                         "(passes.drop_gelu_before_sign; sign(GELU(x)) = sign(x) except where the evaluated GELU underflows "
                         "to 0). The references use gelu(x) = x (with --gelu-fp32-zero-set: x for x >= t, 0 below), i.e. the "
                         "same semantics; --out-dir should be a variant directory (e.g. artifacts/<model>/no_gelu)")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    tasks = {"bibert-bert-base-sst2": "sst-2", "bibert-bert-base-mrpc": "mrpc", "bibert-bert-base-mnli": "mnli", "bibert-bert-base-qqp": "qqp"}
    reg = {m["model_id"]: m for m in json.load(open(args.registry))["models"]}[args.model_id]
    ckpt = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    t0 = time.time()
    actual = sha256_file(ckpt)
    if actual != reg["checkpoint"]["sha256"]:
        print(f"ERROR: checkpoint hash mismatch: registry {reg['checkpoint']['sha256']}, got {actual}", file=sys.stderr)
        return 3
    os.makedirs(args.out_dir, exist_ok=True)
    model = torch_bibert.load_official(args.repo, args.ckpt_dir)
    repo_head = open(os.path.join(args.repo, ".git", "HEAD")).read().strip()
    if repo_head.startswith("ref:"):
        ref = os.path.join(args.repo, ".git", repo_head[5:])
        repo_head = open(ref).read().strip() if os.path.exists(ref) else repo_head

    gelu_zero = None
    if args.gelu_fp32_zero_set:
        import transformer.modeling_quant as mq

        gelu_zero = fp32_gelu_zero_threshold(mq.gelu)
        print("fp32 gelu zero set:", gelu_zero)
    em, ids, tt, mask, labels = load_eval(args.eval_dir)
    if "task" in em and args.model_id in tasks:  # eval sets from tools/build_glue_eval.py carry their task
        assert em["task"] == tasks[args.model_id], f"eval set task {em['task']} != model task {tasks[args.model_id]}"
    if args.max_samples:
        ids, tt, mask, labels = ids[:args.max_samples], tt[:args.max_samples], mask[:args.max_samples], labels[:args.max_samples]
    n, L = ids.shape

    if args.drop_gelu_before_sign:  # the variant's source semantics: GELU -> identity (keeping the zero set, if any)
        import transformer.modeling_quant as mq

        tz = gelu_zero["t"] if gelu_zero is not None else None
        mq.gelu = (lambda x: x) if tz is None else (lambda x: torch.where(x >= tz, x, torch.zeros_like(x)))
    ref32 = torch_logits(model, ids, tt, mask).astype(np.float32)
    m64 = fp64_fixed_binary(model)
    m64n = copy.deepcopy(model).double()
    if gelu_zero is not None and not args.drop_gelu_before_sign:   # float64 gelu with the fp32 zero set, for both float64 references
        import transformer.modeling_quant as mq

        orig_gelu, tz = mq.gelu, gelu_zero["t"]
        mq.gelu = lambda x: torch.where(x >= tz, orig_gelu(x), torch.zeros_like(x)) if x.dtype == torch.float64 else orig_gelu(x)
    ref64 = torch_logits(m64, ids, tt, mask)
    ref64n = torch_logits(m64n, ids, tt, mask)
    ref32.tofile(os.path.join(args.out_dir, "reference_logits.f32"))
    ref64.tofile(os.path.join(args.out_dir, "reference_logits_f64.f64"))
    ref64n.tofile(os.path.join(args.out_dir, "reference_logits_f64_naive.f64"))
    print(f"reference fp32 acc {np.mean(ref32.argmax(1) == labels):.4f}  fp64 {np.mean(ref64.argmax(1) == labels):.4f}  "
          f"fp64-naive {np.mean(ref64n.argmax(1) == labels):.4f}  ({time.time() - t0:.0f}s)")

    report = {"model_id": args.model_id, "n_samples": int(n), "eval_set": em["name"], "eval_limitations": em["limitations"],
              "source": {"repo": "https://github.com/htqin/BiBERT", "repo_commit": repo_head, "checkpoint_sha256": actual,
                         "class": "transformer.modeling_quant.BertForSequenceClassification",
                         "torch": torch.__version__, "python": platform.python_version()},
              "reference": {"top1_accuracy_fp32": float(np.mean(ref32.argmax(1) == labels)),
                            "top1_accuracy_fp64": float(np.mean(ref64.argmax(1) == labels)),
                            "top1_accuracy_fp64_naive": float(np.mean(ref64n.argmax(1) == labels))},
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels),
              "reference_f64_naive_vs_f64": compare_outputs(ref64, ref64n, labels)}
    report["gelu_fp32_underflow_eval_set"] = gelu_underflow_stats(model, ids, tt, mask)
    if gelu_zero is not None:
        report["gelu_fp32_zero_set"] = dict(gelu_zero, note="IR and float64 references force gelu = 0 for x < t")

    if args.full_dev_dir:
        fem, fids, ftt, fmask, flab = load_eval(args.full_dev_dir)
        t2 = time.time()
        f32 = torch_logits(model, fids, ftt, fmask, batch=args.full_dev_batch)
        f32.astype(np.float32).tofile(os.path.join(args.out_dir, "reference_logits_full_dev.f32"))
        report["full_dev"] = {"eval_set": fem["name"], "n": int(len(flab)), "batch_size": args.full_dev_batch,
                              "top1_accuracy_fp32": float(np.mean(f32.argmax(1) == flab)),
                              "claimed": reg["reported_metrics"]["value"],
                              "logits_file": "reference_logits_full_dev.f32"}
        if f32.shape[1] == 2 and fem.get("task") in ("mrpc", "qqp"):
            report["full_dev"]["f1_fp32"] = f1_binary(f32.argmax(1), flab)
        print(f"full dev {fem['name']} ({len(flab)}): fp32 accuracy {report['full_dev']['top1_accuracy_fp32']:.4f}"
              + (f", F1 {report['full_dev']['f1_fp32']:.4f}" if "f1_fp32" in report["full_dev"] else "") + f" ({time.time() - t2:.0f}s)")

    graph, info = torch_bibert.convert(model, args.model_id, seq_len=L, gelu_zero_below=gelu_zero["t"] if gelu_zero else None)
    applied = passes.optimize(graph, fold=False)
    if args.drop_gelu_before_sign:
        applied["drop_gelu_before_sign"] = passes.drop_gelu_before_sign(graph)
        if not applied["drop_gelu_before_sign"]:
            raise ValueError("--drop-gelu-before-sign: no GELU used only through a sign binarization")
        applied["prune_unused_constants"] += passes.prune_unused_constants(graph)
    graph.manifest = {
        "schema": "bitino.manifest.v1", "model_id": args.model_id,
        "variant": "unfused" + (", drop_gelu_before_sign" if args.drop_gelu_before_sign else ""),
        "source": {"framework": "pytorch", "repo": "https://github.com/htqin/BiBERT", "repo_commit": repo_head,
                   "class": "transformer.modeling_quant.BertForSequenceClassification", "checkpoint_file": "pytorch_model.bin",
                   "checkpoint_sha256": actual},
        "conversion": {"tool": "tools/convert_bibert.py", "adapter": "bitino.adapters.torch_bibert", "passes": applied,
                       "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "python": platform.python_version(),
                       "torch": torch.__version__},
        "input": {"names": ["input_ids", "token_type_ids", "attention_mask"], "shape": [1, L],
                  "dtypes": ["i32", "i32", "f32"], "preprocessing": em.get("tokenization") or {
                      "format": "[CLS] tokens [SEP] + zero padding; token_type 0; mask 1/0", "max_seq_length": L,
                      "note": "converted on an eval set without a tokenization record (bitino fetch synthetic set); "
                              "tokenize as BiBERT quant_task_glue.py (BertTokenizer, do_lower_case=True)"}},
        "outputs": ["logits"], "op_versions": {"format": 1},
        "layer_precision": info["layer_precision"], "exact_zero_weights_total": info["exact_zero_weights_total"],
        "unused_parameters": info["unused_parameters"],
        "gelu": info["gelu"] + ("; removed (only used through sign_zero): drop_gelu_before_sign" if args.drop_gelu_before_sign else ""),
    }
    path = os.path.join(args.out_dir, "model.bitino")
    file_sha = bfformat.write(graph, path)
    g2 = bfformat.read(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB) ops {g2.op_counts()}")

    out32, out64, zeros = [], [], {}
    t1 = time.time()
    for i in range(n):
        feed = {"input_ids": ids[i:i + 1], "token_type_ids": tt[i:i + 1], "attention_mask": mask[i:i + 1]}
        env = executor.run(g2, feed, keep_intermediates=True)
        out32.append(env[g2.outputs[0]].reshape(1, -1))
        for k, v in zero_stats(g2, env).items():
            zeros[k] = zeros.get(k, 0) + v
        out64.append(executor.run(g2, feed, float_dtype=np.float64)[g2.outputs[0]].reshape(1, -1))
    out32 = np.concatenate(out32).astype(np.float32)
    out64 = np.concatenate(out64)
    out32.tofile(os.path.join(args.out_dir, "numpy_logits.f32"))
    print(f"numpy executor: {time.time() - t1:.0f}s")
    sem = compare_outputs(ref64, out64, labels)
    dev = np.max(np.abs(out64 - ref64), axis=1)
    deviating = [int(i) for i in np.nonzero(dev > args.sem_tol)[0]]
    report["conversion"] = {"file": "model.bitino", "file_sha256": file_sha, "file_bytes": os.path.getsize(path),
                            "op_counts": g2.op_counts(), "passes": applied,
                            "exact_zero_weights_total": info["exact_zero_weights_total"],
                            "numpy_f32_vs_reference_f32": compare_outputs(ref32, out32, labels),
                            "numpy_f32_vs_reference_f64": compare_outputs(ref64, out32, labels),
                            "semantic_check_ir_f64_vs_reference_f64": sem,
                            "semantic_samples_within_tol": int(n - len(deviating)), "semantic_tol": args.sem_tol,
                            "semantic_max_abs_dev_within_tol": float(np.max(dev[dev <= args.sem_tol])) if np.any(dev <= args.sem_tol) else None,
                            "ir_f64_vs_reference_f64_naive": compare_outputs(ref64n, out64, labels),
                            "ternary_zero_counts_eval_set_ir_f32": zeros}
    print(f"SEMANTIC (IR f64 vs source f64): {n - len(deviating)}/{n} within {args.sem_tol}, max_abs {sem['max_abs_error']:.3e}, "
          f"top1 agree {sem['top1_agreement']:.4f}; deviating {deviating[:20]}")

    flips = []
    for i in deviating:
        feed = {"input_ids": ids[i:i + 1], "token_type_ids": tt[i:i + 1], "attention_mask": mask[i:i + 1]}
        env = executor.run(g2, feed, keep_intermediates=True, float_dtype=np.float64)
        rec = capture_torch(m64, torch.from_numpy(ids[i:i + 1].astype(np.int64)), torch.from_numpy(tt[i:i + 1].astype(np.int64)),
                            torch.from_numpy(mask[i:i + 1].astype(np.int64)))
        r = {"sample": i, "max_abs_logit_dev": float(dev[i])}
        r.update(first_flip(rec, g2, env, model.config.num_hidden_layers))
        flips.append(r)
        print(json.dumps(r))
    json.dump(flips, open(os.path.join(args.out_dir, "first_flip.json"), "w"), indent=1)
    report["first_flip"] = {"n_deviating": len(deviating),
                            "max_abs_pre_at_first_flip": max([f.get("max_abs_pre_at_flip_torch") or 0 for f in flips], default=None),
                            "file": "first_flip.json"}
    json.dump(report, open(os.path.join(args.out_dir, "conversion_report.json"), "w"), indent=1)
    print(json.dumps(report["reference"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
