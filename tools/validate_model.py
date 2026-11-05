#!/usr/bin/env python3
"""Level-4/5 validation: compares runtime outputs (host or device) with references.

Runs (or ingests) `bitino_bench --inputs <eval tensors> --outputs <file>` and
compares the produced logits against:
  * the numpy executor of the same .bitino (float32 intermediates)   -> runtime correctness
  * the source framework in float32 and float64                       -> end-to-end fidelity
Writes a JSON report. Exit code 0 iff the acceptance criteria hold:
  - top-1 agreement with the float64 reference >= --min-top1-agreement (default 0.98), and
  - every top-1 disagreement is explained by the float32 noise floor of the source model
    itself (i.e. its count does not exceed that of float32-vs-float64 reference + slack).

Usage (host):
  python3 tools/validate_model.py --artifacts artifacts/larq-zoo-quicknet-small \
      --bench build/host/bitino_bench --eval-dir .cache/data/eval128 --stage reference
Image-to-image models (restoration / super-resolution, slice S8; --output-kind image): outputs are images, so
there is no top-1. The eval set's manifest carries the ground-truth crops ("targets") and the metric
("image_task"). Reported: per-sample max |diff| and PSNR against the float64 / float32 references and the numpy
executor, and the task PSNR/SSIM against ground truth (candidate vs source). Acceptance (default): the candidate's
mean task PSNR is within the float32 noise floor of the source (|PSNR_f32 - PSNR_f64|) + --max-psnr-drop of the
float64 reference, and no sample is below --min-psnr-vs-f64 against the float64 reference. --accept-exact-numpy and
--expect-identical as for classifiers. --explain: every sample whose output differs from the numpy executor by
more than --image-tol x the output range is traced to its first binarization flip (same rule as for top-1
disagreements); samples below that bound differ by float rounding only.

Usage (device outputs already pulled):
  python3 tools/validate_model.py --artifacts ... --outputs-file pulled.f32 --label device:neon
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
from bitino.validation.metrics import compare_outputs  # noqa: E402
from bitino.validation import image_metrics  # noqa: E402


def flip_explained(ff, args) -> bool:
    """None = no runtime-vs-numpy flip: the disagreement is already present in the numpy executor."""
    if ff is None:
        return True
    pre = max(ff["max_abs_pre_numpy"], ff["max_abs_pre_cpp"])
    if args.flip_rule == "feed":
        return pre <= ff["max_abs_diff_feeding_tensor"]
    return pre < args.flip_rel * ff["typical_abs_pre"]


def rule_text(args) -> str:
    if args.flip_rule == "feed":
        return "first flip at |pre| <= max |C++ - numpy| of the binarized tensor (flip-rule feed)"
    return f"first flip at |pre| < {args.flip_rel} x typical |pre|"


def rejudge(args) -> int:
    rep = json.load(open(args.rejudge))
    traces = rep["explain"]["traces"]
    for t in traces:
        t["explained_as_rounding"] = bool(flip_explained(t.get("first_flip"), args))
    ok = all(t["explained_as_rounding"] for t in traces)
    rep["explain"].update({"rule": rule_text(args), "all_explained": ok, "rejudged_from": os.path.basename(args.rejudge)})
    rep["acceptance"] = {"criterion": "every top-1 disagreement vs fp64 traced to a rounding-scale first flip (--explain, "
                                      + rule_text(args) + ")", "passed": ok}
    rep["label"] += f":{args.flip_rule}"
    path = args.json or args.rejudge.replace(".json", f"_{args.flip_rule}.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(f"{rep['label']}: {len(traces)} traces, all explained {ok} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def trace_samples(args, samples, model_path: str, in_path: str, n: int) -> tuple[list, bool]:
    """Traces each listed sample to its first runtime-vs-numpy binarization flip (tools/first_flip_runtime.py)."""
    import shutil
    traces, all_ok = [], True
    rec = os.path.getsize(in_path) // n          # bytes per sample record (all graph inputs)
    for k in samples:
        d = tempfile.mkdtemp(prefix=f"bf_dump_{k}_")
        # run only sample k (a one-record inputs file): samples are independent, so the dump is the same as
        # with the full file and --dump-sample k, at 1/n of the cost (matters under qemu)
        one = os.path.join(d, "sample.bin")
        with open(in_path, "rb") as f:
            f.seek(k * rec)
            open(one, "wb").write(f.read(rec))
        subprocess.run([args.bench, "--model", model_path, "--inputs", one,
                        "--outputs", "/dev/null", "--stage", args.stage, "--threads", str(args.threads),
                        "--warmup", "0", "--iterations", "1", "--no-reuse-memory", "--dump-dir", d,
                        "--dump-sample", "0"], check=True, capture_output=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "first_flip_runtime.py"), "--model", model_path,
                            "--inputs", one, "--sample", "0", "--dump-dir", d],
                           check=True, capture_output=True, text=True)
        shutil.rmtree(d, ignore_errors=True)
        t0 = json.loads(r.stdout.strip().splitlines()[-1])
        t0["sample"] = k
        r = types.SimpleNamespace(stdout=json.dumps(t0))
        t = json.loads(r.stdout.strip().splitlines()[-1])
        ff = t.get("first_flip")
        # No runtime-vs-numpy flip: the disagreement is already present in the numpy
        # executor itself (source-model noise floor, see conversion report).
        ok = flip_explained(ff, args)
        t["explained_as_rounding"] = bool(ok)
        all_ok &= ok
        traces.append(t)
    return traces, all_ok


def image_main(args, em: dict, pre: dict, n: int, raw: np.ndarray, model_path: str) -> int:
    """--output-kind image: image-to-image models (see the module docstring)."""
    it = em["image_task"]
    rng = it.get("output_range") or it["output_range_by_preprocessing"][args.preprocess]
    data_range = float(rng[1] - rng[0])
    metric = args.image_metric or it["metric"]
    border = int(it.get("crop_border", 0))
    tg = em["targets"]
    targets = np.fromfile(os.path.join(args.eval_dir, tg["file"]), np.uint8).reshape(tg["shape"])[:n]
    out = raw.reshape((n,) + targets.shape[1:])
    ref32 = np.fromfile(os.path.join(args.artifacts, "reference_logits.f32"), np.float32).reshape(out.shape)
    ref64 = np.fromfile(os.path.join(args.artifacts, "reference_logits_f64.f64"), np.float64).reshape(out.shape)

    def quality(imgs):
        q = [image_metrics.image_quality(o, t, metric, border) for o, t in zip(imgs, targets)]
        ss = [b for _, b in q if b is not None]
        return {"psnr_mean": float(np.mean([a for a, _ in q])), "ssim_mean": float(np.mean(ss)) if ss else None,
                "psnr_per_sample": [float(a) for a, _ in q]}

    report = {"label": args.label or f"host:{args.stage}:t{args.threads}", "model": model_path, "n_samples": int(n),
              "output_kind": "image", "eval_set": em["name"], "eval_limitations": em["limitations"],
              "metric": {"name": metric, "crop_border": border, "data_range": data_range},
              "vs_reference_f64": image_metrics.compare_images(ref64, out, data_range),
              "vs_reference_f32": image_metrics.compare_images(ref32, out, data_range),
              "reference_f32_vs_f64": image_metrics.compare_images(ref64, ref32, data_range),
              "quality_vs_ground_truth": {"candidate": quality(out), "reference_f32": quality(ref32), "reference_f64": quality(ref64)}}
    np_path = os.path.join(args.artifacts, args.numpy_logits)
    npy = None
    if os.path.exists(np_path):
        npy = np.fromfile(np_path, np.float32).reshape(out.shape)
        report["vs_numpy_executor_f32"] = image_metrics.compare_images(npy, out, data_range)
    q = report["quality_vs_ground_truth"]
    floor = abs(q["reference_f32"]["psnr_mean"] - q["reference_f64"]["psnr_mean"])
    drop = q["reference_f64"]["psnr_mean"] - q["candidate"]["psnr_mean"]
    pmin = report["vs_reference_f64"]["psnr_vs_reference_min_db"]
    report["acceptance"] = {
        "criterion": (f"task PSNR drop vs the float64 reference ({drop:+.4f} dB) <= float32 noise floor ({floor:.4f} dB) + "
                      f"{args.max_psnr_drop} dB and min per-sample PSNR vs float64 ({pmin:.1f} dB) >= {args.min_psnr_vs_f64} dB"),
        "passed": bool(drop <= floor + args.max_psnr_drop and pmin >= args.min_psnr_vs_f64)}
    if args.require_exact_numpy:
        exact = bool(report.get("vs_numpy_executor_f32", {}).get("exact_match", False))
        report["acceptance"]["criterion"] += " and bit-exact with the numpy executor"
        report["acceptance"]["passed"] = bool(report["acceptance"]["passed"] and exact)
    if args.accept_exact_numpy and report.get("vs_numpy_executor_f32", {}).get("exact_match"):
        report["acceptance"]["passed"] = True
        report["acceptance"]["criterion"] = "bit-exact with the numpy executor (exact stage; numpy vs source is Level 3)"
    if args.expect_identical:
        same = os.path.exists(args.expect_identical) and os.path.getsize(args.expect_identical) == raw.nbytes \
            and bool(np.array_equal(np.fromfile(args.expect_identical, np.float32).view(np.uint32), raw.view(np.uint32)))
        report["identical_to"] = {"file": args.expect_identical, "identical": same}
        if same:
            report["acceptance"]["passed"] = True
            report["acceptance"]["criterion"] = f"bit-identical to {args.expect_identical}"
    if args.explain:
        assert args.bench, "--explain needs --bench (host)"
        assert npy is not None, f"--explain (image) needs the numpy executor outputs {np_path}"
        dev = np.abs(out.reshape(n, -1).astype(np.float64) - npy.reshape(n, -1)).max(axis=1)
        tol = args.image_tol * data_range
        dis = np.where(dev > tol)[0].tolist()
        traces, all_ok = trace_samples(args, dis, model_path, os.path.join(args.eval_dir, pre["file"]), n)
        for t in traces:
            t["max_abs_vs_numpy"] = float(dev[t["sample"]])
        report["explain"] = {"rule": rule_text(args), "trigger": f"max |runtime - numpy| > {args.image_tol} x output range ({tol:g})",
                             "n_samples_traced": len(dis), "n_samples_identical_to_numpy": int(np.sum(dev == 0)),
                             "max_abs_vs_numpy_untraced": float(dev[dev <= tol].max()) if np.any(dev <= tol) else None,
                             "traces": traces, "all_explained": bool(all_ok)}
        report["acceptance"]["passed"] = bool(all_ok)
        report["acceptance"]["criterion"] = (f"every sample deviating from the numpy executor by more than {args.image_tol} x the output range "
                                             "traced to a rounding-scale first flip (--explain)")
    path = args.json or os.path.join(args.artifacts, f"validation_{report['label'].replace(':', '_')}.json")
    json.dump(report, open(path, "w"), indent=1)
    v = report["vs_reference_f64"]
    print(f"{report['label']}: task PSNR {q['candidate']['psnr_mean']:.4f} dB (f64 ref {q['reference_f64']['psnr_mean']:.4f}, "
          f"f32 ref {q['reference_f32']['psnr_mean']:.4f}), PSNR vs f64 min {v['psnr_vs_reference_min_db']:.1f} dB, max_abs {v['max_abs_error']:.3e}"
          + (f", vs numpy max_abs {report['vs_numpy_executor_f32']['max_abs_error']:.3e} exact={report['vs_numpy_executor_f32']['exact_match']}"
             if "vs_numpy_executor_f32" in report else "")
          + (f", traced {report['explain']['n_samples_traced']}" if "explain" in report else "")
          + (f", identical_to_expected={report['identical_to']['identical']}" if "identical_to" in report else "")
          + f" -> {'PASS' if report['acceptance']['passed'] else 'FAIL'}")
    return 0 if report["acceptance"]["passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--model", default="model.bitino")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--preprocess", default="larq_zoo")
    ap.add_argument("--bench", help="host bitino_bench binary (runs it)")
    ap.add_argument("--outputs-file", help="pre-computed outputs (e.g. pulled from device)")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--numpy-logits", default="numpy_logits_fused.f32")
    ap.add_argument("--min-top1-agreement", type=float, default=0.98)
    ap.add_argument("--json", help="report path (default: <artifacts>/validation_<label>.json)")
    ap.add_argument("--explain", action="store_true",
                    help="(host, with --bench) trace every top-1 disagreement to its first binarization flip; "
                         "accept iff all start at |pre| < --flip-rel * typical |pre|")
    ap.add_argument("--flip-rel", type=float, default=1e-4)
    ap.add_argument("--flip-rule", choices=("relative", "feed"), default="relative",
                    help="--explain acceptance of a first runtime-vs-numpy flip: 'relative' (|pre| < --flip-rel x typical |pre|) "
                         "or 'feed' (|pre| <= max |C++ - numpy| of the tensor being binarized, i.e. the flipped value lies inside "
                         "the float deviation of that tensor; for models with unscaled, large activations such as bitorch)")
    ap.add_argument("--rejudge", metavar="REPORT",
                    help="re-evaluate the --explain traces stored in REPORT with --flip-rule/--flip-rel (no runtime run)")
    ap.add_argument("--num-classes", type=int, default=0,
                    help="logits per sample at the start of each output record (default: auto, 1000 for ImageNet "
                         "models, else the whole record)")
    ap.add_argument("--require-exact-numpy", action="store_true",
                    help="additionally require outputs bit-identical to the numpy executor (--numpy-logits)")
    ap.add_argument("--accept-exact-numpy", action="store_true",
                    help="exact stages: PASS if the outputs are bit-identical to the numpy executor running the same "
                         ".bitino (its fidelity to the source is the Level-3 check), whatever the top-1 rule says")
    ap.add_argument("--expect-identical", metavar="OUTPUTS",
                    help="also compare bit-for-bit with these outputs (e.g. the emulated-ARM run of the same build); "
                         "PASS if identical, otherwise the other criteria decide")
    ap.add_argument("--output-kind", choices=("logits", "embedding", "image"), default="logits",
                    help="embedding: outputs are embeddings (face / speech models). A sample 'disagrees' when its cosine "
                         "similarity with the float64 reference is below --embedding-cos (instead of a top-1 change); the "
                         "acceptance rule, the float32 noise floor and --explain use that set. The whole output record is "
                         "compared unless --num-classes is given. image: image-to-image models (eval set with targets + "
                         "image_task; see the docstring). Default (logits) behaviour is unchanged.")
    ap.add_argument("--embedding-cos", type=float, default=0.999)
    ap.add_argument("--image-metric", choices=image_metrics.METRICS, help="override the eval set's image_task metric")
    ap.add_argument("--image-tol", type=float, default=1e-3,
                    help="image --explain: trace samples with max |runtime - numpy| > image-tol x output range")
    ap.add_argument("--max-psnr-drop", type=float, default=0.02, help="image acceptance slack (dB) beyond the float32 noise floor")
    ap.add_argument("--min-psnr-vs-f64", type=float, default=30.0, help="image acceptance: per-sample PSNR floor vs the float64 reference")
    args = ap.parse_args()
    if args.rejudge:
        return rejudge(args)

    em = json.load(open(os.path.join(args.eval_dir, "manifest.json")))
    pre = em["preprocessing"][args.preprocess]
    n = pre["shape"][0]
    labels = np.array([s["label"] for s in em["samples"]][:n])
    model_path = os.path.join(args.artifacts, args.model)

    if args.outputs_file:
        out_path = args.outputs_file
    else:
        assert args.bench, "need --bench or --outputs-file"
        out_path = tempfile.mktemp(suffix=".f32")
        subprocess.run([args.bench, "--model", model_path, "--inputs", os.path.join(args.eval_dir, pre["file"]),
                        "--outputs", out_path, "--stage", args.stage, "--threads", str(args.threads),
                        "--warmup", "0", "--iterations", "1"], check=True, capture_output=True)
    raw = np.fromfile(out_path, np.float32)
    if args.output_kind == "image":
        return image_main(args, em, pre, n, raw, model_path)
    per = raw.size // n
    assert per * n == raw.size, f"outputs size {raw.size} not divisible by {n}"
    out = raw.reshape(n, per)
    n_cls = args.num_classes or (per if args.output_kind == "embedding" else 1000 if per in (1000, 2000) else per)
    logits = out[:, :n_cls]

    ref32 = np.fromfile(os.path.join(args.artifacts, "reference_logits.f32"), np.float32).reshape(n, -1)
    ref64 = np.fromfile(os.path.join(args.artifacts, "reference_logits_f64.f64"), np.float64).reshape(n, -1)
    report = {"label": args.label or f"host:{args.stage}:t{args.threads}", "model": model_path, "n_samples": int(n),
              "eval_set": em["name"], "eval_limitations": em["limitations"],
              "vs_reference_f64": compare_outputs(ref64, logits, labels),
              "vs_reference_f32": compare_outputs(ref32, logits, labels),
              "reference_f32_vs_f64": compare_outputs(ref64, ref32, labels)}
    np_path = os.path.join(args.artifacts, args.numpy_logits)
    if os.path.exists(np_path):
        report["vs_numpy_executor_f32"] = compare_outputs(np.fromfile(np_path, np.float32).reshape(n, -1), logits, labels)
    floor = report["reference_f32_vs_f64"]["n_top1_disagreements"]
    got = report["vs_reference_f64"]["n_top1_disagreements"]
    slack = max(1, int(round(0.01 * n)))
    report["acceptance"] = {
        "criterion": f"top1_agreement_vs_f64 >= {args.min_top1_agreement} and top1 disagreements <= float32 noise floor ({floor}) + slack ({slack})",
        "passed": bool(report["vs_reference_f64"]["top1_agreement"] >= args.min_top1_agreement and got <= floor + slack),
    }
    if args.output_kind == "embedding":
        def cosines(a, b):
            a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
            return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)
        c64, cfl = cosines(logits, ref64), cosines(ref32, ref64)
        dis_emb = np.where(c64 < args.embedding_cos)[0]
        floor_e = int(np.sum(cfl < args.embedding_cos))
        report["embedding"] = {"cosine_threshold": args.embedding_cos,
                               "cosine_vs_f64_min": float(c64.min()), "cosine_vs_f64_mean": float(c64.mean()),
                               "cosine_ref_f32_vs_f64_min": float(cfl.min()),
                               "n_below_threshold": int(len(dis_emb)), "samples_below_threshold": [int(i) for i in dis_emb],
                               "ref_f32_n_below_threshold": floor_e}
        if os.path.exists(np_path):
            report["embedding"]["cosine_vs_numpy_min"] = float(cosines(
                logits, np.fromfile(np_path, np.float32).reshape(n, -1)).min())
        report["acceptance"] = {
            "criterion": f"samples with cosine vs f64 < {args.embedding_cos} <= float32 noise floor ({floor_e}) + slack ({slack})",
            "passed": bool(len(dis_emb) <= floor_e + slack)}
    if args.require_exact_numpy:
        exact = bool(report.get("vs_numpy_executor_f32", {}).get("exact_match", False))
        report["acceptance"]["criterion"] += " and bit-exact with the numpy executor"
        report["acceptance"]["passed"] = bool(report["acceptance"]["passed"] and exact)
    if args.accept_exact_numpy and report.get("vs_numpy_executor_f32", {}).get("exact_match"):
        report["acceptance"]["passed"] = True
        report["acceptance"]["criterion"] = "bit-exact with the numpy executor (exact stage; numpy vs source is Level 3)"
    if args.expect_identical:
        same = os.path.exists(args.expect_identical) and os.path.getsize(args.expect_identical) == raw.nbytes \
            and bool(np.array_equal(np.fromfile(args.expect_identical, np.float32).view(np.uint32), raw.view(np.uint32)))
        report["identical_to"] = {"file": args.expect_identical, "identical": same}
        if same:
            report["acceptance"]["passed"] = True
            report["acceptance"]["criterion"] = f"bit-identical to {args.expect_identical}"
    if args.explain:
        assert args.bench, "--explain needs --bench (host)"
        dis = (np.where(logits.argmax(1) != ref64.argmax(1))[0].tolist() if args.output_kind == "logits"
               else [int(i) for i in report["embedding"]["samples_below_threshold"]])
        traces, all_ok = trace_samples(args, dis, model_path, os.path.join(args.eval_dir, pre["file"]), n)
        report["explain"] = {"rule": rule_text(args), "traces": traces, "all_explained": bool(all_ok)}
        report["acceptance"]["passed"] = bool(all_ok)
        report["acceptance"]["criterion"] = ("every top-1 disagreement vs fp64" if args.output_kind == "logits" else
                                             f"every sample with cosine vs fp64 < {args.embedding_cos}") + \
            " traced to a rounding-scale first flip (--explain)"
    path = args.json or os.path.join(args.artifacts, f"validation_{report['label'].replace(':', '_')}.json")
    json.dump(report, open(path, "w"), indent=1)
    v = report["vs_reference_f64"]
    print(f"{report['label']}: top1 acc {v['candidate_top1_accuracy']:.4f} (ref {v['reference_top1_accuracy']:.4f}), "
          f"top1 agree vs f64 {v['top1_agreement']:.4f}, max_abs {v['max_abs_error']:.3e}, cos_min {v['cosine_min']:.5f}"
          + (f", vs numpy max_abs {report['vs_numpy_executor_f32']['max_abs_error']:.3e} exact={report['vs_numpy_executor_f32']['exact_match']}"
             if "vs_numpy_executor_f32" in report else "")
          + (f", identical_to_expected={report['identical_to']['identical']}" if "identical_to" in report else "")
          + (f", embedding: {report['embedding']['n_below_threshold']} below cos {args.embedding_cos} (floor "
             f"{report['embedding']['ref_f32_n_below_threshold']})" if "embedding" in report else "")
          + (f", explain: {len(report['explain']['traces'])} traced" if "explain" in report and "embedding" in report else "")
          + f" -> {'PASS' if report['acceptance']['passed'] else 'FAIL'}")
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
