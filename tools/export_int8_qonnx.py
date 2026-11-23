#!/usr/bin/env python3
"""Writer of step 5 of the QONNX route (docs/qonnx-frontend.md § Plan, docs/qonnx-step5.md): a published int8_full file
(model_int8_full.bitino / model_int8_full_log2.bitino, the I8* graph the quantizer wrote) -> canonical QONNX file of
the Bitino profile (docs/qonnx-profile.md § 5, ``bitino-qonnx/1``), one ONNX group per IR node
(python/bitino/qonnx_int8.py, group mapping of step 3).

Runs in .cache/venv-qonnx (numpy + onnx), python -I:

  .cache/venv-qonnx/bin/python -I tools/export_int8_qonnx.py --model-id ab-bnn-cifar10-reactnet18 \
      --file model_int8_full.bitino --out model.onnx [--artifacts-dir artifacts] [--json export.json]

The source is the published int8_full file itself (read only): it is the quantizer's output, and the quantizer
(tools/quantize_int8_full.py) reads calibration images that are not redistributed (registry/recipes.json: the int8
steps are "reconstructed", not runnable recipes). Before writing, the file's data and graph SHA-256 are checked against
registry/recipes.json -> expected[<file>]. The model metadata (§ 6) comes from the recipe (calibration, licence,
source code, eval set), registry/eval_sets via the eval manifest (preprocessing) and the file's own manifest
(framework, checkpoint SHA-256). The written file is refused unless it passes check-qonnx with no error and no
warning.

`bitino fetch --int8` (python/bitino/fetch.py) uses --source instead: the source is the quantizer output of the
user's own run (tools/quantize_int8_full.py with the user's calibration images), written as is; its hashes are
recorded and compared with the recipe's (equal only when the calibration samples and the quantizer are the recorded
ones), never required. The calibration metadata then comes from that file's own manifest (quantization.calibration),
the preprocessing from --eval-dir. fetch re-imports the canonical file and requires the result to equal the
quantizer output (data + graph SHA-256).

  .cache/venv-qonnx/bin/python -I tools/export_int8_qonnx.py --model-id ID --file model_int8_full.bitino \
      --source <quantizer output> --eval-dir <eval set> --out model.onnx [--json export.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "python"))

PROFILE = "bitino-qonnx/1"
FILES = {"model_int8_full.bitino": "int8_full", "model_int8_full_log2.bitino": "int8_full_log2"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def recipes() -> dict:
    return json.load(open(os.path.join(ROOT, "registry", "recipes.json")))["recipes"]


def int8_files() -> list:
    """(model id, published int8_full file) of every recipe, in registry order."""
    return [(mid, f) for mid, r in sorted(recipes().items()) for f in r.get("expected", {}) if f in FILES]


def labels_of(em: dict, n_out: int) -> str:
    name = (em or {}).get("name", "")
    for key, n, label_set in (("imagenet", 1000, "imagenet-1k"), ("cifar100", 100, "cifar100"), ("cifar10", 10, "cifar10"),
                              ("glue-mnli", 3, "glue-mnli"), ("glue-", 2, None), ("sst", 2, "glue-sst-2")):
        if key in name and n_out == n:
            return json.dumps({"num_classes": n, "set": label_set or name.split("-eval")[0]})
    if "imagenet" in name and n_out == 1001:
        return json.dumps({"num_classes": 1001, "set": "imagenet-1k + background (index 0)"})
    return json.dumps({"num_classes": n_out, "set": name}) if n_out > 1 else "none"


def model_metadata(mid: str, fn: str, g, cache: str, shapes: dict, eval_dir: str | None = None, from_source: bool = False) -> dict:
    """bitino.* model metadata (docs/qonnx-profile.md § 6) of the canonical int8_full file. from_source: the file is a
    quantizer output of this machine (bitino fetch --int8): its own manifest's calibration, not the recipe's."""
    rec = recipes()[mid]
    ev = rec["eval"]
    em = json.load(open(os.path.join(eval_dir or os.path.join(cache, "data", ev["dir"]), "manifest.json")))
    pre = em["preprocessing"][ev["preprocess"]]
    man = g.manifest or {}
    src = man.get("source", {})
    conv = man.get("conversion", {})
    q = man.get("quantization", {})
    entry = next(e for e in rec.get("int8", []) if e["file"] == fn)
    cal = entry["calibration"]
    if from_source:
        mc = q.get("calibration") or {}
        cal = {"set": mc.get("set") or "unknown", "split": mc.get("split") or "unknown", "n": int(mc.get("n") or 0),
               "tensor_sha256": mc.get("tensor_sha256") or "unknown", "range": mc.get("range", "?"),
               "preprocessing": mc.get("preprocessing", "?")}
    gathers = {n.inputs[1]: g.constants[n.inputs[0]].shape[0] for n in g.nodes
               if n.op == "I8Gather" and len(n.inputs) > 1 and n.inputs[0] in g.constants}
    if len(g.inputs) == 1 and g.inputs[0]["dtype"] == "f32":
        dom = "float32"
    else:
        dom = json.dumps({i["name"]: ("float32" if i["dtype"] == "f32" else f"ids:{gathers[i['name']]}") for i in g.inputs})
    fw_name = src.get("framework") or rec.get("framework") or "unknown"
    vers = [f"{k} {conv[k]}" for k in ("torch", "tensorflow", "larq", "larq_zoo", "onnx", "openvino", "mindspore", "nncf",
                                        "brevitas", "theano", "tensorpack", "transformers", "timm") if conv.get(k)]
    ck = src.get("checkpoint_sha256") or next((d["sha256"] for d in rec.get("downloads", []) if d.get("sha256")), None)
    if isinstance(ck, dict):
        ck = json.dumps(ck)
    n_out = int(shapes[g.outputs[0]][-1])
    meta = {"bitino.profile": PROFILE, "bitino.model_id": mid, "bitino.variant": FILES[fn], "bitino.layout": "NCHW",
            "bitino.input_domain": dom,
            "bitino.preprocessing": json.dumps({"key": ev["preprocess"], "definition": pre["definition"]}),
            "bitino.labels": labels_of(em, n_out),
            "bitino.source.framework": fw_name, "bitino.source.framework_version": ", ".join(vers) or "see the float file",
            "bitino.source.checkpoint_sha256": ck or "0" * 64,
            "bitino.weights_license": (rec.get("licence") or {}).get("weights") or "unknown",
            "bitino.exporter": f"tools/export_int8_qonnx.py@{git_commit()} (writer of the published int8_full graph, "
                               f"bitino.qonnx_int8 group mapping)",
            "bitino.calibration": json.dumps({"set": cal["set"], "split": cal["split"], "n": int(cal["n"]),
                                              "tensor_sha256": cal["tensor_sha256"]}),
            "bitino.notes": (f"{'quantizer output' if from_source else 'published file'} {fn} (quantized {q.get('date', entry.get('date', '?'))}, "
                             f"{q.get('scheme', 'int8_full')[:60]}...; options {json.dumps(entry.get('options', {}), sort_keys=True)}); "
                             f"calibration range {cal.get('range', '?')}, preprocessing {cal.get('preprocessing', '?')}; "
                             f"group mapping (one ONNX group per int8 IR node, bitino.import)")}
    srcs = rec.get("sources") or []
    if srcs and srcs[0].get("repo") and srcs[0].get("commit"):
        meta["bitino.source.code"] = f"{srcs[0]['repo']}@{srcs[0]['commit']}"
    if not ck:
        meta["bitino.notes"] += "; checkpoint SHA-256 unknown in the manifest"
    return meta


def export(mid: str, fn: str, out: str, artifacts: str, cache: str, shapes_kinds=None, source: str | None = None,
           eval_dir: str | None = None) -> dict:
    """Write the canonical file; returns the export record (raises if the published file does not have the recipe's
    hashes or the written file does not pass check-qonnx). source: a quantizer output to write instead of the
    published file (bitino fetch --int8); its hashes are recorded and compared with the recipe's, not required."""
    import onnx

    from bitino import format as bff, qonnx_int8 as qi
    from bitino.fetch import data_sha256, graph_sha256
    from bitino.qonnx_profile import check_model
    pub = source or os.path.join(artifacts, mid, fn)
    exp = recipes()[mid]["expected"][fn]
    hp = {"data_sha256": data_sha256(pub), "graph_sha256": graph_sha256(pub)}
    same = hp["data_sha256"] == exp["data_sha256"] and hp["graph_sha256"] == exp["graph_sha256"]
    if not same and not source:
        raise RuntimeError(f"{mid}/{fn}: the published file does not have the recipe's expected hashes")
    g = bff.read(pub)
    shapes, kinds = shapes_kinds or qi.ir_shapes_kinds(g)
    spec, rep = qi.write_spec_int8(g, shapes, kinds)
    meta = model_metadata(mid, fn, g, cache, shapes, eval_dir, from_source=bool(source))
    spec.meta.update(meta)
    m = qi.build_model(spec, "bitino tools/export_int8_qonnx.py", extra_meta={"bitino.exporter": meta["bitino.exporter"]
                                                                             + f"; onnx {onnx.__version__}"})
    del spec
    cr = check_model(m)
    if cr.errors or cr.warnings:
        raise RuntimeError(f"{mid}/{fn}: the written file does not pass check-qonnx: "
                         + "; ".join(f"{f.code} {f.message[:160]}" for f in cr.errors + cr.warnings))
    onnx.save(m, out)
    rec = {"model_id": mid, "file": fn, ("source" if source else "published"): hp, "writer": rep, "ir_nodes": len(g.nodes), "ir_constants": len(g.constants),
           "onnx_file": os.path.basename(out), "onnx_sha256": sha256_file(out), "onnx_bytes": os.path.getsize(out),
           "onnx_nodes": len(m.graph.node), "onnx": onnx.__version__, "metadata": meta,
           "profile_check": {"passed": True, "summary": cr.summary, "warnings": []}}
    if source:
        rec["source_file"] = os.path.abspath(source)
        rec["source_equals_recipe"] = same
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--file", required=True, choices=sorted(FILES))
    ap.add_argument("--out", required=True, help="canonical QONNX file to write")
    ap.add_argument("--artifacts-dir", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, ".cache"))
    ap.add_argument("--json", help="export record (JSON)")
    ap.add_argument("--source", help="write this quantizer output instead of the published file (bitino fetch --int8)")
    ap.add_argument("--eval-dir", help="eval set for the preprocessing metadata (default <cache>/data/<recipe eval dir>)")
    a = ap.parse_args()
    rec = export(a.model_id, a.file, a.out, a.artifacts_dir, a.cache_dir, source=a.source, eval_dir=a.eval_dir)
    if a.json:
        json.dump(rec, open(a.json, "w"), indent=1, default=str)
    print(f"exported {a.model_id}/{a.file} -> {a.out} ({rec['onnx_bytes']} bytes, sha256 {rec['onnx_sha256'][:16]}, "
          f"{rec['onnx_nodes']} nodes, {rec['writer']['groups']} groups); check-qonnx passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
