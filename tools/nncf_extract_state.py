#!/usr/bin/env python3
"""Extract the tensors-only model state from an NNCF training checkpoint with the restricted unpickler.

The NNCF zoo files (resnet18_imagenet_binarization_{xnor,dorefa}.pth) are training checkpoints that
pickle a ``nncf.api.compression.CompressionStage`` and a whole ``torch.optim.Adam`` object, so a plain
``torch.load(weights_only=True)`` refuses them. This tool keeps the restricted (weights_only) unpickler
and registers *inert stub classes* under those two names with ``torch.serialization.add_safe_globals``
(torch >= 2.4; run with the base env, torch 2.6): no NNCF or torch.optim code runs, the stubs only
swallow their state. The output holds only the model ``state_dict`` (``module.`` / ``nncf_module.``
prefixes stripped) and a JSON string with plain metadata (epoch, best_acc1, compression_state with the
stage as an int, source SHA-256); it is loaded later with ``weights_only=True`` by tools/convert_nncf.py.
The derived file goes under .cache/ and is never committed.

  python3 -I tools/nncf_extract_state.py <ckpt.pth> <out.state.pt> [<other ckpt to compare tensor by tensor>]
"""
import collections
import hashlib
import json
import sys

import torch


class _Inert:
    def __init__(self, *a, **k):
        self._args = a

    def __setstate__(self, st):
        self._state = st

    def __reduce_ex__(self, p):
        raise RuntimeError("inert stub")


def stub(module, name):
    cls = type(name, (_Inert,), {"__module__": module})
    return cls


class CompressionStage(int):
    pass


CompressionStage.__module__ = "nncf.api.compression"
CompressionStage.__qualname__ = "CompressionStage"
Adam = stub("torch.optim.adam", "Adam")
torch.serialization.add_safe_globals([CompressionStage, Adam, collections.defaultdict, dict])


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # report the exact refusal
        raise SystemExit(f"weights_only load failed: {e}")


def plain(o):
    if isinstance(o, CompressionStage):
        return int(o)
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() < 16 else f"<tensor {list(o.shape)}>"
    return f"<{type(o).__name__}>"


def strip(sd):
    out = collections.OrderedDict()
    for k, v in sd.items():
        k2 = k[len("module."):] if k.startswith("module.") else k
        k2 = k2[len("nncf_module."):] if k2.startswith("nncf_module.") else k2
        out[k2] = v
    return out


def main():
    ck = load(sys.argv[1])
    sd = strip(ck["state_dict"])
    h = hashlib.sha256()
    with open(sys.argv[1], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    meta = {"source_file": sys.argv[1].rsplit("/", 1)[-1], "source_sha256": h.hexdigest(),
            "torch_used_for_extraction": torch.__version__,
            "top_keys": sorted(ck.keys()), "epoch": ck.get("epoch"), "arch": ck.get("arch"),
            "best_acc1": float(ck["best_acc1"]) if "best_acc1" in ck else None,
            "compression_state": plain(ck.get("compression_state")),
            "n_state_dict": len(sd), "key_prefix_example": next(iter(ck["state_dict"].keys()))}
    torch.save({"state_dict": sd, "meta_json": json.dumps(meta)}, sys.argv[2])
    print(json.dumps({k: meta[k] for k in ("top_keys", "epoch", "arch", "best_acc1", "n_state_dict", "key_prefix_example")}))
    print("compression_state:", json.dumps(meta["compression_state"])[:2000])
    if len(sys.argv) > 3:
        ck2 = load(sys.argv[3])
        sd2 = strip(ck2["state_dict"])
        same_keys = list(sd.keys()) == list(sd2.keys())
        neq = [k for k in sd if k not in sd2 or not torch.equal(sd[k], sd2[k])]
        o1, o2 = ck.get("original_model_state_dict", {}), ck2.get("original_model_state_dict", {})
        o1, o2 = strip(o1), strip(o2)
        oneq = [k for k in o1 if k not in o2 or not torch.equal(o1[k], o2[k])]
        print(json.dumps({"compare_with": sys.argv[3], "same_keys_after_strip": same_keys, "n_state_dict_other": len(sd2),
                          "state_dict_tensors_not_equal": neq, "n_original_model_state_dict": len(o1),
                          "original_not_equal": oneq, "best_acc1_other": float(ck2.get("best_acc1", float("nan"))),
                          "epoch_other": ck2.get("epoch"),
                          "compression_state_equal": plain(ck.get("compression_state")) == plain(ck2.get("compression_state"))}))


if __name__ == "__main__":
    main()
