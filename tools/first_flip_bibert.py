#!/usr/bin/env python3
"""First binarization disagreement between the BiBERT torch model and its IR.

Walks the binarization sites of the network in execution order and finds the FIRST one
whose binarized value differs between the source (torch, any dtype) and the IR (numpy
executor, any float dtype):

  per layer: query/key/value input sign(h) | sign(Q) | sign(K) | sign(V) | probs = scores>0 |
             sign(context) at attention.output.dense | sign(LN out) at intermediate.dense |
             sign(GELU) at output.dense;   then sign(CLS) at bert.pooler.dense.

At that site it reports the number of flips and |pre-binarization value| at the flipped
positions next to the typical |value|. A flip at |pre| many orders of magnitude below the
typical value is float rounding; a flip at a large |pre| is a conversion bug.
Used by tools/convert_bibert.py (function first_flip) and standalone.
"""
from __future__ import annotations

import numpy as np


def _sites(n_layers: int):
    out = []
    for i in range(n_layers):
        p = f"bert.encoder.layer.{i}"
        out += [(p + ".attention.self.query", "in", "sign"), (p + ".attention.self.query", "out", "sign"),
                (p + ".attention.self.key", "out", "sign"), (p + ".attention.self.value", "out", "sign"),
                (p + ".attention.self", "scores", "gt0"), (p + ".attention.output.dense", "in", "sign"),
                (p + ".intermediate.dense", "in", "sign"), (p + ".output.dense", "in", "sign")]
    out.append(("bert.pooler.dense", "in", "sign"))
    return out


def capture_torch(model, input_ids, token_type_ids, attention_mask) -> dict:
    """Runs the torch model once and records pre-binarization tensors at every site."""
    import torch

    rec: dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "QuantizeLinear":
            def hook(m, inp, out, name=name):
                rec[(name, "in")] = inp[0].detach().double().numpy()
                rec[(name, "out")] = out.detach().double().numpy()
            hooks.append(mod.register_forward_hook(hook))
        elif type(mod).__name__ == "BertSelfAttention":
            def hook2(m, inp, out, name=name):
                rec[(name, "scores")] = out[1].detach().double().numpy()  # scores/8 + mask, before ZMean
            hooks.append(mod.register_forward_hook(hook2))
    with torch.no_grad():
        logits = model(input_ids, token_type_ids, attention_mask)[0]
    for h in hooks:
        h.remove()
    rec["logits"] = logits.detach().double().numpy()
    return rec


def ir_value(graph, env: dict, site) -> np.ndarray:
    name, kind, _ = site
    nodes = {n.name: n for n in graph.nodes}
    if kind == "scores":
        return np.asarray(env[nodes[name + "/masked_scores"].outputs[0]], np.float64)
    n = nodes[name]
    return np.asarray(env[n.inputs[0]] if kind == "in" else env[n.outputs[0]], np.float64)


def first_flip(torch_rec: dict, graph, env: dict, n_layers: int) -> dict:
    for site in _sites(n_layers):
        tv = torch_rec[(site[0], site[1])]
        iv = ir_value(graph, env, site).reshape(tv.shape)
        if site[2] == "gt0":
            tb, ib = tv > 0, iv > 0
        else:
            tb, ib = np.sign(tv), np.sign(iv)
        flips = tb != ib
        if flips.any():
            nz = np.abs(tv[np.abs(tv) < 1e3])  # exclude masked (-10000) scores from the typical value
            return {"first_flip_site": f"{site[0]}:{site[1]}", "binarize": site[2], "n_flips": int(flips.sum()),
                    "n_values": int(flips.size), "max_abs_pre_at_flip_torch": float(np.max(np.abs(tv[flips]))),
                    "max_abs_pre_at_flip_ir": float(np.max(np.abs(iv[flips]))),
                    "torch_values_at_flip": [float(v) for v in tv[flips][:8]], "ir_values_at_flip": [float(v) for v in iv[flips][:8]],
                    "typical_abs_pre": float(np.median(nz)) if nz.size else None,
                    "max_abs_diff_at_site": float(np.max(np.abs(tv - iv)))}
    return {"first_flip_site": None}
