"""BiBERT adapter tests on a tiny randomly initialized official model (htqin/BiBERT modeling_quant).

Needs the BiBERT clone (BITINO_BIBERT_REPO or <main checkout>/.cache/repos/BiBERT); skipped otherwise.
Checks the float64 semantics of the converted IR against the float64 source with the fp32-derived binary
weights (tools/convert_bibert.py fp64_fixed_binary), for the exact GELU and for the fp32-GELU-zero-set variant
(gelu forced to 0 below the float32 underflow threshold, the semantics of the fp32 source).
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
REPO = os.environ.get("BITINO_BIBERT_REPO", os.path.join(MAIN, ".cache", "repos", "BiBERT"))

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "transformer", "modeling_quant.py")),
                                reason="BiBERT clone not available")

from bitino import executor  # noqa: E402
from bitino.adapters import torch_bibert  # noqa: E402

T_ZERO = -5.546368598937988   # fp32 gelu underflow threshold of torch on CPU (tools/convert_bibert.py)


def _tiny_model(tmp_path, seed=0):
    for name in ("matplotlib", "matplotlib.pyplot", "seaborn"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    # Other tests may have imported a different top-level "transformer" package (e.g. another BERT clone).
    for name in [n for n in sys.modules if n == "transformer" or n.startswith("transformer.")]:
        if not (getattr(sys.modules[name], "__file__", "") or "").startswith(REPO):
            del sys.modules[name]
    sys.dont_write_bytecode = True
    from transformer.modeling_quant import BertForSequenceClassification
    from transformer.configuration import BertConfig

    cfg = {"attention_probs_dropout_prob": 0.1, "clip_val": 2.5, "embedding_bits": 1, "hidden_act": "gelu",
           "hidden_dropout_prob": 0.1, "hidden_size": 32, "initializer_range": 0.02, "input_bits": 1, "intermediate_size": 64,
           "layer_norm_eps": 1e-12, "max_position_embeddings": 16, "num_attention_heads": 2, "num_hidden_layers": 2,
           "num_labels": 3, "output_attentions": True, "output_hidden_states": True, "output_intermediate": True,
           "quantize_act": True, "type_vocab_size": 2, "vocab_size": 50, "weight_bits": 1}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    torch.manual_seed(seed)
    model = BertForSequenceClassification(BertConfig.from_json_file(str(p)), num_labels=3)
    with torch.no_grad():
        for n, w in model.named_parameters():
            if "LayerNorm" not in n:
                w.copy_(torch.randn_like(w) * 0.5)
        for layer in model.bert.encoder.layer:   # push part of the FFN pre-activations below the underflow threshold
            layer.intermediate.dense.bias.add_(-4.0)
    return model.eval()


def _inputs(L=8, seed=1):
    rng = np.random.default_rng(seed)
    ids = rng.integers(1, 50, (1, L)).astype(np.int32)
    tt = np.r_[np.zeros(L // 2), np.ones(L - L // 2)].astype(np.int32)[None]
    mask = np.ones((1, L), np.float32)
    mask[0, -2:] = 0
    return ids, tt, mask


@pytest.mark.parametrize("variant", ["exact", "fp32_zero_set"])
def test_tiny_bibert_float64_semantics(tmp_path, variant):
    from convert_bibert import fp64_fixed_binary, torch_logits

    model = _tiny_model(tmp_path)
    import transformer.modeling_quant as mq
    ids, tt, mask = _inputs()
    tz = T_ZERO if variant == "fp32_zero_set" else None
    g, info = torch_bibert.convert(model, "tiny", seq_len=ids.shape[1], gelu_zero_below=tz)
    assert g.op_counts().get("Mul", 0) == (2 if tz is not None else 0)
    orig = mq.gelu
    hits = {"below": 0}

    def gelu_ref(x):
        if tz is None or x.dtype != torch.float64:
            return orig(x)
        hits["below"] += int((x < tz).sum())
        return torch.where(x >= tz, orig(x), torch.zeros_like(x))
    mq.gelu = gelu_ref
    try:
        ref = torch_logits(fp64_fixed_binary(model), ids, tt, mask)
    finally:
        mq.gelu = orig
    out = executor.run(g, {"input_ids": ids, "token_type_ids": tt, "attention_mask": mask}, float_dtype=np.float64)[g.outputs[0]]
    assert np.max(np.abs(out.reshape(ref.shape) - ref)) < 1e-9
    if tz is not None:
        assert hits["below"] > 0   # the zero set is exercised


def test_fp32_gelu_zero_threshold_matches_torch():
    from convert_bibert import fp32_gelu_zero_threshold

    def gelu(x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    z = fp32_gelu_zero_threshold(gelu)
    t = np.float32(z["t"])
    x = torch.tensor([t, np.nextafter(t, np.float32(0)), np.nextafter(t, np.float32(-1e9))], dtype=torch.float32)
    g = gelu(x)
    assert g[0] != 0 and g[1] != 0 and g[2] == 0
