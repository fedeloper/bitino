"""BiBERT (Qin et al., ICLR 2022; github.com/htqin/BiBERT) -> Bitino IR.

Input: the official ``transformer.modeling_quant.BertForSequenceClassification``
instantiated from the repository and loaded with the released state_dict. Every
module is mapped explicitly; unmapped parameterized modules raise.

Source semantics (BiBERT commit 014399c), all reproduced exactly:
  * QuantizeLinear (utils_quant.py:219-242), weight_bits=1:
      alpha = mean(|W|, dim=1)  (UNcentered), Wb = alpha * sign(W - mean_row(W))   [sign(0)=0]
      input: sign(x)  (torch.sign, sign(0)=0, no scale / threshold)
      out = linear(sign(x), Wb) + bias
    -> BDense(input_binarize=sign_zero, w=sign bits, post_mul=alpha, post_add=bias,
              w_zero=mask of exact-zero weights when present)
    Wb is computed here with the *same torch fp32 expressions* as the source, so the IR
    holds exactly the binary weights the fp32 source model uses.
  * QuantizeEmbedding (utils_quant.py:259-272): same per-row binarization of the word
    table -> Gather(bit table, ids, scale=alpha, zero_mask).
  * position / token-type embeddings (nn.Embedding), all LayerNorms: fp32.
  * BertSelfAttention.forward (modeling_quant.py:141-184): Q, K = sign(linear + bias);
    scores = Q.K^T / sqrt(64) + mask; softmax removed; probs = ZMeanBinaryQuantizer(scores)
    = 1 if score > 0 else 0 (utils_quant.py:28-35); V = sign(V); context = probs @ V.
    -> BMatMul(sign_zero x sign_zero, scale 1/8), AddBroadcast(mask), BMatMul(gt0 x sign_zero).
  * mask: (1 - m) * -10000 (modeling_quant.py:444-447) -> ChannelAffine(m, 10000, -10000)
    (identical for m in {0,1}).
  * BertIntermediate: gelu(x) = x*0.5*(1+erf(x/sqrt 2)) (modeling_quant.py:50-56, 228-232), fp32.
    The IR evaluates it in double (exact). Optional variant (gelu_zero_below): reproduce the fp32 zero set
    (1 + erf rounds to 0 for x <~ -5.5, so sign(gelu) = 0) with ChannelAffine + Activation(sign_zero) +
    ChannelAffine + Mul (existing ops).
  * BertPooler (modeling_quant.py:300-314): QuantizeLinear on token 0 + tanh.
  * classifier (modeling_quant.py:463): nn.Linear, fp32.
  * Dropout: identity (eval). Unused in forward: ``*.scale`` (0.0), ``*.gate``, clip buffers,
    query/key/value "scores" side outputs (distillation only).
"""
from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

from ..ir import Graph


def load_official(repo_dir: str, ckpt_dir: str, weights_file: str = "pytorch_model.bin"):
    """Imports the official model definition and loads the checkpoint (torch.load weights_only=True)
    through the repository's own BertPreTrainedModel.from_pretrained(state_dict=...)."""
    import torch

    for name in ("matplotlib", "matplotlib.pyplot", "seaborn"):  # imported but unused by the model code
        sys.modules.setdefault(name, types.ModuleType(name))
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from transformer.modeling_quant import BertForSequenceClassification  # noqa: E402

    import json

    sd = torch.load(os.path.join(ckpt_dir, weights_file), map_location="cpu", weights_only=True)
    # num_labels from the checkpoint's config.json (quant_task_glue.py: len(processor.get_labels()); MNLI = 3)
    num_labels = int(json.load(open(os.path.join(ckpt_dir, "config.json"))).get("num_labels", 2))
    model = BertForSequenceClassification.from_pretrained(ckpt_dir, state_dict=sd, num_labels=num_labels)
    model.eval()
    return model


def binarize_like_source(weight):
    """utils_quant.py:221-224 in the weight's own dtype: returns (alpha[O], sign[O,K]) as numpy."""
    import torch

    w = weight.detach()
    scaling_factor = torch.mean(abs(w), dim=1, keepdim=True)
    real_weights = w - torch.mean(w, dim=-1, keepdim=True)
    s = torch.sign(real_weights)
    return scaling_factor[:, 0].numpy(), s.numpy().astype(np.int8)


def _np(t) -> np.ndarray:
    return t.detach().cpu().numpy()


def _sign_planes(s: np.ndarray):
    """{-1,0,+1} int8 -> (±1 sign array for the bit tensor, zero mask (-1 = zero) or None)."""
    pm = np.where(s < 0, -1, 1).astype(np.int8)
    zeros = s == 0
    return pm, (np.where(zeros, -1, 1).astype(np.int8) if zeros.any() else None)


class _Builder:
    def __init__(self, g: Graph):
        self.g = g
        self.precision: list[dict] = []

    def binary_linear(self, x: str, mod, name: str) -> str:
        assert type(mod).__name__ == "QuantizeLinear", (name, type(mod))
        assert mod.weight_bits == 1 and mod.input_bits == 1 and mod.quantize_act, name
        alpha, s = binarize_like_source(mod.weight)
        pm, zm = _sign_planes(s)
        g = self.g
        ins = [x, g.add_constant(name + ".w_sign", pm, "bit", "OI_bit"), g.add_constant(name + ".alpha", alpha),
               g.add_constant(name + ".bias", _np(mod.bias)), ""]
        if zm is not None:
            ins.append(g.add_constant(name + ".w_zero", zm, "bit", "OI_bit"))
        y = g.add_node("BDense", ins, {"input_binarize": "sign_zero"}, name=name)
        self.precision.append({"layer": name, "type": "QuantizeLinear", "weights": "binary (alpha_row * sign(W - mean_row W))",
                               "activations": "sign (torch.sign, 0 -> 0)", "bias": "fp32", "shape": list(mod.weight.shape),
                               "exact_zero_weights": int((s == 0).sum())})
        return y

    def layernorm(self, x: str, mod, name: str) -> str:
        assert type(mod).__name__ == "LayerNorm" and len(mod.normalized_shape) == 1 and mod.elementwise_affine, name
        y = self.g.add_node("LayerNorm", [x, self.g.add_constant(name + ".weight", _np(mod.weight)),
                                          self.g.add_constant(name + ".bias", _np(mod.bias))], {"eps": float(mod.eps)}, name=name)
        self.precision.append({"layer": name, "type": "LayerNorm", "precision": "fp32", "eps": float(mod.eps)})
        return y


def convert(model, model_id: str, seq_len: int = 64, gelu_zero_below: float | None = None) -> tuple[Graph, dict]:
    """Returns (graph, info). Graph inputs: input_ids i32[1,L], token_type_ids i32[1,L],
    attention_mask f32[1,L] (values 0/1). Output: logits f32[1, num_labels].

    gelu_zero_below: optional float32 threshold t; the FFN GELU output is forced to 0 for x < t, reproducing the
    zero set of the source's float32 GELU (tools/convert_bibert.py --gelu-fp32-zero-set). Default: exact GELU."""
    cfg = model.config
    assert cfg.weight_bits == 1 and cfg.input_bits == 1 and cfg.quantize_act, "only the fully binary (1-bit) BiBERT path"
    assert cfg.hidden_size % cfg.num_attention_heads == 0
    L, Dm, H = seq_len, cfg.hidden_size, cfg.num_attention_heads
    Dh = Dm // H
    assert L <= cfg.max_position_embeddings
    g = Graph()
    b = _Builder(g)
    ids = g.add_input("input_ids", [1, L], "i32")
    tt = g.add_input("token_type_ids", [1, L], "i32")
    am = g.add_input("attention_mask", [1, L], "f32")
    bert = model.bert
    mapped: set[str] = set()

    # ---- embeddings (modeling_quant.py:76-90)
    emb = bert.embeddings
    we = emb.word_embeddings
    assert type(we).__name__ == "QuantizeEmbedding" and we.weight_bits == 1
    alpha, s = binarize_like_source(we.weight)
    pm, zm = _sign_planes(s)
    w_in = [g.add_constant("bert.embeddings.word_embeddings.w_sign", pm, "bit", "VD_bit"), ids,
            g.add_constant("bert.embeddings.word_embeddings.alpha", alpha),
            g.add_constant("bert.embeddings.word_embeddings.w_zero", zm, "bit", "VD_bit") if zm is not None else ""]
    x_w = g.add_node("Gather", w_in, name="bert.embeddings.word_embeddings")
    b.precision.append({"layer": "bert.embeddings.word_embeddings", "type": "QuantizeEmbedding",
                        "weights": "binary per vocab row (alpha_row * sign(W - mean_row W))", "shape": list(we.weight.shape),
                        "exact_zero_weights": int((s == 0).sum())})
    pos_ids = g.add_constant("bert.embeddings.position_ids", np.arange(L, dtype=np.int32)[None], "i32")
    x_p = g.add_node("Gather", [g.add_constant("bert.embeddings.position_embeddings.weight", _np(emb.position_embeddings.weight)), pos_ids],
                     name="bert.embeddings.position_embeddings")
    x_t = g.add_node("Gather", [g.add_constant("bert.embeddings.token_type_embeddings.weight", _np(emb.token_type_embeddings.weight)), tt],
                     name="bert.embeddings.token_type_embeddings")
    b.precision += [{"layer": "bert.embeddings.position_embeddings", "type": "nn.Embedding", "precision": "fp32"},
                    {"layer": "bert.embeddings.token_type_embeddings", "type": "nn.Embedding", "precision": "fp32"}]
    # words + position + token_type, left to right as in modeling_quant.py:87
    e = g.add_node("Add", [g.add_node("Add", [x_w, x_p], name="bert.embeddings/add_pos"), x_t], name="bert.embeddings/add_type")
    h = b.layernorm(e, emb.LayerNorm, "bert.embeddings.LayerNorm")
    mapped |= {"bert.embeddings", "bert.embeddings.word_embeddings", "bert.embeddings.position_embeddings",
               "bert.embeddings.token_type_embeddings", "bert.embeddings.LayerNorm"}

    # ---- additive mask (modeling_quant.py:444-447): (1 - m) * -10000 == m*10000 - 10000 for m in {0,1}
    m = g.add_node("ChannelAffine", [am, g.add_constant("attention_mask.scale", np.full(L, 10000.0, np.float32)),
                                     g.add_constant("attention_mask.bias", np.full(L, -10000.0, np.float32))], name="bert/extended_mask")
    mask4 = g.add_node("Reshape", [m], {"shape": [1, 1, 1, L]}, name="bert/extended_mask4")

    def heads(x: str, name: str) -> str:
        r = g.add_node("Reshape", [x], {"shape": [1, L, H, Dh]}, name=name + "/split")
        return g.add_node("Transpose", [r], {"perm": [0, 2, 1, 3]}, name=name + "/heads")

    for li, layer in enumerate(bert.encoder.layer):
        p = f"bert.encoder.layer.{li}"
        sa = layer.attention.self
        assert sa.num_attention_heads == H and sa.attention_head_size == Dh and sa.quantize_act and sa.input_bits == 1
        q = heads(b.binary_linear(h, sa.query, p + ".attention.self.query"), p + ".attention.self.query")
        k = heads(b.binary_linear(h, sa.key, p + ".attention.self.key"), p + ".attention.self.key")
        v = heads(b.binary_linear(h, sa.value, p + ".attention.self.value"), p + ".attention.self.value")
        sc = g.add_node("BMatMul", [q, k], {"transpose_b": 1, "a_binarize": "sign_zero", "b_binarize": "sign_zero",
                                            "scale": 1.0 / math.sqrt(Dh)}, name=p + ".attention.self/scores")
        sc = g.add_node("AddBroadcast", [sc, mask4], name=p + ".attention.self/masked_scores")
        ctx = g.add_node("BMatMul", [sc, v], {"transpose_b": 0, "a_binarize": "gt0", "b_binarize": "sign_zero", "scale": 1.0},
                         name=p + ".attention.self/context")
        ctx = g.add_node("Transpose", [ctx], {"perm": [0, 2, 1, 3]}, name=p + ".attention.self/merge")
        ctx = g.add_node("Reshape", [ctx], {"shape": [1, L, Dm]}, name=p + ".attention.self/context_flat")
        b.precision.append({"layer": p + ".attention.self", "type": "BiBERT attention",
                            "scores": "sign(Q)·sign(K)^T / sqrt(d) + mask (exact)", "softmax": "removed",
                            "probs": "Heaviside {0,1}: score > 0", "values": "sign(V) (0 -> 0)", "context": "exact integer"})
        o = b.binary_linear(ctx, layer.attention.output.dense, p + ".attention.output.dense")
        a1 = b.layernorm(g.add_node("Add", [o, h], name=p + ".attention.output/residual"), layer.attention.output.LayerNorm,
                         p + ".attention.output.LayerNorm")
        f1 = b.binary_linear(a1, layer.intermediate.dense, p + ".intermediate.dense")
        x1 = f1
        f1 = g.add_node("Activation", [f1], {"type": "gelu"}, name=p + ".intermediate/gelu")
        b.precision.append({"layer": p + ".intermediate/gelu", "type": "gelu (erf)", "precision": "fp32 source; IR evaluates in double"})
        if gelu_zero_below is not None:
            # fp32 GELU zero set: the source's fp32 gelu returns exactly 0 for x < gelu_zero_below (1 + erf rounds to 0),
            # so sign(gelu) = 0 there. keep = (sign_zero(x - t) + 1) / 2 is 1 for x > t, 1/2 at x == t, 0 below; the
            # product keeps the sign of gelu(x) for x >= t and gives (+/-)0 below. x - t is exact near t (Sterbenz).
            Fd = int(layer.intermediate.dense.weight.shape[0])
            t = g.add_node("ChannelAffine", [x1, "", g.add_constant(p + ".intermediate/gelu_zero.shift",
                                                                    np.full(Fd, -gelu_zero_below, np.float32))],
                           name=p + ".intermediate/gelu_zero/shift")
            s = g.add_node("Activation", [t], {"type": "sign_zero"}, name=p + ".intermediate/gelu_zero/sign")
            keep = g.add_node("ChannelAffine", [s, g.add_constant(p + ".intermediate/gelu_zero.half", np.full(Fd, 0.5, np.float32)),
                                                g.add_constant(p + ".intermediate/gelu_zero.half_b", np.full(Fd, 0.5, np.float32))],
                              name=p + ".intermediate/gelu_zero/keep")
            f1 = g.add_node("Mul", [f1, keep], name=p + ".intermediate/gelu_zero/mul")
        f2 = b.binary_linear(f1, layer.output.dense, p + ".output.dense")
        h = b.layernorm(g.add_node("Add", [f2, a1], name=p + ".output/residual"), layer.output.LayerNorm, p + ".output.LayerNorm")
        mapped |= {p, p + ".attention", p + ".attention.self", p + ".attention.self.query", p + ".attention.self.key",
                   p + ".attention.self.value", p + ".attention.self.dropout", p + ".attention.output",
                   p + ".attention.output.dense", p + ".attention.output.LayerNorm", p + ".attention.output.dropout",
                   p + ".intermediate", p + ".intermediate.dense", p + ".output", p + ".output.dense", p + ".output.LayerNorm",
                   p + ".output.dropout"}

    # ---- pooler: hidden[:, 0] -> QuantizeLinear -> tanh (modeling_quant.py:309-314)
    flat = g.add_node("Reshape", [h], {"shape": [L, Dm]}, name="bert.pooler/flat")
    cls = g.add_node("Gather", [flat, g.add_constant("bert.pooler.cls_index", np.zeros(1, np.int32), "i32")], name="bert.pooler/cls")
    pooled = b.binary_linear(cls, bert.pooler.dense, "bert.pooler.dense")
    pooled = g.add_node("Activation", [pooled], {"type": "tanh"}, name="bert.pooler/tanh")
    # ---- classifier: nn.Linear fp32 (modeling_quant.py:463)
    clf = model.classifier
    assert type(clf).__name__ == "Linear"
    logits = g.add_node("Dense", [pooled, g.add_constant("classifier.weight", _np(clf.weight)), g.add_constant("classifier.bias", _np(clf.bias))],
                        name="classifier")
    b.precision.append({"layer": "classifier", "type": "nn.Linear", "precision": "fp32", "shape": list(clf.weight.shape)})
    g.set_outputs(logits)
    mapped |= {"", "bert", "bert.encoder", "bert.encoder.layer", "bert.pooler", "bert.pooler.dense", "bert.pooler.activation",
               "classifier", "dropout", "bert.embeddings.dropout"}

    unmapped = [n for n, _ in model.named_modules() if n not in mapped]
    if unmapped:
        raise ValueError(f"unmapped modules: {unmapped[:10]}")
    info = {"layer_precision": b.precision, "seq_len": L, "num_labels": int(clf.weight.shape[0]),
            "exact_zero_weights_total": int(sum(p.get("exact_zero_weights", 0) for p in b.precision)),
            "unused_parameters": "*.scale (0.0, only set by reset_scale), *.gate, weight/act clip buffers, clip_query/key/value/attn",
            "gelu": "exact (double)" if gelu_zero_below is None else f"fp32 zero set: 0 for x < {gelu_zero_below!r}"}
    return g, info
