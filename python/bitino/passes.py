"""Semantics-preserving graph rewrites applied at conversion time.

Each pass returns the number of rewrites; the converter records applied passes
in the model manifest so fused and unfused variants can be benchmarked and
validated separately (benchmark stage "fusion").
"""
from __future__ import annotations

import numpy as np

from .ir import Graph, Node


def _consumers(g: Graph) -> dict[str, list]:
    cons: dict[str, list] = {}
    for n in g.nodes:
        for i in n.inputs:
            if i:
                cons.setdefault(i, []).append(n)
    for o in g.outputs:
        cons.setdefault(o, []).append(None)
    return cons


def _const(g: Graph, name: str):
    return g.constants[name].data if name else None


def _replace_uses(g: Graph, old: str, new: str) -> None:
    for n in g.nodes:
        n.inputs = [new if i == old else i for i in n.inputs]
    g.outputs = [new if o == old else o for o in g.outputs]


def fold_affine_into_conv(g: Graph) -> int:
    """Conv2D(act=none) -> ChannelAffine  ==>  Conv2D with scaled weights/bias.
    BConv2D(act=none) -> ChannelAffine    ==>  BConv2D with updated post_mul/post_add.
    QBConv2D / QBDense(act=none) -> ChannelAffine  ==>  same, post_mul/post_add updated.
    """
    count = 0
    changed = True
    while changed:
        changed = False
        cons = _consumers(g)
        for node in list(g.nodes):
            if node.op != "ChannelAffine":
                continue
            src_name = node.inputs[0]
            prod = next((p for p in g.nodes if src_name in p.outputs), None)
            if prod is None or prod.op not in ("Conv2D", "BConv2D", "QBConv2D", "QBDense") or len(cons.get(src_name, [])) != 1:
                continue
            if prod.attrs.get("activation", "none") != "none" or prod.attrs.get("output", "f32") != "f32":
                continue
            scale = _const(g, node.inputs[1]) if len(node.inputs) > 1 else None
            shift = _const(g, node.inputs[2]) if len(node.inputs) > 2 else None
            oc = (g.constants[prod.inputs[1]].shape)[0]
            scale = np.ones(oc, np.float32) if scale is None else scale.astype(np.float32)
            shift = np.zeros(oc, np.float32) if shift is None else shift.astype(np.float32)
            tag = g.unique(prod.name + "/folded")
            if prod.op == "Conv2D":
                w = g.constants[prod.inputs[1]].data
                b = _const(g, prod.inputs[2]) if len(prod.inputs) > 2 else None
                new_w = (w.astype(np.float64) * scale.astype(np.float64)[:, None, None, None]).astype(np.float32)
                new_b = ((0.0 if b is None else b.astype(np.float64)) * scale + shift).astype(np.float32)
                prod.inputs = [prod.inputs[0], g.add_constant(tag + "_w", new_w, layout="OHWI"), g.add_constant(tag + "_b", new_b)]
            else:
                mul = _const(g, prod.inputs[2])
                add = _const(g, prod.inputs[3])
                mul = np.ones(oc, np.float32) if mul is None else mul
                add = np.zeros(oc, np.float32) if add is None else add
                new_mul = (mul.astype(np.float64) * scale).astype(np.float32)
                new_add = (add.astype(np.float64) * scale + shift).astype(np.float32)
                new_inputs = [prod.inputs[0], prod.inputs[1], g.add_constant(tag + "_mul", new_mul),
                              g.add_constant(tag + "_add", new_add)]
                if prod.op == "BConv2D":  # QBConv2D / QBDense take no threshold input
                    new_inputs.append(prod.inputs[4] if len(prod.inputs) > 4 else "")
                prod.inputs = new_inputs
            _replace_uses(g, node.outputs[0], src_name)
            g.nodes.remove(node)
            count += 1
            changed = True
            break
    return count


def fold_relu_into_producer(g: Graph) -> int:
    """Conv2D/BConv2D/Add(act=none) -> Activation(relu)  ==> fused activation attr."""
    count = 0
    cons = _consumers(g)
    for node in list(g.nodes):
        if node.op != "Activation" or node.attrs.get("type") not in ("relu", "relu6"):
            continue
        src = node.inputs[0]
        prod = next((p for p in g.nodes if src in p.outputs), None)
        if prod is None or prod.op not in ("Conv2D", "BConv2D", "Add", "Dense") or len(cons.get(src, [])) != 1:
            continue
        if prod.attrs.get("activation", "none") != "none" or prod.attrs.get("output", "f32") != "f32":
            continue
        prod.attrs["activation"] = node.attrs["type"]
        _replace_uses(g, node.outputs[0], src)
        g.nodes.remove(node)
        cons = _consumers(g)
        count += 1
    return count


_SIGN_MODES = ("sign_zero", "ge", "gt")


def _sign_only_uses(g: Graph, t: str, cons: dict, producers: dict) -> list | None:
    """The uses of tensor t if every one of them depends on t only through the sign of its values
    (sign(t), and whether an element is exactly zero), else None. Accepted uses:
      * BConv2D / BDense input 0 with input_binarize ge / gt / sign_zero and no threshold;
      * BMatMul operand a or b with mode ge / gt / sign_zero / gt0;
      * Activation sign / sign_zero;
      * Activation hardtanh with min < 0 < max (sign-preserving clip, zero stays zero) and
        Mul(t, k) with k >= 0 by construction (see _nonnegative), whose own uses are accepted
        recursively (they are returned as uses to rewire, the clip / Mul itself stays).
    """
    uses = []
    for n in cons.get(t, []):
        if n is None:  # graph output
            return None
        slots = [i for i, x in enumerate(n.inputs) if x == t]
        if n.op in ("BConv2D", "BDense"):
            if slots != [0] or n.attrs.get("input_binarize", "ge") not in _SIGN_MODES:
                return None
            if len(n.inputs) > 4 and n.inputs[4]:
                return None
        elif n.op == "BMatMul":
            for i in slots:
                if i > 1 or n.attrs.get("ab"[i] + "_binarize", "sign_zero") not in _SIGN_MODES + ("gt0",):
                    return None
        elif n.op == "Activation" and n.attrs.get("type") in ("sign", "sign_zero"):
            pass
        elif n.op == "Activation" and n.attrs.get("type") == "hardtanh":
            if float(n.attrs.get("min", -1.0)) >= 0 or float(n.attrs.get("max", 1.0)) <= 0:
                return None
            if _sign_only_uses(g, n.outputs[0], cons, producers) is None:
                return None
        elif n.op == "Mul":
            if slots != [0] and slots != [1]:
                return None
            if not _nonnegative(g, n.inputs[1 - slots[0]], producers):
                return None
            if _sign_only_uses(g, n.outputs[0], cons, producers) is None:
                return None
        else:
            return None
        uses.append(n)
    return uses


def _nonnegative(g: Graph, t: str, producers: dict) -> bool:
    """t >= 0 (or NaN-free zero) for every input, by construction: a constant with no negative entry,
    a relu, or ChannelAffine(s, scale c, bias b) of a sign / sign_zero output s in [-1, 1] with
    b - |c| >= 0 per channel (e.g. the (sign_zero(x - t) + 1) / 2 keep mask of the BiBERT fp32 GELU
    zero-set variant)."""
    if t in g.constants:
        d = g.constants[t]
        return d.dtype == "f32" and bool(np.all(d.data >= 0))
    p = producers.get(t)
    if p is None:
        return False
    if p.op == "Activation" and p.attrs.get("type") in ("relu", "relu6"):
        return True
    if p.op == "ChannelAffine" and len(p.inputs) > 2 and p.inputs[1] in g.constants and p.inputs[2] in g.constants:
        s = producers.get(p.inputs[0])
        if s is not None and s.op == "Activation" and s.attrs.get("type") in ("sign", "sign_zero"):
            c = g.constants[p.inputs[1]].data.astype(np.float64)
            b = g.constants[p.inputs[2]].data.astype(np.float64)
            return bool(np.all(b - np.abs(c) >= 0))
    return False


# Smallest float32 x < 0 at which the IR's double GELU x*0.5*(1+erf(x/sqrt 2)) is not zero: below it 1 + erf
# rounds to 0 in double (glibc and bionic libm), so the GELU is exactly -0 for x <= -8.37438965 (the largest such
# float, kGeluCutoff in runtime/kernels/float_opt/transformer_leaf.h) and non-zero with the sign of x above.
F64_GELU_NONZERO_FROM = float(np.nextafter(np.float32(-8.37438965), np.float32(0)))


def drop_gelu_before_sign(g: Graph, keep_zero_below: float | None = None) -> int:
    """Activation(gelu) whose output is used only through its sign  ==>  the uses read the GELU's input.

    GELU(x) = x * Phi(x) with Phi > 0, so sign(GELU(x)) = sign(x) for every real x (and GELU(x) = 0
    iff x = 0); a clip with min < 0 < max and a product with a non-negative factor keep that. The
    binarization mode of each use is kept, so exact zeros of x still give 0 under sign_zero. The only
    numerical difference is a floating-point artifact of evaluating GELU: 1 + erf(x / sqrt 2) rounds
    to 0 for x <= -8.374 in double (x <= -5.54 for a float32 erf), so the evaluated GELU is (-)0 there
    and sign_zero gives 0 (ge gives +1) where the exact value is -1. Without the GELU the binarization
    sees x itself, i.e. the mathematically exact sign. Not bit-identical: a model variant
    (docs/numerical-correctness.md, "GELU in front of sign binarizations").

    keep_zero_below = t keeps the evaluated GELU's zero set instead: the GELU becomes x * keep with
    keep = (sign_zero(x - t) + 1) / 2 in {0, 1/2, 1} (ChannelAffine -> Activation sign_zero -> ChannelAffine -> Mul,
    the construction of the BiBERT fp32 zero-set variant), which is 0 for x < t and has the sign of x for x >= t.
    With t = F64_GELU_NONZERO_FROM (the IR's double GELU is zero exactly for x < t) every sign binarization
    sees the same value class as from the GELU (also NaN and +-inf: 0, 0, +1), so the model is bit-identical
    and the transcendental GELU is replaced by four exact elementwise ops that fuse into one pass."""
    count = 0
    changed = True
    while changed:
        changed = False
        cons = _consumers(g)
        producers = {o: n for n in g.nodes for o in n.outputs}
        for node in list(g.nodes):
            if node.op != "Activation" or node.attrs.get("type") != "gelu" or not node.inputs or not node.inputs[0]:
                continue
            y, x = node.outputs[0], node.inputs[0]
            uses = _sign_only_uses(g, y, cons, producers)
            if not uses:
                continue
            src = x
            idx = g.nodes.index(node)
            if keep_zero_below is not None:
                c = _channels(g, x, producers)
                if c is None:
                    continue
                tag = g.unique(node.name + "/zero_mask")
                t32 = np.float32(keep_zero_below)
                shift = g.add_constant(tag + "/shift", np.full(c, -t32, np.float32))
                half = g.add_constant(tag + "/half", np.full(c, 0.5, np.float32))
                mask_nodes = []
                for op, ins, attrs, nm in (("ChannelAffine", [x, "", shift], {}, "shift"),
                                           ("Activation", [None], {"type": "sign_zero"}, "sign"),
                                           ("ChannelAffine", [None, half, half], {}, "keep"),
                                           ("Mul", [x, None], {}, "mul")):
                    prev = mask_nodes[-1].outputs[0] if mask_nodes else None
                    ins = [prev if i is None else i for i in ins]
                    mask_nodes.append(Node(op, ins, [f"{tag}/{nm}:out"], attrs, f"{tag}/{nm}"))
                g.nodes[idx:idx + 1] = mask_nodes
                src = mask_nodes[-1].outputs[0]
            else:
                g.nodes.remove(node)
            for n in uses:
                n.inputs = [src if i == y else i for i in n.inputs]
            count += 1
            changed = True
            break
    return count


def _channels(g: Graph, t: str, producers: dict) -> int | None:
    """Last-dimension size of tensor t when it is the output of a BDense / Dense (weights [O, K]) or a constant."""
    if t in g.constants:
        return int(g.constants[t].shape[-1])
    p = producers.get(t)
    if p is not None and p.op in ("BDense", "Dense") and p.inputs[1] in g.constants:
        return int(g.constants[p.inputs[1]].shape[0])
    if p is not None and p.op in ("BConv2D", "Conv2D") and p.inputs[1] in g.constants:
        return int(g.constants[p.inputs[1]].shape[0])
    return None


def prune_unused_constants(g: Graph) -> int:
    used = {i for n in g.nodes for i in n.inputs if i}
    dead = [k for k in g.constants if k not in used]
    for k in dead:
        del g.constants[k]
    return len(dead)


def optimize(g: Graph, fold: bool = True) -> dict[str, int]:
    applied: dict[str, int] = {}
    if fold:
        applied["fold_affine_into_conv"] = fold_affine_into_conv(g)
        applied["fold_relu_into_producer"] = fold_relu_into_producer(g)
    applied["prune_unused_constants"] = prune_unused_constants(g)
    return applied
