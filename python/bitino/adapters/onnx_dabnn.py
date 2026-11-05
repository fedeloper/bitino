"""Adapter: daBNN-style Bi-Real ONNX (binary convs stored as all-+-1 float Conv weights) -> Bitino IR.

Source files: JDAI-CV/dabnn model zoo `birealnet18.onnx` (ir 3, opset 9, pytorch 0.4) and
`birealnet18stem.onnx` (ir 3, opset 6, pytorch 0.3). The ONNX graphs contain no Sign op: daBNN's
converter (tools/onnx2bnn, "aggressive" mode, the default) turns every Conv whose weights are all
exactly +-1 into a binary convolution whose *input* is binarized implicitly (docs/model_conversion.md).
This adapter applies the same rule and raises on any op, attribute or layout it does not know.

Semantics (daBNN runtime, verified in the source at .cache/repos/dabnn):
  * BinConv input binarization: aarch64 `pack_128_opt` packs the IEEE sign bit and inverts it
    (dabnn/bitpack.h): x > 0 and +0.0 -> +1, x < 0 and -0.0 -> -1. The portable `pack_64_bitfield`
    (common/common_bitpack.h) uses x > 0 (0 -> -1). Bitino `ge` (x >= 0 -> +1) equals the aarch64
    path for every value except -0.0; `gt` equals the portable path. The converter counts exact zeros
    and -0.0 at every binarization input, so the choice is checked to be immaterial (or reported).
  * BinConv padding: the packed input is padded with bit 0 = -1 (dabnn/pad.h; im2col path fills
    float 0, which pack maps to bit 0 as well) -> Bitino `pad_value = -1` (default). The training
    framework of these weights is undocumented (the official Bi-Real code zero-pads); `pad_value = 0`
    is available as an option for the comparison.
  * Weights: +-1, no scale (the per-channel scale is folded into the following BatchNormalization).
  * Everything else is float: stem Conv(s), 1x1 downsample Convs, BatchNormalization (opset 6/9
    inference formula (x - mean) / sqrt(var + eps) * gamma + beta), Add, Relu, MaxPool (padding acts
    as -inf), AveragePool (no padding), Concat, final AveragePool over the full 7x7 map and a 1x1
    Conv with bias as the classifier.

Mapping: NCHW -> NHWC, OIHW -> OHWI; BN -> ChannelAffine (scale/shift folded in float64, rounded once
to float32; the float64 values are returned in info["float64_constants"] for the Level-3 check);
global AveragePool -> GlobalAvgPool, classifier 1x1 Conv on a 1x1 map -> Reshape-free Dense on [N, C];
Pad nodes must be all-zero (they are no-ops in these files) and are dropped.
"""
from __future__ import annotations

import numpy as np

from ..ir import Graph

KNOWN_OPS = {"Conv", "BatchNormalization", "Add", "Pad", "AveragePool", "MaxPool", "Relu", "Concat"}


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise NotImplementedError(msg)


def _attrs(node) -> dict:
    from onnx import helper

    out = {}
    for a in node.attribute:
        v = helper.get_attribute_value(a)
        out[a.name] = v.decode() if isinstance(v, bytes) else v
    return out


def _pads_tblr(p) -> list[int]:
    """ONNX 2-D pads [top, left, bottom, right] -> Bitino [top, bottom, left, right]."""
    p = list(p) if p else [0, 0, 0, 0]
    _expect(len(p) == 4, f"unsupported pads {p}")
    return [int(p[0]), int(p[2]), int(p[1]), int(p[3])]


def is_binary_weight(w: np.ndarray) -> bool:
    return bool(w.ndim == 4 and np.all((w == 1.0) | (w == -1.0)))


def binary_convs(model) -> list[str]:
    """Output names of the Convs onnx2bnn (aggressive mode) treats as binary, in graph order."""
    from onnx import numpy_helper

    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    return [n.output[0] for n in model.graph.node if n.op_type == "Conv" and n.input[1] in init and is_binary_weight(init[n.input[1]])]


def convert(model, binarize: str = "ge", pad_value: int = -1, input_name: str = "input") -> tuple[Graph, dict]:
    """model: onnx.ModelProto (original daBNN file). Returns (graph, info)."""
    from onnx import numpy_helper

    _expect(binarize in ("ge", "gt"), f"binarize {binarize}")
    _expect(pad_value in (-1, 0), f"pad_value {pad_value}")
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    gin = [i for i in model.graph.input if i.name not in init]
    _expect(len(gin) == 1 and len(model.graph.output) == 1, "expected one graph input and one output")
    shp = [d.dim_value for d in gin[0].type.tensor_type.shape.dim]
    _expect(len(shp) == 4 and shp[0] == 1, f"input shape {shp}")
    g = Graph()
    x = g.add_input(input_name, [1, shp[2], shp[3], shp[1]])
    env = {gin[0].name: x}              # ONNX tensor -> IR tensor
    shape = {gin[0].name: [1, shp[1], shp[2], shp[3]]}   # NCHW, or [N, C] after global pooling
    precision, exact = [], {}
    used_init = set()

    for k, node in enumerate(model.graph.node):
        op, a = node.op_type, _attrs(node)
        _expect(op in KNOWN_OPS, f"node {k}: unsupported op {op}")
        out = node.output[0]
        name = f"{op.lower()}_{out}"
        src = node.input[0]
        _expect(src in env, f"node {k} ({op}): input {src} not produced yet")
        s = shape[src]
        if op == "Conv":
            w = init[node.input[1]]
            used_init.add(node.input[1])
            b = init[node.input[2]] if len(node.input) > 2 and node.input[2] else None
            if b is not None:
                used_init.add(node.input[2])
            _expect(a.get("group", 1) == 1 and list(a.get("dilations", [1, 1])) == [1, 1], f"{name}: groups/dilation")
            o, ci, kh, kw = w.shape
            _expect(list(a.get("kernel_shape", [kh, kw])) == [kh, kw], f"{name}: kernel_shape")
            st = [int(v) for v in a.get("strides", [1, 1])]
            pads = _pads_tblr(a.get("pads"))
            if len(s) == 2:   # classifier: 1x1 conv on the globally pooled [N, C] vector
                _expect(kh == kw == 1 and pads == [0, 0, 0, 0] and not is_binary_weight(w), f"{name}: expected fp 1x1 classifier")
                y = g.add_node("Dense", [env[src], g.add_constant(f"{name}/kernel", w.reshape(o, ci).astype(np.float32)),
                                         g.add_constant(f"{name}/bias", b.astype(np.float32)) if b is not None else ""],
                               {"activation": "none"}, name=name)
                shape[out] = [s[0], o]
                precision.append({"layer": name, "op": "Dense", "weights": "fp32", "activations": "fp32", "shape": [o, ci]})
            else:
                _expect(s[1] == ci, f"{name}: channel mismatch {s} vs {w.shape}")
                oh = (s[2] + pads[0] + pads[1] - kh) // st[0] + 1
                ow = (s[3] + pads[2] + pads[3] - kw) // st[1] + 1
                attrs = {"strides": st, "dilations": [1, 1], "pads": pads, "groups": 1, "activation": "none"}
                wt = np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1)))
                if is_binary_weight(w):
                    _expect(b is None, f"{name}: binary conv with bias")
                    attrs.update({"pad_value": pad_value, "input_binarize": binarize, "output": "f32"})
                    y = g.add_node("BConv2D", [env[src], g.add_constant(f"{name}/kernel_bits", wt.astype(np.int8), dtype="bit", layout="OHWI_bit"),
                                               "", "", ""], attrs, name=name)
                    precision.append({"layer": name, "op": "BConv2D", "weights": "binary +-1 (no scale)",
                                      "activations": f"binary ({binarize}), pad_value {pad_value}", "kernel": [kh, kw], "strides": st, "pads": pads})
                else:
                    attrs["pad_value"] = 0.0
                    y = g.add_node("Conv2D", [env[src], g.add_constant(f"{name}/kernel", wt.astype(np.float32), layout="OHWI"),
                                              g.add_constant(f"{name}/bias", b.astype(np.float32)) if b is not None else ""], attrs, name=name)
                    precision.append({"layer": name, "op": "Conv2D", "weights": "fp32", "activations": "fp32", "kernel": [kh, kw], "strides": st, "pads": pads})
                shape[out] = [s[0], o, oh, ow]
        elif op == "BatchNormalization":
            gm, bt, mu, var = (init[i].astype(np.float64) for i in node.input[1:5])
            used_init.update(node.input[1:5])
            _expect(len(node.output) == 1 or all(not o for o in node.output[1:]), f"{name}: training-mode outputs")
            _expect(a.get("is_test", 1) == 1 and int(a.get("spatial", 1)) == 1, f"{name}: only inference/spatial BN")
            eps = float(a.get("epsilon", 1e-5))
            # ONNX evaluates eps in float32 (attribute is a float); fold with that exact value.
            sc = gm / np.sqrt(var + np.float64(np.float32(eps)))
            sh = bt - mu * sc
            exact[f"{name}/scale"], exact[f"{name}/shift"] = sc, sh
            y = g.add_node("ChannelAffine", [env[src], g.add_constant(f"{name}/scale", sc.astype(np.float32)),
                                             g.add_constant(f"{name}/shift", sh.astype(np.float32))], name=name)
            shape[out] = s
        elif op == "Add":
            o2 = node.input[1]
            _expect(o2 in env and shape[o2] == s, f"{name}: Add of two activations with equal shapes expected")
            y = g.add_node("Add", [env[src], env[o2]], {"activation": "none"}, name=name)
            shape[out] = s
        elif op == "Relu":
            y = g.add_node("Activation", [env[src]], {"type": "relu"}, name=name)
            shape[out] = s
        elif op == "Pad":
            _expect(a.get("mode", "constant") == "constant" and not any(a.get("pads", [])) and len(node.input) == 1,
                    f"{name}: only all-zero (no-op) Pad supported")
            env[out], shape[out] = env[src], s
            continue
        elif op in ("MaxPool", "AveragePool"):
            ks = [int(v) for v in a["kernel_shape"]]
            st = [int(v) for v in a.get("strides", ks)]
            pads = _pads_tblr(a.get("pads"))
            # floor-mode output size (ceil_mode 0) in ONNX and in Bitino's pool geometry
            _expect(a.get("auto_pad", "NOTSET") in ("NOTSET", b"NOTSET") and not a.get("ceil_mode", 0), f"{name}: auto_pad/ceil_mode")
            if op == "AveragePool" and ks == s[2:] and pads == [0, 0, 0, 0]:
                y = g.add_node("GlobalAvgPool", [env[src]], {"keepdims": 0}, name=name)
                shape[out] = [s[0], s[1]]
            else:
                if op == "AveragePool":
                    _expect(pads == [0, 0, 0, 0], f"{name}: padded AveragePool")
                y = g.add_node("MaxPool2D" if op == "MaxPool" else "AvgPool2D", [env[src]],
                               {"kernel": ks, "strides": st, "pads": pads}, name=name)
                shape[out] = [s[0], s[1], (s[2] + pads[0] + pads[1] - ks[0]) // st[0] + 1, (s[3] + pads[2] + pads[3] - ks[1]) // st[1] + 1]
        elif op == "Concat":
            _expect(a.get("axis") == 1, f"{name}: only channel concat")
            for i in node.input:
                _expect(i in env and shape[i][2:] == s[2:], f"{name}: concat input {i}")
            y = g.add_node("Concat", [env[i] for i in node.input], {"axis": -1}, name=name)
            shape[out] = [s[0], sum(shape[i][1] for i in node.input)] + s[2:]
        env[out] = y
    gout = model.graph.output[0].name
    _expect(len(shape[gout]) == 2, "graph output must be the classifier vector")
    g.set_outputs(env[gout])
    unused = sorted(set(init) - used_init)
    info = {"layer_precision": precision, "op_counts": g.op_counts(), "float64_constants": exact,
            "unused_initializers": unused, "binarize": binarize, "pad_value": pad_value,
            "n_binary_convs": g.op_counts().get("BConv2D", 0)}
    return g, info
