"""Slice S13: BNAS cell networks (adapter torch_bnas, official yonseivnl/bnas code). No new operator.

* BinOp selection: torch_bnas.binop_targets == the official bin_utils.BinOp target list for both source versions
  (first release cdb58f7: preprocess_res convs binarized, stale-variable 'res' test; head 675ecc7: float), incl. the
  counting quirk that leaves the last #preprocess_res non-res convs float, and torch_bnas.xnor_binarize ==
  BinOp.binarization().
* Official NetworkCIFAR (latest_cell_zeroise: Zero ops, all-zero states, dead preprocess1) and NetworkImageNet
  (latest_cell: dilated binary convs, max pools, reduction residual convs) with random weights / BN statistics, small
  widths: IR in float64 with float64 constants == the official float64 forward (sign_zero), and the quirk convs are
  Activation(sign) + Conv2D.
* numpy vs C++ bit-exact on the small converted CIFAR network (reference sign_zero, reference ge, neon ge).
* ImageNet class order permutation (ImageFolder over folders named 0..999).
The tests that need the official code skip when the source trees (.cache/repos/bnas-s13/src-cdb58f7, .../src) are absent.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from bitino import executor, format as bfformat  # noqa: E402
from bitino.adapters import torch_bnas  # noqa: E402

BENCH = os.environ.get("BITINO_BENCH", os.path.join(ROOT, "build", "host", "bitino_bench"))
MAIN = os.environ.get("BITINO_DATA_ROOT", ROOT)
# first release (cdb58f7: the code the checkpoints were trained with) and head (675ecc7: BinOp "typo fix")
BNAS_REPO = os.environ.get("BITINO_BNAS_REPO", os.path.join(MAIN, ".cache", "repos", "bnas-s13", "src-cdb58f7"))
BNAS_REPO_HEAD = os.environ.get("BITINO_BNAS_REPO_HEAD", os.path.join(MAIN, ".cache", "repos", "bnas-s13", "src"))


def test_imagenet_class_perm():
    from convert_bnas import imagenet_class_perm
    perm = imagenet_class_perm()
    names = sorted(str(i) for i in range(1000))
    assert sorted(perm.tolist()) == list(range(1000))
    assert all(int(names[perm[c]]) == c for c in range(1000))
    assert perm[0] == 0 and perm[1] == 1 and perm[10] == 2 and perm[100] == 3 and perm[999] == 999


def _randomize(model):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.3)
                m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
                if m.affine:
                    m.weight.copy_(torch.randn(m.num_features, generator=g))   # negative scales included
                    m.bias.copy_(torch.randn(m.num_features, generator=g) * 0.3)
            elif isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.2)
                if m.bias is not None:
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)


def _source(model_id: str, cfg: dict, repo: str = BNAS_REPO):
    pytest.importorskip("torch")
    if not os.path.exists(os.path.join(repo, "src", "model_imagenet.py")):
        pytest.skip("BNAS source tree not available")
    from convert_bnas import load_source
    return load_source(model_id, repo, None, cfg_override=cfg, prepare=_randomize)


CASES = {
    "cifar": ("bnas-a-cifar10", {"init_channels": 4, "layers": 5}, (32, 32), None),
    "imagenet": ("bnas-d-imagenet", {"init_channels": 8, "layers": 3}, (224, 224), True),
}


@pytest.mark.parametrize("version", ["first_release", "head"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_binop_selection_and_float64_equivalence(case, version):
    torch = pytest.importorskip("torch")
    from convert_bnas import imagenet_class_perm, run_reference
    model_id, cfg, hw, use_perm = CASES[case]
    model, targets, info = _source(model_id, cfg, BNAS_REPO if version == "first_release" else BNAS_REPO_HEAD)
    b = info["binop"]
    assert b["reimplementation_matches_official_selection"] and b["reimplementation_weight_mismatches"] == 0
    assert b["version"].startswith("first release" if version == "first_release" else "head")
    # quirk: the last (number of preprocess_res convs) non-res convs keep float weights (both versions)
    n_res = b["n_conv2d"] - b["n_nonres_conv2d"]
    assert n_res == 2 and len(b["nonres_not_binarized"]) == b["num_skip"] + n_res
    # first release: the 'res' exclusion is void, the preprocess_res convs are binarized too
    assert len(b["res_convs_binarized"]) == (2 if version == "first_release" else 0)
    perm = imagenet_class_perm() if use_perm else None
    g, info_g = torch_bnas.convert(model, targets, binarize="sign_zero", class_perm=perm, input_hw=hw)
    assert info_g["float_weight_convs_with_binary_input"] == b["nonres_not_binarized"][b["num_skip"]:]
    assert info_g["binarized_float_input_convs"] == b["res_convs_binarized"]
    if case == "cifar":   # reduction-cell preprocess1 only feeds Zero ops: not emitted
        assert info_g["unused_in_forward"] == ["cells.1.preprocess1", "cells.1.preprocess1.bn", "cells.1.preprocess1.op.0",
                                               "cells.3.preprocess1", "cells.3.preprocess1.bn", "cells.3.preprocess1.op.0"] or \
            all(".preprocess1" in u for u in info_g["unused_in_forward"])
        assert g.op_counts().get("ChannelAffine", 0) > 0 and any(n.name.endswith("/zeros") for n in g.nodes)
    else:
        assert info_g["unused_in_forward"] == []
    for name, v in info_g["float64_constants"].items():
        g.constants[name].data = v.reshape(g.constants[name].data.shape)
    x = np.random.default_rng(1).normal(size=(2, hw[0], hw[1], 3)).astype(np.float32)
    ref64, _ = run_reference(model, x, torch.float64, info_g["binarization_points"], perm, collect_stats=False)
    for i in range(2):
        out = executor.run(g, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]].ravel()
        assert np.abs(out - ref64[i]).max() < 1e-9 * max(1.0, np.abs(ref64).max())


@pytest.mark.skipif(not os.path.exists(BENCH), reason="host bitino_bench not built")
@pytest.mark.parametrize("stage,binarize", [("reference", "sign_zero"), ("reference", "ge"), ("neon", "ge")])
def test_bnas_cifar_numpy_bit_exact_with_cpp(tmp_path, stage, binarize):
    model, targets, _ = _source("bnas-a-cifar10", {"init_channels": 4, "layers": 5})
    g, _ = torch_bnas.convert(model, targets, binarize=binarize)
    path = tmp_path / "m.bitino"
    bfformat.write(g, str(path))
    g2 = bfformat.read(str(path))
    xs = np.random.default_rng(3).normal(size=(2, 32, 32, 3)).astype(np.float32)
    xs.tofile(tmp_path / "in.f32")
    subprocess.run([BENCH, "--model", str(path), "--inputs", str(tmp_path / "in.f32"), "--outputs", str(tmp_path / "out.f32"),
                    "--warmup", "0", "--iterations", "1", "--stage", stage, "--threads", "3"], check=True, capture_output=True)
    got = np.fromfile(tmp_path / "out.f32", np.float32).reshape(2, -1)
    for i in range(2):
        np.testing.assert_array_equal(got[i], executor.run(g2, {"input": xs[i:i + 1]})[g2.outputs[0]].ravel())


def test_ge_zero_channel_correction():
    """Structurally zero channels (layer-0 cell's Zero states) reaching an affine=False BN with running_mean 0 (as in
    the BNAS-A checkpoint, cells.7.preprocess0): sign(0) = 0 in the source; the ge variant must add the exact
    correction so that it equals the source on inputs without data-dependent ties."""
    torch = pytest.importorskip("torch")
    from convert_bnas import run_reference

    def prep(model):
        _randomize(model)
        with torch.no_grad():   # cell 2 (after reduction cell 1): preprocess0 = BN(affine=False) on cell 0's output
            bn = model.cells[2].preprocess0.bn
            assert not bn.affine
            bn.running_mean[2 * 4:] = 0.0
    pytest.importorskip("torch")
    if not os.path.exists(os.path.join(BNAS_REPO, "src", "model.py")):
        pytest.skip("BNAS clone not available")
    from convert_bnas import load_source
    model, targets, _ = load_source("bnas-a-cifar10", BNAS_REPO, None, cfg_override={"init_channels": 4, "layers": 5}, prepare=prep)
    x = np.random.default_rng(5).normal(size=(2, 32, 32, 3)).astype(np.float32)
    outs = {}
    for binarize in ("ge", "sign_zero"):
        g, info = torch_bnas.convert(model, targets, binarize=binarize)
        if binarize == "ge":
            assert [c["layer"] for c in info["zero_channel_corrections"]] == ["cells.2.preprocess0"]
            assert info["zero_channel_corrections"][0]["zero_input_channels"] == 8
        for name, v in info["float64_constants"].items():
            g.constants[name].data = v.reshape(g.constants[name].data.shape)
        ref64, st = run_reference(model, x, torch.float64, info["binarization_points"])
        assert st["sign_input_exact_zero"] > 0
        outs[binarize] = np.stack([executor.run(g, {"input": x[i:i + 1]}, float_dtype=np.float64)[g.outputs[0]].ravel() for i in range(2)])
        assert np.abs(outs[binarize] - ref64).max() < 1e-9 * max(1.0, np.abs(ref64).max())
    # without the correction the ge model would differ
    g, info = torch_bnas.convert(model, targets, binarize="ge")
    node = next(n for n in g.nodes if n.name == "cells.2.preprocess0/zero_channel_correction")
    g.constants[node.inputs[1]].data[:] = 0
    raw = executor.run(g, {"input": x[:1]}, float_dtype=np.float64)[g.outputs[0]].ravel()
    assert np.abs(raw - outs["ge"][0]).max() > 1e-3


def test_adapter_rejects_unbinarized_target():
    model, targets, _ = _source("bnas-a-cifar10", {"init_channels": 4, "layers": 5})
    torch = pytest.importorskip("torch")
    name = targets[3]
    with torch.no_grad():
        dict(model.named_modules())[name].weight[0, 0, 0, 0] *= 1.5
    with pytest.raises(NotImplementedError, match="not sign"):
        torch_bnas.convert(model, targets)
