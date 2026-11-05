# -*- coding: utf-8 -*-
"""Trusted reference for the original BNN-PYNQ Theano/Lasagne checkpoints (slice S7). PYTHON 2.7.

Runs the official network definitions of Xilinx/BNN-PYNQ (bnn/src/training/{cnv,lfc,quantization,
quantized_net}.py, Python 2 code) with Theano 1.0.4 + Lasagne 0.2.dev1 on the CPU, exactly as the training
scripts evaluate the test set: ``lasagne.layers.get_output(net, deterministic=True)`` after
``lasagne.layers.set_all_param_values(net, [npz['arr_%d' % i] ...])`` (the npz files were written with
``np.savez(path, *lasagne.layers.get_all_param_values(net))``). The npz is opened with allow_pickle=False.

Environment (project-local, nothing else touched): conda env .cache/env-theano27 (python 2.7, theano 1.0.4,
numpy 1.16.5, scipy), Lasagne master zip in .cache/downloads-s7/lasagne (sha256 in the notes). Run with
``-E -s`` (no PYTHON* env, no user site) from a directory without downloaded files, e.g.

  THEANO_FLAGS=base_compiledir=/tmp/theanoc,blas.ldflags= .cache/env-theano27/bin/python -E -s \
      tools/bnn_pynq_theano_reference.py --arch cnv --num-outputs 10 --npz <npz> --inputs <f32 NHWC file> \
      --shape 128,32,32,3 --floatx float32 --lasagne-dir .cache/downloads-s7/lasagne/Lasagne-master \
      --bnnpynq-dir .cache/repos/bnn-pynq/bnn/src/training --out <dump.npz>

Writes <dump.npz> with: logits [N, num_outputs]; pre_<k> = input of the k-th binary activation
(NonlinearityLayer with FixedHardTanH(QuantizationBinary)), i.e. the BN output, [N, ...] in NCHW; qin = output
of the 8-bit input quantizer (CNV); wq_<k> = the quantized weights of the k-th quantized layer as Theano
computes them (QuantizationBinary.quantizeWeights); layer_names (repr of the Lasagne layers).
"""
from __future__ import print_function

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["cnv", "lfc"])
    ap.add_argument("--num-outputs", type=int, required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--floatx", required=True, choices=["float32", "float64"])
    ap.add_argument("--lasagne-dir", required=True)
    ap.add_argument("--bnnpynq-dir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--logits-only", action="store_true", help="dump only logits and wq (large eval sets)")
    ap.add_argument("--binary-weights", help="dump.npz of a previous (float32) run: its wq_<k> (exactly +-1) replace the latent "
                                             "W of the quantized layers, so a float64 run uses the float32-derived binary weights "
                                             "(quantizeWeights(+-1) = +-1 in any precision)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    flags = os.environ.get("THEANO_FLAGS", "")
    os.environ["THEANO_FLAGS"] = ",".join([f for f in [flags, "floatX=" + args.floatx, "device=cpu"] if f])
    sys.path.insert(0, os.path.abspath(args.lasagne_dir))
    sys.path.insert(1, os.path.abspath(args.bnnpynq_dir))

    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne
    import quantization as q
    import quantized_net as qn

    assert theano.config.floatX == args.floatx

    class LearningParameters(object):
        pass

    lp = LearningParameters()
    lp.activation_bits = 1
    lp.weight_bits = 1
    lp.W_LR_scale = "Glorot"
    lp.epsilon = 1e-4
    lp.alpha = .1
    lp.dropout_in = .2       # lfc only; DropoutLayer is the identity with deterministic=True
    lp.dropout_hidden = .5

    shape = [int(s) for s in args.shape.split(",")]
    x = np.fromfile(args.inputs, dtype=np.float32).reshape(shape).transpose(0, 3, 1, 2)   # NHWC -> NCHW (bc01)
    inp = T.tensor4("inputs")
    if args.arch == "cnv":
        import cnv
        net = cnv.genCnv(inp, args.num_outputs, lp)
    else:
        import lfc
        net = lfc.genLfc(inp, args.num_outputs, lp)

    f = np.load(args.npz, allow_pickle=False)
    params = [f["arr_%d" % i] for i in range(len(f.files))]
    layers = lasagne.layers.get_all_layers(net)
    ql = [l for l in layers if isinstance(l, (qn.Conv2DLayer, qn.DenseLayer))]
    if args.binary_weights:
        bw = np.load(args.binary_weights, allow_pickle=False)
        allp = lasagne.layers.get_all_params(net)
        for k, l in enumerate(ql):
            i = [j for j, p in enumerate(allp) if p is l.W][0]
            assert params[i].shape == bw["wq_%d" % k].shape and np.all(np.abs(bw["wq_%d" % k]) == 1)
            params[i] = bw["wq_%d" % k].astype(np.float32)
    lasagne.layers.set_all_param_values(net, params)

    pts, qin = [], None
    for l in layers:
        if isinstance(l, lasagne.layers.NonlinearityLayer) and isinstance(l.nonlinearity, qn.FixedHardTanH):
            if isinstance(l.nonlinearity.quantization, q.QuantizationBinary):
                pts.append(l.input_layer)
            elif qin is None:
                qin = l
    if args.logits_only:
        pts, qin = [], None
    outs = [net] + pts + ([qin] if qin is not None else [])
    fn = theano.function([inp], lasagne.layers.get_output(outs, deterministic=True))
    res = None
    for s in range(0, len(x), args.batch):
        r = fn(x[s:s + args.batch].astype(args.floatx))
        res = [[v] for v in r] if res is None else [a + [v] for a, v in zip(res, r)]
    res = [np.concatenate(v) for v in res]
    dump = {"logits": res[0]}
    for k in range(len(pts)):
        dump["pre_%d" % k] = res[1 + k]
    if qin is not None:
        dump["qin"] = res[-1]
    for k, l in enumerate(ql):
        dump["wq_%d" % k] = theano.function([], l.quantize(l.W))()
    dump["layer_names"] = np.array([type(l).__name__ for l in layers])
    dump["versions"] = np.array(["theano " + theano.__version__, "lasagne " + lasagne.__version__, "numpy " + np.__version__,
                                 "python " + sys.version.split()[0], "floatX " + theano.config.floatX])
    np.savez(args.out, **dump)
    print("wrote %s: logits %s, %d binarization points, %d quantized layers, %s" % (args.out, res[0].shape, len(pts), len(ql),
                                                                                    theano.config.floatX))


if __name__ == "__main__":
    main()
