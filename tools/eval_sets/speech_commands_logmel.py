#!/usr/bin/env python3
"""Speech Commands v0.02 test set -> BRILLsson log-mel features (host-side front-end of the BRILLsson models).

BRILLsson (HarlinLee/brillsson model.py) computes its features *inside* the Keras graph with a `Spectrogram`
layer; Bitino runs only the binary network, so the same TensorFlow ops run here, on the host:

  waveform = int16 / 32767 (eval.py prepare_example: tf.cast(w, float32) / float(tf.int16.max)), centre zero-padded
             to 16000 samples (left = pad // 2)
  stft     = tf.signal.stft(waveform, frame_length=400, frame_step=160, fft_length=1024)       (pad_end False)
  mel      = tf.tensordot(|stft|, tf.signal.linear_to_mel_weight_matrix(64, 513, 16000, 40.0, 7800.0), 1)
  feature  = log(mel + 1e-6)[..., newaxis]    -> (98, 64, 1) for a 1 s clip

(`Spectrogram.__init__` as released references an undefined name `augment`; the attribute is never used in call(),
so the call() body is reproduced verbatim here.) Needs TensorFlow (Larq reference env):

  .cache/venv-larq/bin/python -I tools/eval_sets/speech_commands_logmel.py --root <extracted test set> \
      --tar-sha256 cc2a00c1... --n 128 --out .cache/data/eval128_speechcommands

Selection: classes = sorted directory names (12: 10 words, _silence_, _unknown_); sample i takes class i % 12 and
the (i // 12)-th file of that class in sorted order (n = 0: every file, class-major order). Audio is NOT
redistributed (Speech Commands: CC BY 4.0; the manifest records file names and SHA-256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave

import numpy as np

SR, FRAME, STEP, FFT, MELS, FMIN, FMAX = 16000, 400, 160, 1024, 64, 40.0, 7800.0


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR, path
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="extracted speech_commands_test_set_v0.02 directory")
    ap.add_argument("--tar-sha256", default="", help="SHA-256 of the source tarball (recorded only)")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    classes = sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    files = {c: sorted(f for f in os.listdir(os.path.join(args.root, c)) if f.endswith(".wav")) for c in classes}
    if args.n:
        sel = [(classes[i % len(classes)], files[classes[i % len(classes)]][i // len(classes)]) for i in range(args.n)]
    else:
        sel = [(c, f) for c in classes for f in files[c]]
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    mel_w = tf.signal.linear_to_mel_weight_matrix(MELS, FFT // 2 + 1, SR, FMIN, FMAX)
    os.makedirs(args.out, exist_ok=True)
    h = hashlib.sha256()
    samples, n_frames = [], None
    with open(os.path.join(args.out, "brillsson_logmel.f32"), "wb") as fo:
        for c, f in sel:
            p = os.path.join(args.root, c, f)
            raw = read_wav(p)
            assert raw.size <= SR, (p, raw.size)
            w = tf.cast(tf.constant(raw), tf.float32) / float(tf.int16.max)
            pad = SR - raw.size
            w = tf.pad(w, [[pad // 2, pad - pad // 2]])
            stfts = tf.signal.stft(w[tf.newaxis], frame_length=FRAME, frame_step=STEP, fft_length=FFT)
            mel = tf.tensordot(tf.abs(stfts), mel_w, 1)
            feat = tf.math.log(mel + 1e-6)[..., tf.newaxis][0].numpy().astype(np.float32)
            n_frames = n_frames or feat.shape[0]
            assert feat.shape == (n_frames, MELS, 1), feat.shape
            b = np.ascontiguousarray(feat).tobytes()
            fo.write(b)
            h.update(b)
            samples.append({"file": f"{c}/{f}", "audio_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                            "label": classes.index(c), "class_name": c, "n_samples_raw": int(raw.size)})
    manifest = {
        "schema": "bitino.eval_set.v1",
        "name": f"speechcommands-v0.02-test-{'balanced-' + str(len(sel)) if args.n else 'all-' + str(len(sel))}",
        "task": "speech (non-semantic embedding; keyword spotting labels)",
        "source": "Speech Commands v0.02 official test set (http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz, "
                  "CC BY 4.0); not redistributed",
        "source_files": {"speech_commands_test_set_v0.02.tar.gz": {"sha256": args.tar_sha256,
                                                                   "md5_equals_gcs_etag": "854c580ee90bff80c516491c84544e32"}},
        "selection_rule": ("sample i: class i % 12 (sorted dirs), (i // 12)-th sorted file" if args.n
                           else "every file, classes in sorted order, files sorted"),
        "class_index": {"classes": classes},
        "limitations": "BRILLsson outputs embeddings; labels serve only the linear-probe sanity check",
        "samples": samples,
        "preprocessing": {"brillsson_logmel": {
            "shape": [len(sel), n_frames, MELS, 1], "dtype": "float32", "layout": "NHWC", "file": "brillsson_logmel.f32",
            "sha256": h.hexdigest(),
            "definition": (f"tensorflow {tf.__version__}: int16/32767, centre zero-pad to 16000, tf.signal.stft(400, 160, 1024), "
                           "|.| @ linear_to_mel_weight_matrix(64, 513, 16000, 40, 7800), log(x + 1e-6), [..., None] "
                           "(HarlinLee/brillsson model.py Spectrogram.call + eval.py prepare_example)")}},
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"{args.out}: {len(sel)} clips, features {n_frames}x{MELS}x1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
