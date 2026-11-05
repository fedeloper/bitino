# Model discovery

How the registry (`registry/models.json`, `registry/CATALOG.md`) was built, and what it does and does not contain.

## Method

The search (October 2026) covered eight families, each with its own pass over papers, repositories and
forward-pass code: Larq Zoo; Bi-Real / ReActNet / IR-Net; classical BNNs (BinaryNet, XNOR-Net, ABC-Net, DoReFa,
FINN / Brevitas); modern binary CNNs (2020–2026); binary BERT and LLMs; binary ViTs; non-classification tasks; and
a sweep of model hubs (Hugging Face API, GitHub search, curated lists).

1. **Discovery.** For every candidate: paper, code, published checkpoint, binarization semantics (sign at 0,
   padding of binary convolutions, scaling, full-precision layers) with file:line citations. Checkpoints up to
   1.5 GB were downloaded, hashed (SHA-256) and inspected without executing code (`torch.load(weights_only=True)`,
   h5py, zip listings).
2. **Adversarial verification.** A second, independent pass per family re-hashed every local file, re-checked URL
   reachability, re-read the cited source lines, re-fetched licences and downgraded unsupported claims
   (160 corrections).
3. **Completeness check.** Gaps found in the first pass (OpenVINO OMZ binary IR models, NNCF binarized ResNet-18,
   BURN self-supervised ReActNet, 2025 task papers, skipped downloads) were searched separately.
4. **Second sweep (2026-10-30).** 106 more entries, in four scopes: classification CNNs; ViTs, dense tasks and
   other modalities; LLMs and NLP; hardware toolchains and hubs.
5. **Registry rules.** `registry_status = verified-pretrained` only if the checkpoint was downloaded, a SHA-256 was
   recorded and the local file re-hashes to the same value. Otherwise `download-failed` (a URL was tried and
   failed), `unverified` (URL known, not downloaded) or `no-public-weights`. A verified entry is never replaced
   by an unverified one.

## Results

`registry/CATALOG.md` lists every entry by family. In total 326 entries; strict BNNs (binary weights **and**
activations in the body) with verified weights: 163. Near-BNNs (W1A2, binary-weight-only, ternary) are recorded but
out of scope for conversion.

## Key negative findings

* **ReActNet (official ReActNet-A and ResNet)**: every published link points to a deleted HKUST OneDrive account
  (HTTP 404), across four README revisions; no mirror was found. The same applies to the official AdamBNN
  checkpoint.
* **Bi-Real Net (official)** and **IR-Net (official)**: no checkpoints were ever published. Bi-Real weights *are*
  available via Larq Zoo (Keras) and daBNN (ONNX / .dab).
* Papers with code but without weights are recorded as `no-public-weights` and never substituted with a
  different checkpoint.

## Semantics that differ between families (they matter for conversion)

| family | sign(0) | binary-conv padding | scaling |
|---|---|---|---|
| Larq Zoo | +1 | `pad_values` (QuickNet +1, others 0) | MagnitudeAwareSign per-channel mean \|W\| (Bi-RealNet) or none |
| PyTorch ReActNet-style (AB-BNN, BNext, …) | 0 (`torch.sign`) | zero | per-channel mean \|W\|, learnable biases, RPReLU |
| Bi-Real Caffe / daBNN | −1 / path-dependent | −1 (daBNN) | none |

Bitino's deployable binarization maps 0 to +1; where a source uses `torch.sign` (0 stays 0), the converters
also write a `sign_zero` variant for the reference kernels and report every sample where exact zeros occur
(`docs/numerical-correctness.md`).

## Licences

Code and weights licences are recorded separately for every entry. Many weight releases carry **no explicit
licence**; they are recorded as such ("no explicit weights licence (inherits repo licence? unverified)"). Bitino
redistributes a converted file only when the weights licence is explicitly permissive (`bitino list --release`);
for everything else the registry references the upstream URL and SHA-256, and `bitino fetch` converts locally.
