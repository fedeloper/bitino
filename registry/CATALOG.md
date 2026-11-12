# Bitino model catalog

Generated 2026-11-05T17:18:11Z by `tools/build_registry.py` from `registry/models.json`. Do not edit by hand.

Status meanings: **verified-pretrained**: checkpoint downloaded and SHA-256 re-verified locally. **download-failed**: documented link is dead or blocked. **no-public-weights**: none published. **unverified**: link known but not downloaded. *strict BNN* = binary weights AND activations in the body (W1A1 / W1A1+FP / W1A1+INT8).

Filter with `bitino registry list --family F --task T --precision P --license L --status S [--json]`.

| status | count |
|---|---|
| download-failed | 10 |
| no-public-weights | 46 |
| strict_bnn_verified | 180 |
| unverified | 64 |
| verified-pretrained | 223 |

## binary-bert-language

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `arb-llm-ptq` | binary-bert-language | causal-language-modeling | research-only (no verified weights) |  | Apache-2.0 / ? | no-public-weights |  |
| `bebert-bert-base-glue` | binary-bert-language | glue | research-only (no verified weights), multi-bit/low-bit (not strict BNN) |  | Apache-2.0 / n/a | no-public-weights |  |
| `bibert-bert-base-glue-other-tasks` | binary-bert-language | glue-cola/mnli/qnli/qqp/rte/st | W1A1+FP ★ |  | No LICENSE file (GitHub lice / no explicit weights licence (inherits re | unverified |  |
| `bibert-bert-base-mrpc` | binary-bert-language | glue-mrpc | W1A1+FP ★ | 438.1 MB | No LICENSE file (GitHub lice / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bibert-bert-base-sst2` | binary-bert-language | glue-sst2 | W1A1+FP ★ | 438.1 MB | No LICENSE file in repo (Git / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending (S22 unavailable 2026-10-29), device: pending (physical device). Preliminary host x86 latency only (seq 64, stage neon, medians): 1 thr ~591 ms, 4 thr ~243 ms, 8 thr ~188 ms |
| `billm-llm-ptq` | binary-bert-language | causal-language-modeling | research-only (no verified weights) |  | MIT / ? | no-public-weights |  |
| `binarybert-bert-base-glue` | binary-bert-language | glue / squad | research-only (no verified weights), multi-bit/low-bit (not strict BNN) |  | Apache-2.0 (BinaryBERT/LICEN / n/a (no weights) | no-public-weights |  |
| `bipft-base-pretrain-step500000` | binary-bert-language | masked-lm pretraining (require | W1A1+FP ★ | 441.1 MB | No LICENSE file (GitHub lice / no explicit weights licence (inherits re | verified-pretrained | deferred, verif: not-started, device: not-started |
| `bit-bert-base-glue` | binary-bert-language | glue (all tasks) / squad-v1.1 | research-only (no verified weights), W1A1+FP ★ |  | CC-BY-NC-4.0 (LICENSE file;  / CC-BY-NC-4.0 per README statement 'BiT i | download-failed |  |
| `fbi-llm-1.3b-7b` | binary-bert-language | causal-language-modeling | W1A-fp | 5460.8 MB | No LICENSE in GitHub repo / Apache-2.0 (HF license tag on both repos | unverified |  |
| `fbi-llm-130m` | binary-bert-language | causal-language-modeling | W1A-fp | 537.2 MB | No LICENSE in GitHub repo (l / Apache-2.0 (HF model card metadata / lic | verified-pretrained |  |
| `microsoft-bitnet-b1.58-2b-4t` | binary-bert-language | causal-language-modeling (chat | ternary (not strict BNN) | 1178.6 MB | MIT / MIT | verified-pretrained |  |
| `onebit-llama2-7b` | binary-bert-language | causal-language-modeling | W1A-fp | 1869.4 MB | MIT / MIT (README: 'This code and checkpoints  | unverified |  |
| `pb-llm-ptq` | binary-bert-language | causal-language-modeling | research-only (no verified weights), multi-bit/low-bit (not strict BNN) |  | MIT / ? | no-public-weights |  |
| `prism-bonsai-1.7b-q1_0-gguf` | binary-bert-language | causal-language-modeling (chat | W1A-fp | 248.3 MB | llama.cpp MIT; PrismML fork  / Apache-2.0 (LICENSE + NOTICE.txt: built  | verified-pretrained |  |

## binary-vit

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `bhvit-small-binary` | binary-vit | imagenet-classification | W1A1+FP ★ | 273.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bhvit-small-somefp` | binary-vit | imagenet-classification | W1A1+FP ★ | 273.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bhvit-tiny-binary` | binary-vit | imagenet-classification | W1A1+FP ★ | 166.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bhvit-tiny-somefp` | binary-vit | imagenet-classification | W1A1+FP ★ | 166.6 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host only), device: pending |
| `bi-vit-deit-small` | binary-vit | imagenet-classification | W1A1+FP ★ | 269.2 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bi-vit-deit-tiny` | binary-vit | imagenet-classification | W1A1+FP ★ | 71.1 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `binaryattention-deit-tiny` | binary-vit | imagenet-classification | multi-bit/low-bit (not strict BNN) | 99.6 MB | Apache-2.0 / no explicit weights licence; the HF repo | verified-pretrained |  |
| `binaryvit-le-li-small` | binary-vit | imagenet-classification | research-only (no verified weights) |  | none (no LICENSE file; GitHu / n/a | no-public-weights |  |
| `binaryvit-xiao` | binary-vit | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `bivit-he-iccv23` | binary-vit | imagenet-classification | research-only (no verified weights) |  | Apache-2.0 / ? | no-public-weights |  |
| `bvt-ima` | binary-vit | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `didb-vit` | binary-vit | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `gsb-deit-s-cifar100-stage1` | binary-vit | cifar100-classification | W1A-fp | 261.6 MB | MIT / no explicit weights licence (inherits re | verified-pretrained |  |
| `gsb-deit-s-cifar100-stage2` | binary-vit | cifar100-classification | W1A1+INT8, multi-bit/low-bit (not strict BNN) ★ | 295.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `nakahara-binary-vit` | binary-vit | unknown | unsupported/incomplete | 118.5 MB | none / none | verified-pretrained |  |
| `qvit-deit-s-2bit` | binary-vit | imagenet-classification | multi-bit/low-bit (not strict BNN) | 271.1 MB | none (no LICENSE file; GitHu / none | verified-pretrained |  |
| `reactnet-deit-small-baseline` | binary-vit | imagenet-classification | research-only (no verified weights) |  | none (no LICENSE) / ? | no-public-weights |  |
| `si-bivit` | binary-vit | imagenet-classification | research-only (no verified weights) |  | none (no LICENSE; GitHub API / ? | no-public-weights |  |

## bnn-tasks-2025-iccv-cvpr-iros

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `bicd-levir-cd` | bnn-tasks-2025-iccv-cvpr-iros | change-detection (LEVIR-CD rem | unsupported/incomplete | 150.9 MB | BiCD MIT. DGMA2-Net licence  / Not stated | verified-pretrained |  |
| `bicd-pcd` | bnn-tasks-2025-iccv-cvpr-iros | change-detection (PCD street-v | W1A1+FP, W1A-fp ★ | 435.7 MB | MIT / Not stated separately (MIT repo; weights | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bicd-vl-cmu-cd` | bnn-tasks-2025-iccv-cvpr-iros | change-detection (VL-CMU-CD st | W1A1+FP, W1A-fp ★ | 314.6 MB | MIT / Not stated separately | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `biisnet` | bnn-tasks-2025-iccv-cvpr-iros | infrared-small-target-detectio | research-only (no verified weights) |  | none stated / ? | unverified |  |
| `binarydemoire` | bnn-tasks-2025-iccv-cvpr-iros | image-demoireing | research-only (no verified weights) |  | none stated (no LICENSE file / ? | unverified |  |
| `bivoc` | bnn-tasks-2025-iccv-cvpr-iros | speech-vocoder | research-only (no verified weights) |  | MIT (placeholder repo) / ? | unverified |  |
| `bmtnet-hybridevs` | bnn-tasks-2025-iccv-cvpr-iros | raw-image demosaicing (Quad Ba | W1A1+FP ★ | 149.6 MB | No licence file (GitHub API  / No licence stated | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bu-net` | bnn-tasks-2025-iccv-cvpr-iros | underwater-image-enhancement | research-only (no verified weights) |  | none stated / ? | unverified |  |
| `pathfinder-rs-lvf` | bnn-tasks-2025-iccv-cvpr-iros | road/path segmentation from Li | W1A1+FP, multi-bit/low-bit (not strict BNN) ★ | 161.7 MB | MIT / Not stated separately | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `pathfinder-sensaturban` | bnn-tasks-2025-iccv-cvpr-iros | urban point-cloud semantic seg | W1A1+FP, multi-bit/low-bit (not strict BNN) ★ | 161.7 MB | MIT / Not stated separately | verified-pretrained | unsupported, verif: not-started, device: not-started |

## budget-skipped-downloadable-checkpoints

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `bnas-f-imagenet` | BNAS | imagenet-classification | W1A1+FP ★ | 318.8 MB | none / ? | unverified |  |
| `bnas-g-imagenet` | BNAS | imagenet-classification | W1A1+FP ★ | 299.1 MB | none / ? | unverified |  |
| `bnas-e-imagenet` | BNAS (binary NAS, modern binar | imagenet-classification | W1A1+FP ★ | 256.4 MB | none (no LICENSE in repo) / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnas-h-imagenet` | BNAS (binary NAS, modern binar | imagenet-classification | W1A1+FP ★ | 1281.0 MB | none / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bibert-bert-base-glue-cola-qnli-rte-stsb` | BiBERT | glue-cola, glue-qnli, glue-rte | W1A1+FP ★ |  | ? / ? | unverified |  |
| `bibert-bert-base-mnli` | BiBERT (binary BERT) | glue-mnli | W1A1+FP ★ | 438.1 MB | no LICENSE file (Apache-2.0  / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bibert-bert-base-qqp` | BiBERT (binary BERT) | glue-qqp | W1A1+FP ★ | 438.1 MB | no LICENSE file / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `biper-resnet34-imagenet` | BiPer (modern binary CNN) | imagenet-classification | W1A1+FP ★ | 174.7 MB | MIT / unspecified (MIT repo; no separate weigh | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `allenai-xnor-net-alexnet-bwn` | XNOR-Net (classical early BNN) | imagenet-classification | W1A-fp | 499.3 MB | allenai SOFTWARE-LICENSE-AGR / same non-commercial agreement (assumed t | verified-pretrained |  |
| `larq-zoo-binary-densenet28-notop` | larq-zoo | imagenet-feature-extractor (ba | W1A1+FP ★ | 18.5 MB | Apache-2.0 / unspecified | verified-pretrained | deferred, verif: not-started, device: not-started |
| `larq-zoo-quicknet-notop` | larq-zoo | imagenet-feature-extractor (ba | W1A1+FP ★ | 51.1 MB | Apache-2.0 / unspecified (release assets of Apache-2. | verified-pretrained | deferred, verif: not-started, device: not-started |

## burn-self-supervised-binary-reactnet

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `burn-moco-fp-teacher` | burn-self-supervised-binary-re | imagenet self-supervised repre | unsupported/incomplete | 94.3 MB | CC BY-NC 4.0 (BURN repo); Mo / unverified | verified-pretrained |  |
| `burn-reacta-linear-eval-imagenet` | burn-self-supervised-binary-re | imagenet-classification | research-only (no verified weights) |  | CC BY-NC 4.0 / ? | unverified |  |
| `burn-reacta-stage1` | burn-self-supervised-binary-re | imagenet self-supervised repre | W-fp-A1 | 174.3 MB | CC BY-NC 4.0 / No separate statement; assume CC BY-NC 4 | verified-pretrained |  |
| `burn-reacta-stage2` | burn-self-supervised-binary-re | imagenet self-supervised repre | W1A1+FP ★ | 117.7 MB | CC BY-NC 4.0 (NAVER Corp.);  / No separate statement. Distributed via t | verified-pretrained | converted, verif: passed (host + emulated arm64), feature-level only; physical device pending, device: pending |
| `s2bnn-reacta-contrastive-distill` | burn-self-supervised-binary-re | imagenet self-supervised repre | W1A1+FP ★ | 595.8 MB | none (no LICENSE) / ? | unverified |  |
| `s2bnn-reacta-contrastive-only` | burn-self-supervised-binary-re | imagenet self-supervised repre | W1A1+FP ★ | 550.9 MB | none (no LICENSE) / ? | unverified |  |
| `s2bnn-reacta-distill-only` | burn-self-supervised-binary-re | imagenet self-supervised repre | W1A1+FP ★ | 595.8 MB | No LICENSE file (GitHub API  / No statement | verified-pretrained | converted, verif: passed (host + emulated arm64), feature-level only; physical device pending, device: pending |
| `s2bnn-reacta-mocov2-baseline` | burn-self-supervised-binary-re | imagenet self-supervised repre | W1A1+FP ★ | 432.9 MB | none (no LICENSE) / ? | unverified |  |

## classical-early-bnn

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `abc-net` | ABC-Net | imagenet-classification | research-only (no verified weights), multi-bit/low-bit (not strict BNN) |  | MIT (unofficial layog repo o / ? | no-public-weights |  |
| `bnn-plus` | BNN+ | imagenet/cifar10-classificatio | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `binaryconnect-courbariaux` | BinaryConnect | mnist/cifar10/svhn-classificat | research-only (no verified weights), W1A-fp |  | GPL-2.0 (GitHub API) / ? | no-public-weights |  |
| `larq-zoo-binary-densenet-xref` | BinaryDenseNet | imagenet-classification | W1A1+FP ★ |  | ? / ? | no-public-weights |  |
| `binarynet-courbariaux-theano` | BinaryNet | mnist/cifar10/svhn-classificat | research-only (no verified weights) |  | BSD-3-Clause (Courbariaux re / ? | no-public-weights |  |
| `tensorpack-dorefa-alexnet-bwn-w1a32` | DoReFa-Net | imagenet-classification | W1A-fp, multi-bit/low-bit (not strict BNN) | 227.6 MB | Apache-2.0 / no explicit weights licence (inherits re | unverified |  |
| `tensorpack-dorefa-alexnet-w1a1-g32` | DoReFa-Net | imagenet-classification | W1A1+FP ★ | 227.7 MB | Apache-2.0 (tensorpack) / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `tensorpack-dorefa-alexnet-w1a2-g6` | DoReFa-Net | imagenet-classification | multi-bit/low-bit (not strict BNN) | 227.5 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained |  |
| `tensorpack-dorefa-resnet18-w1a4` | DoReFa-Net | imagenet-classification | multi-bit/low-bit (not strict BNN) | 42.4 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained |  |
| `bnn-pynq-cnv-w1a1-cifar10` | FINN/BNN-PYNQ | cifar10-classification | W1A1+INT8 ★ | 6.2 MB | BSD-3-Clause (repo LICENSE ( / BSD-3-Clause by inclusion in the repo tr | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnn-pynq-cnv-w1a1-gtsrb` | FINN/BNN-PYNQ | gtsrb-classification (43 class | W1A1+INT8 ★ | 6.3 MB | BSD-3-Clause / BSD-3-Clause by inclusion in the repo tr | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnn-pynq-lfc-w1a1-mnist` | FINN/BNN-PYNQ | mnist-classification | W1A1 ★ | 11.7 MB | BSD-3-Clause / BSD-3-Clause by inclusion in the repo tr | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `brevitas-bnn-pynq-cnv-1w1a` | FINN/Brevitas BNN-PYNQ | cifar10-classification | W1A1+INT8 ★ | 6.2 MB | BSD-3-Clause / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + qemu-emulated arm64); device pending, device: pending |
| `brevitas-bnn-pynq-cnv-1w2a` | FINN/Brevitas BNN-PYNQ | cifar10-classification | ternary (not strict BNN), multi-bit/low-bit (not strict BNN) | 6.2 MB | BSD-3-Clause / no explicit weights licence (inherits re | verified-pretrained |  |
| `brevitas-bnn-pynq-lfc-1w1a` | FINN/Brevitas BNN-PYNQ | mnist-classification | W1A1 ★ | 11.7 MB | BSD-3-Clause / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + qemu-emulated arm64); device pending, device: pending |
| `brevitas-bnn-pynq-sfc-1w1a` | FINN/Brevitas BNN-PYNQ | mnist-classification | W1A1 ★ | 1.4 MB | BSD-3-Clause / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + qemu-emulated arm64); device pending, device: pending |
| `brevitas-bnn-pynq-tfc-1w1a` | FINN/Brevitas BNN-PYNQ | mnist-classification | W1A1 ★ | 0.2 MB | BSD-3-Clause (Brevitas LICEN / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + qemu-emulated arm64); device pending, device: pending |
| `finn-examples-bnn-pynq-onnx` | FINN/Brevitas BNN-PYNQ | cifar10-classification / mnist | W1A1, W1A1+INT8 ★ | 1.6 MB | BSD-3-Clause (finn-examples, / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `hwgq-resnet18-w1a2-bw` | HWGQ | imagenet-classification | multi-bit/low-bit (not strict BNN) | 46.8 MB | UC Regents licence: educatio / no explicit weights licence (presumably  | verified-pretrained |  |
| `lq-nets-resnet18-w1a2` | LQ-Nets | imagenet-classification | research-only (no verified weights), multi-bit/low-bit (not strict BNN) |  | MIT / ? | no-public-weights |  |
| `larq-zoo-real-to-bin-net-xref` | Real-to-Binary | imagenet-classification | W1A1+FP ★ |  | ? / ? | no-public-weights |  |
| `allenai-xnor-net-alexnet-xnor` | XNOR-Net | imagenet-classification | W1A1+FP ★ | 499.3 MB | AI2 non-commercial Software  / covered by the same agreement (it covers | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `xnor-net-plus-plus` | XNOR-Net | imagenet-classification | research-only (no verified weights) |  | BSD-3-Clause (1adrianb/binar / ? | no-public-weights |  |
| `xnor-net-pytorch-alexnet-imagenet` | XNOR-Net | imagenet-classification | research-only (no verified weights) |  | none stated / ? | download-failed |  |
| `xnor-net-pytorch-lenet5-mnist` | XNOR-Net | mnist-classification | research-only (no verified weights) |  | none stated (no LICENSE file / ? | download-failed |  |
| `xnor-net-pytorch-nin-cifar10` | XNOR-Net | cifar10-classification | research-only (no verified weights) |  | none (no LICENSE file; GitHu / n/a | download-failed |  |

## hub-sweep

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `ab-bnn-cifar10-reactnet18` | A&B BNN | cifar10-classification | W1A1+FP ★ | 44.8 MB | none declared / MIT (HF card) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ab-bnn-cifar10-reactneta` | A&B BNN | cifar10-classification | W1A1+FP ★ | 113.3 MB | none declared / MIT (HF card) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ab-bnn-cifar100-reactnet18` | A&B BNN | cifar100-classification | W1A1+FP ★ | 45.0 MB | none declared / MIT (HF card) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ab-bnn-cifar100-reactneta` | A&B BNN | cifar100-classification | W1A1+FP ★ | 113.7 MB | none declared / MIT (HF card) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ab-bnn-imagenet-reactnet34` | A&B BNN | imagenet-classification | W1A1+FP ★ | 87.4 MB | none declared / MIT (HF card) | verified-pretrained | converted, verif: semantically exact; host bit-exact; default top-1 acceptance FAILS (source fp32 noise floor), device: not-run |
| `ab-bnn-imagenet-reactneta` | A&B BNN | imagenet-classification | W1A1+FP ★ | 117.4 MB | none declared / MIT (HF card) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ab-bnn-imagenet-reactnet18` | A&B BNN (normalizer-free ReAct | imagenet-classification | W1A1+FP ★ | 46.8 MB | none declared (GitHub repo h / MIT (HF model card metadata 'license: mi | verified-pretrained | converted, verif: passed (host); device: pre-merge runtime bit-exact on S22, post-merge pending (physical device), device: pending |
| `bifsmn-release-v1-teacher` | BiFSMN (binary keyword spottin | keyword-spotting (Google Speec | research-only (no verified weights) | 4.9 MB | none declared (GitHub API li / no explicit weights licence | verified-pretrained |  |
| `efficient-ml-llama3-8b-billm-fake` | BiLLM (PTQ binarized LLM) | text-generation (LLM) | W1A-fp | 16060.6 MB | ? / not declared (no card) | unverified |  |
| `prism-bonsai-1.7b-q1_0` | Bonsai 1-bit LLMs (PrismML) | text-generation (LLM) | W1A-fp | 248.3 MB | Apache-2.0 (forks; not indiv / Apache-2.0 (card metadata + LICENSE file | verified-pretrained |  |
| `prism-bonsai-image-binary-4b` | Bonsai Image (1-bit diffusion  | text-to-image | W1A-fp | 1080.4 MB | ? / Apache-2.0 (card) | unverified |  |
| `fracbnn-cifar10` | FracBNN | cifar10-classification | multi-bit/low-bit (not strict BNN) | 51.6 MB | BSD-3-Clause / no explicit weights licence (inherits re | verified-pretrained |  |
| `fracbnn-imagenet` | FracBNN | imagenet-classification | multi-bit/low-bit (not strict BNN) | 837.0 MB | BSD-3-Clause (Copyright 2021 / no explicit weights licence (inherits re | verified-pretrained |  |
| `groupnet-resnet18-softgates` | Group-Net (structured binary) | imagenet-classification | multi-bit/low-bit (not strict BNN) | 668.3 MB | README states GPLv3-or-later / no explicit weights licence (README term | verified-pretrained |  |
| `hyperbeeai-shallow007-ap2b8b` | HyperbeeAI MAX78000 quantized/ | cifar100-classification | multi-bit/low-bit (not strict BNN) | 1.4 MB | conflicting: card metadata a / same conflict | verified-pretrained |  |
| `nas-bnn-supernet` | NAS-BNN | imagenet-classification | research-only (no verified weights) | 798.8 MB | academic research only; comm / same academic-only terms (README) | unverified |  |
| `liqunma-onebit-mamba2-780m` | OneBit (W1A16 LLM) | text-generation (LLM) | research-only (no verified weights) | 3431.2 MB | ? / ? | unverified |  |
| `primeneural-vit-b16-d50-1bit` | PNN 1-bit sparse ViT (propriet | imagenet-classification | W1A-fp | 8.9 MB | proprietary (card license: o / proprietary, 'published for evaluation', | verified-pretrained |  |
| `rbnn-imagenet-resnet34` | RBNN | imagenet-classification | W1A1+FP ★ | 87.9 MB | none declared / no explicit weights licence | verified-pretrained | duplicate, verif: not-started, device: not-started |
| `rbnn-imagenet-resnet18` | RBNN (Rotated Binary Neural Ne | imagenet-classification | W1A1+FP ★ | 47.1 MB | none declared (GitHub API li / no explicit weights licence (repo has no | verified-pretrained | duplicate, verif: not-started, device: not-started |
| `s2bnn-distill-only-link` | S2-BNN | imagenet-self-supervised (line | research-only (no verified weights) | 227.2 MB | none declared (API license n / no explicit weights licence | verified-pretrained |  |
| `xnornet-pytorch-alexnet` | XNOR-Net (PyTorch port) | imagenet-classification | research-only (no verified weights) |  | none declared (API license n / ? | download-failed |  |
| `kanakmalpani-bnn-lab-canaries` | bnn-lab | codec/shape tests | unsupported/incomplete | 4.3 MB | ? / MIT (card) | unverified |  |
| `arotting-bitforge-1bit` | toy binary-weight MLP | digits-classification (8x8, 64 | W1A-fp | 0.0 MB | ? / Apache-2.0 (card) | verified-pretrained |  |
| `opentransformer-purebit` | toy binary-weight bit-level LM | text-generation (bit-level, to | W1A-fp | 18.9 MB | ? / MIT (card) | verified-pretrained |  |
| `chanjeong-resnet-binary-weights` | unknown (timm ResNet) | imagenet-classification (presu | research-only (no verified weights) | 140.7 MB | ? / MIT (card metadata) | unverified |  |

## larq-zoo

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `larq-zoo-binary-alexnet` | larq-zoo | imagenet-classification | W1A1 ★ | 247.5 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |
| `larq-zoo-binary-densenet28` | larq-zoo | imagenet-classification | W1A1+FP ★ | 20.8 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |
| `larq-zoo-binary-densenet37` | larq-zoo | imagenet-classification | W1A1+FP ★ | 35.2 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64, --explain); device pending, device: pending |
| `larq-zoo-binary-densenet37-dilated` | larq-zoo | imagenet-classification | W1A1+FP ★ | 35.2 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64) except host optimized --explain (one float_opt-induced flip at 1.7e-4 x typical); device pending, device: pending |
| `larq-zoo-binary-densenet45` | larq-zoo | imagenet-classification | W1A1+FP ★ | 56.1 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64, --explain); device pending, device: pending |
| `larq-zoo-binary-resnet-e18` | larq-zoo | imagenet-classification | W1A1+FP ★ | 47.0 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |
| `larq-zoo-birealnet` | larq-zoo | imagenet-classification | W1A1+FP ★ | 46.9 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host), device: pending |
| `larq-zoo-dorefanet` | larq-zoo | imagenet-classification | multi-bit/low-bit (not strict BNN) | 249.7 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending; not strict BNN (W1A2), device: pending |
| `larq-zoo-meliusnet22` | larq-zoo | imagenet-classification | W1A1+FP ★ | 28.2 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |
| `larq-zoo-quicknet` | larq-zoo | imagenet-classification | W1A1+FP ★ | 53.2 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host), device: pending |
| `larq-zoo-quicknet-large` | larq-zoo | imagenet-classification | W1A1+FP ★ | 93.7 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |
| `larq-zoo-quicknet-small` | larq-zoo | imagenet-classification | W1A1+FP ★ | 50.8 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + device), device: pending |
| `larq-zoo-quicknet-xl-legacy` | larq-zoo | imagenet-classification | W1A1+FP ★ | 88.7 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `larq-zoo-real-to-binary-net` | larq-zoo | imagenet-classification | W1A1+FP ★ | 48.3 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |
| `larq-zoo-xnornet` | larq-zoo | imagenet-classification | W1A1+FP ★ | 249.6 MB | Apache-2.0 / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); device pending, device: pending |

## missed-bnns-2026-10-30

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `aleph-alpha-1bw-llama-30b-1bit` | 1-bit-wonder (k-means QAT LLM) | causal-language-modeling (chat | binary-weight-only (W1A-fp) | 4997.2 MB | NOASSERTION (GitHub API) / Open Aleph License (HF card license othe | unverified |  |
| `bmxnet-v1-lenet-mnist` | BMXNet (HPI) | mnist-classification | W1A1+FP ★ | 0.2 MB | Apache-2.0 / unspecified (Apache-2.0 repo) | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bmxnet-v1-resnet18-cifar10` | BMXNet (HPI) | cifar10-classification | W1A1+FP ★ | 1.5 MB | Apache-2.0 / unspecified (Apache-2.0 repo) | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bmxnet-v1-resnet18-imagenet` | BMXNet (HPI) | imagenet-classification | W1A1+FP ★ | 3.6 MB | Apache-2.0 / unspecified (Apache-2.0 repo) | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bmxnet-v1-resnet18-imagenet-fp1st` | BMXNet (HPI) | imagenet-classification | W1A1+FP ★ | 4.1 MB | Apache-2.0 / unspecified (Apache-2.0 repo) | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bmxnet-v2-android-binary-densenet28` | BMXNet-v2 (HPI) | imagenet-classification | W1A1+FP ★ | 4.3 MB | GPL-3.0 (android-image-class / unspecified; shipped in a GPL-3.0 repo | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bmxnet-v2-android-resnete18` | BMXNet-v2 (HPI) | imagenet-classification | W1A1+FP ★ | 4.2 MB | GPL-3.0 (android-image-class / unspecified; shipped in a GPL-3.0 repo | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `brillsson-binarydensenet28` | BRILLsson (binary non-semantic | non-semantic speech representa | W1A1+FP ★ | 20.8 MB | none (no LICENSE file; GitHu / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `brillsson-meliusnet22` | BRILLsson (binary non-semantic | non-semantic speech representa | W1A1+FP ★ | 28.1 MB | none (no LICENSE file; GitHu / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `brillsson-quicknet` | BRILLsson (binary non-semantic | non-semantic speech representa | W1A1+FP ★ | 53.1 MB | none (no LICENSE file; GitHu / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `brve-llrvd` | BRVE (binarized raw video enha | low-light-raw-video-enhancemen | W1A1+FP ★ | 54.8 MB | MIT / unspecified (MIT repo; no separate weigh | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `brve-smoid` | BRVE (binarized raw video enha | low-light-raw-video-enhancemen | W1A1+FP ★ | 54.8 MB | MIT / unspecified (MIT repo; no separate weigh | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bsrc2023-binary-sr-x2-v0` | BSRC-2023 (Binary Super-Resolu | single-image-super-resolution- | W1A1+FP ★ | 5.6 MB | none (no LICENSE file) / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bsrc2023-binary-sr-x2-v1` | BSRC-2023 (Binary Super-Resolu | single-image-super-resolution- | W1A1+FP ★ | 7.0 MB | none (no LICENSE file) / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bsrc2023-binary-sr-x4-v0` | BSRC-2023 (Binary Super-Resolu | single-image-super-resolution- | W1A1+FP ★ | 3.0 MB | none (no LICENSE file) / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bsrc2023-binary-sr-x4-v1` | BSRC-2023 (Binary Super-Resolu | single-image-super-resolution- | W1A1+FP ★ | 3.2 MB | none (no LICENSE file) / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `shuaiz-bireal20-cifar10` | Bi-Real Net (third-party reimp | cifar10-classification | W1A1+FP ★ | 3.3 MB | none (GitHub API license nul / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bimacosr-realsr-x4` | BiMaCoSR (binary one-step diff | real-world-image-super-resolut | W1A1+FP ★ | 486.5 MB | MIT / unspecified (MIT repo) | unverified |  |
| `biper-vgg-small-cifar10` | BiPer | cifar10-classification | W1A1+FP ★ | 37.3 MB | MIT / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bulat-binary-face-alignment-aflw` | Binarized landmark localizers  | face-alignment (68 landmarks,  | W1A1+FP ★ | 1.4 MB | BSD-3-Clause-like (Universit / unspecified (same repo licence presumabl | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bulat-binary-human-pose-mpii` | Binarized landmark localizers  | human-pose-estimation (MPII, 1 | W1A1+FP ★ | 1.3 MB | BSD-3-Clause-like (Universit / unspecified (same repo licence presumabl | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `prism-bonsai-1.7b-mlx-1bit` | Bonsai 1-bit LLMs (PrismML) | causal-language-modeling (chat | binary-weight-only (W1A-fp) | 269.1 MB | Apache-2.0 (HF cardData.lice / Apache-2.0 | verified-pretrained |  |
| `prism-bonsai-27b-q1_0-gguf` | Bonsai 1-bit LLMs (PrismML) | causal-language-modeling (chat | binary-weight-only (W1A-fp) | 3803.5 MB | Apache-2.0 (HF cardData.lice / Apache-2.0 | unverified |  |
| `prism-bonsai-4b-q1_0-gguf` | Bonsai 1-bit LLMs (PrismML) | causal-language-modeling (chat | binary-weight-only (W1A-fp) | 572.3 MB | Apache-2.0 (HF cardData.lice / Apache-2.0 | unverified |  |
| `prism-bonsai-8b-q1_0-gguf` | Bonsai 1-bit LLMs (PrismML) | causal-language-modeling (chat | binary-weight-only (W1A-fp) | 1158.7 MB | Apache-2.0 (HF cardData.lice / Apache-2.0 | unverified |  |
| `eevbnn-mnist-cifar10-verifiable-bnns` | EEV verifiable BNNs (Jia & Rin | mnist-classification / cifar10 | W1A1+FP ★ | 428.0 MB | none (no LICENSE) / unspecified | unverified |  |
| `ebconv-expert-binary-resnet-imagenet` | Expert Binary Networks (Bulat  | imagenet-classification | W1A1+FP ★ | 173.7 MB | MIT / unspecified (MIT repo, weights on person | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `qnn-mo-pynq-dorefanet-w1a2-imagenet` | FINN / QNN-MO-PYNQ | imagenet-classification | W1A2 |  | BSD-3-Clause / unspecified (BSD-3-Clause repo) | unverified |  |
| `qnn-mo-pynq-tinier-yolo-w1a3-voc` | FINN / QNN-MO-PYNQ | object-detection (PASCAL VOC) | multi-bit/low-bit |  | BSD-3-Clause / unspecified (BSD-3-Clause repo) | unverified |  |
| `bnn-pynq-cnv-w1a1-svhn-folded` | FINN/BNN-PYNQ | svhn-classification | W1A1+INT8 ★ |  | BSD-3-Clause / unspecified (BSD-3-Clause repo) | unverified |  |
| `bnn-pynq-cnv-w1a2-cifar10` | FINN/BNN-PYNQ | cifar10-classification | W1A2 | 6.2 MB | BSD-3-Clause / unspecified (BSD-3-Clause repo) | verified-pretrained |  |
| `bnn-pynq-lfc-w1a1-nist-chars-folded` | FINN/BNN-PYNQ | character-classification (NIST | W1A1+INT8 ★ |  | BSD-3-Clause / unspecified (BSD-3-Clause repo) | unverified |  |
| `bnn-pynq-lfc-w1a2-mnist` | FINN/BNN-PYNQ | mnist-classification | W1A2 | 11.7 MB | BSD-3-Clause / unspecified (BSD-3-Clause repo) | verified-pretrained |  |
| `finn-examples-resnet50-w1a2-imagenet` | FINN/Brevitas | imagenet-classification | W1A2 | 8.3 MB | BSD-3-Clause / unspecified (BSD-3-Clause repo) | verified-pretrained |  |
| `aliasgharshinwari-finn-cnv-w1a1-cifar10` | FINN/Brevitas BNN-PYNQ | cifar10-classification | W1A1+INT8 ★ | 6.3 MB | none declared (GitHub API li / unspecified | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `brevitas-bnn-pynq-lfc-1w2a` | FINN/Brevitas BNN-PYNQ | mnist-classification | W1A2 | 11.7 MB | BSD-3-Clause-style (Brevitas / unspecified (released with Brevitas) | verified-pretrained |  |
| `brevitas-bnn-pynq-sfc-1w2a` | FINN/Brevitas BNN-PYNQ | mnist-classification | W1A2 | 1.4 MB | BSD-3-Clause-style (Brevitas / unspecified (released with Brevitas) | verified-pretrained |  |
| `brevitas-bnn-pynq-tfc-1w2a` | FINN/Brevitas BNN-PYNQ | mnist-classification | W1A2 | 0.3 MB | BSD-3-Clause-style (Brevitas / unspecified (released with Brevitas) | verified-pretrained |  |
| `finn-examples-cnv-w1a1-gtsrb` | FINN/Brevitas BNN-PYNQ | gtsrb-classification (traffic  | W1A1+INT8 ★ | 0.4 MB | BSD-3-Clause (finn-examples) / unspecified separately; QONNX zoo folder | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ftl-deepfake-bnext-t-cocofake` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 119.8 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-t-cocofake-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 358.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed: host x86 reference/neon bit-exact with numpy, stage all PASS (--explain); emulated arm64 neon bit-exact, optimized PASS (--explain); OnePlus Open and Moto e13 outputs (neon, optimized; 128 samples) byte-identical to the emulated references, device: see speed |
| `ftl-deepfake-bnext-s-cocofake` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 268.8 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-s-cocofake-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 805.4 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed: host x86 reference/neon bit-exact with numpy, stage all PASS (--explain); emulated arm64 neon bit-exact, optimized PASS (--explain); OnePlus Open and Moto e13 outputs (neon, optimized; 128 samples) byte-identical to the emulated references, device: see speed |
| `ftl-deepfake-bnext-m-cocofake` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 532.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-m-cocofake-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 1596.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed: host x86 reference/neon bit-exact with numpy, stage all PASS (--explain); emulated arm64 neon bit-exact, optimized PASS (--explain); OnePlus Open and Moto e13 outputs (neon, optimized; 128 samples) byte-identical to the emulated references, device: see speed |
| `ftl-deepfake-bnext-t-dffd` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 119.8 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-t-dffd-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 358.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-s-dffd` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 268.8 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-s-dffd-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 805.4 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-m-dffd` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 532.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-m-dffd-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 1596.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-t-cifake` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 119.8 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-t-cifake-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 358.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-s-cifake` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 268.8 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-s-cifake-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 805.4 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-m-cifake` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 532.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ftl-deepfake-bnext-m-cifake-unfrozen` | Faster Than Lies (binary deepf | deepfake-image-detection (bina | W1A1+FP ★ | 1596.7 MB | none: the repository has no  / no explicit weights licence (no LICENSE  | verified-pretrained | converted, verif: passed (Level 3, numpy executor), device: not run (same network as the measured model of the same size) |
| `ir2net-resnet18-imagenet-cifar` | IR2Net (information restrictio | imagenet-classification | W1A1+FP ★ |  | none (no LICENSE file) / unspecified | unverified |  |
| `larqface-binarydensenet28-msceleb` | Larq binary face recognition ( | face-recognition (embedding; v | W1A1+FP ★ | 21.2 MB | MIT / non-commercial research only (README: 't | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `larqface-binarydensenet37-msceleb` | Larq binary face recognition ( | face-recognition (embedding; v | W1A1+FP ★ | 35.7 MB | MIT / non-commercial research only (README: 't | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `larqface-binarydensenet45-msceleb` | Larq binary face recognition ( | face-recognition (embedding; v | W1A1+FP ★ | 56.7 MB | MIT / non-commercial research only (README: 't | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `larqface-binaryfacenet-msceleb` | Larq binary face recognition ( | face-recognition (embedding; v | W1A1+FP ★ | 2.4 MB | MIT / non-commercial research only (README: 't | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `larqface-quicknet-msceleb` | Larq binary face recognition ( | face-recognition (embedding; v | W1A1+FP ★ | 53.5 MB | MIT / non-commercial research only (README: 't | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `monodiffuse-cifar10-w1a1-core-seed0` | MonoDiffuse (one-bit diffusion | unconditional-image-generation | W1A1+FP ★ | 0.9 MB | MIT / MIT (repo artifacts; no separate terms) | verified-pretrained | deferred, verif: not-started, device: not-started |
| `ovsw-resnet-vgg-1w1a` | OvSW (Overcoming Silent Weight | imagenet-classification | W1A1+FP ★ |  | none (GitHub API license non / unspecified | download-failed |  |
| `rbnn-resnet20-bireal-cifar10` | RBNN | cifar10-classification | W1A1+FP ★ | 1.2 MB | none (no LICENSE file; GitHu / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `rbnn-vgg-small-cifar10` | RBNN | cifar10-classification | W1A1+FP ★ | 18.8 MB | none (no LICENSE file; GitHu / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `recu-resnet18-cifar10` | ReCU | cifar10-classification | W1A1+FP ★ | 89.5 MB | none (no LICENSE; GitHub API / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `recu-vgg-small-cifar10` | ReCU | cifar10-classification | W1A1+FP ★ | 37.3 MB | none (no LICENSE; GitHub API / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `svnet-dgcnn-binary-kd-modelnet40` | SVNet (SO(3)-equivariant binar | point-cloud-classification (Mo | W1A1+FP, binary-weight-only (W1A-fp) ★ | 18.8 MB | none (no LICENSE file) / unspecified | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `svnet-dgcnn-binary-kd-scanobjectnn` | SVNet (SO(3)-equivariant binar | point-cloud-classification (Sc | W1A1+FP, binary-weight-only (W1A-fp) ★ | 18.7 MB | none (no LICENSE file) / unspecified | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `svnet-dgcnn-binary-kd-shapenet` | SVNet (SO(3)-equivariant binar | point-cloud-part-segmentation  | W1A1+FP, binary-weight-only (W1A-fp) ★ | 16.4 MB | none (no LICENSE file) / unspecified | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `svnet-dgcnn-binary-modelnet40` | SVNet (SO(3)-equivariant binar | point-cloud-classification (Mo | W1A1+FP, binary-weight-only (W1A-fp) ★ | 18.8 MB | none (no LICENSE file) / unspecified | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `svnet-pointnet-binary-modelnet40` | SVNet (SO(3)-equivariant binar | point-cloud-classification (Mo | W1A1+FP, binary-weight-only (W1A-fp) ★ | 29.9 MB | none (no LICENSE file) / unspecified | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `siman-resnet18-cifar10` | SiMaN | cifar10-classification | W1A1+FP ★ | 134.2 MB | none (no LICENSE; GitHub API / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `siman-resnet20-cifar10` | SiMaN | cifar10-classification | W1A1+FP ★ | 3.3 MB | none (no LICENSE; GitHub API / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `siman-vgg-small-cifar10` | SiMaN | cifar10-classification | W1A1+FP ★ | 55.6 MB | none (no LICENSE; GitHub API / unspecified (no separate weights licence | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `snn-subbit-reactnet18-w0.67a1-imagenet` | Sub-bit Neural Networks (SNN,  | imagenet-classification | W1A1+FP ★ | 140.8 MB | MIT / unspecified (MIT repo) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `snn-subbit-reactnet18-w1a1-imagenet` | Sub-bit Neural Networks (SNN,  | imagenet-classification | W1A1+FP ★ | 140.6 MB | MIT / unspecified (MIT repo) | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `terdit-4.2b-imagenet256` | TerDiT (ternary diffusion tran | class-conditional-image-genera | ternary (W1.58A16) | 1099.5 MB | MIT / MIT (HF card license: mit) | unverified |  |
| `terdit-4.2b-imagenet512` | TerDiT (ternary diffusion tran | class-conditional-image-genera | ternary (W1.58A16) | 1099.5 MB | MIT / MIT (HF card license: mit) | unverified |  |
| `terdit-600m-imagenet256` | TerDiT (ternary diffusion tran | class-conditional-image-genera | ternary (W1.58A16) | 175.1 MB | MIT / MIT (HF card license: mit) | verified-pretrained |  |
| `liqunma-onebit-mamba2-1.3b` | bi-mamba (1-bit SSM) | causal-language-modeling | binary-weight-only (W1A-fp) | 5788.9 MB | none found / unspecified (no card, no licence) | unverified |  |
| `liqunma-onebit-mamba2-2.7b` | bi-mamba (1-bit SSM) | causal-language-modeling | binary-weight-only (W1A-fp) | 11328.5 MB | none found / unspecified (no card, no licence) | unverified |  |
| `binarymos-llama2-7b-e4` | binarymos (token-adaptive bina | causal-language-modeling | binary-weight-only (W1A-fp) | 4949.6 MB | MIT (GitHub API) / CC-BY-NC-4.0 (HF card) | unverified |  |
| `bitmamba2-1b` | bitmamba (ternary SSM) | causal-language-modeling | ternary (W1.58A8) | 644.3 MB | MIT / MIT | unverified |  |
| `bitmamba2-255m` | bitmamba (ternary SSM) | causal-language-modeling | ternary (W1.58A8) | 258.3 MB | MIT (GitHub API, BitMamba-2  / MIT (HF card) | verified-pretrained |  |
| `bitorch-densenet28-imagenet` | bitorch model hub (hpi-xnor) | imagenet-classification | W1A1+FP ★ | 61.8 MB | GPL-3.0 (hpi-xnor/bitorch) / unspecified (no licence on the W&B artif | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bitorch-meliusnet22-imagenet` | bitorch model hub (hpi-xnor) | imagenet-classification | W1A1+FP ★ | 186.3 MB | GPL-3.0 (hpi-xnor/bitorch) / unspecified (no licence on the W&B artif | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bitorch-resnete18-imagenet` | bitorch model hub (hpi-xnor) | imagenet-classification | W1A1+FP ★ | 243.0 MB | GPL-3.0 (hpi-xnor/bitorch) / unspecified (no licence on the W&B artif | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `euclidstellar-tinystories-1bit-llm` | community ternary/1-bit LMs | causal-language-modeling (Tiny | ternary (W1.58A8) | 2.3 MB | MIT / MIT (HF card) | unverified |  |
| `qep-onebit-command-a-1bit-extreme` | community ternary/1-bit LMs | causal-language-modeling (chat | binary-weight-only (W1A-fp) | 32424.1 MB | n/a / cc-by-nc-4.0 | unverified |  |
| `tbt-bart-base-w1a1-summarization` | community ternary/1-bit LMs | abstractive-summarization (XSU | W1A1+FP ★ |  | NOASSERTION (GitHub API) / unknown | download-failed |  |
| `ternarybert-bert-base-mnli` | community ternary/1-bit LMs | glue-mnli | ternary (W1.58A8) | 438.0 MB | none (GitHub API license nul / unspecified (no card) | unverified |  |
| `ternarybert-bert-base-sst2` | community ternary/1-bit LMs | glue-sst2 | ternary (W1.58A8) | 438.0 MB | none (GitHub API license nul / unspecified (no card) | unverified |  |
| `thox-ai-thoxmicro-1bit-16m` | community ternary/1-bit LMs | causal-language-modeling (Tiny | ternary (W1.58A8) | 8.0 MB | n/a / other ('pending-review') | unverified |  |
| `hls4ml-keras-3layer-binary-jettag` | hls4ml example-models (binary/ | jet-tagging (LHC jet substruct | W1A1+FP ★ | 0.1 MB | none declared (example-model / unspecified | verified-pretrained | deferred, verif: not-started, device: not-started |
| `hls4ml-keras-3layer-binarydense-relu-jettag` | hls4ml example-models (binary/ | jet-tagging (LHC jet substruct | binary-weight-only (W1A-fp) | 0.1 MB | none declared (example-model / unspecified | verified-pretrained |  |
| `hls4ml-keras-3layer-ternary-jettag` | hls4ml example-models (binary/ | jet-tagging (LHC jet substruct | multi-bit/low-bit | 0.1 MB | none declared (example-model / unspecified | verified-pretrained |  |
| `hls4ml-keras-bnn-jettag` | hls4ml example-models (binary/ | jet-tagging (LHC jet substruct | W1A1+FP ★ | 0.1 MB | none declared (example-model / unspecified | verified-pretrained | deferred, verif: not-started, device: not-started |
| `rajmali-zebracam-bcnn-v3` | hobby FPGA BNN | zebra-crossing detection (bina | W1A1+FP ★ | 0.4 MB | MIT / MIT (repo) | verified-pretrained | deferred, verif: not-started, device: not-started |
| `microsoft-bitnet-b1.58-2b-4t-bf16` | microsoft-bitnet | causal-language-modeling (chat | ternary (W1.58A8) | 4825.7 MB | MIT / MIT | unverified |  |
| `microsoft-bitnet-embedding-0p6b` | microsoft-bitnet | text-embedding (MTEB / retriev | ternary (W1.58A8) | 427.9 MB | MIT / MIT | unverified |  |
| `microsoft-bitnet-embedding-270m` | microsoft-bitnet | text-embedding (MTEB / retriev | ternary (W1.58A8) | 367.5 MB | MIT / MIT | unverified |  |
| `microsoft-vibevoice-asr-bitnet` | microsoft-bitnet | automatic-speech-recognition | ternary (W1.58A8) | 992.9 MB | MIT / MIT | unverified |  |
| `hf1bitllm-llama3-8b-1.58-100b-tokens` | microsoft-bitnet (HF fine-tune | causal-language-modeling (chat | ternary (W1.58A8) | 3846.8 MB | n/a / unspecified on card (Llama 3 base licenc | unverified |  |
| `1bitllm-bitnet-b1.58-3b` | microsoft-bitnet (community re | causal-language-modeling | ternary (W1.58A8) | 4993.7 MB | MIT (HF card) / MIT | unverified |  |
| `1bitllm-bitnet-b1.58-large` | microsoft-bitnet (community re | causal-language-modeling | ternary (W1.58A8) | 2915.4 MB | MIT (HF card) / MIT | unverified |  |
| `1bitllm-bitnet-b1.58-xl` | microsoft-bitnet (community re | causal-language-modeling | ternary (W1.58A8) | 4960.1 MB | MIT (HF card) / MIT | unverified |  |
| `nousresearch-olmo-bitnet-1b` | microsoft-bitnet (community re | causal-language-modeling | ternary (W1.58A8) | 4707.4 MB | Apache-2.0 (HF card) / Apache-2.0 | unverified |  |
| `openbmb-bitcpm4-0p5b` | openbmb-bitcpm (ternary LLM) | causal-language-modeling | ternary (W1.58A-fp) | 867.8 MB | Apache-2.0 (OpenBMB/MiniCPM) / Apache-2.0 | unverified |  |
| `openbmb-bitcpm4-1b` | openbmb-bitcpm (ternary LLM) | causal-language-modeling | ternary (W1.58A-fp) | 2720.6 MB | Apache-2.0 (OpenBMB/MiniCPM) / Apache-2.0 | unverified |  |
| `facebook-mobilellm-paretoq-125m-1bit` | paretoq-mobilellm (Meta low-bi | causal-language-modeling | binary-weight-only (W1A-fp) | 249.4 MB | BSD-3-Clause per README (Git / other (FAIR licence; gated, manual appro | unverified |  |
| `facebook-mobilellm-paretoq-1p5b-1bit` | paretoq-mobilellm (Meta low-bi | causal-language-modeling | binary-weight-only (W1A-fp) | 3022.5 MB | BSD-3-Clause per README (Git / other (FAIR licence; gated, manual appro | unverified |  |
| `prism-ternary-bonsai-2-2b-ptq1_0-gguf` | prism-bonsai (ternary LLM) | causal-language-modeling (chat | ternary (W1.58A-fp) | 429.2 MB | Apache-2.0 (HF cardData.lice / Apache-2.0 | unverified |  |
| `prism-ternary-bonsai-8b-q2_0-gguf` | prism-bonsai (ternary LLM) | causal-language-modeling (chat | ternary (W1.58A-fp) | 2182.2 MB | Apache-2.0 (HF cardData.lice / Apache-2.0 | unverified |  |
| `ptq161-llama2-7b` | ptq161 (binary PTQ LLM) | causal-language-modeling | binary-weight-only (W1A-fp) | 4939.0 MB | none (GitHub API license nul / unspecified (no card) | unverified |  |
| `spectra-trilm-3.9b-unpacked` | spectra-trilm (ternary LLM sui | causal-language-modeling | ternary (W1.58A16) | 4992.5 MB | none (GitHub API null) / Apache-2.0 | unverified |  |
| `spectra-trilm-99m-unpacked` | spectra-trilm (ternary LLM sui | causal-language-modeling | ternary (W1.58A16) | 199.5 MB | none (GitHub API license=nul / Apache-2.0 (HF card) | verified-pretrained |  |
| `tiiuae-falcon-e-1b-instruct` | tiiuae-falcon-bitnet | causal-language-modeling (chat | ternary (W1.58A8) | 665.0 MB | NOASSERTION (tiiuae/onebitll / Falcon-LLM License (TII Falcon License 2 | unverified |  |
| `tiiuae-falcon-e-3b-instruct` | tiiuae-falcon-bitnet | causal-language-modeling (chat | ternary (W1.58A8) | 998.6 MB | NOASSERTION (tiiuae/onebitll / Falcon-LLM License (TII Falcon License 2 | unverified |  |
| `tiiuae-falcon3-10b-instruct-1.58bit` | tiiuae-falcon-bitnet | causal-language-modeling (chat | ternary (W1.58A8) | 3986.2 MB | NOASSERTION (tiiuae/onebitll / Falcon-LLM License (TII Falcon License 2 | unverified |  |
| `tiiuae-falcon3-1b-instruct-1.58bit` | tiiuae-falcon-bitnet | causal-language-modeling (chat | ternary (W1.58A8) | 1357.0 MB | NOASSERTION (tiiuae/onebitll / Falcon-LLM License (TII Falcon License 2 | unverified |  |

## modern-binary-cnn

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `adabin-resnet20-cifar10` | AdaBin | cifar10-classification | W1A1+FP ★ | 1.2 MB | MIT (BinaryNetworks/AdaBin/L / MIT by co-location in the same directory | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `adabin-resnet20-cifar10-mindspore` | AdaBin | cifar10-classification | W1A1+FP ★ | 1.1 MB | Apache-2.0 (mindspore/models / no explicit weights licence (inherits re | verified-pretrained | deferred, verif: not-started, device: not-started |
| `bcdnet-a-imagenet` | BCDNet | imagenet-classification | W1A1+FP ★ | 494.2 MB | none (gh api license null) / none stated | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bnas-d-imagenet` | BNAS | imagenet-classification | W1A1+FP ★ | 227.7 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnas-a-cifar10` | BNAS (Learning Architectures f | cifar10-classification | W1A1+FP ★ | 22.5 MB | none (gh api license null) / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnext-large` | BNext | imagenet-classification | W1A1+FP ★ | 1307.5 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnext-middle` | BNext | imagenet-classification | W1A1+FP ★ | 538.8 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnext-small` | BNext | imagenet-classification | W1A1+FP ★ | 274.9 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bnext-tiny` | BNext | imagenet-classification | W1A1+FP ★ | 123.9 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ftbnn-birealnet-v2` | Bi-Real-v2 | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `biper-resnet18-cifar10` | BiPer | cifar10-classification | W1A1+FP ★ | 88.2 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `biper-resnet18-imagenet` | BiPer | imagenet-classification | W1A1+FP ★ | 93.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `biper-resnet20-cifar10` | BiPer | cifar10-classification | W1A1+FP ★ | 2.3 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bd-bnn` | Bimodal BNN | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `boolnet` | BoolNet | imagenet-classification | research-only (no verified weights) |  | none / ? | no-public-weights |  |
| `dybnn` | DyBNN | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `fda-bnn-resnet20-cifar10-mindspore` | FDA-BNN | cifar10-classification | W1A1+FP ★ | 1.1 MB | Apache-2.0 (source file head / no explicit weights licence (inherits Ap | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `insta-bnn` | INSTA-BNN | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `lcr-bnn` | LCR-BNN | imagenet-classification | research-only (no verified weights) |  | none / ? | no-public-weights |  |
| `pokebnn` | PokeBNN | imagenet-classification | research-only (no verified weights), W1A1+INT8 ★ |  | Apache-2.0 / ? | no-public-weights |  |
| `proxybnn` | ProxyBNN | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `rbnn-resnet18-cifar10` | RBNN | cifar10-classification | W1A1+FP ★ | 45.1 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `rbnn-resnet18-imagenet` | RBNN | imagenet-classification | W1A1+FP ★ | 47.1 MB | none (no LICENSE; gh api lic / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `rbnn-resnet20-cifar10` | RBNN | cifar10-classification | W1A1+FP ★ | 1.2 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `rbnn-resnet34-imagenet` | RBNN | imagenet-classification | W1A1+FP ★ | 87.9 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `rbonn-reactnet-resnet18-imagenet` | RBONN | imagenet-classification | W1A1+FP ★ | 47.0 MB | none (gh api license null) / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `rebnn-reactnet-resnet18` | ReBNN | imagenet-classification | research-only (no verified weights) |  | none / ? | no-public-weights |  |
| `recu-resnet18-imagenet` | ReCU | imagenet-classification | W1A1+FP ★ | 93.6 MB | none (mirror has no LICENSE) / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `recu-resnet20-cifar10` | ReCU | cifar10-classification | W1A1+FP ★ | 2.2 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `recu-resnet34-imagenet` | ReCU | imagenet-classification | W1A1+FP ★ | 174.6 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `sa-bnn` | SA-BNN | imagenet-classification | W1A1+FP ★ |  | ? / ? | no-public-weights |  |
| `sa-bnn-resnet18-imagenet` | SA-BNN | imagenet-classification | W1A1+FP ★ | 140.5 MB | none stated / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bats` | Searched BNN | imagenet-classification | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `siman-resnet18-imagenet` | SiMaN | imagenet-classification | W1A1+FP ★ | 137.6 MB | none (gh api license null) / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `siman-resnet34-imagenet` | SiMaN | imagenet-classification | W1A1+FP ★ | 259.0 MB | none / none stated | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |

## nncf-binarized-resnet18-xnor-dorefa

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `nncf-resnet18-binarization-dorefa` | nncf-binarized-resnet18-xnor-d | imagenet-classification | W1A1+FP ★ | 327.7 MB | Apache-2.0 / Not stated separately. Presumably Apache | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `nncf-resnet18-binarization-dorefa-v2.7.0` | nncf-binarized-resnet18-xnor-d | imagenet-classification | W1A1+FP ★ | 327.8 MB | Apache-2.0 / Not stated separately (unverified) | verified-pretrained | duplicate, verif: not separately run (identical tensors), device: see duplicate_of |
| `nncf-resnet18-binarization-xnor` | nncf-binarized-resnet18-xnor-d | imagenet-classification | W1A1+FP ★ | 327.7 MB | Apache-2.0 / Not stated separately. Presumably Apache | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `nncf-resnet18-binarization-xnor-v2.7.0` | nncf-binarized-resnet18-xnor-d | imagenet-classification | W1A1+FP ★ | 327.8 MB | Apache-2.0 / Not stated separately (unverified) | verified-pretrained | duplicate, verif: not separately run (identical tensors), device: see duplicate_of |

## openvino-omz-binary-ir-models

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `omz-face-detection-adas-binary-0001` | OpenVINO Open Model Zoo binary | face-detection (SSD) | W1A1+FP ★ | 1.8 MB | Apache-2.0 / Apache-2.0 (model.yml license: opencv/op | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `omz-pedestrian-detection-adas-binary-0001` | OpenVINO Open Model Zoo binary | pedestrian-detection (SSD) | W1A1+FP ★ | 2.3 MB | Apache-2.0 / Apache-2.0 | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `omz-resnet18-xnor-binary-onnx-0001` | OpenVINO Open Model Zoo binary | imagenet-classification | W1A1+FP ★ | 4.2 MB | Apache-2.0 / Apache-2.0 | verified-pretrained | converted, verif: passed (host + qemu-emulated arm64); device pending, device: pending |
| `omz-resnet18-xnor-binary-onnx-0001-fp16` | OpenVINO Open Model Zoo binary | imagenet-classification | W1A1+FP ★ | 2.8 MB | Apache-2.0 / Apache-2.0 | verified-pretrained | converted, verif: passed (host), device: pending |
| `omz-resnet50-binary-0001` | OpenVINO Open Model Zoo binary | imagenet-classification | W1A1+FP ★ | 22.2 MB | Apache-2.0 / Apache-2.0 (model.yml license field poin | verified-pretrained | converted, verif: passed (host + qemu-emulated arm64, --explain); device pending, device: pending |
| `omz-resnet50-binary-0001-fp16` | OpenVINO Open Model Zoo binary | imagenet-classification | W1A1+FP ★ | 12.4 MB | Apache-2.0 / Apache-2.0 | verified-pretrained | converted, verif: passed (host, --explain), device: pending |
| `omz-vehicle-detection-adas-binary-0001` | OpenVINO Open Model Zoo binary | vehicle-detection (SSD) | W1A1+FP ★ | 2.2 MB | Apache-2.0 / Apache-2.0 | verified-pretrained | unsupported, verif: not-started, device: not-started |

## other-tasks-bnn

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `lws-det` | 1-bit detectors (Beihang) | object-detection | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `bbcu-deblocking` | BBCU (Basic Binary Convolution | jpeg-artifact-reduction (grays | W1A1+FP ★ | 2.7 MB | MIT (BBCU-denoiseAndblocking / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bbcu-denoise` | BBCU (Basic Binary Convolution | color-image-denoising (AWGN bl | W1A1+FP ★ | 2.7 MB | MIT for the BBCU-denoiseAndb / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bbcu-sr-l-x2` | BBCU (Basic Binary Convolution | single-image-super-resolution- | W1A1+FP ★ | 5.6 MB | Apache-2.0 (BBCU-SR folder); / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bbcu-sr-l-x4` | BBCU (Basic Binary Convolution | single-image-super-resolution- | W1A1+FP ★ | 6.2 MB | BBCU-SR folder: Apache-2.0 ( / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bbcu-sr-m-x2` | BBCU (Basic Binary Convolution | single-image-super-resolution- | W1A1+FP ★ | 5.6 MB | Apache-2.0 (BBCU-SR folder) / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bbcu-sr-m-x4` | BBCU (Basic Binary Convolution | single-image-super-resolution- | W1A1+FP ★ | 6.2 MB | Apache-2.0 (BBCU-SR folder) / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bi-diffsr-x2` | BI-DiffSR | single-image-super-resolution- | W1A1+FP ★ | 205.7 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bi-diffsr-x4` | BI-DiffSR | single-image-super-resolution- | W1A1+FP ★ | 205.7 MB | Apache-2.0 / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bidet-faster-rcnn` | BiDet | object-detection-voc/coco | research-only (no verified weights) |  | MIT / no explicit weights licence (inherits re | no-public-weights |  |
| `bidet-ssd300-voc` | BiDet | object-detection-voc07 | W1A1+FP ★ | 131.0 MB | MIT (Copyright (c) 2019 Ziyi / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bifsmn-speechcommands-v1-12` | BiFSMN | keyword-spotting (Google Speec | unsupported/incomplete | 4.9 MB | none (no LICENSE file; GitHu / no explicit weights licence (repo has no | verified-pretrained |  |
| `bifsmnv2` | BiFSMN | keyword-spotting | research-only (no verified weights) |  | none (GitHub API license=nul / ? | no-public-weights |  |
| `bimatting-bimobilenet-backbone` | BiMatting | imagenet-classification (backb | W1A1+FP ★ | 53.4 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `bimatting-rvm-stage4` | BiMatting | video-matting (VideoMatte240K) | W1A1+FP ★ | 17.4 MB | MIT (BiMatting LICENSE, Copy / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bipointnet` | BiPointNet | point-cloud-classification (Mo | research-only (no verified weights) |  | MIT / ? | no-public-weights |  |
| `bisci-bbcu-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A1+FP ★ | 2.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bisci-biconnect-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A-fp | 2.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained |  |
| `bisci-bireal-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A1+FP ★ | 2.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bisci-bnn-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A1+FP ★ | 2.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bisci-btm-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A1+FP ★ | 2.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bisci-irnet-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A1+FP ★ | 2.7 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bisci-reactnet-sci` | BiSCI (Binarized Spectral Comp | hyperspectral-sci-reconstructi | W1A1+FP ★ | 2.8 MB | MIT / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bisrnet-cassi-real` | BiSCI (Binarized Spectral Comp | hyperspectral-snapshot-compres | unsupported/incomplete | 2.9 MB | MIT / no explicit weights licence (inherits re | verified-pretrained |  |
| `bisrnet-cassi-simulation` | BiSCI (Binarized Spectral Comp | hyperspectral-snapshot-compres | W1A1+FP ★ | 2.8 MB | MIT (Copyright (c) 2023 Yuan / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |
| `bidm-ldm` | Binarized diffusion | unconditional-image-generation | research-only (no verified weights) |  | none (GitHub API license=nul / ? | no-public-weights |  |
| `binarydm-ldm` | Binarized diffusion | unconditional-image-generation | research-only (no verified weights) |  | none (GitHub API license=nul / ? | no-public-weights |  |
| `bi-gcn` | Binary graph nets | node-classification (citation  | research-only (no verified weights) |  | none (GitHub API license=nul / ? | no-public-weights |  |
| `binary-gnn-bahri` | Binary graph nets | point-cloud / graph classifica | research-only (no verified weights) |  | MIT / ? | no-public-weights |  |
| `bidense` | Binary segmentation/depth | semantic-segmentation / depth- | research-only (no verified weights) |  | Apache-2.0 / ? | no-public-weights |  |
| `bam-sr` | Binary super-resolution | single-image-super-resolution | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `btm-sr-original` | Binary super-resolution | single-image-super-resolution | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `frb-sr` | Binary super-resolution | single-image-super-resolution | research-only (no verified weights) |  | ? / ? | no-public-weights |  |
| `wonnado-binary-nets-resnet18-duke` | Group-Net third-party reimplem | person-reid (DukeMTMC-reID) | unsupported/incomplete | 46.2 MB | none (GitHub API license=nul / no explicit weights licence (repo has no | verified-pretrained |  |
| `ida-det-bires18-imagenet-backbone` | IDa-Det (1-bit detectors, Beih | imagenet-classification (backb | W1A1+FP ★ | 47.0 MB | none (no LICENSE) / no explicit weights licence (repo has no | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `ida-det-faster-rcnn-bir18-voc` | IDa-Det (1-bit detectors, Beih | object-detection-voc07 | W1A1+FP ★ | 133.4 MB | No LICENSE file (GitHub API  / no explicit weights licence (inherits re | verified-pretrained | unsupported, verif: not-started, device: not-started |

## reactnet-birealnet-irnet

Research log: the development log (not published)

| model_id | family | task | precision | size | code / weights licence | status | Bitino |
|---|---|---|---|---|---|---|---|
| `birealnet-official` | Bi-Real Net | imagenet-classification | research-only (no verified weights) |  | none stated (no LICENSE file / n/a (no weights) | no-public-weights |  |
| `dabnn-bireal18` | Bi-Real Net | imagenet-classification | W1A1+FP ★ | 47.0 MB | BSD-3-Clause (daBNN LICENSE, / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `dabnn-bireal18-stem` | Bi-Real Net | imagenet-classification | W1A1+FP ★ | 47.0 MB | BSD-3-Clause (daBNN; GitHub  / No explicit weights licence (inherits re | verified-pretrained | converted, verif: passed (host + emulated arm64); physical device pending, device: pending |
| `sajaddarabi-bireal18-fp` | Bi-Real Net | imagenet-classification | unsupported/incomplete | 46.8 MB | none stated (no LICENSE; Git / none stated | verified-pretrained |  |
| `irnet-official` | IR-Net | imagenet-classification / cifa | research-only (no verified weights) |  | none stated (no LICENSE file / n/a (no weights) | no-public-weights |  |
| `adambnn-reactnet-a-official` | ReActNet | imagenet-classification | research-only (no verified weights) |  | none stated (no LICENSE; Git / none stated | download-failed |  |
| `reactnet-a-mobilenet-official` | ReActNet | imagenet-classification | research-only (no verified weights) |  | none stated (no LICENSE file / none stated (weights unavailable) | download-failed |  |
| `reactnet-resnet18-official` | ReActNet | imagenet-classification | research-only (no verified weights) |  | none stated (no LICENSE; Git / none stated | download-failed |  |
| `lucifer443-mmcls-bnn-reimpl` | ReActNet / IR-Net / Bi-Real Ne | imagenet-classification | research-only (no verified weights) |  | Apache-2.0 (repo LICENSE, in / ? | no-public-weights |  |

## Searched without usable public weights

- **larq-zoo**: Real-to-Binary FP / BAN variants, StrongBaselineNet (BAN/BNN) and ResNet18FP (larq_zoo/literature/real_to_bin_nets.py): defined in code but have no ImageNet weights; _get_imagenet_weights_path raises 'No stored weights'.
- **larq-zoo**: BinaryResNetE 34/50/101/152 specs exist in resnet_e.py, but only the 18-layer weights are published.
- **larq-zoo**: Legacy QuickNet v0.2.1 and QuickNetLarge v0.2.1 (and v0.1.x/v0.2.0): superseded by v1.0. Release assets and hashes exist in history (@40f4cdc29f) but were not downloaded (deprecated).
- **larq-zoo**: Older releases (birealnet-v0.1.0/v0.2.0, binary_alexnet-v0.1.0/v0.2.0, xnornet-v0.1.0) were superseded and not downloaded.
- **larq-zoo**: Notop (include_top=False) weights for all models: hashes are in the source but the files were not downloaded or verified.
- **larq-zoo**: LCE conversion of each model was not run locally (that would require executing larq_zoo code). Support status comes from docs.larq.dev/zoo and the LCE converter source.
- **reactnet-birealnet-irnet**: ReActNet official ReActNet-A / ResNet checkpoints: all HKUST OneDrive links (4 revisions) return 404; owner site deleted; no Wayback or Hugging Face mirror
- **reactnet-birealnet-irnet**: AdamBNN official ReActNet-A checkpoint: same deleted OneDrive account (404)
- **reactnet-birealnet-irnet**: Rep_AdamBNN (imfinethanks): pan.seu.edu.cn link; host did not resolve from this machine (curl error 6), not tested further
- **reactnet-birealnet-irnet**: Bi-Real Net official Caffe/PyTorch weights: never published (issues #14, #19 unanswered)
- **reactnet-birealnet-irnet**: IR-Net official weights (ImageNet + CIFAR-10): never published; BiBench (htqin) README has no checkpoints
- **reactnet-birealnet-irnet**: Hugging Face hub search for reactnet/birealnet/bireal/irnet: no matching models
- **reactnet-birealnet-irnet**: S2-BNN (szq0214) ReActNet-A self-supervised checkpoints: MBZUAI SharePoint reachable (distill-only = 595,766,899 bytes, HEAD only); different method (self-supervised, likely no classifier head), left to a self-supervised-BNN family
- **reactnet-birealnet-irnet**: ReActNet-A-architecture derivatives with Google Drive weights (Sense-GVT/BCDNet, naver-ai/burn, VITA-Group/BNN_NoBN, Ruichen0424/AB-BNN): separate methods, not investigated here
- **reactnet-birealnet-irnet**: larq-zoo BiRealNet (Keras): verified weights exist but belong to the larq-zoo family (already in .cache/checkpoints/larq-zoo-birealnet)
- **reactnet-birealnet-irnet**: ShixiangLi/ReActNet_FPGA: FPGA C headers (conv_weight.h) only, no documented checkpoint; not investigated further
- **reactnet-birealnet-irnet**: MindSpore lyqcom/BiRealNet (Apache-2.0): training code only, no checkpoint link found
- **classical-early-bnn**: LQ-Nets (microsoft/LQ-Nets): training code only; no checkpoints in README, releases or issues
- **classical-early-bnn**: BinaryNet Theano (MatthieuCourbariaux/BinaryNet), Torch7 (itayhubara/BinaryNet), BinaryNet.pytorch: no weight files
- **classical-early-bnn**: BinaryConnect (MatthieuCourbariaux/BinaryConnect): no weights
- **classical-early-bnn**: ABC-Net: no official code; unofficial layog TF repo has no weights
- **classical-early-bnn**: BNN+ (arXiv 1812.11800): no code or weights located
- **classical-early-bnn**: XNOR-Net++ (arXiv 1909.13863): no official weights; 1adrianb/binary-networks-pytorch has no pretrained models
- **classical-early-bnn**: jiecaoyu/XNOR-Net-PyTorch LeNet-5 / NIN / AlexNet: all 4 Google Drive IDs return HTTP 404 (gdown: cannot retrieve public link)
- **classical-early-bnn**: BNN-PYNQ bnn/params/*: only FINN-folded hardware .bin weight/threshold tiles, not a standard checkpoint (not downloaded)
- **modern-binary-cnn**: ReCU ReActNet-ResNet18 (66.4%): cited by RBONN/ReBNN with a link to the official z-hXu/ReCU repo, which now returns 404; the yuanchunyu mirror README does not include it
- **modern-binary-cnn**: AdaBin ImageNet (ResNet-18 63.1/66.4, ResNet-34, AlexNet): not linked in the PyTorch repo or the MindSpore gitee README; only CIFAR ResNet-20 checkpoints exist
- **modern-binary-cnn**: FDA-BNN ImageNet / PyTorch: only the MindSpore CIFAR-10 ResNet-20 checkpoint exists
- **modern-binary-cnn**: Bi-Real++: no paper or repo by that name found (closest is Bi-Real Net V2 / FTBNN, which has no weights)
- **modern-binary-cnn**: Bethge 'Back to Simplicity' / BinaryDenseNet: official BMXNet-v2 / wiki holds result tables only; weights exist only via larq zoo (binary_densenet*, handled by the larq-zoo family) [CORRECTED 2026-10-30: BinaryDenseNet-28 weights are also public outside Larq Zoo: PyTorch via the bitorch model hub (bitorch-densenet28-imagenet) and an MXNet/BMXNet-v2 export in hpi-xnor/android-image-classification (bmxnet-v2-android-binary-densenet28). See the research notes (not published)]
- **modern-binary-cnn**: MeliusNet official BMXNet-v2 weights: none; only the larq zoo MeliusNet22 copy [CORRECTED 2026-10-30: HPI publishes PyTorch MeliusNet-22 ImageNet weights via the bitorch model hub (bitorch-meliusnet22-imagenet). See the research notes (not published)]
- **modern-binary-cnn**: ReBNN: README model link is '-'
- **modern-binary-cnn**: PokeBNN: AQT has training configs only; no checkpoint
- **modern-binary-cnn**: LCR-BNN, BoolNet, BATS, FTBNN (Bi-Real v2), BD-BNN: repos exist but have no weights
- **modern-binary-cnn**: INSTA-BNN, DyBNN, SA-BNN, ProxyBNN: no official code repository found
- **modern-binary-cnn**: E2FIF: binarized super-resolution (not classification); repo pppLang/E2FIF has no weights linked
- **modern-binary-cnn**: 'ReActNet with distillation' family: ReActNet itself is covered by the reactnet-birealnet-irnet family; RBONN, BCDNet and BNext (distillation-based) are covered here
- **modern-binary-cnn**: Not downloaded for size budget (available, ids in notes): BNAS-E/F/G/H ImageNet, BiPer-ResNet34 ImageNet, RBNN 120/200-epoch variants, ReCU vanilla variants, SiMaN CIFAR, BCDNet step-1
- **binary-bert-language**: BiT (facebookresearch/bit) checkpoints: all 15 SharePoint links in README return HTTP 404 on 2026-10-29; no HF mirror found
- **binary-bert-language**: BinaryBERT binarized checkpoints: not released (only FP DynaBERT teacher on Google Drive)
- **binary-bert-language**: BEBERT checkpoints: code repo TTTTTTris/BEBERT has none; arXiv lists no model URL
- **binary-bert-language**: Original BitNet (arXiv 2310.11453, W1A8): no official Microsoft weights found; only b1.58 ternary models released
- **binary-bert-language**: BinaryFormer: only a vision ViT paper (IEEE TII 2024) found, no language model and no code
- **binary-bert-language**: OneBit LLaMA-7B/13B, LLaMA2-13B: Drive folders exist but larger than LLaMA2-7B (>1.5 GB), not sized/downloaded
- **binary-bert-language**: BiLLM / PB-LLM / ARB-LLM: PTQ code only; third-party yang31210999 HF uploads are dense fake-quantized weights with no model card
- **binary-bert-language**: 1bitLLM/bitnet_b1_58-large (community reproduction, 2.9 GB fp safetensors): ternary and over cap, not downloaded [UPDATED 2026-10-30: recorded as unverified entries 1bitllm-bitnet-b1.58-large/-xl/-3b. See the research notes (not published)]
- **binary-bert-language**: prism-ml Ternary-Bonsai models: ternary (not strict BNN), not investigated further [UPDATED 2026-10-30: recorded as prism-ternary-bonsai-8b-q2_0-gguf and prism-ternary-bonsai-2-2b-ptq1_0-gguf. See the research notes (not published)]
- **binary-vit**: BiBench (AI-Efficiency/BiBench): no ViT/DeiT/Swin configs (only resnet18/20/34, bireal18, reactnet, vgg16, basebert, pointnet, dfsmn) and no released checkpoints (issue #2 open)
- **binary-vit**: BinaryViT Le & Li (CVPRW'23): training code only; author refused weights (issue #1)
- **binary-vit**: ReActNet-DeiT-S baseline: scripts only in the BinaryViT repo, no weights
- **binary-vit**: BiViT He et al. (ICCV'23): repo has 2 helper files, no weights
- **binary-vit**: SI-BiViT (ACM MM'24): simple-version code, no weights
- **binary-vit**: BVT-IMA (AAAI'24): no code found
- **binary-vit**: BinaryViT Xiao et al. (arXiv 2305.14730): withdrawn, no code
- **binary-vit**: DIDB-ViT (arXiv 2507.02222): no code
- **binary-vit**: Bi-ViT Swin-T/S and DeiT-B: paper results only; Drive folder holds only DeiT-T/S
- **binary-vit**: GSB ImageNet: not released (only CIFAR-100, Flowers102, Chaoyang; Flowers/Chaoyang not downloaded)
- **binary-vit**: BiT (facebookresearch/bit): NLP/BERT only, no vision checkpoint
- **binary-vit**: 'BiFormer': no binary ViT under that name (BiFormer is an fp bi-level routing ViT)
- **binary-vit**: Bi-ViT README visible link folder 1StJBp_-aQqOe2S5YWo5P8HWPjhGEORKG: contains rebnn_66_9.pth (not a ViT; not downloaded)
- **other-tasks-bnn**: BiPointNet (htqin/BiPointNet): MIT training code only; no weights in the README (current or 2021 revision), releases or tree
- **other-tasks-bnn**: BiFSMN binary model: only the FP teacher DfsmnModel_pre is released (release v1.0); no binary weights
- **other-tasks-bnn**: BiFSMNv2 (htqin/BiFSMNv2): README explicitly says there is no checkpoint link
- **other-tasks-bnn**: BiDet Faster R-CNN / COCO models: not released (only SSD300-VOC)
- **other-tasks-bnn**: LWS-Det (CVPR 2021): no code found
- **other-tasks-bnn**: BAM (Xin et al., ECCV 2020) binary SR: no code found
- **other-tasks-bnn**: BTM (Jiang et al., AAAI 2021) binary SR: no official code; only the BiSCI CASSI re-implementation has weights
- **other-tasks-bnn**: FRB (ICML 2024) and SCALES binary SR: no repositories found
- **other-tasks-bnn**: Bi-GCN (bywmm/Bi-GCN): code only
- **other-tasks-bnn**: Binary GNN (mbahri/binary_gnn): promised LARQ release never shipped
- **other-tasks-bnn**: BinaryDM and BiDM (Xingyu-Zheng): LDM patch code only; the BiDM CelebA link is upstream FP DDIM
- **other-tasks-bnn**: BiDense (frickyinn/BiDense): code only, no checkpoints
- **other-tasks-bnn**: BiBench (AI-Efficiency/BiBench): no checkpoints
- **other-tasks-bnn**: Binary DAD-Net / other binary segmentation nets: no public checkpoints found
- **other-tasks-bnn**: Binary GANs: no public binary-GAN checkpoint located in this pass
- **other-tasks-bnn**: DA-BNN: could not identify the paper or repo
- **other-tasks-bnn**: IR-Net / ReActNet official CIFAR-10 weights: none in htqin/IR-Net or liuzechun/ReActNet; CIFAR-10 BNN checkpoints (RBNN, ReCU, AdaBin, FracBNN, BNAS, Brevitas/FINN CNV, XNOR-Net NIN) are handled by other families and already present in .cache/checkpoints
- **hub-sweep**: htqin/BiBench: code and configs only, no checkpoints
- **hub-sweep**: BiFSMN binary weights: release asset is the fp teacher (see entry); BiFSMNv2 README lists no weights
- **hub-sweep**: BMXNet-v2 / bitorch / BinaryBrain / BoolNet / BNN_NoBN / real2binary / SKD-BNN / ftbnn / BinaryNet.pytorch / rethinking-bnn-optimization: no pretrained weight links found in README or releases [CORRECTED 2026-10-30: bitorch has a model hub (bitorch/models/base.py points to a W&B table) with ResNetE-18, MeliusNet-22, BinaryDenseNet-28 ImageNet checkpoints (bitorch-*-imagenet); BMXNet v1 commits packed binary models in smd_hpi/binary_models (bmxnet-v1-*). See the research notes (not published)]
- **hub-sweep**: OvSW (ECCV 2024): only a --pretrained placeholder, no link [UPDATED 2026-10-30: the README now links 5 models on OpenI, which return 404 anonymously (ovsw-resnet-vgg-1w1a, download-failed). See the research notes (not published)]
- **hub-sweep**: Riptide, BinaryDuo, ReSTE, biprop, lbcnn.torch, TCBNN, BinaryDM, BiDM, BiMatting: no weight links in README
- **hub-sweep**: Ternary_Binary_Transformer (facebookresearch), AdamBNN: weights only on HKUST SharePoint, not attempted (no direct-download URL)
- **hub-sweep**: TBN: Baidu Pan only, not attempted
- **hub-sweep**: jiecaoyu XNOR-Net-PyTorch: Drive links dead/permission-denied (gdown error recorded)
- **hub-sweep**: kellian-cottart/BinaryMetaplasticityFromUncertainty: HF repo has README only
- **hub-sweep**: Kcarochun/ZeroFree-1Bit-MNIST: code and PNGs only, no weights
- **hub-sweep**: tuanio/binary_bert: empty HF repo (only .gitattributes)
- **hub-sweep**: ewdlop/bnn-mnist, lightning-uq-box/*bnn*, sams-tom/*bnn*: Bayesian NNs, not binary
- **hub-sweep**: Ruichen0424/DSN and IS-SNN: spiking neural networks, not BNNs
- **hub-sweep**: phanerozoic/threshold-xnor*: hand-built threshold-logic gates, not trained BNNs
- **hub-sweep**: HF tag 'larq' and author 'larq'/'plumerai': no models on the Hub
- **hub-sweep**: WebSearch for 2025-26 HF/GitHub W1A1 ImageNet releases: nothing beyond AB-BNN and dabnn
- **openvino-omz-binary-ir-models**: FP16-INT1 variants of face/pedestrian/vehicle-detection-adas-binary-0001: not listed in any checked model.yml (2019_R3 to 2020.4). Only FP32-INT1 (INT1 in 2019_R3) was published.
- **openvino-omz-binary-ir-models**: README.md and per-model accuracy-check.yml for the 3 ADAS detectors at tag 2020.4: 404. The descriptions live in description/<name>.md and the accuracy configs in tools/accuracy_checker/configs/<name>.yml.
- **openvino-omz-binary-ir-models**: ADAS binary detectors at OMZ tag 2021.1: model.yml 404, so they were removed from OMZ.
- **openvino-omz-binary-ir-models**: Paper or training code for resnet50-binary-0001 and resnet18-xnor-binary-onnx-0001: none cited by OMZ. NNCF binarization is the likely toolchain (inferred), but no PyTorch checkpoint of these exact models was found.
- **nncf-binarized-resnet18-xnor-dorefa**: NNCF v2.8.0, v2.8.1, v2.9.0 and v2.10.0+ versioned storage paths for resnet18_imagenet_binarization_{xnor,dorefa}.pth: these return HTTP 200 text/html with 1061 bytes (a soft-404 page, identical for a bogus version), not the file. v2.8/v2.9 docs link to the develop/ path instead.
- **nncf-binarized-resnet18-xnor-dorefa**: NNCF binarization algorithm in v2.10.0 through v3.4.0 (latest, 2026-09-17): nncf/torch/binarization returns 404. ReleaseNotes says it was deprecated in v2.9.0 and removed in v2.10.0, so newer releases host no new binarized checkpoints.
- **nncf-binarized-resnet18-xnor-dorefa**: Any NNCF binarized model other than ResNet-18 (XNOR/DoReFa): none is listed in the README v2.5.0 or ModelZoo.md v2.6.0-v2.9.0.
- **nncf-binarized-resnet18-xnor-dorefa**: Upstream SHA-256/MD5 for the NNCF checkpoints: none published; only S3 multipart ETags.
- **burn-self-supervised-binary-reactnet**: BSSL (Kim & Choi, 'Self-Supervised Learning for Binary Networks by Joint Classifier Training'): this is arXiv 2110.08851 v1, the earlier version of BURN. Same repo (naver-ai/burn), no separate weights.
- **burn-self-supervised-binary-reactnet**: BURN ImageNet linear-eval classifier weights: the README link is empty ('[here]()'). The repo has no issues and no releases.
- **burn-self-supervised-binary-reactnet**: killawhale2/burn (clone URL in the BURN README): GitHub returns 404, so there is no alternative weight source.
- **burn-self-supervised-binary-reactnet**: S2-BNN real-valued ResNet-50 student (Google Drive 1eY-iVkYxxJfLhfeFBvStDzhtY3ns0p7S): not a BNN, so not downloaded by this run. Another run had already stored a 227 MB checkpoint_0099.pth.tar in .cache/checkpoints/s2bnn-distill-only/; this run did not inspect it.
- **bnn-tasks-2025-iccv-cvpr-iros**: BinaryDemoire (zhengchen1999): only Supplementary_Material.pdf as a release asset; code and models still a TODO
- **bnn-tasks-2025-iccv-cvpr-iros**: BiisNet (joeyxin-del): model/BiisNet.py only, no releases or weight links
- **bnn-tasks-2025-iccv-cvpr-iros**: BiVoC (Andong-Li-speech): LICENSE + README placeholder only
- **bnn-tasks-2025-iccv-cvpr-iros**: BU-Net (Sy-ling): README says code and pretrained models will be released upon publication
- **bnn-tasks-2025-iccv-cvpr-iros**: Hugging Face mirrors for BMTNet/BiCD/Pathfinder/BinaryDemoire/BiisNet/BiVoC/BU-Net: none relevant found
- **bnn-tasks-2025-iccv-cvpr-iros**: BiCD LEVIR-CD binarized model code: not in the repo (the checkpoint has the DGMA2-Net FP layout)
- **bnn-tasks-2025-iccv-cvpr-iros**: BiCD 'binary_version_ReActNet' module and train_ours_auxdecoder_c3po.py: referenced but missing from the repo
- **budget-skipped-downloadable-checkpoints**: BNAS-F and BNAS-G ImageNet: public (HEAD 200, 318,773,673 B and 299,147,631 B) but not downloaded; the task asked only for the smallest (E) and largest (H)
- **budget-skipped-downloadable-checkpoints**: BiBERT CoLA/QNLI/RTE/STS-B pytorch_model.bin: public in the same Drive folder, not downloaded in this pass (MNLI+QQP were the priority)
- **budget-skipped-downloadable-checkpoints**: tensorpack DoReFa AlexNet-1,32,32.npz BWN, RBNN 120/200-epoch, ReCU vanilla, SiMaN CIFAR, BCDNet step-1: not addressed in this pass (outside the listed priorities)
- **budget-skipped-downloadable-checkpoints**: Google Drive quota/permission errors: none occurred; every gdown call succeeded
- **missed-bnns-2026-10-30**: VITA-Group/BNN_NoBN (BN-free ReActNet, CVPRW 2021): README has no checkpoint links; no releases (https://github.com/VITA-Group/BNN_NoBN)
- **missed-bnns-2026-10-30**: cvlab-yonsei/EWGS (W1A1 ResNet-20 CIFAR, CVPR 2021, GPL-3.0): training code only, no checkpoints in README/releases
- **missed-bnns-2026-10-30**: liyunqianggyn/Equal-Bits-BNN (AAAI 2022, Bi-half): code only, no weights
- **missed-bnns-2026-10-30**: 42Shawn/CMIM: code only, no weights; 42Shawn/LCR_BNN already recorded as no-weights
- **missed-bnns-2026-10-30**: DravenALG/ReSTE (ICCV 2023): code only (README mentions only --resume)
- **missed-bnns-2026-10-30**: pingxue-hfut/sd-bnn (Self-Distribution BNN, MIT): README tables (ResNet-18 ImageNet 66.5%) but no weight links; pingxue-hfut/DWR (AAAI 2023 depth-width reshaping): code only
- **missed-bnns-2026-10-30**: two-tiger/FSG (AAAI 2024): only an FP pretrained model inside Results/ for MetaQuant-style training, no released binary model
- **missed-bnns-2026-10-30**: bemc22/DeSign (2026, JSTSP dithered sign activation): code only
- **missed-bnns-2026-10-30**: RWTH-IDS/efficient-binary-neural-networks, gushan/SGDAT, robustml-eurecom/SBNN, ShuaiZ1037/bnn-xnor-bireal, lucamocerino/Binary-Neural-Networks-PyTorch-2.x, PSCLab-ASU/BNNPruning, shekhovt/PSA-Neurips2020, zhaohui-yang/Binary-Neural-Networks: no weight links in README/releases
- **missed-bnns-2026-10-30**: Wzixin/SKD-BNN: no weights (also in existing searched list)
- **missed-bnns-2026-10-30**: htqin/DIR-Net and Sense-GVT/DIR-Net: GitHub 404; htqin/IR-Net README (pushed 2026-09) still has no checkpoints
- **missed-bnns-2026-10-30**: sfalkena/LAB (Learnable Activation Binarizer, WACV 2023, Larq, Apache-2.0): only the paper PDF as a release asset; no trained models
- **missed-bnns-2026-10-30**: fedeloper/CycleBNN (2024): README has training commands only, no weight links
- **missed-bnns-2026-10-30**: huawei-noah/Efficient-Computing/BinaryNetworks: only AdaBin (already in registry); BiMLP code/weights not found on GitHub (search "BiMLP" 0 hits)
- **missed-bnns-2026-10-30**: MindSpore: gitee mindspore/models research/cv lists only AdaBin and FDA-BNN as BNNs (both in registry); download.mindspore.cn/model_zoo/{research,community,r1.x,official} has only FDA_BNN/fdabnn.ckpt (BNN) - the rest are INT8 quant models or IBN-Net
- **missed-bnns-2026-10-30**: hpi-xnor/bitorch hub: 5th row resnete18 CIFAR-10 (epoch 0, 40.4% acc) is an untrained artefact, not downloaded; dlrm row is a recommender, out of scope
- **missed-bnns-2026-10-30**: hpi-xnor/bitorch-engine: repo moved to GreenBitAI/bitorch-engine (Apache-2.0), kernels only, no classification checkpoints; GreenBitAI HF models are low-bit LLMs (out of scope)
- **missed-bnns-2026-10-30**: huawei-noah/bolt: BNN kernels (compute/tensor/src/cpu/arm/bnn, daBNN-format support) but no model files
- **missed-bnns-2026-10-30**: JDAI-CV/dabnn: only the two Bi-Real-18 models (both already in registry)
- **missed-bnns-2026-10-30**: IR2Net (pingxue-hfut): weights only on Baidu Pan (entry recorded as unverified)
- **missed-bnns-2026-10-30**: jia-kai/eevbnn: 428 MB Dropbox tar.xz, over cap (entry recorded unverified)
- **missed-bnns-2026-10-30**: S2-BNN, BURN, BCDNet, AB-BNN, BNext, RBONN, ReCU, RBNN, SiMaN, BiPer ImageNet: already in registry
- **missed-bnns-2026-10-30**: RBNN ImageNet 120/200-epoch, ReCU vanilla (R18/R20/VGG CIFAR, R18/R34 ImageNet), SNN 0.56W/0.44W, ReCU VGG-Small finetune (folder 1bskc10Hb8RkNp-Btd9xak2aU4PNPacEo), Faster-Than-Lies S/M and unfrozen variants: public but not downloaded (budget / redundancy)
- **missed-bnns-2026-10-30**: Rep_AdamBNN (imfinethanks): pan.seu.edu.cn only (already in existing searched list)
- **missed-bnns-2026-10-30**: OpenI-hosted OvSW models: anonymous 404 (entry recorded)
- **missed-bnns-2026-10-30**: qonnx_model_zoo CIFAR10 CNV_1W1A/1W2A/2W2A, MNIST LFC/TFC 1W1A/1W2A/2W2A: QONNX re-exports of the Brevitas BNN-PYNQ models already in the registry (brevitas-bnn-pynq-*, finn-examples) - not added as separate entries
- **missed-bnns-2026-10-30**: Xilinx/QNN-MO-PYNQ binparam-dorefanet (W1A2 DoReFa ImageNet) and tiny-YOLO W1A3: FINN-folded hardware weight/threshold tiles (.bin per PE), not a standard checkpoint - not downloaded (same reasoning as BNN-PYNQ params)
- **missed-bnns-2026-10-30**: HirokiNakahara/GUINNESS (Chainer FPGA BNN GUI, GPL-2.0): tutorial PDF only, no trained models linked
- **missed-bnns-2026-10-30**: Hugging Face search (reactnet, bireal, birealnet, xnor, xnornet, binarized, binary resnet, 1w1a, w1a1, bnn-imagenet, bnn-cifar, bnext, irnet, binarynet, binaryconnect, larq, quicknet, meliusnet, ebconv, adabin, recu, rbnn, bnn): no new classification BNNs; hits are LLM "ultrafeedback_binarized" DPO models, binary-classification ResNets, Bayesian NNs, Chanjeong/resnet_binary_weights (already in registry), AB-BNN and HyperbeeAI (already in registry)
- **missed-bnns-2026-10-30**: GitHub repository search ("binary neural network" pushed>2023-06, reactnet, bireal, 1w1a, W1A1 resnet, xnor-net, FleXOR, MoBiNet, Elastic-Link, CycleBNN): no further repos with classification checkpoints; FleXOR, MoBiNet and Elastic-Link have no findable official code
- **missed-bnns-2026-10-30**: htqin/awesome-model-quantization (2023-2026 binary sections) and michaeltinsley/awesome-binary-neural-networks: no additional binary-CNN repos with weights beyond those checked
- **missed-bnns-2026-10-30**: BiDRN (ZHITENGLI/BiDRN, binarized 3D whole-body human mesh recovery, arXiv 2311.14323): README '[pretrained models]' has no link; only arxiv_supp.pdf release asset; no weights
- **missed-bnns-2026-10-30**: BinRVR (Tony1882880/BinRVR, 'Binarized High-Efficiency RAW Video Restoration and Beyond', TPAMI 2026 per repo description): repository is empty (no files) as of 2026-10-30
- **missed-bnns-2026-10-30**: Rectified Binary Network for SISR (TNNLS 2024, jwxintt/Rectified-BSR): repository empty
- **missed-bnns-2026-10-30**: BSSR (nntk-wb/BSSR, 'Beyond Binarization: semi-structured sparse ... SR'): repository empty
- **missed-bnns-2026-10-30**: E2FIF (pppLang/E2FIF), SISRBNN (pppLang), binary-srgan-pytorch (twice154), lixcli/binsr: code only, no checkpoints
- **missed-bnns-2026-10-30**: BOB-YOLO (534-quant/BOB-YOLO, FAIA 2024 binary YOLO): repository empty
- **missed-bnns-2026-10-30**: BinaryCenterNet (brenoskuk/BinaryCenterNet, Larq CenterNet detector): code + notebook only, no weight files/links
- **missed-bnns-2026-10-30**: BiDRN/BinaryHPE-style binary pose 2023-26: no other public binary pose weights found besides Bulat 2017 Torch7 models
- **missed-bnns-2026-10-30**: Group-Net semantic segmentation (bohanzhuang/Group-Net-semantic-segmentation): only ImageNet backbone init link (groupnet already in registry), no segmentation checkpoint
- **missed-bnns-2026-10-30**: PBNet (weiguangzhao/PBNet, ICCV'23 point-wise binarization for instance seg): 'binarization' refers to point-wise labels, network is FP - not a BNN
- **missed-bnns-2026-10-30**: SparkNet (jsvir/sparknet, Interspeech 2024 'Sparse Binarization for KWS'): checkpoints exist (ckpt/kws_C_*.ckpt) but the method binarizes a sparse input representation followed by a linear classifier - not a BNN (inferred from abstract)
- **missed-bnns-2026-10-30**: Binary GNN (mbahri/binary_gnn): pretrained/ contains only FP DGCNN teacher model.t7 files; binary LARQ models never released (registry entry binary-gnn-bahri stays no-public-weights)
- **missed-bnns-2026-10-30**: BRILLsson: no LICENSE; weights only on Dropbox (recorded as entries)
- **missed-bnns-2026-10-30**: One-bit Deep Hashing (hly1998/ODH): code only, no checkpoints
- **missed-bnns-2026-10-30**: BNN-VAD-pytorch (loujc), BNN EO segmentation (abhisheksinghIN), Salient-Object-Detection-Using-BNN (Akash210085, only FP u2netp.pth): no BNN weights
- **missed-bnns-2026-10-30**: ECG5000 binary NN (CameronDetig), patbinn (YuvalHarary): out of scope / classification of 1-D signals, not investigated further
- **missed-bnns-2026-10-30**: Binary CLIP / binary multimodal, binary NeRF (1-bit network weights), binary event-camera nets, binary trackers, binary GANs (salu133445/binarygan = binary neurons for music, not BNN weights), binary medical segmentation: GitHub + HF search found no public 1-bit checkpoints
- **missed-bnns-2026-10-30**: Binary ViT/MLP (CIFAR/ImageNet) beyond registry: GitHub/HF searches ('binarized vit', 'binary vit', 'bivit', '1-bit vit', 'binarized swin', 'binarized mlp-mixer') found nothing new with weights (BinaryFormer mattiaspaul = Hamming attention for medical 3D, no weights; npq-vit = binary spherical quantization tokenizer, not BNN)
- **missed-bnns-2026-10-30**: BinaryDM / BiDM / BI-DiffSR follow-ups: no new binary diffusion checkpoints except BiMaCoSR (entry) and TerDiT (ternary, entry)
- **missed-bnns-2026-10-30**: HF API searches 'binarized' (DPO 'ultrafeedback_binarized' LLMs), 'binary transformer', 'bivit', 'binary vit' (binary-classification ViTs), '1-bit vit', 'bnn sr', 'binary super resolution', 'terdit' (-> lucky-lance/TerDiT), 'binarized diffusion', 'bimacosr', 'brve', 'bbcu', 'bidm', 'binarydm', '1-bit diffusion' (CoreML palettized SD, not BNN), 'binary-weight', 'binary detection': only TerDiT relevant
- **missed-bnns-2026-10-30**: Channel Correlation Loss for BNNs (CVPR 2026 Findings): classification, out of scope; code not checked
- **missed-bnns-2026-10-30**: llm-nlp: strict binary (W1A1) BERT/encoder checkpoints beyond BiBERT/BiPFT: none found. HF searches for binarized-bert, bnn-bert, 1bit-bert, bitbert, bit-bert, bitnet-bert, binarized-transformer, fully-binarized and bibench all returned 0 relevant results. The 'bibert' hits (jhu-clsp/bibert-ende, HCKLab/BiBert-*, naist-nlp/bibert-*) are bilingual or bi-encoder BERTs, not binary.
- **missed-bnns-2026-10-30**: llm-nlp: tuanio/binary_bert-* (about 20 repos, incl. SST-2 fine-tunes): config shows a standard BF16 BertForSequenceClassification. 'binary' refers to binary classification, not a BNN. tuanio/binary-bert-for-sequence-classification is actually a wav2vec2-xls-r ASR fine-tune.
- **missed-bnns-2026-10-30**: llm-nlp: TBT (facebookresearch/Ternary_Binary_Transformer, ACL 2023) W1A1/W1A8/W2A2/W2A8 BART checkpoints: all 8 SharePoint links 404 on 2026-10-30 (recorded as entry tbt-bart-base-w1a1-summarization, download-failed)
- **missed-bnns-2026-10-30**: llm-nlp: BWTA (arXiv 2604.03957, binary weights + ternary activations Transformer, 2026): no code or weights linked
- **missed-bnns-2026-10-30**: llm-nlp: Binarized Neural Machine Translation (Google, NeurIPS 2023): only a third-party reproduction repo (the-sergiu/Binarized-Neural-Machine-Translation) and AkshatSh/BinarizedNMT (2019); no weights
- **missed-bnns-2026-10-30**: llm-nlp: SensiMix (snudm-starlab, 1-bit value + 8-bit index BERT): code only, no checkpoint link in README
- **missed-bnns-2026-10-30**: llm-nlp: MotionBitBERT (CyperStone): 1-bit MotionBERT (pose, not NLP); README links only the FP MotionBERT weights
- **missed-bnns-2026-10-30**: llm-nlp: STBLLM (pprp, ICLR'25), HBLLM (Yeyke, NeurIPS'25), ARB-LLM, PBS2P (XIANGLONGYAN), RaBiT (SamsungLabs, ICML'26): PTQ/QAT code only; no HF checkpoints (SamsungLabs has no HF models; README shows a placeholder 'username/rabit-llama-7b-2bit')
- **missed-bnns-2026-10-30**: llm-nlp: BTC-LLM: no official repo or HF weights found (search hits are crypto-trading LLMs; Chooovy/BTC-LLM is an empty personal repo)
- **missed-bnns-2026-10-30**: llm-nlp: DB-LLM: only Efficient-ML/LLaMA-3-8B-DB-LLM-2bit-fake (2-bit, not binary). Efficient-ML/LLaMA-3-8B-PB-LLM-1.7bit-fake is a dense fake-quant like the already-listed BiLLM one (not added)
- **missed-bnns-2026-10-30**: llm-nlp: BitNet a4.8 / BitNet v2 / BitDistill (Microsoft 2025): no official weights on HF (microsoft org only has bitnet-b1.58-2B-4T{,-bf16,-gguf}, bitnet-embedding-270m/0.6b, VibeVoice-ASR-BitNet). ustccuigm/bitdistill is an unrelated RL distillation repo. BitDistiller/* HF models are 2-bit (w2g64)
- **missed-bnns-2026-10-30**: llm-nlp: OneBit (xuyuzhuang11): no HF mirror; the HF 'onebit' hits are nlp-project-uw/quant_1_58b_impl_OneBit_* (student course runs, ternary) and NoamFelix/OneBit (empty)
- **missed-bnns-2026-10-30**: llm-nlp: FBI-LLM: the only new form is LiqunMa/FBI-ckpt-backup (565 GB of intermediate distcp/pytorch training checkpoints for 130M/1.3B/7B). Not added: existing fbi-llm entries cover the final models
- **missed-bnns-2026-10-30**: llm-nlp: litert-community/1Bit-Bonsai-1.7B-PoC (550 MB .litertlm int1): its own card says it is an unrunnable proof of concept (no INT1 kernel in LiteRT), derived from Ternary-Bonsai. Not added
- **missed-bnns-2026-10-30**: llm-nlp: OneBitAI/cronus-150M: native ternary BitLinear SLM, but only a 150M-token smoke run (loss 5.39). Not added (low value)
- **missed-bnns-2026-10-30**: llm-nlp: Thox-ai/ThoxMicro-1bit-9M / -16M: tiny BitNet b1.58 TinyStories models (TQ2_0 GGUF 6.25 MB, card sha256 published), licence 'pending review'. Not added (toy, unclear licence)
- **missed-bnns-2026-10-30**: llm-nlp: OpenTransformer/binary-transformers 'purebit' (4.7M, binary weights, GELU FP activations, byte/bit-level LM, hobby experiment Jan 2026): not added
- **missed-bnns-2026-10-30**: llm-nlp: AQLM 1-bit (ISTA-DASLab *-AQLM-PV-1Bit-1x16), HQQ 1-bit (mobiuslabsgmbh *_1bitgs8_hqq, eewer/Qwen3-4B-QAT-1bit with HQQ meta tensors), CommVQ, LittleBit (Haverbex/Qwen3.8-27B-LittleBit-1bit), JANG/turboquant Bonsai re-quants: vector-quantized or affine re-quantizations, not sign-binary BNNs. Not added
- **missed-bnns-2026-10-30**: llm-nlp: Bonsai MLX twins (4B 629 MB, 8B 1.28 GB, 27B 5.13 GB) and unpacked FP16 repos (1.7B 3.4 GB to 27B 54.7 GB): same weights as the GGUF entries; sizes noted in those entries' download_status
- **missed-bnns-2026-10-30**: llm-nlp: community BitNet GGUF re-quants (mradermacher, RichardErkhov, QuantFactory, tdh111, mlx-community 4/6/8-bit): derivative or not ternary-preserving. Not added
- **missed-bnns-2026-10-30**: llm-nlp: BitCPM-CANN-0.5B/1B/3B/8B (openbmb, 2026, Ascend NPU target, pytorch_model.bin + unquantized + gguf): noted under the BitCPM4 entries, not separate entries
- **missed-bnns-2026-10-30**: llm-nlp: HF1BitLLM Llama3-8B-1.58-Linear-10B-tokens and -Sigmoid-k100-10B-tokens: same 3.85 GB packed format as the 100B-token entry. Not added separately
- **missed-bnns-2026-10-30**: llm-nlp: Falcon3-3B/7B 1.58bit, Falcon-E-1B/3B-Base, *-GGUF (I2_S), *-prequantized revisions: same family as the 4 recorded Falcon entries
- **missed-bnns-2026-10-30**: llm-nlp: MobileLLM-ParetoQ 350M/600M/1B 1-bit and all 1.58-bit sizes: gated (manual). Only 125M and 1.5B 1-bit recorded; sizes in the 125M entry
- **missed-bnns-2026-10-30**: llm-nlp: BinaryMoS Llama-2-13B, Llama-1-7B/13B/30B (dongwonjo): same format as the recorded 7B; ptq161/llama2-13b and llama-65b-ptq161: same as the recorded 7B
- **missed-bnns-2026-10-30**: llm-nlp: rileyseaburg/bitmamba-*, rasatavohary/BitMamba-2-* (mirrors), cs-cmyk/df-ssm (1-bit Mamba-2 distillation; no HF weights): not added
- **missed-bnns-2026-10-30**: finn-examples releases: kws (MLP W3A3), cybersecurity (UNSW-NB15 MLP W2A2), radioml (VGG10 W4A4), mobilenetv1 (W4A4) -> not 1-bit, skipped. binary-cop (BinaryCoP, W1A1, MaskedFace-Net): only Pynq-Z1 bitfile zip in release 'binary-cop', no ONNX/weights; NaelFasfous/BinaryCoP repo 404. https://github.com/Xilinx/finn-examples/releases
- **missed-bnns-2026-10-30**: MLPerf Tiny v0.7 open/hls4ml-finn CNV-W1A1 cnv-w1a1.onnx: sha256 47007fe6... identical to finn-examples v0.0.1a cnv-w1a1.onnx (already registered). Other MLPerf Tiny rounds v0.5/v1.0-v1.3: no 1-bit model files (hls4ml RN06/RN08, AD are multi-bit). https://github.com/mlcommons/tiny_results_v0.7
- **missed-bnns-2026-10-30**: al-nafisah/fpga-video-classifier build/finn/models/cnv-w1a1.onnx: same size (6,223,756 B) as the finn-examples cnv-w1a1 export, presumably a copy (not downloaded).
- **missed-bnns-2026-10-30**: QONNX model zoo (fastmachinelearning/qonnx_model_zoo): CNV/TFC/LFC 1W1A/1W2A are exports of the Brevitas bnn_pynq checkpoints (registered or added here); other models (mobilenet 4W4A, kws W3A3, UNSW W2A2, CloudSatNet 4b, rn8 2W2A/3W3A, resnet18 float8) are not 1-bit.
- **missed-bnns-2026-10-30**: Brevitas examples beyond bnn_pynq: imagenet_classification (MobileNet-v1 4b, ProxylessNAS 4b/4b5b/hadamard), speech_to_text QuartzNet 4b/8b, bnn_pynq resnet18_4w4a: none 1-bit. sfc_2w2a/tfc_2w2a/cnv_2w2a are 2-bit (skipped).
- **missed-bnns-2026-10-30**: Xilinx/ResNet50-PYNQ (archived): HLS-packed W1A2 weights under src/w1a2-v1.0 (no ONNX/pth); the ONNX is in finn-examples (entry added).
- **missed-bnns-2026-10-30**: OpenVINO OMZ tags 2019_R1, 2019_R3, 2020.4, 2021.4, master: only the 5 already-registered *-binary-*/xnor models exist; no other INT1 models.
- **missed-bnns-2026-10-30**: huawei-noah/bolt: BNN (XNOR/DoReFa) ARM kernels + X2bolt BNN_FP16 conversion, but no released BNN models (README/USER_HANDBOOK).
- **missed-bnns-2026-10-30**: JDAI-CV/dabnn: only the two Bi-Real-18 models (already registered); docs/onnx2bnn.md has no further models.
- **missed-bnns-2026-10-30**: larq/compute-engine: no model files in repo (benchmarks use larq-zoo models, already registered); Plumerai: no public BNN weights (MLPerf Tiny closed results only).
- **missed-bnns-2026-10-30**: microsoft/nni: examples/compression/quantization/bnn_example.py + BNNQuantizer, no pretrained weights.
- **missed-bnns-2026-10-30**: PaddleSlim: example/quantization/bnn (XNOR/BiReal/ReAct/IR-Net/ReCU/ReSTE/FDA binarizers, CIFAR-10 training script) and example/BinaryDM, example/BiFSMN: code only, no checkpoints.
- **missed-bnns-2026-10-30**: quic/aimet-model-zoo: no binary/1-bit models in tree. IntelLabs/distiller: repo 404 (archived/removed); no BNN checkpoints known.
- **missed-bnns-2026-10-30**: cornell-zhang/bnn-fpga, ShixiangLi/ReActNet_FPGA: no weight files in tree (ReActNet_FPGA = C headers only, already noted).
- **missed-bnns-2026-10-30**: HyperbeeAI/vision-bnn-benchmarks-hf: 16 checkpoint folders (maxim000/001, shallow001-012, 998/999) are 2b/8b MAX78000 nets; only shallow007 AP2b8b is registered and the rest are multi-bit, not BNN.
- **missed-bnns-2026-10-30**: hls4ml example-models onnx/bnn_model_fc_1layer.onnx (2,321 B): a single-layer test graph, not a trained model (not recorded).
- **missed-bnns-2026-10-30**: Hugging Face sweep (37 search terms + createdAt/lastModified sorts, 2,581 ids): non-BNN hits dominate (Bayesian 'bnn', 'binary' classifiers, BiT ResNetV2 'bit', bitfit, bittensor). Relevant: euclidstellar/tinystories-1bit-llm, Thox-ai/ThoxMicro-1bit-16M/9M, Feyd89/BitBop-125M-babylm (ternary, 53.8 MB model.pt), OneBitAI/cronus-150M (ternary, 604 MB), qep/qep-1bit-extreme (OneBit, 32 GB). snowian/binarized_ternary_ViT_example, NoamFelix/OneBit, mamad2500/xnor: README only. h4rr9/mnist_1bit*: OPT-125m text models, not BNN. Arvyniitb/1bitModel, arunb74/Luxeai-anu-1-bit-70M, xfcxcxcdfdfd/1-bit, justheuristic/test-1bit (AQLM), BananaMind KV1-8M 1Bit (custom code, no card): not verifiable BNNs. Rabe3/1-bit-embedding-general: binary embeddings. Kinastra/tiny-1bit-autoregressive-generator: 1.19 GB undocumented .pt.
- **missed-bnns-2026-10-30**: GitHub sweep (12 queries pushed>2023-06/2024): FPGA hobby BNNs mostly without weights (Vzzzz/bitladder, GaryGaoUW/tiny-hex-vision, Sudip-329/Memory-Efficient-BNN, SepehrMohammady/Efficient-NAS-BNN-Pipeline, b1shtream/bnext-imagenet-test, KelanuRanganath/ME0_SegmentNN, Dsekar75/FKD-HABNet, KarthikaSivakumar2005/Improved_BNN_CAN_IDS, johmyzhang/BNN-NIDS: no weight files in tree). elimkwan/Entropy-Driven-Adaptive-Filter: BNN-PYNQ cifar10 folded params copy. ryuz/BinaryBrain: LUT-net framework, no released weights (already in hub-sweep list).
- **missed-bnns-2026-10-30**: hpi-xnor/ios-image-classification predict-ios/Models/binarized_32_resnet-18-binary-0001.params (3,553,824 B): 32-bit-packed export, same size as the BMXNet v1 ImageNet ResNet-18 (presumably same model, different word size); not downloaded. BMXNet v1 binarized_mnist_32bit-0001.params likewise.
- **missed-bnns-2026-10-30**: jwfromm/zoo@riptide (Riptide branch of larq zoo): riptide/resnet_e.py and dorefanet.py download weights via larq-zoo release files (resnet_e v0.1.0 resnet_e_18_weights.h5 hash bde4a64d..., dorefanet v0.1.0 645d7839...), i.e. the larq-zoo weights already registered; no separate Riptide-trained weights found.
- **missed-bnns-2026-10-30**: Kaggle Models: WebSearch for kaggle.com/models BNN/XNOR 1-bit returned no Kaggle pages; Kaggle API not configured (not checked further). MATLAB, Edge Impulse, ncnn/MNN model zoos: not checked in this pass.
- **missed-bnns-2026-10-30**: HF ?author= sweep for 35 BNN authors/orgs (htqin, liuzechun, hpi-xnor, Xilinx, fastmachinelearning, plumerai, larq, Sense-GVT, VITA-Group, JDAI-CV, huawei-noah, ...): only huawei-noah TernaryBERT (added) and Ruichen0424 (AB-BNN registered; I2E/DSN/IS-SNN are SNNs).
