#!/usr/bin/env bash
# Paired exact (stage neon) vs final INT8 config (stage optimized_int8) accuracy for every converted
# model on the largest local eval set of its family (ImageNet: 5,000 val images; CIFAR-10 / CIFAR-100 / MNIST:
# full 10,000 test sets; GLUE full dev sets: SST-2 872, MRPC 408, MNLI-m 9,815, QQP 40,430 -- the QQP run is
# long, BERT-base at L=128 costs ~0.3-0.5 s per sample and config on the host). Native host runtime; the INT8 kernels are
# bit-exact with int8_reference on every ISA, so the INT8 side equals the device's INT8 config.
#   scripts/int8_accuracy_all.sh            MODELS="..." JOBS=3 THREADS=8 scripts/int8_accuracy_all.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
JOBS="${JOBS:-3}"; THREADS="${THREADS:-8}"; BENCH="${BENCH:-build/host/bitino_bench}"
MODELS="${MODELS:-$(for d in artifacts/*/; do [[ -f "$d/model.bitino" ]] && basename "$d"; done | tr '\n' ' ')}"
big_eval_of() {
  case "$1" in
    ab-bnn-cifar10-*)  echo ".cache/data/eval10000_cifar10_norm cifar10_imagenet_norm 10 cifar10test10000" ;;
    ab-bnn-cifar100-*) echo ".cache/data/eval10000_cifar100_norm cifar100_imagenet_norm 100 cifar100test10000" ;;
    ab-bnn-*|rbonn-*) echo ".cache/data/eval5000 torchvision_imagenet 1000 eval5000" ;;
    sa-bnn-resnet18-imagenet|ida-det-bires18-imagenet-backbone) echo ".cache/data/eval5000 torchvision_imagenet 1000 eval5000" ;;
    bitorch-*) echo ".cache/data/eval5000_bitorch bitorch_imagenet 1000 eval5000" ;;
    recu-*-imagenet|rbnn-*-imagenet|siman-*-imagenet|biper-*-imagenet|snn-subbit-*-imagenet) echo ".cache/data/eval5000 torchvision_imagenet 1000 eval5000" ;;
    recu-*-cifar10|rbnn-*-cifar10|siman-*-cifar10|biper-*-cifar10) echo ".cache/data/eval10000_cifar10_norm cifar10_imagenet_norm 10 cifar10test10000" ;;
    adabin-resnet20-cifar10|fda-bnn-resnet20-cifar10-mindspore) echo ".cache/data/eval10000_cifar10_norm cifar10_imagenet_norm 10 cifar10test10000" ;;
    tensorpack-dorefa-alexnet-*) echo ".cache/data/eval5000_tensorpack_bgr tensorpack_bgr 1000 eval5000" ;;
    bnas-a-cifar10) echo ".cache/data/eval10000_cifar10_darts cifar10_darts_norm 10 cifar10test10000" ;;   # S13
    bnas-*-imagenet) echo ".cache/data/eval5000 torchvision_imagenet 1000 eval5000" ;;   # S13
    bimatting-bimobilenet-backbone) echo ".cache/data/eval5000 torchvision_imagenet 1000 eval5000" ;;   # S11
    # S11 self-supervised backbones: no classifier released (features only), so no top-1 to compare
    burn-reacta-stage2|s2bnn-reacta-distill-only) echo "" ;;
    shuaiz-bireal20-cifar10) echo ".cache/data/eval10000_cifar10_norm cifar10_bireal_norm 10 cifar10test10000" ;;
    bnext-*)  echo ".cache/data/eval5000_timm_bicubic_crop095 timm_bicubic_crop095 1000 eval5000" ;;
    bhvit-*)  echo ".cache/data/eval5000_bhvit bhvit 1000 eval5000" ;;
    bi-vit-deit-*) echo ".cache/data/eval5000_bhvit bhvit 1000 eval5000" ;;
    gsb-deit-s-cifar100-*) echo ".cache/data/eval10000_cifar100_deit cifar100_deit 100 cifar100test10000" ;;
    omz-*)    echo ".cache/data/eval5000_omz_bgr omz_bgr 1000 eval5000" ;;
    bibert-bert-base-mrpc) echo ".cache/data/glue/mrpc/dev408 bibert 2 mrpcdev408" ;;
    bibert-bert-base-mnli) echo ".cache/data/glue/mnli/dev9815 bibert 3 mnlidevmatched9815" ;;
    bibert-bert-base-qqp)  echo ".cache/data/glue/qqp/dev40430 bibert 2 qqpdev40430" ;;
    bibert-*) echo ".cache/data/sst2/dev872 bibert 2 sst2dev872" ;;
    brevitas-bnn-pynq-cnv-*|aliasgharshinwari-finn-cnv-*) echo ".cache/data/eval10000_cifar10_brevitas cifar10_totensor 10 cifar10test10000" ;;
    brevitas-bnn-pynq-*)     echo ".cache/data/eval10000_mnist_brevitas mnist_totensor 10 mnisttest10000" ;;
    dabnn-*|nncf-resnet18-binarization-*) echo ".cache/data/eval5000 torchvision_imagenet 1000 eval5000" ;;
    finn-examples-bnn-pynq-cnv-*) echo ".cache/data/eval10000_cifar10_finn cifar10_pixels 10 cifar10test10000" ;;
    finn-examples-bnn-pynq-tfc-*) echo ".cache/data/eval10000_mnist_finn mnist_pixels 10 mnisttest10000" ;;
    finn-examples-cnv-w1a1-gtsrb) echo ".cache/data/eval12630_gtsrb gtsrb_totensor 44 gtsrbtest12630" ;;
    bnn-pynq-cnv-w1a1-cifar10) echo ".cache/data/eval10000_cifar10_finn cifar10_bnnpynq 10 cifar10test10000" ;;
    bnn-pynq-cnv-w1a1-gtsrb)   echo ".cache/data/eval12630_gtsrb gtsrb_bnnpynq 44 gtsrbtest12630" ;;
    bnn-pynq-lfc-w1a1-mnist)   echo ".cache/data/eval10000_mnist_finn mnist_bnnpynq 10 mnisttest10000" ;;
    bbcu-*|bsrc2023-*) echo "IMAGE" ;;   # image-to-image (S8): full-image PSNR, tools/restoration_eval.py --stages neon optimized_int8
    larq-zoo-*) echo ".cache/data/eval5000 larq_zoo 1000 eval5000" ;;
    # S6 embedding models have no top-1: their task metrics are tools/face_verification_eval.py (LFW / AgeDB-30 pairs,
    # .cache/data/eval12000_lfw, eval12000_agedb30; a config "int8=bench:<model>:optimized_int8") and
    # tools/speech_probe_eval.py (.cache/data/eval4890_speechcommands), so they are skipped here.
    larqface-*|brillsson-*) echo "" ;;
    *) echo "" ;;
  esac
}
one() {
  m="$1"; read -r edir pp ncls tag <<<"$(big_eval_of "$m")"
  [[ -n "$edir" ]] || { echo "$m: SKIP (no large eval set for this family yet)"; return; }
  [[ "$edir" != IMAGE ]] || { echo "$m: SKIP (image-to-image model: INT8 PSNR via tools/restoration_eval.py --stages neon optimized_int8)"; return; }
  out="reports/accuracy/${m}_exact_vs_int8-final_${tag}.json"
  M="artifacts/$m/model.bitino"
  python3 tools/accuracy_eval.py --eval-dir "$edir" --preprocess "$pp" --a "$M:neon" --b "$M:optimized_int8" \
      --bench "$BENCH" --classes "$ncls" --threads "$THREADS" --json "$out" >/dev/null 2>"$out.err" \
    && { rm -f "$out.err"; python3 -c "
import json;d=json.load(open('$out'));a,b,p=d['configs']['a'],d['configs']['b'],d['paired']
print(f\"$m: exact {a['top1']*100:.2f}% int8 {b['top1']*100:.2f}% diff {p['top1_diff_b_minus_a']*100:+.2f} pt p={p['mcnemar_exact_p']:.2g} agree {p['top1_prediction_agreement']*100:.1f}% n={d['n']}\")"; } \
    || echo "$m: FAILED ($(tail -1 "$out.err"))"
}
export -f one big_eval_of; export BENCH THREADS
for m in $MODELS; do echo "$m"; done | xargs -P "$JOBS" -L1 bash -c 'one "$0"'
