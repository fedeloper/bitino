# Per-family eval set: eval_of <model_id> -> "<eval dir> <preprocessing key> <extra validate_model args>".
# Sourced by scripts/device_final_suite.sh and scripts/validate_all_host.sh.
eval_of() {
  case "$1" in
    ab-bnn-cifar10-*)  echo ".cache/data/eval128_cifar10_norm cifar10_imagenet_norm --num-classes 10" ;;
    ab-bnn-cifar100-*) echo ".cache/data/eval128_cifar100_norm cifar100_imagenet_norm --num-classes 100" ;;
    ab-bnn-*) echo ".cache/data/eval128 torchvision_imagenet" ;;
    bitorch-*) echo ".cache/data/eval128_bitorch bitorch_imagenet --flip-rule feed" ;;
    rbonn-*)  echo ".cache/data/eval128 torchvision_imagenet" ;;
    sa-bnn-resnet18-imagenet|ida-det-bires18-imagenet-backbone) echo ".cache/data/eval128 torchvision_imagenet" ;;
    recu-*-imagenet|rbnn-*-imagenet|siman-*-imagenet|biper-*-imagenet|snn-subbit-*-imagenet) echo ".cache/data/eval128 torchvision_imagenet" ;;
    recu-*-cifar10|rbnn-*-cifar10|siman-*-cifar10|biper-*-cifar10) echo ".cache/data/eval128_cifar10_norm cifar10_imagenet_norm --num-classes 10" ;;
    adabin-resnet20-cifar10|fda-bnn-resnet20-cifar10-mindspore) echo ".cache/data/eval128_cifar10_norm cifar10_imagenet_norm --num-classes 10" ;;   # S12
    tensorpack-dorefa-alexnet-*) echo ".cache/data/eval128_tensorpack_bgr tensorpack_bgr" ;;   # S12: tensorpack BGR eval path (tools/eval_sets/tensorpack_bgr.py)
    bnas-a-cifar10) echo ".cache/data/eval128_cifar10_darts cifar10_darts_norm --num-classes 10" ;;   # S13: DARTS CIFAR mean/std (tools/eval_sets/cifar10_darts.py)
    bnas-*-imagenet) echo ".cache/data/eval128 torchvision_imagenet" ;;   # S13: test_imagenet.py Resize 256 / CenterCrop 224 / ImageNet norm
    bimatting-bimobilenet-backbone) echo ".cache/data/eval128 torchvision_imagenet" ;;   # S11: ReActNet-A-like classifier backbone
    burn-reacta-stage2|s2bnn-reacta-distill-only) echo ".cache/data/eval128 torchvision_imagenet --output-kind embedding --embedding-cos 0.99999" ;;   # S11: self-supervised backbones, no head (1024-d features)
    shuaiz-bireal20-cifar10) echo ".cache/data/eval128_cifar10_norm cifar10_bireal_norm --num-classes 10" ;;
    bnext-*)  echo ".cache/data/eval128_timm_bicubic_crop095 timm_bicubic_crop095" ;;
    ftl-deepfake-*) echo ".cache/data/eval128_cifake_ftl ftl_rgb01 --num-classes 2" ;;   # Faster Than Lies detectors: CIFAKE test subset, RGB in [0, 1] (tools/eval_sets/ftl_deepfake.py)
    bhvit-*)  echo ".cache/data/eval128_bhvit bhvit" ;;
    bibert-bert-base-mrpc) echo ".cache/data/glue/mrpc/eval128 bibert --numpy-logits numpy_logits.f32 --num-classes 2" ;;
    bibert-bert-base-mnli) echo ".cache/data/glue/mnli/eval128 bibert --numpy-logits numpy_logits.f32 --num-classes 3" ;;
    bibert-bert-base-qqp)  echo ".cache/data/glue/qqp/eval128 bibert --numpy-logits numpy_logits.f32 --num-classes 2" ;;
    bi-vit-deit-*) echo ".cache/data/eval128_bhvit bhvit" ;;   # DeiT eval transform (Resize 256 bicubic, crop 224) = BHViT's, bit-identical file
    gsb-deit-s-cifar100-*) echo ".cache/data/eval128_cifar100_deit cifar100_deit --num-classes 100" ;;
    bibert-*) echo ".cache/data/sst2/eval128 bibert --numpy-logits numpy_logits.f32 --num-classes 2" ;;
    brevitas-bnn-pynq-cnv-*|aliasgharshinwari-finn-cnv-*) echo ".cache/data/eval128_cifar10_brevitas cifar10_totensor --num-classes 10 --require-exact-numpy" ;;
    brevitas-bnn-pynq-*)     echo ".cache/data/eval128_mnist_brevitas mnist_totensor --num-classes 10 --require-exact-numpy" ;;
    dabnn-*)  echo ".cache/data/eval128 torchvision_imagenet" ;;
    nncf-resnet18-binarization-*) echo ".cache/data/eval128 torchvision_imagenet" ;;
    finn-examples-bnn-pynq-cnv-*) echo ".cache/data/eval128_cifar10_finn cifar10_pixels --num-classes 10 --require-exact-numpy" ;;
    finn-examples-bnn-pynq-tfc-*) echo ".cache/data/eval128_mnist_finn mnist_pixels --num-classes 10 --require-exact-numpy" ;;
    finn-examples-cnv-w1a1-gtsrb) echo ".cache/data/eval128_gtsrb gtsrb_totensor --num-classes 44 --require-exact-numpy" ;;
    bnn-pynq-cnv-w1a1-cifar10) echo ".cache/data/eval128_cifar10_finn cifar10_bnnpynq --num-classes 10 --require-exact-numpy" ;;
    bnn-pynq-cnv-w1a1-gtsrb)   echo ".cache/data/eval128_gtsrb gtsrb_bnnpynq --num-classes 44 --require-exact-numpy" ;;
    bnn-pynq-lfc-w1a1-mnist)   echo ".cache/data/eval128_mnist_finn mnist_bnnpynq --num-classes 10 --require-exact-numpy" ;;
    # image-to-image (slice S8): crops + ground truth, validate_model.py --output-kind image (PSNR, no top-1)
    bbcu-denoise)    echo ".cache/data/eval68_cbsd68_sigma25_c96 bbcu_denoise_s25 --output-kind image" ;;
    bbcu-deblocking) echo ".cache/data/eval68_jpeg_y_q10q40_c96 bbcu_jpeg_y --output-kind image" ;;
    bbcu-sr-*-x2)    echo ".cache/data/eval64_sr_x2_lr48 sr_x2_unit --output-kind image" ;;
    bbcu-sr-*-x4)    echo ".cache/data/eval64_sr_x4_lr32 sr_x4_unit --output-kind image" ;;
    bsrc2023-binary-sr-x2-*) echo ".cache/data/eval64_sr_x2_lr48 sr_x2_255 --output-kind image --image-metric bsrc_y" ;;
    bsrc2023-binary-sr-x4-*) echo ".cache/data/eval64_sr_x4_lr32 sr_x4_255 --output-kind image --image-metric bsrc_y" ;;
    omz-*)    echo "$(ls -d .cache/data/eval128_omz* 2>/dev/null | head -1) $(python3 -c "import json,glob;print(next(iter(json.load(open(glob.glob('.cache/data/eval128_omz*/manifest.json')[0]))['preprocessing'])))" 2>/dev/null)" ;;
    larqface-*)  echo ".cache/data/eval128_lfw insightface --output-kind embedding --embedding-cos 0.99999" ;;  # face embeddings (S6)
    brillsson-*) echo ".cache/data/eval128_speechcommands brillsson_logmel --output-kind embedding --embedding-cos 0.99999" ;;  # speech embeddings (S6); fp32 noise floor cos ~0.9997
    *)        echo ".cache/data/eval128 larq_zoo" ;;
  esac
}
