# Wanted: binary networks with public weights that are not in the zoo yet

Every network below has public weights that we downloaded and hash-checked (registry state `verified-pretrained`), but
no Bitino conversion yet. Each has a full entry in [`registry/models.json`](../registry/models.json): paper, code,
checkpoint URL and SHA-256, licences, binarization notes. List as of 2026-11-20.

Want to take one? Open an issue with the **Add a model** template so that nobody duplicates the work, then follow
[converting-a-new-bnn.md](converting-a-new-bnn.md) (QONNX route) or [adding-a-model.md](adding-a-model.md) (a
converter script).

## Good first conversions

Image classifiers and small networks built from operators Bitino already has (binary / float convolution, dense,
pooling, element-wise). The work is mostly reading the checkpoint and matching the source framework's semantics.

| Model id | Task | Source | Notes |
|---|---|---|---|
| `larq-zoo-quicknet-notop`, `larq-zoo-binary-densenet28-notop` | ImageNet backbones | Larq Zoo | the converted classifiers without their heads; `tools/convert_larq_zoo.py` |
| `hls4ml-keras-bnn-jettag`, `hls4ml-keras-3layer-binary-jettag` | jet tagging (16 features → 5 classes) | Keras (hls4ml) | binary MLPs |
| `rajmali-zebracam-bcnn-v3` | zebra-crossing detection (80×60 grey) | PyTorch | small binary CNN |
| `adabin-resnet20-cifar10-mindspore` | CIFAR-10 | MindSpore | AdaBin binarizer (learned per-channel centre and scale) |
| `bmxnet-v1-resnet18-imagenet`, `bmxnet-v1-resnet18-cifar10`, `bmxnet-v1-lenet-mnist` | classification | MXNet (BMXNet) | needs an MXNet `.params` reader (plain binary format) and BMXNet's bit-packing order |
| `bmxnet-v2-android-resnete18`, `bmxnet-v2-android-binary-densenet28` | ImageNet | MXNet (BMXNet v2) | the networks behind BMXNet's Android demo |

## Multi-bit activations (binary weights)

Bitino runs binary weights with k-bit activations (`QBConv2D` / `QBDense`, DoReFa style). These use 2- or 4-bit
activations: `brevitas-bnn-pynq-{cnv,lfc,sfc,tfc}-1w2a`, `bnn-pynq-cnv-w1a2-cifar10`, `bnn-pynq-lfc-w1a2-mnist`,
`finn-examples-resnet50-w1a2-imagenet`, `hwgq-resnet18-w1a2-bw`, `tensorpack-dorefa-alexnet-w1a2-g6`,
`tensorpack-dorefa-resnet18-w1a4`, `fracbnn-cifar10`, `fracbnn-imagenet`. The Brevitas and FINN ones should import
through QONNX (`tools/export_brevitas_qonnx.py`).

## Need new operators

| Model ids | Task | What is missing |
|---|---|---|
| `omz-{face,pedestrian,vehicle}-detection-adas-binary-0001`, `bidet-ssd300-voc`, `ida-det-faster-rcnn-bir18-voc` | object detection | SSD / Faster R-CNN heads (prior boxes, decoding, NMS) |
| `bulat-binary-face-alignment-aflw`, `bulat-binary-human-pose-mpii` | landmarks, pose | hourglass networks; heat-map decoding |
| `bimatting-rvm-stage4`, `brve-llrvd`, `brve-smoid` | video matting, low-light video | recurrent (ConvGRU) state across frames |
| `svnet-*` (5 models) | point clouds | k-NN graph features (DGCNN), point-wise max pooling |
| `bisci-*`, `bisrnet-cassi-*` (8 models) | hyperspectral reconstruction | spectral shifts / masks of the CASSI pipeline |
| `bicd-pcd`, `bicd-vl-cmu-cd` | change detection | two-image (siamese) inputs |
| `bi-diffsr-x2`, `bi-diffsr-x4`, `monodiffuse-cifar10-w1a1-core-seed0` | diffusion | sampling loop, time embeddings |
| `bmtnet-hybridevs`, `pathfinder-*` | demosaicing, segmentation | task-specific pre- and post-processing |
| `ebconv-expert-binary-resnet-imagenet` | ImageNet | per-image expert selection (one of 4 weight banks per layer) |
| `bcdnet-a-imagenet` | ImageNet | circular shifts of binary activations |
| `allenai-xnor-net-alexnet-xnor` | ImageNet (the original XNOR-Net) | a Torch7 reader; `sign(0) = 0` gives ternary activations; non-commercial weights (fetch recipe only) |

Language models with 1- or 1.58-bit weights (`microsoft-bitnet-b1.58-2b-4t`, `fbi-llm-130m`, Bonsai, BitMamba-2, …)
are in the registry but out of scope: they need a decoder runtime (KV cache, sampling), which projects such as
bitnet.cpp already provide.
