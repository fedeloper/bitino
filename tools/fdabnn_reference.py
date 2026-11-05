#!/usr/bin/env python3
"""PyTorch port of the official FDA-BNN CIFAR-10 ResNet-20 (MindSpore model zoo, research/cv/FDA-BNN), used as
the float32 / float64 reference of slice S12 (MindSpore cannot run float64 convolutions on CPU).

Ported line by line from the official files (``.cache/repos/mindspore-models-FDA-BNN``, read only):
  * quan.py  Signer.construct (l.32-35): ``P.Sign()(inp * 1)``; MindSpore Sign(0) = 0 -> torch.sign.
             ScaleSigner.construct (l.60-62): ``sign(w) * ReduceMean(Abs(w))`` (mean over the whole tensor).
             QuanConv.construct (l.120-125): ``conv2d(AQuan(inp), WQuan(weight))``, P.Conv2D(mode=1,
             pad_mode="pad", pad=padding, stride, dilation=1, group=1) (l.111-118): zero padding.
  * resnet.py PReLU.construct (l.107-113): ``P.PReLU()(x, w)`` with one slope (channel=1, l.78/134-135).
             BasicCell.construct (l.144-152): ``out = bn1(relu1(conv1(x))); out = bn2(relu2(conv2(out)));
             x = x[:, :, ::2, ::2] if stride != 1 or in != planes; x = nn.Pad(((0,0),(planes//4,)*2,(0,0),(0,0)))(x);
             out += x`` (no activation after the add).
             ResNet.construct (l.207-220): ``bn1(conv1(x))`` (no activation; _conv3x3 = nn.Conv2d(3, 16, 3,
             padding=1, pad_mode='pad'), no bias), layer1-3 ([3, 3, 3], 16/32/64), ``relu``,
             ``ReduceMean(keep_dims=True)(out, (2, 3))``, Flatten, Dense(64, 10) with bias.
             _bn = nn.BatchNorm2d(channel) (l.62-63): eps 1e-5, eval mode uses moving_mean / moving_variance.
Parameter names are the checkpoint's (``layerX.Y.reluN.w``, ``bnN.gamma/beta/moving_mean/moving_variance``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5   # mindspore.nn.BatchNorm2d default


class MSBatchNorm2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.zeros(c))
        self.register_buffer("moving_mean", torch.zeros(c))
        self.register_buffer("moving_variance", torch.ones(c))

    def forward(self, x):
        return F.batch_norm(x, self.moving_mean, self.moving_variance, self.gamma, self.beta, False, 0.0, EPS)


class Sign(nn.Module):   # quan.py Signer / AQuan (hook point of the binarization input)
    def forward(self, x):
        return torch.sign(x * 1)


class QuanConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cout, cin, k, k))
        self.stride, self.padding = stride, padding
        self.quan_a = Sign()

    def quantized_weight(self):
        w = self.weight
        return torch.sign(w) * torch.mean(torch.abs(w))

    def forward(self, x):
        return F.conv2d(self.quan_a(x), self.quantized_weight(), None, self.stride, self.padding)


class PReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.full((1,), 0.25))

    def forward(self, x):
        return F.prelu(x, self.w)


class BasicCell(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = QuanConv(in_planes, planes, 3, stride=stride, padding=1)
        self.bn1 = MSBatchNorm2d(planes)
        self.conv2 = QuanConv(planes, planes, 3, stride=1, padding=1)
        self.bn2 = MSBatchNorm2d(planes)
        self.relu1 = PReLU()
        self.relu2 = PReLU()
        self.stride, self.in_planes, self.planes = stride, in_planes, planes

    def forward(self, x):
        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.bn2(self.relu2(self.conv2(out)))
        if self.stride != 1 or self.in_planes != self.planes:
            x = x[:, :, ::2, ::2]
            x = F.pad(x, (0, 0, 0, 0, self.planes // 4, self.planes // 4))
        return out + x


class ResNet20(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = MSBatchNorm2d(16)
        cells, cin = [], 16
        for planes, stride in ((16, 1), (32, 2), (64, 2)):
            layer = []
            for s in [stride, 1, 1]:
                layer.append(BasicCell(cin, planes, s))
                cin = planes
            cells.append(nn.Sequential(*layer))
        self.layer1, self.layer2, self.layer3 = cells
        self.linear = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.relu(out)
        out = out.mean((2, 3), keepdim=True)
        return self.linear(torch.flatten(out, 1))


def build(params: dict) -> ResNet20:
    """ResNet20 with the checkpoint tensors (numpy arrays from bitino.adapters.mindspore_ckpt.read), strict."""
    model = ResNet20()
    sd = {k: torch.from_numpy(v.copy()) for k, v in params.items()}
    model.load_state_dict(sd, strict=True)
    return model.eval()
