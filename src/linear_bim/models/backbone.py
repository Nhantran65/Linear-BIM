"""Paper ULW backbone and its linear-Dense-64 variants.

The layer names and initialization order intentionally match the canonical
historical implementation.  ``LinearPaperULW`` is the author-compatible
variant: the penultimate Dense-64 has no activation.  The configurable class
is used for S3's ten-modality, 320-dimensional input while preserving the
same state-dict naming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


WIDTHS = (8, 16, 32)


def glorot_uniform_logical_(
    tensor: torch.Tensor, fan_in: int, fan_out: int, generator: torch.Generator
) -> None:
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-bound, bound, generator=generator)


def keras_same_max_pool1d(value: torch.Tensor, kernel: int = 2, stride: int = 2) -> torch.Tensor:
    length = int(value.shape[-1])
    output = math.ceil(length / stride)
    total = max((output - 1) * stride + kernel - length, 0)
    left, right = total // 2, total - total // 2
    if left or right:
        value = F.pad(value, (left, right), value=-torch.inf)
    return F.max_pool1d(value, kernel_size=kernel, stride=stride)


class KerasSeparableConv1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel: int, stride: int):
        super().__init__()
        if kernel not in (1, 3) or stride not in (1, 2):
            raise ValueError((kernel, stride))
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel = kernel
        self.stride = stride
        self.depthwise = nn.Conv1d(
            input_channels,
            input_channels,
            kernel,
            stride=stride,
            padding=1 if kernel == 3 else 0,
            groups=input_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_channels, output_channels, 1, bias=True)

    def reset_keras_parameters(self, generator: torch.Generator) -> None:
        # Keras logical shapes are [kernel, in_channels, depth_multiplier=1]
        # and [1, in_channels, out_channels].
        glorot_uniform_logical_(
            self.depthwise.weight,
            fan_in=self.kernel * self.input_channels,
            fan_out=self.kernel,
            generator=generator,
        )
        glorot_uniform_logical_(
            self.pointwise.weight,
            fan_in=self.input_channels,
            fan_out=self.output_channels,
            generator=generator,
        )
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(value))


class KerasBatchNorm1d(nn.Module):
    """Keras BN with population batch variance and old-stat momentum 0.99."""

    def __init__(self, channels: int, eps: float = 1e-3, update_rate: float = 0.01):
        super().__init__()
        self.eps = eps
        self.momentum = update_rate
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def reset_keras_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        self.running_mean.zero_()
        self.running_var.fill_(1.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = value.mean(dim=(0, 2))
            variance = (value - mean[None, :, None]).square().mean(dim=(0, 2))
            with torch.no_grad():
                self.running_mean.mul_(1.0 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * variance.detach())
        else:
            mean = self.running_mean
            variance = self.running_var
        normalized = (value - mean[None, :, None]) * torch.rsqrt(variance[None, :, None] + self.eps)
        return normalized * self.weight[None, :, None] + self.bias[None, :, None]


class DSSCBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.main1 = KerasSeparableConv1d(input_channels, output_channels, 3, 1)
        self.bn1 = KerasBatchNorm1d(output_channels, eps=1e-3, update_rate=0.01)
        self.main2 = KerasSeparableConv1d(output_channels, output_channels, 3, 1)
        self.bn2 = KerasBatchNorm1d(output_channels, eps=1e-3, update_rate=0.01)
        self.skip1 = KerasSeparableConv1d(input_channels, output_channels, 1, 2)
        self.skip2 = KerasSeparableConv1d(output_channels, output_channels, 1, 2)
        self.dropout = nn.Dropout(0.1)

    def reset_keras_parameters(self, generator: torch.Generator) -> None:
        for layer in (self.main1, self.main2, self.skip1, self.skip2):
            layer.reset_keras_parameters(generator)
        for bn in (self.bn1, self.bn2):
            bn.reset_keras_parameters()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        main = keras_same_max_pool1d(F.relu(self.bn1(self.main1(value))))
        main = keras_same_max_pool1d(F.relu(self.bn2(self.main2(main))))
        skip = self.skip2(self.skip1(value))
        if main.shape != skip.shape:
            raise RuntimeError(f"DSSC shape mismatch: {main.shape} versus {skip.shape}")
        return self.dropout(main + skip)


class PaperULW(nn.Module):
    """Published four-modality ULW topology with the original Dense ReLU."""

    def __init__(self, seed: int, modalities: int = 4):
        super().__init__()
        self.modalities = modalities
        channels = (1,) + WIDTHS[:-1]
        self.blocks = nn.ModuleList(
            [DSSCBlock(input_channels, output_channels) for input_channels, output_channels in zip(channels, WIDTHS, strict=True)]
        )
        self.feature_dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(modalities * WIDTHS[-1], 64)
        self.classifier = nn.Linear(64, 5)
        self.reset_keras_parameters(seed)

    @property
    def separable_layers(self) -> tuple[KerasSeparableConv1d, ...]:
        return tuple(layer for block in self.blocks for layer in (block.main1, block.main2, block.skip1, block.skip2))

    @property
    def separable_kernels(self) -> tuple[torch.Tensor, ...]:
        return tuple(kernel for layer in self.separable_layers for kernel in (layer.depthwise.weight, layer.pointwise.weight))

    def reset_keras_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for block in self.blocks:
            block.reset_keras_parameters(generator)
        glorot_uniform_logical_(self.dense.weight, self.modalities * WIDTHS[-1], 64, generator)
        nn.init.zeros_(self.dense.bias)
        glorot_uniform_logical_(self.classifier.weight, 64, 5, generator)
        nn.init.zeros_(self.classifier.bias)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (self.modalities, 3000):
            raise ValueError(f"expected [B,{self.modalities},3000], got {tuple(value.shape)}")
        batch = int(value.shape[0])
        feature = value.reshape(batch * self.modalities, 1, 3000)
        for block in self.blocks:
            feature = block(feature)
        return feature.mean(dim=-1).reshape(batch, self.modalities * WIDTHS[-1])

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(value)
        hidden = F.relu(self.dense(self.feature_dropout(embedding)))
        return self.classifier(hidden), embedding

    def l2_penalty(self, coefficient: float = 0.001) -> torch.Tensor:
        return coefficient * sum(torch.sum(kernel.square()) for kernel in self.separable_kernels)


class LinearPaperULW(PaperULW):
    """Author-compatible backbone with a linear penultimate Dense-64."""

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(value)
        hidden = self.dense(self.feature_dropout(embedding))
        return self.classifier(hidden), embedding


class ConfigurableLinearULW(LinearPaperULW):
    """Linear Dense-64 ULW accepting a configured modality count."""

    def __init__(self, seed: int, modalities: int = 4):
        super().__init__(seed=seed, modalities=modalities)


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def parameter_ledger(model: PaperULW) -> dict[str, int]:
    row = {
        "feature_extractor": int(sum(parameter.numel() for block in model.blocks for parameter in block.parameters())),
        "dense_128_to_64": int(sum(parameter.numel() for parameter in model.dense.parameters())),
        "classifier_64_to_5": int(sum(parameter.numel() for parameter in model.classifier.parameters())),
        "total": trainable_parameter_count(model),
    }
    row["dense_input_dim"] = int(model.modalities * WIDTHS[-1])
    row["dense_to_64"] = row["dense_128_to_64"]
    return row


@dataclass(frozen=True)
class OperationLedger:
    convolution_macs: int
    dense_macs: int
    total_macs: int
    two_flop_per_mac: int


def operation_ledger(modalities: int = 4) -> OperationLedger:
    length = 3000
    input_channels = 1
    per_modality = 0
    for output_channels in WIDTHS:
        per_modality += length * (input_channels * 3 + input_channels * output_channels)
        first_pool = math.ceil(length / 2)
        per_modality += first_pool * (output_channels * 3 + output_channels * output_channels)
        output_length = math.ceil(first_pool / 2)
        per_modality += first_pool * (input_channels + input_channels * output_channels)
        per_modality += output_length * (output_channels + output_channels * output_channels)
        length = output_length
        input_channels = output_channels
    convolution_macs = modalities * per_modality
    dense_macs = modalities * WIDTHS[-1] * 64 + 64 * 5
    total = convolution_macs + dense_macs
    return OperationLedger(convolution_macs, dense_macs, total, 2 * total)


def tensorflow_cosine_decay_restarts(
    step: int,
    initial_lr: float = 1e-3,
    first_decay_steps: int = 10,
    t_mul: float = 2.0,
    m_mul: float = 1.0,
    alpha: float = 0.0,
) -> float:
    if step < 0:
        raise ValueError(step)
    cycle_start = 0
    cycle_length = first_decay_steps
    cycle_index = 0
    while step >= cycle_start + cycle_length:
        cycle_start += cycle_length
        cycle_length = int(round(cycle_length * t_mul))
        cycle_index += 1
    fraction = (step - cycle_start) / cycle_length
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    decayed = (1.0 - alpha) * cosine + alpha
    return float(initial_lr * (m_mul**cycle_index) * decayed)
