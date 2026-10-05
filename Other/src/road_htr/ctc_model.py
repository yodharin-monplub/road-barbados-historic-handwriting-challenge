"""A compact anisotropic CRNN that keeps horizontal CTC time resolution."""

from __future__ import annotations

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int, batch_norm: bool = True):
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not batch_norm)
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.SiLU(inplace=True))
    return layers


class CRNN(nn.Module):
    """CNN + BiLSTM recognizer with only 4x horizontal downsampling."""

    horizontal_stride = 4

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 384,
        lstm_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_block(3, 64, False),
            nn.MaxPool2d(2, 2),
            *_conv_block(64, 128),
            nn.MaxPool2d(2, 2),
            *_conv_block(128, 256),
            *_conv_block(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),
            *_conv_block(256, 384),
            *_conv_block(384, 384),
            nn.MaxPool2d((2, 1), (2, 1)),
            *_conv_block(384, 512),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).squeeze(2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(self.dropout(sequence))

    @classmethod
    def output_lengths(cls, image_widths: torch.Tensor) -> torch.Tensor:
        return torch.div(image_widths, cls.horizontal_stride, rounding_mode="floor")

