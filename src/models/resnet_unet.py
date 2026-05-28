from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ChannelBridge, DecoderBlock, ResNet50Encoder, SegmentationHead


class ResNetUNet(nn.Module):
    """ResNet-50 encoder with a U-Net decoder for semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 9,
        in_channels: int = 3,
        pretrained_encoder: bool = True,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ResNet50Encoder(in_channels=in_channels, pretrained=pretrained_encoder)

        self.bridge3 = ChannelBridge(1024, 512)
        self.bridge2 = ChannelBridge(512, 256)
        self.bridge1 = ChannelBridge(256, 128)
        self.bridge0 = ChannelBridge(64, 64)

        self.decoder1 = DecoderBlock(2048, 512, skip_channels=512, dropout=decoder_dropout)
        self.decoder2 = DecoderBlock(512, 256, skip_channels=256, dropout=decoder_dropout)
        self.decoder3 = DecoderBlock(256, 128, skip_channels=128, dropout=decoder_dropout)
        self.decoder4 = DecoderBlock(128, 64, skip_channels=64, dropout=decoder_dropout)
        self.head = SegmentationHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self.encoder(x, include_layer4=True)

        x = self.decoder1(features["layer4"], self.bridge3(features["layer3"]))
        x = self.decoder2(x, self.bridge2(features["layer2"]))
        x = self.decoder3(x, self.bridge1(features["layer1"]))
        x = self.decoder4(x, self.bridge0(features["stem"]))
        logits = self.head(x)

        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits

