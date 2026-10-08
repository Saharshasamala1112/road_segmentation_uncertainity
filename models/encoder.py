"""CNN Encoder backbone for feature extraction.

Supports ResNet and EfficientNet architectures with multi-scale feature output.
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    """ResNet-based encoder producing multi-scale feature maps."""

    def __init__(self, variant: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if variant == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        elif variant == "resnet34":
            backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported ResNet variant: {variant}")

        # Extract layers at different depths for skip connections
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 1/4 resolution
        self.layer2 = backbone.layer2  # 1/8 resolution
        self.layer3 = backbone.layer3  # 1/16 resolution
        self.layer4 = backbone.layer4  # 1/32 resolution

        self.channels = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            List of feature maps at scales [1/4, 1/8, 1/16, 1/32]
        """
        x = self.stem(x)
        f1 = self.layer1(x)   # (B, 256, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 512, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 1024, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 2048, H/32, W/32)
        return [f1, f2, f3, f4]


class EfficientNetEncoder(nn.Module):
    """EfficientNet-based encoder producing multi-scale feature maps."""

    def __init__(self, variant: str = "efficientnet_b4", pretrained: bool = True):
        super().__init__()
        if variant == "efficientnet_b4":
            backbone = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif variant == "efficientnet_b0":
            backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        self.features = backbone.features
        # EfficientNet feature extraction points
        self.channels = [24, 32, 56, 160, 1792]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning multi-scale features."""
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Capture features at specific indices for skip connections
            if i in [1, 2, 4, 6, 8]:
                features.append(x)
        return features


def build_encoder(name: str, pretrained: bool = True) -> nn.Module:
    """Factory function to build an encoder by name.

    Args:
        name: Encoder architecture name
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        Encoder module
    """
    encoders = {
        "resnet50": lambda: ResNetEncoder("resnet50", pretrained),
        "resnet34": lambda: ResNetEncoder("resnet34", pretrained),
        "efficientnet_b4": lambda: EfficientNetEncoder("efficientnet_b4", pretrained),
        "efficientnet_b0": lambda: EfficientNetEncoder("efficientnet_b0", pretrained),
    }
    if name not in encoders:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(encoders.keys())}")
    return encoders[name]()
