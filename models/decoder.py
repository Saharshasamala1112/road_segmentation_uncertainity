"""UNet-style decoder with skip connections and MC Dropout.

Progressively upsamples feature maps while fusing skip connections from
the encoder. Dropout layers remain active during inference for Monte Carlo
uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample and fuse with skip connection.

        Args:
            x: (B, C_in, H, W) lower resolution feature
            skip: (B, C_skip, H*2, W*2) skip connection from encoder

        Returns:
            (B, C_out, H*2, W*2) upsampled and fused feature
        """
        x = self.up(x)

        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """UNet decoder producing dense segmentation with MC Dropout."""

    def __init__(
        self,
        encoder_channels: list[int],
        num_classes: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        # Reverse channels for decoder path (deepest first)
        channels = list(reversed(encoder_channels))

        # Build upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, skip_ch in enumerate(channels[1:]):
            out_ch = skip_ch
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch, dropout))
            in_ch = out_ch

        # Final upsampling to full resolution
        self.final_up = nn.ConvTranspose2d(in_ch, 64, kernel_size=2, stride=2)
        self.final_conv = DoubleConv(64, 64, dropout)

        # Segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Decode multi-scale features to segmentation map.

        Args:
            features: List of encoder features [f1, f2, f3, f4]
                      at scales [1/4, 1/8, 1/16, 1/32]

        Returns:
            (B, num_classes, H, W) segmentation logits
        """
        # Start from deepest feature
        x = features[-1]

        # Apply upsampling blocks with skip connections
        skip_features = features[:-1][::-1]  # Reverse to match decoder order
        for up_block, skip in zip(self.up_blocks, skip_features):
            x = up_block(x, skip)

        # Final upsampling to full resolution
        x = self.final_up(x)
        x = self.final_conv(x)

        # Segmentation head
        x = self.seg_head(x)

        # Ensure output matches input resolution
        return x
