"""Transformer module for global context modeling.

Applies multi-head self-attention over spatial feature tokens to capture
long-range dependencies that CNNs miss.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal position encoding for 2D feature maps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (B, N, C) where N = H * W

        Returns:
            x with positional encoding added
        """
        B, N, C = x.shape
        device = x.device

        # Generate 2D positional encoding
        y_pos = torch.arange(int(math.sqrt(N)), device=device).unsqueeze(1).float()
        x_pos = torch.arange(int(math.sqrt(N)), device=device).unsqueeze(0).float()

        div_term = torch.exp(torch.arange(0, C, 2, device=device).float() * (-math.log(10000.0) / C))

        pe_y = torch.zeros(1, int(math.sqrt(N)), C, device=device)
        pe_x = torch.zeros(1, int(math.sqrt(N)), C, device=device)

        pe_y[0, :, 0::2] = torch.sin(y_pos * div_term[: C // 2])
        pe_y[0, :, 1::2] = torch.cos(y_pos * div_term[: C // 2])
        pe_x[0, :, 0::2] = torch.sin(x_pos * div_term[: C // 2])
        pe_x[0, :, 1::2] = torch.cos(x_pos * div_term[: C // 2])

        pe = pe_y + pe_x  # (1, H, C)
        pe = pe.repeat(1, int(math.sqrt(N)), 1)  # (1, H*W, C) - simplified

        return x + pe[:, :N, :]


class SpatialTransformer(nn.Module):
    """Transformer encoder for spatial feature maps.

    Flattens the feature map into a sequence of tokens, applies multi-head
    self-attention, then reshapes back to spatial format.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # Project input channels to embedding dimension
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Position encoding
        self.pos_encoding = SinusoidalPositionEncoding(embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to original channels
        self.output_proj = nn.Conv2d(embed_dim, in_channels, kernel_size=1)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer to spatial feature map.

        Args:
            x: (B, C, H, W) feature map

        Returns:
            (B, C, H, W) feature map with global context
        """
        B, C, H, W = x.shape

        # Project to embedding dimension
        x = self.input_proj(x)  # (B, embed_dim, H, W)

        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer
        x = self.transformer(x)  # (B, H*W, embed_dim)

        # Layer norm
        x = self.norm(x)

        # Reshape back to spatial
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        # Project back to original channels
        x = self.output_proj(x)  # (B, C, H, W)

        return x
