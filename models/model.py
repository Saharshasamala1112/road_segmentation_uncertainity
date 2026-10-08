"""Complete hybrid CNN-Transformer segmentation model with MC Dropout."""

import torch
import torch.nn as nn
from .encoder import build_encoder
from .transformer import SpatialTransformer
from .decoder import UNetDecoder


class RoadSegmentationModel(nn.Module):
    """Hybrid CNN + Transformer model for uncertainty-aware road segmentation.

    Architecture:
        1. CNN Encoder (ResNet/EfficientNet) extracts multi-scale local features
        2. Transformer module models global context via self-attention
        3. UNet Decoder upsamples with skip connections
        4. MC Dropout layers enable uncertainty quantification at inference
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        pretrained: bool = True,
        transformer_dim: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.5,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        # Encoder
        self.encoder = build_encoder(encoder, pretrained)
        encoder_channels = self.encoder.channels

        # Transformer on deepest feature map
        self.transformer = SpatialTransformer(
            in_channels=encoder_channels[-1],
            embed_dim=transformer_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=0.1,
        )

        # Decoder
        self.decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            (B, num_classes, H, W) segmentation logits
        """
        input_size = x.shape[2:]

        # Encode
        features = self.encoder(x)

        # Apply transformer to deepest feature
        features[-1] = self.transformer(features[-1])

        # Decode
        out = self.decoder(features)

        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = nn.functional.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out

    def enable_mc_dropout(self):
        """Enable dropout layers for Monte Carlo inference.

        Call this before running multiple stochastic forward passes
        for uncertainty estimation.
        """
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def predict_with_uncertainty(
        self, x: torch.Tensor, mc_samples: int = 30
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run MC Dropout inference.

        Args:
            x: (B, 3, H, W) input image tensor
            mc_samples: Number of stochastic forward passes

        Returns:
            Tuple of (mean_prediction, uncertainty_map)
            - mean_prediction: (B, num_classes, H, W) sigmoid probabilities
            - uncertainty_map: (B, 1, H, W) variance across MC samples
        """
        self.eval()
        self.enable_mc_dropout()

        predictions = []
        with torch.no_grad():
            for _ in range(mc_samples):
                logits = self(x)
                probs = torch.sigmoid(logits)
                predictions.append(probs)

        # Stack predictions: (mc_samples, B, num_classes, H, W)
        predictions = torch.stack(predictions)

        # Mean prediction
        mean_pred = predictions.mean(dim=0)

        # Uncertainty (variance)
        uncertainty = predictions.var(dim=0)

        return mean_pred, uncertainty
