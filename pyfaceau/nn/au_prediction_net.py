"""
AUPredictionNet - Neural Network for Action Unit Intensity Prediction

Replaces HOG+SVM AU prediction with a single forward pass neural network.

Input: 112x112x3 RGB aligned face image
Output: 17 AU intensities (0-5 scale)

AUs predicted:
    AU01_r (Inner Brow Raiser)
    AU02_r (Outer Brow Raiser)
    AU04_r (Brow Lowerer)
    AU05_r (Upper Lid Raiser)
    AU06_r (Cheek Raiser)
    AU07_r (Lid Tightener)
    AU09_r (Nose Wrinkler)
    AU10_r (Upper Lip Raiser)
    AU12_r (Lip Corner Puller)
    AU14_r (Dimpler)
    AU15_r (Lip Corner Depressor)
    AU17_r (Chin Raiser)
    AU20_r (Lip Stretcher)
    AU23_r (Lip Tightener)
    AU25_r (Lips Part)
    AU26_r (Jaw Drop)
    AU45_r (Blink)

Architecture:
  - EfficientNet-lite backbone (more accurate than MobileNetV2 for fine-grained tasks)
  - Multi-task attention heads for different AU groups
  - Smooth L1 loss with per-AU weighting

Target: 20-30 FPS on ARM Mac with >0.90 correlation to pyCLNF AU output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


# AU metadata
AU_NAMES = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]

NUM_AUS = 17

# AU groups by facial region (for attention)
AU_GROUPS = {
    'upper_face': [0, 1, 2, 3, 4, 5],  # AU01-AU07 (brows, eyes)
    'mid_face': [6, 7, 8],              # AU09-AU12 (nose, upper lip, cheeks)
    'lower_face': [9, 10, 11, 12, 13, 14, 15],  # AU14-AU26 (mouth, chin, jaw)
    'eyes': [16],                        # AU45 (blink)
}


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation attention block."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# =============================================================================
# ENHANCED ATTENTION MODULES FOR MAX ACCURACY
# =============================================================================

class ChannelAttention(nn.Module):
    """Channel attention module (part of CBAM)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module (part of CBAM)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Applies channel attention followed by spatial attention.
    From: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AUCorrelationTransformer(nn.Module):
    """
    Transformer to model AU correlations.

    AUs are often correlated (e.g., AU6+AU12 = smile, AU1+AU4 = sadness).
    This module learns these relationships.
    """

    def __init__(self, num_aus: int = 17, hidden_dim: int = 64, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.num_aus = num_aus
        self.hidden_dim = hidden_dim

        # Project each AU embedding to hidden dim
        self.input_proj = nn.Linear(1, hidden_dim)

        # Learnable AU position embeddings
        self.au_embeddings = nn.Parameter(torch.randn(num_aus, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, au_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            au_features: (B, num_aus) initial AU predictions

        Returns:
            refined_aus: (B, num_aus) refined AU predictions
        """
        B = au_features.shape[0]

        # Project each AU to hidden dim: (B, num_aus, 1) -> (B, num_aus, hidden_dim)
        x = self.input_proj(au_features.unsqueeze(-1))

        # Add AU position embeddings
        x = x + self.au_embeddings.unsqueeze(0)

        # Apply transformer
        x = self.transformer(x)

        # Project back to AU intensities
        refined = self.output_proj(x).squeeze(-1)

        # Residual connection
        return au_features + refined


class RegionAUHead(nn.Module):
    """
    Region-specific AU prediction head.

    Uses spatial features from specific face regions for better accuracy.
    """

    def __init__(self, in_channels: int, au_indices: List[int], hidden_dim: int = 128):
        super().__init__()
        self.au_indices = au_indices
        num_aus = len(au_indices)

        # Spatial attention for this region
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

        # AU prediction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_aus),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) spatial features

        Returns:
            au_preds: (B, num_region_aus) predictions for this region's AUs
        """
        # Compute attention weights
        attn = self.attention(features)  # (B, 1, H, W)

        # Weighted average pooling
        weighted = (features * attn).sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)

        # Predict AUs
        return F.relu(self.fc(weighted))


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv block (EfficientNet building block)."""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        use_se: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        ])

        # Squeeze-Excitation
        if use_se:
            layers.append(SqueezeExcitation(hidden_dim))

        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class EfficientNetLiteBackbone(nn.Module):
    """
    Lightweight EfficientNet-inspired backbone for face analysis.

    Optimized for 112x112 input with fewer parameters than full EfficientNet.
    """

    def __init__(self, width_mult: float = 1.0):
        super().__init__()

        # Config: [expand_ratio, channels, num_blocks, stride, use_se]
        settings = [
            [1, 16, 1, 1, False],
            [6, 24, 2, 2, False],
            [6, 40, 2, 2, True],
            [6, 80, 3, 2, True],
            [6, 112, 3, 1, True],
            [6, 192, 4, 2, True],
            [6, 320, 1, 1, True],
        ]

        input_channel = int(32 * width_mult)

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.SiLU(inplace=True),
        )

        # Build blocks
        self.blocks = nn.ModuleList()
        for t, c, n, s, se in settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(MBConv(input_channel, output_channel, stride, t, se))
                input_channel = output_channel

        # Head
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.head = nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.SiLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        x = self.avgpool(x)
        return x.flatten(1)


class AUHead(nn.Module):
    """
    AU prediction head with optional attention.

    Predicts AU intensities in range [0, 5].
    """

    def __init__(
        self,
        in_features: int,
        num_aus: int = NUM_AUS,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_aus = num_aus

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_aus),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output clamped to [0, 5] during inference
        out = self.fc(x)
        # Use ReLU to ensure non-negative, but allow > 5 during training
        # for gradient flow
        return F.relu(out)


class AUPredictionNet(nn.Module):
    """
    Neural network for Action Unit intensity prediction.

    Replaces HOG+SVM pipeline with end-to-end learned features.

    Input: (batch, 3, 112, 112) RGB face image in [0, 1]
    Output: (batch, 17) AU intensities in [0, 5]

    Usage:
        model = AUPredictionNet()
        au_intensities = model(image_batch)  # (B, 17)
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        dropout: float = 0.3,
        use_pretrained_backbone: bool = False,
    ):
        super().__init__()

        # Backbone
        self.backbone = EfficientNetLiteBackbone(width_mult=width_mult)
        backbone_features = self.backbone.last_channel

        # AU prediction head
        self.au_head = AUHead(
            in_features=backbone_features,
            num_aus=NUM_AUS,
            hidden_dim=256,
            dropout=dropout,
        )

        # Store AU names for reference
        self.au_names = AU_NAMES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (batch, 3, 112, 112) in range [0, 1]

        Returns:
            AU intensities (batch, 17) in range [0, 5+]
        """
        features = self.backbone(x)
        au_intensities = self.au_head(features)
        return au_intensities

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with named outputs.

        Returns:
            Dictionary mapping AU names to intensity tensors
        """
        intensities = self.forward(x)
        return {name: intensities[:, i] for i, name in enumerate(self.au_names)}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-time prediction with clamped output.

        Args:
            x: Input image tensor

        Returns:
            AU intensities clamped to [0, 5]
        """
        with torch.no_grad():
            out = self.forward(x)
            return torch.clamp(out, 0, 5)


class EnhancedAUPredictionNet(nn.Module):
    """
    Enhanced AU prediction network with attention mechanisms for maximum accuracy.

    Improvements over basic AUPredictionNet:
    1. CBAM attention after backbone (channel + spatial attention)
    2. Region-specific prediction heads (upper face, mid face, lower face, eyes)
    3. AU correlation transformer to model AU relationships
    4. Multi-scale feature fusion

    Target: r >= 0.99 correlation while maintaining 30+ FPS

    Input: (batch, 3, 112, 112) RGB face image in [0, 1]
    Output: (batch, 17) AU intensities in [0, 5]
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        use_correlation_transformer: bool = True,
        transformer_layers: int = 2,
    ):
        super().__init__()

        self.use_correlation_transformer = use_correlation_transformer

        # Backbone (EfficientNet-lite)
        self.backbone = EfficientNetLiteBackbone(width_mult=width_mult)

        # We need spatial features, so modify backbone to not pool
        # Get the channel count before final pooling
        backbone_spatial_channels = int(320 * width_mult)  # Before head conv
        backbone_final_channels = self.backbone.last_channel  # After head conv (1280)

        # CBAM attention on spatial features
        self.cbam = CBAM(backbone_final_channels, reduction=16, kernel_size=7)

        # Global feature extraction (with attention)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Region-specific heads
        # Each head learns to attend to its relevant face region
        self.region_heads = nn.ModuleDict({
            'upper_face': RegionAUHead(backbone_final_channels, AU_GROUPS['upper_face'], hidden_dim=128),
            'mid_face': RegionAUHead(backbone_final_channels, AU_GROUPS['mid_face'], hidden_dim=96),
            'lower_face': RegionAUHead(backbone_final_channels, AU_GROUPS['lower_face'], hidden_dim=128),
            'eyes': RegionAUHead(backbone_final_channels, AU_GROUPS['eyes'], hidden_dim=64),
        })

        # Global AU head (for ensemble with region heads)
        self.global_head = nn.Sequential(
            nn.Linear(backbone_final_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, NUM_AUS),
        )

        # Fusion weights (learnable combination of global and region predictions)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # AU correlation transformer
        if use_correlation_transformer:
            self.au_transformer = AUCorrelationTransformer(
                num_aus=NUM_AUS,
                hidden_dim=64,
                num_heads=4,
                num_layers=transformer_layers,
            )

        # Store AU names
        self.au_names = AU_NAMES

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize new layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input image tensor (batch, 3, 112, 112) in range [0, 1]

        Returns:
            AU intensities (batch, 17) in range [0, 5+]
        """
        B = x.shape[0]

        # Get spatial features from backbone (before global pooling)
        x = self.backbone.stem(x)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.head(x)  # (B, 1280, 4, 4) for 112x112 input

        # Apply CBAM attention
        x_attended = self.cbam(x)  # (B, 1280, 4, 4)

        # Region-specific predictions
        region_preds = {}
        for region_name, head in self.region_heads.items():
            region_preds[region_name] = head(x_attended)

        # Assemble region predictions into full AU vector
        region_aus = torch.zeros(B, NUM_AUS, device=x.device)
        for region_name, indices in AU_GROUPS.items():
            preds = region_preds[region_name]
            for i, au_idx in enumerate(indices):
                region_aus[:, au_idx] = preds[:, i]

        # Global prediction
        global_features = self.global_pool(x_attended).flatten(1)
        global_aus = F.relu(self.global_head(global_features))

        # Fuse global and region predictions
        alpha = torch.sigmoid(self.fusion_weight)
        fused_aus = alpha * global_aus + (1 - alpha) * region_aus

        # Apply AU correlation transformer for refinement
        if self.use_correlation_transformer:
            refined_aus = self.au_transformer(fused_aus)
        else:
            refined_aus = fused_aus

        return F.relu(refined_aus)

    def forward_dict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with named outputs."""
        intensities = self.forward(x)
        return {name: intensities[:, i] for i, name in enumerate(self.au_names)}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time prediction with clamped output."""
        with torch.no_grad():
            out = self.forward(x)
            return torch.clamp(out, 0, 5)


class ConcordanceCorrelationLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss.

    Better than MSE for agreement between predicted and target values.
    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Loss = 1 - CCC (so lower is better)
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute means
        pred_mean = pred.mean(dim=0)
        target_mean = target.mean(dim=0)

        # Compute variances
        pred_var = pred.var(dim=0) + self.eps
        target_var = target.var(dim=0) + self.eps

        # Compute covariance
        covar = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)

        # CCC per AU
        ccc = (2 * covar) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + self.eps)

        # Average CCC loss (1 - CCC)
        return (1 - ccc).mean()


class AUPredictionLoss(nn.Module):
    """
    Combined loss for AU prediction.

    Uses:
    - Smooth L1 loss for robust regression
    - CCC loss for better correlation
    - Per-AU weighting (optional)
    """

    def __init__(
        self,
        smooth_l1_weight: float = 1.0,
        ccc_weight: float = 0.5,
        au_weights: Optional[List[float]] = None,
    ):
        super().__init__()

        self.smooth_l1_weight = smooth_l1_weight
        self.ccc_weight = ccc_weight

        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.ccc_loss = ConcordanceCorrelationLoss()

        # Per-AU weights (e.g., upweight rare AUs)
        if au_weights is not None:
            self.register_buffer('au_weights', torch.tensor(au_weights))
        else:
            self.au_weights = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss.

        Args:
            pred: Predicted AU intensities (batch, 17)
            target: Ground truth AU intensities (batch, 17)

        Returns:
            Dictionary with 'total', 'smooth_l1', 'ccc' losses
        """
        # Smooth L1 loss
        l1_loss = self.smooth_l1(pred, target)

        if self.au_weights is not None:
            l1_loss = l1_loss * self.au_weights
        l1_loss = l1_loss.mean()

        # CCC loss (only if batch size > 1)
        if pred.size(0) > 1:
            ccc_loss = self.ccc_loss(pred, target)
        else:
            ccc_loss = torch.tensor(0.0, device=pred.device)

        # Total loss
        total_loss = (
            self.smooth_l1_weight * l1_loss +
            self.ccc_weight * ccc_loss
        )

        return {
            'total': total_loss,
            'smooth_l1': l1_loss,
            'ccc': ccc_loss,
        }


def export_au_to_onnx(
    model: AUPredictionNet,
    output_path: str,
    opset_version: int = 12,
):
    """
    Export AU model to ONNX format.

    Args:
        model: Trained model
        output_path: Path to save .onnx file
        opset_version: ONNX opset version
    """
    # Move model to CPU for ONNX export (ONNX is device-agnostic)
    model = model.cpu()
    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['au_intensities'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'au_intensities': {0: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def export_au_to_coreml(
    model: AUPredictionNet,
    output_path: str,
):
    """
    Export AU model to CoreML format.

    Args:
        model: Trained model
        output_path: Path to save .mlpackage
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools required. Install with: pip install coremltools")

    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112)
    traced_model = torch.jit.trace(model, dummy_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 112, 112), scale=1/255.0)],
        outputs=[ct.TensorType(name="au_intensities")],
        minimum_deployment_target=ct.target.macOS13,
    )

    mlmodel.save(output_path)
    print(f"Exported CoreML model to {output_path}")


if __name__ == "__main__":
    # Quick test
    model = AUPredictionNet(width_mult=1.0)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(4, 3, 112, 112)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    # Test loss
    loss_fn = AUPredictionLoss()
    target = torch.rand(4, NUM_AUS) * 5  # Random AUs in [0, 5]
    losses = loss_fn(output, target)
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Smooth L1: {losses['smooth_l1'].item():.4f}")
    print(f"CCC: {losses['ccc'].item():.4f}")
