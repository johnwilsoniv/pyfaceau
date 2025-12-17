"""
UnifiedLandmarkPoseNet - Neural Network for Landmark and Pose Prediction

Replaces pyCLNF iterative optimization with a single forward pass neural network.

Input: 112x112x3 RGB aligned face image
Output:
  - 68 2D landmarks (136 values) in image coordinates
  - 6 global params [scale, rx, ry, rz, tx, ty]
  - 34 local params (PDM shape coefficients)

Architecture:
  - MobileNetV2 backbone (efficient for ARM Mac)
  - Multi-head regression for landmarks, global params, local params
  - Wing loss for landmarks, MSE for params

Target: 20-30 FPS on ARM Mac with >95% correlation to pyCLNF output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Backbone(nn.Module):
    """
    Lightweight MobileNetV2 backbone optimized for 112x112 face images.

    Reduces channels compared to full MobileNetV2 for faster inference
    while maintaining accuracy on the face analysis task.
    """

    def __init__(self, width_mult: float = 1.0):
        super().__init__()

        # MobileNetV2 config: [expand_ratio, channels, num_blocks, stride]
        # Reduced from standard config for 112x112 input
        settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = int(32 * width_mult)

        # First conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )

        # Inverted residual blocks
        self.blocks = nn.ModuleList()
        for t, c, n, s in settings:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # Last conv
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
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
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        return x.flatten(1)


class LandmarkHead(nn.Module):
    """
    Regression head for 68 facial landmarks.

    Predicts landmarks in normalized coordinates [0, 1] relative to 112x112 image.
    """

    def __init__(self, in_features: int, num_landmarks: int = 68):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_landmarks * 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: (batch, 136) -> landmarks in [0, 1] normalized coords
        out = self.fc(x)
        # Sigmoid to constrain to [0, 1]
        return torch.sigmoid(out)


class GlobalParamsHead(nn.Module):
    """
    Regression head for 6 global pose parameters.

    Predicts: [scale, rx, ry, rz, tx, ty]
    - scale: positive (use softplus)
    - rx, ry, rz: rotation in radians (typically [-pi/4, pi/4])
    - tx, ty: translation in pixels (relative to image center)
    """

    def __init__(self, in_features: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        # Apply constraints:
        # - scale: softplus to ensure positive
        # - rotations: tanh * pi/2 to constrain to [-pi/2, pi/2]
        # - translations: no constraint (can be any value)
        scale = F.softplus(out[:, 0:1]) + 0.1  # Minimum scale 0.1
        rotations = torch.tanh(out[:, 1:4]) * (math.pi / 2)
        translations = out[:, 4:6] * 100  # Scale to typical translation range

        return torch.cat([scale, rotations, translations], dim=1)


class LocalParamsHead(nn.Module):
    """
    Regression head for 34 PDM local shape parameters.

    These are PCA coefficients that control facial shape variations.
    Typically in range [-3*sqrt(eigenvalue), 3*sqrt(eigenvalue)].
    """

    def __init__(self, in_features: int, num_params: int = 34):
        super().__init__()
        self.num_params = num_params

        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output: (batch, 34) - PDM shape coefficients
        return self.fc(x)


class UnifiedLandmarkPoseNet(nn.Module):
    """
    Unified network for landmark and pose prediction.

    Replaces the iterative pyCLNF optimization with a single forward pass.

    Input: (batch, 3, 112, 112) RGB face image
    Output: dict with keys:
        - 'landmarks': (batch, 68, 2) in image coordinates [0, 112]
        - 'global_params': (batch, 6) [scale, rx, ry, rz, tx, ty]
        - 'local_params': (batch, 34) PDM shape coefficients

    Usage:
        model = UnifiedLandmarkPoseNet()
        output = model(image_batch)
        landmarks = output['landmarks']  # (B, 68, 2)
        global_params = output['global_params']  # (B, 6)
        local_params = output['local_params']  # (B, 34)
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        num_landmarks: int = 68,
        num_local_params: int = 34,
        image_size: int = 112
    ):
        super().__init__()

        self.image_size = image_size
        self.num_landmarks = num_landmarks

        # Backbone
        self.backbone = MobileNetV2Backbone(width_mult=width_mult)
        backbone_features = self.backbone.last_channel

        # Regression heads
        self.landmark_head = LandmarkHead(backbone_features, num_landmarks)
        self.global_params_head = GlobalParamsHead(backbone_features)
        self.local_params_head = LocalParamsHead(backbone_features, num_local_params)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image tensor (batch, 3, 112, 112) in range [0, 1]

        Returns:
            Dictionary with 'landmarks', 'global_params', 'local_params'
        """
        # Extract features
        features = self.backbone(x)

        # Predict each output
        landmarks_norm = self.landmark_head(features)  # (B, 136) in [0, 1]
        global_params = self.global_params_head(features)  # (B, 6)
        local_params = self.local_params_head(features)  # (B, 34)

        # Reshape landmarks to (B, 68, 2) and scale to image coordinates
        landmarks = landmarks_norm.view(-1, self.num_landmarks, 2) * self.image_size

        return {
            'landmarks': landmarks,
            'global_params': global_params,
            'local_params': local_params,
        }

    def forward_flat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with flat outputs (for ONNX export).

        Returns:
            Tuple of (landmarks, global_params, local_params)
        """
        output = self.forward(x)
        return (
            output['landmarks'].view(-1, self.num_landmarks * 2),
            output['global_params'],
            output['local_params']
        )


class WingLoss(nn.Module):
    """
    Wing loss for landmark regression.

    Better handles small and medium errors compared to L2 loss.
    From: "Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks" (Feng et al., 2018)
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.abs(pred - target)

        # Wing loss formula
        loss = torch.where(
            x < self.w,
            self.w * torch.log(1 + x / self.epsilon),
            x - self.C
        )

        return loss.mean()


class LandmarkPoseLoss(nn.Module):
    """
    Combined loss for landmark and pose prediction.

    Loss = w_lm * WingLoss(landmarks) + w_gp * MSE(global_params) + w_lp * L1(local_params)
    """

    def __init__(
        self,
        landmark_weight: float = 1.0,
        global_params_weight: float = 0.1,
        local_params_weight: float = 0.01,
        wing_w: float = 2.0,
        wing_epsilon: float = 0.5,
    ):
        super().__init__()

        self.landmark_weight = landmark_weight
        self.global_params_weight = global_params_weight
        self.local_params_weight = local_params_weight

        # Tighter Wing loss for sub-pixel accuracy
        # w=2.0: switch to linear beyond 2px error
        # epsilon=0.5: steep penalty for sub-pixel errors
        self.wing_loss = WingLoss(w=wing_w, epsilon=wing_epsilon)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Model output dict
            target: Ground truth dict with same keys

        Returns:
            Dict with 'total', 'landmark', 'global_params', 'local_params' losses
        """
        # Landmark loss (flatten for wing loss)
        lm_pred = pred['landmarks'].view(-1, 68 * 2)
        lm_target = target['landmarks'].view(-1, 68 * 2)
        lm_loss = self.wing_loss(lm_pred, lm_target)

        # Global params loss
        gp_loss = self.mse_loss(pred['global_params'], target['global_params'])

        # Local params loss (L1 for sparsity)
        lp_loss = self.l1_loss(pred['local_params'], target['local_params'])

        # Total loss
        total_loss = (
            self.landmark_weight * lm_loss +
            self.global_params_weight * gp_loss +
            self.local_params_weight * lp_loss
        )

        return {
            'total': total_loss,
            'landmark': lm_loss,
            'global_params': gp_loss,
            'local_params': lp_loss,
        }


# =============================================================================
# HIGH-ACCURACY HEATMAP-BASED LANDMARK NETWORK
# =============================================================================

class SoftArgmax2D(nn.Module):
    """
    Soft-argmax for sub-pixel coordinate extraction from heatmaps.

    Computes expected value of coordinates weighted by softmax of heatmap.
    This is differentiable and provides sub-pixel accuracy.
    """

    def __init__(self, heatmap_size: int = 56, temperature: float = 1.0):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.temperature = temperature

        # Create coordinate grids (normalized to [0, 1])
        coords = torch.linspace(0, 1, heatmap_size)
        self.register_buffer('coord_x', coords.view(1, 1, 1, -1))  # (1, 1, 1, W)
        self.register_buffer('coord_y', coords.view(1, 1, -1, 1))  # (1, 1, H, 1)

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract coordinates from heatmaps using soft-argmax.

        Args:
            heatmaps: (B, num_landmarks, H, W) heatmaps

        Returns:
            coords: (B, num_landmarks, 2) coordinates in [0, 1]
        """
        B, N, H, W = heatmaps.shape

        # Apply temperature-scaled softmax over spatial dimensions
        heatmaps_flat = heatmaps.view(B, N, -1)
        weights = F.softmax(heatmaps_flat / self.temperature, dim=-1)
        weights = weights.view(B, N, H, W)

        # Compute expected x and y coordinates
        x = (weights * self.coord_x).sum(dim=(2, 3))  # (B, N)
        y = (weights * self.coord_y).sum(dim=(2, 3))  # (B, N)

        # Stack to (B, N, 2)
        coords = torch.stack([x, y], dim=-1)

        return coords


class ConvBNReLU(nn.Module):
    """Conv + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class HeatmapLandmarkNet(nn.Module):
    """
    High-accuracy landmark network using heatmap regression + refinement.

    Architecture:
    1. EfficientNet-style backbone for feature extraction
    2. Heatmap decoder: Predicts 68 heatmaps at 56x56 resolution
    3. Soft-argmax: Extracts sub-pixel coordinates from heatmaps
    4. Refinement head: Predicts small offset corrections (±2px)
    5. Pose heads: Global and local parameters

    Target accuracy: <0.5px MAE on 112x112 images
    Target speed: 30+ FPS on ARM Mac
    """

    def __init__(
        self,
        num_landmarks: int = 68,
        heatmap_size: int = 56,
        image_size: int = 112,
        num_global_params: int = 6,
        num_local_params: int = 34,
    ):
        super().__init__()

        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        self.image_size = image_size

        # =====================================================================
        # BACKBONE: Efficient feature extraction (ResNet-18 style, optimized)
        # =====================================================================
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, kernel=3, stride=2),   # 112 -> 56
            ConvBNReLU(32, 64, kernel=3, stride=1),
        )

        # Stage 1: 56x56
        self.stage1 = nn.Sequential(
            self._make_residual_block(64, 64),
            self._make_residual_block(64, 64),
        )

        # Stage 2: 56 -> 28
        self.stage2 = nn.Sequential(
            self._make_residual_block(64, 128, stride=2),
            self._make_residual_block(128, 128),
        )

        # Stage 3: 28 -> 14
        self.stage3 = nn.Sequential(
            self._make_residual_block(128, 256, stride=2),
            self._make_residual_block(256, 256),
        )

        # Stage 4: 14 -> 7
        self.stage4 = nn.Sequential(
            self._make_residual_block(256, 512, stride=2),
            self._make_residual_block(512, 512),
        )

        # =====================================================================
        # HEATMAP DECODER: Multi-scale feature fusion + heatmap prediction
        # =====================================================================
        # Upsample from 7x7 back to 56x56
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fuse3 = ConvBNReLU(512, 256, kernel=1)  # 256 + 256 -> 256

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 14 -> 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse2 = ConvBNReLU(256, 128, kernel=1)  # 128 + 128 -> 128

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 28 -> 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fuse1 = ConvBNReLU(128, 64, kernel=1)  # 64 + 64 -> 64

        # Final heatmap prediction
        self.heatmap_head = nn.Sequential(
            ConvBNReLU(64, 64, kernel=3),
            ConvBNReLU(64, 64, kernel=3),
            nn.Conv2d(64, num_landmarks, kernel_size=1),  # 68 heatmaps
        )

        # Soft-argmax for sub-pixel coordinate extraction
        self.soft_argmax = SoftArgmax2D(heatmap_size=heatmap_size, temperature=1.0)

        # =====================================================================
        # REFINEMENT HEAD: Predict small offset corrections
        # =====================================================================
        # Takes concatenated [coarse_coords, local_features] and predicts offsets
        self.refinement_head = nn.Sequential(
            nn.Linear(num_landmarks * 2 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_landmarks * 2),  # Offset for each coordinate
            nn.Tanh(),  # Bound offsets to [-1, 1], scaled later
        )
        self.max_offset = 2.0 / image_size  # Max ±2px offset (normalized)

        # Global feature pooling for refinement
        self.global_pool_refine = nn.AdaptiveAvgPool2d(1)
        self.refine_proj = nn.Linear(512, 64)

        # =====================================================================
        # POSE HEADS: Global and local parameters
        # =====================================================================
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.global_params_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_global_params),
        )

        self.local_params_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_local_params),
        )

        self._init_weights()

    def _make_residual_block(self, in_ch: int, out_ch: int, stride: int = 1):
        """Create a residual block with optional downsampling."""
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        return ResidualBlock(in_ch, out_ch, stride, downsample)

    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize heatmap head to output near-zero (uniform) heatmaps initially
        nn.init.zeros_(self.heatmap_head[-1].weight)
        nn.init.zeros_(self.heatmap_head[-1].bias)

        # Initialize refinement to output zero offsets initially
        nn.init.zeros_(self.refinement_head[-2].weight)
        nn.init.zeros_(self.refinement_head[-2].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, 3, 112, 112) input images

        Returns:
            Dict with 'landmarks', 'heatmaps', 'global_params', 'local_params'
        """
        B = x.shape[0]

        # Backbone forward with skip connections
        x = self.stem(x)  # (B, 64, 56, 56)

        f1 = self.stage1(x)   # (B, 64, 56, 56)
        f2 = self.stage2(f1)  # (B, 128, 28, 28)
        f3 = self.stage3(f2)  # (B, 256, 14, 14)
        f4 = self.stage4(f3)  # (B, 512, 7, 7)

        # Heatmap decoder with skip connections (U-Net style)
        d4 = self.up4(f4)  # (B, 256, 14, 14)
        d4 = self.fuse3(torch.cat([d4, f3], dim=1))  # (B, 256, 14, 14)

        d3 = self.up3(d4)  # (B, 128, 28, 28)
        d3 = self.fuse2(torch.cat([d3, f2], dim=1))  # (B, 128, 28, 28)

        d2 = self.up2(d3)  # (B, 64, 56, 56)
        d2 = self.fuse1(torch.cat([d2, f1], dim=1))  # (B, 64, 56, 56)

        # Predict heatmaps
        heatmaps = self.heatmap_head(d2)  # (B, 68, 56, 56)

        # Extract coarse coordinates via soft-argmax (in [0, 1])
        coarse_coords = self.soft_argmax(heatmaps)  # (B, 68, 2)

        # Refinement: predict small offset corrections
        global_feat = self.global_pool_refine(f4).view(B, -1)  # (B, 512)
        refine_feat = self.refine_proj(global_feat)  # (B, 64)
        refine_input = torch.cat([coarse_coords.view(B, -1), refine_feat], dim=1)
        offsets = self.refinement_head(refine_input)  # (B, 136)
        offsets = offsets.view(B, self.num_landmarks, 2) * self.max_offset

        # Final coordinates = coarse + offset, scaled to image size
        refined_coords = coarse_coords + offsets
        landmarks = refined_coords * self.image_size  # Scale to pixel coordinates

        # Pose predictions
        pooled = self.global_pool(f4).view(B, -1)  # (B, 512)
        global_params = self.global_params_head(pooled)
        local_params = self.local_params_head(pooled)

        return {
            'landmarks': landmarks,
            'heatmaps': heatmaps,
            'coarse_landmarks': coarse_coords * self.image_size,
            'global_params': global_params,
            'local_params': local_params,
        }


class ResidualBlock(nn.Module):
    """Basic residual block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HeatmapLoss(nn.Module):
    """
    Loss function for heatmap-based landmark prediction.

    Combines:
    1. MSE loss on heatmaps (encourages correct spatial distribution)
    2. Wing loss on final coordinates (penalizes landmark errors)
    3. Consistency loss between coarse and refined coordinates
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        coord_weight: float = 10.0,
        consistency_weight: float = 0.5,
        global_params_weight: float = 0.01,
        local_params_weight: float = 0.001,
        heatmap_size: int = 56,
        image_size: int = 112,
        sigma: float = 1.5,
    ):
        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.consistency_weight = consistency_weight
        self.global_params_weight = global_params_weight
        self.local_params_weight = local_params_weight
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.sigma = sigma

        self.wing_loss = WingLoss(w=2.0, epsilon=0.5)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Pre-compute Gaussian kernel for heatmap generation
        self._init_gaussian_kernel()

    def _init_gaussian_kernel(self):
        """Initialize Gaussian kernel for generating target heatmaps."""
        size = int(self.sigma * 6) | 1  # Ensure odd size
        x = torch.arange(size).float() - size // 2
        gaussian_1d = torch.exp(-x**2 / (2 * self.sigma**2))
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        self.register_buffer('gaussian_kernel', gaussian_2d)

    def generate_target_heatmaps(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Generate target heatmaps from landmark coordinates.

        Args:
            landmarks: (B, 68, 2) landmark coordinates in pixel space

        Returns:
            heatmaps: (B, 68, H, W) target heatmaps
        """
        B, N, _ = landmarks.shape
        H = W = self.heatmap_size
        device = landmarks.device

        # Scale landmarks to heatmap coordinates
        scale = self.heatmap_size / self.image_size
        lm_scaled = landmarks * scale  # (B, N, 2)

        # Create target heatmaps
        heatmaps = torch.zeros(B, N, H, W, device=device)

        # For each landmark, place a Gaussian
        kernel = self.gaussian_kernel
        k_size = kernel.shape[0]
        k_half = k_size // 2

        for b in range(B):
            for n in range(N):
                x, y = lm_scaled[b, n]
                x_int, y_int = int(x.round()), int(y.round())

                # Bounds for placing kernel
                x1 = max(0, x_int - k_half)
                x2 = min(W, x_int + k_half + 1)
                y1 = max(0, y_int - k_half)
                y2 = min(H, y_int + k_half + 1)

                # Corresponding kernel bounds
                kx1 = max(0, k_half - x_int)
                kx2 = k_size - max(0, x_int + k_half + 1 - W)
                ky1 = max(0, k_half - y_int)
                ky2 = k_size - max(0, y_int + k_half + 1 - H)

                if x1 < x2 and y1 < y2:
                    heatmaps[b, n, y1:y2, x1:x2] = kernel[ky1:ky2, kx1:kx2]

        return heatmaps

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Model output with 'landmarks', 'heatmaps', 'coarse_landmarks', etc.
            target: Ground truth with 'landmarks', 'global_params', 'local_params'
        """
        # Generate target heatmaps from ground truth landmarks
        target_heatmaps = self.generate_target_heatmaps(target['landmarks'])

        # Heatmap loss (MSE)
        heatmap_loss = self.mse_loss(pred['heatmaps'], target_heatmaps)

        # Coordinate loss (Wing loss on final landmarks)
        coord_loss = self.wing_loss(pred['landmarks'], target['landmarks'])

        # Consistency loss (coarse should also be close)
        if 'coarse_landmarks' in pred:
            consistency_loss = self.wing_loss(pred['coarse_landmarks'], target['landmarks'])
        else:
            consistency_loss = torch.tensor(0.0, device=pred['landmarks'].device)

        # Pose losses
        gp_loss = self.mse_loss(pred['global_params'], target['global_params'])
        lp_loss = self.l1_loss(pred['local_params'], target['local_params'])

        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.coord_weight * coord_loss +
            self.consistency_weight * consistency_loss +
            self.global_params_weight * gp_loss +
            self.local_params_weight * lp_loss
        )

        return {
            'total': total_loss,
            'heatmap': heatmap_loss,
            'coord': coord_loss,
            'consistency': consistency_loss,
            'global_params': gp_loss,
            'local_params': lp_loss,
        }


def export_to_onnx(
    model: UnifiedLandmarkPoseNet,
    output_path: str,
    opset_version: int = 12
):
    """
    Export model to ONNX format.

    Args:
        model: Trained model
        output_path: Path to save .onnx file
        opset_version: ONNX opset version
    """
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 3, 112, 112)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['landmarks', 'global_params', 'local_params'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'landmarks': {0: 'batch_size'},
            'global_params': {0: 'batch_size'},
            'local_params': {0: 'batch_size'},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {output_path}")


def export_to_coreml(
    model: UnifiedLandmarkPoseNet,
    output_path: str,
):
    """
    Export model to CoreML format for ARM Mac.

    Args:
        model: Trained model
        output_path: Path to save .mlpackage
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("coremltools required for CoreML export. Install with: pip install coremltools")

    model.eval()

    # Trace model
    dummy_input = torch.randn(1, 3, 112, 112)
    traced_model = torch.jit.trace(model, dummy_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 112, 112), scale=1/255.0)],
        outputs=[
            ct.TensorType(name="landmarks"),
            ct.TensorType(name="global_params"),
            ct.TensorType(name="local_params"),
        ],
        minimum_deployment_target=ct.target.macOS13,
    )

    mlmodel.save(output_path)
    print(f"Exported CoreML model to {output_path}")


if __name__ == "__main__":
    # Quick test
    model = UnifiedLandmarkPoseNet(width_mult=1.0)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Test forward pass
    x = torch.randn(2, 3, 112, 112)
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Landmarks shape: {output['landmarks'].shape}")
    print(f"Global params shape: {output['global_params'].shape}")
    print(f"Local params shape: {output['local_params'].shape}")

    # Test loss
    loss_fn = LandmarkPoseLoss()
    target = {
        'landmarks': torch.randn(2, 68, 2) * 112,
        'global_params': torch.randn(2, 6),
        'local_params': torch.randn(2, 34),
    }
    losses = loss_fn(output, target)
    print(f"Total loss: {losses['total'].item():.4f}")
