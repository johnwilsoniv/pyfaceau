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
    ):
        super().__init__()

        self.landmark_weight = landmark_weight
        self.global_params_weight = global_params_weight
        self.local_params_weight = local_params_weight

        self.wing_loss = WingLoss(w=10.0, epsilon=2.0)
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
