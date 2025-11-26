"""
Inference wrapper for UnifiedLandmarkPoseNet

Provides a unified interface for landmark and pose prediction with support for:
- PyTorch (CPU, CUDA, MPS)
- ONNX Runtime (Intel optimized)
- CoreML (Apple Silicon optimized)

Usage:
    # Auto-select best backend
    predictor = LandmarkPosePredictor.from_checkpoint('checkpoint_best.pt')

    # Or specify backend explicitly
    predictor = LandmarkPosePredictor.from_onnx('landmark_pose.onnx')
    predictor = LandmarkPosePredictor.from_coreml('landmark_pose.mlpackage')

    # Inference
    aligned_face = get_aligned_face(frame, bbox)  # 112x112x3 BGR
    result = predictor.predict(aligned_face)
    landmarks = result['landmarks']  # (68, 2)
    global_params = result['global_params']  # (6,)
    local_params = result['local_params']  # (34,)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import cv2


class LandmarkPosePredictor:
    """
    Unified interface for landmark and pose prediction.

    Automatically selects the best backend based on available hardware
    and installed libraries.
    """

    def __init__(self):
        self.backend = None
        self.model = None
        self._predictor = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        width_mult: float = 1.0,
    ) -> 'LandmarkPosePredictor':
        """
        Load from PyTorch checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            width_mult: MobileNetV2 width multiplier (must match checkpoint)

        Returns:
            LandmarkPosePredictor instance
        """
        predictor = cls()
        predictor.backend = 'pytorch'
        predictor._predictor = _PyTorchPredictor(checkpoint_path, device, width_mult)
        return predictor

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, Path],
        providers: Optional[list] = None,
    ) -> 'LandmarkPosePredictor':
        """
        Load from ONNX model.

        Args:
            onnx_path: Path to .onnx file
            providers: ONNX Runtime execution providers (None for auto)

        Returns:
            LandmarkPosePredictor instance
        """
        predictor = cls()
        predictor.backend = 'onnx'
        predictor._predictor = _ONNXPredictor(onnx_path, providers)
        return predictor

    @classmethod
    def from_coreml(
        cls,
        mlmodel_path: Union[str, Path],
    ) -> 'LandmarkPosePredictor':
        """
        Load from CoreML model.

        Args:
            mlmodel_path: Path to .mlpackage or .mlmodel

        Returns:
            LandmarkPosePredictor instance
        """
        predictor = cls()
        predictor.backend = 'coreml'
        predictor._predictor = _CoreMLPredictor(mlmodel_path)
        return predictor

    @classmethod
    def from_auto(
        cls,
        model_dir: Union[str, Path],
        prefer_coreml: bool = True,
    ) -> 'LandmarkPosePredictor':
        """
        Automatically select best available model.

        Checks for CoreML, ONNX, and PyTorch checkpoints in order.

        Args:
            model_dir: Directory containing model files
            prefer_coreml: Prefer CoreML on Apple Silicon

        Returns:
            LandmarkPosePredictor instance
        """
        model_dir = Path(model_dir)

        # Check for CoreML (best for Apple Silicon)
        if prefer_coreml:
            coreml_path = model_dir / 'landmark_pose.mlpackage'
            if coreml_path.exists():
                try:
                    return cls.from_coreml(coreml_path)
                except ImportError:
                    pass

        # Check for ONNX
        onnx_path = model_dir / 'landmark_pose.onnx'
        if onnx_path.exists():
            try:
                return cls.from_onnx(onnx_path)
            except ImportError:
                pass

        # Fall back to PyTorch
        checkpoint_path = model_dir / 'checkpoint_best.pt'
        if checkpoint_path.exists():
            return cls.from_checkpoint(checkpoint_path)

        raise FileNotFoundError(
            f"No model found in {model_dir}. "
            "Expected landmark_pose.mlpackage, landmark_pose.onnx, or checkpoint_best.pt"
        )

    def predict(
        self,
        aligned_face: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict landmarks and pose from aligned face image.

        Args:
            aligned_face: Aligned face image (112, 112, 3) in BGR format (uint8)
            return_normalized: If True, return landmarks in [0, 1] range

        Returns:
            Dictionary with:
                - 'landmarks': (68, 2) 2D landmarks in image coordinates (or normalized)
                - 'global_params': (6,) [scale, rx, ry, rz, tx, ty]
                - 'local_params': (34,) PDM shape coefficients
        """
        return self._predictor.predict(aligned_face, return_normalized)

    def predict_batch(
        self,
        aligned_faces: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict landmarks and pose for a batch of faces.

        Args:
            aligned_faces: Batch of aligned faces (N, 112, 112, 3) in BGR format
            return_normalized: If True, return landmarks in [0, 1] range

        Returns:
            Dictionary with:
                - 'landmarks': (N, 68, 2) 2D landmarks
                - 'global_params': (N, 6) pose parameters
                - 'local_params': (N, 34) shape parameters
        """
        return self._predictor.predict_batch(aligned_faces, return_normalized)


class _PyTorchPredictor:
    """PyTorch backend for inference."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        width_mult: float = 1.0,
    ):
        import torch
        from .landmark_pose_net import UnifiedLandmarkPoseNet

        # Select device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = torch.device(device)

        # Load model
        self.model = UnifiedLandmarkPoseNet(width_mult=width_mult)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, aligned_face: np.ndarray) -> 'torch.Tensor':
        """Convert BGR image to model input tensor."""
        import torch

        # BGR -> RGB
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to CHW
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # HWC -> CHW

        # Add batch dimension
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor

    def _preprocess_batch(self, aligned_faces: np.ndarray) -> 'torch.Tensor':
        """Convert batch of BGR images to model input tensor."""
        import torch

        batch_size = len(aligned_faces)
        tensors = []

        for face in aligned_faces:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            tensors.append(tensor)

        batch = torch.from_numpy(np.stack(tensors)).to(self.device)
        return batch

    def predict(
        self,
        aligned_face: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        import torch

        with torch.no_grad():
            input_tensor = self._preprocess(aligned_face)
            output = self.model(input_tensor)

            landmarks = output['landmarks'][0].cpu().numpy()
            global_params = output['global_params'][0].cpu().numpy()
            local_params = output['local_params'][0].cpu().numpy()

            if return_normalized:
                landmarks = landmarks / 112.0

            return {
                'landmarks': landmarks,
                'global_params': global_params,
                'local_params': local_params,
            }

    def predict_batch(
        self,
        aligned_faces: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        import torch

        with torch.no_grad():
            input_tensor = self._preprocess_batch(aligned_faces)
            output = self.model(input_tensor)

            landmarks = output['landmarks'].cpu().numpy()
            global_params = output['global_params'].cpu().numpy()
            local_params = output['local_params'].cpu().numpy()

            if return_normalized:
                landmarks = landmarks / 112.0

            return {
                'landmarks': landmarks,
                'global_params': global_params,
                'local_params': local_params,
            }


class _ONNXPredictor:
    """ONNX Runtime backend for inference."""

    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: Optional[list] = None,
    ):
        import onnxruntime as ort

        # Select providers (execution backends)
        if providers is None:
            available = ort.get_available_providers()
            # Prefer in order: CoreML (macOS), CUDA, CPU
            if 'CoreMLExecutionProvider' in available:
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            elif 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Convert BGR image to model input array."""
        # BGR -> RGB
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and convert to CHW
        array = rgb.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))  # HWC -> CHW

        # Add batch dimension
        return array[np.newaxis, ...]

    def _preprocess_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        """Convert batch of BGR images to model input array."""
        batch = []
        for face in aligned_faces:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            array = rgb.astype(np.float32) / 255.0
            array = np.transpose(array, (2, 0, 1))
            batch.append(array)
        return np.stack(batch)

    def predict(
        self,
        aligned_face: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        input_array = self._preprocess(aligned_face)

        outputs = self.session.run(None, {self.input_name: input_array})

        # ONNX outputs: landmarks (flattened), global_params, local_params
        landmarks = outputs[0][0].reshape(68, 2)
        global_params = outputs[1][0]
        local_params = outputs[2][0]

        if return_normalized:
            landmarks = landmarks / 112.0

        return {
            'landmarks': landmarks,
            'global_params': global_params,
            'local_params': local_params,
        }

    def predict_batch(
        self,
        aligned_faces: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        input_array = self._preprocess_batch(aligned_faces)

        outputs = self.session.run(None, {self.input_name: input_array})

        landmarks = outputs[0].reshape(-1, 68, 2)
        global_params = outputs[1]
        local_params = outputs[2]

        if return_normalized:
            landmarks = landmarks / 112.0

        return {
            'landmarks': landmarks,
            'global_params': global_params,
            'local_params': local_params,
        }


class _CoreMLPredictor:
    """CoreML backend for inference (Apple Silicon optimized)."""

    def __init__(self, mlmodel_path: Union[str, Path]):
        import coremltools as ct

        self.model = ct.models.MLModel(str(mlmodel_path))

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Convert BGR image to model input array."""
        # BGR -> RGB
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        # CoreML expects CHW format with float32 in [0, 1]
        array = rgb.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))  # HWC -> CHW

        # Add batch dimension
        return array[np.newaxis, ...]

    def predict(
        self,
        aligned_face: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        import coremltools as ct

        input_array = self._preprocess(aligned_face)

        # CoreML prediction
        result = self.model.predict({'image': input_array})

        landmarks = np.array(result['landmarks']).reshape(68, 2)
        global_params = np.array(result['global_params']).flatten()
        local_params = np.array(result['local_params']).flatten()

        if return_normalized:
            landmarks = landmarks / 112.0

        return {
            'landmarks': landmarks,
            'global_params': global_params,
            'local_params': local_params,
        }

    def predict_batch(
        self,
        aligned_faces: np.ndarray,
        return_normalized: bool = False,
    ) -> Dict[str, np.ndarray]:
        # CoreML doesn't support dynamic batching well,
        # so we process one at a time
        results = {
            'landmarks': [],
            'global_params': [],
            'local_params': [],
        }

        for face in aligned_faces:
            result = self.predict(face, return_normalized)
            results['landmarks'].append(result['landmarks'])
            results['global_params'].append(result['global_params'])
            results['local_params'].append(result['local_params'])

        return {
            'landmarks': np.stack(results['landmarks']),
            'global_params': np.stack(results['global_params']),
            'local_params': np.stack(results['local_params']),
        }


# Convenience function
def load_predictor(
    model_path: Union[str, Path],
    backend: Optional[str] = None,
) -> LandmarkPosePredictor:
    """
    Load a landmark/pose predictor from a model file or directory.

    Args:
        model_path: Path to model file (.pt, .onnx, .mlpackage) or directory
        backend: Force specific backend ('pytorch', 'onnx', 'coreml', or None for auto)

    Returns:
        LandmarkPosePredictor instance
    """
    model_path = Path(model_path)

    # If directory, auto-select
    if model_path.is_dir():
        return LandmarkPosePredictor.from_auto(model_path)

    # Determine from extension
    suffix = model_path.suffix.lower()

    if backend == 'pytorch' or suffix == '.pt':
        return LandmarkPosePredictor.from_checkpoint(model_path)
    elif backend == 'onnx' or suffix == '.onnx':
        return LandmarkPosePredictor.from_onnx(model_path)
    elif backend == 'coreml' or suffix in ('.mlpackage', '.mlmodel'):
        return LandmarkPosePredictor.from_coreml(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path}")


if __name__ == '__main__':
    # Quick test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python landmark_pose_inference.py <model_path> [image_path]")
        sys.exit(1)

    model_path = sys.argv[1]
    predictor = load_predictor(model_path)

    print(f"Loaded predictor with backend: {predictor.backend}")

    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            sys.exit(1)

        # Assume image is already aligned 112x112
        result = predictor.predict(image)

        print(f"Landmarks shape: {result['landmarks'].shape}")
        print(f"Global params: {result['global_params']}")
        print(f"Local params shape: {result['local_params'].shape}")
    else:
        # Test with random input
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = predictor.predict(dummy_face)

        print(f"Landmarks shape: {result['landmarks'].shape}")
        print(f"Global params shape: {result['global_params'].shape}")
        print(f"Local params shape: {result['local_params'].shape}")
