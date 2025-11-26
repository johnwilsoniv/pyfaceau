"""
Inference wrapper for AUPredictionNet

Provides a unified interface for AU intensity prediction with support for:
- PyTorch (CPU, CUDA, MPS)
- ONNX Runtime (Intel optimized)
- CoreML (Apple Silicon optimized)

Usage:
    # Auto-select best backend
    predictor = AUPredictor.from_checkpoint('checkpoint_best.pt')

    # Or specify backend explicitly
    predictor = AUPredictor.from_onnx('au_prediction.onnx')
    predictor = AUPredictor.from_coreml('au_prediction.mlpackage')

    # Inference
    aligned_face = get_aligned_face(frame, bbox)  # 112x112x3 BGR
    result = predictor.predict(aligned_face)
    au_intensities = result['au_intensities']  # (17,)
    au_dict = result['au_dict']  # {'AU01_r': 0.5, 'AU02_r': 1.2, ...}
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import cv2

from .au_prediction_net import AU_NAMES, NUM_AUS


class AUPredictor:
    """
    Unified interface for AU intensity prediction.

    Automatically selects the best backend based on available hardware
    and installed libraries.
    """

    def __init__(self):
        self.backend = None
        self._predictor = None
        self.au_names = AU_NAMES

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        width_mult: float = 1.0,
        dropout: float = 0.3,
    ) -> 'AUPredictor':
        """
        Load from PyTorch checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            width_mult: Backbone width multiplier (must match checkpoint)
            dropout: Dropout rate (must match checkpoint)

        Returns:
            AUPredictor instance
        """
        predictor = cls()
        predictor.backend = 'pytorch'
        predictor._predictor = _PyTorchAUPredictor(
            checkpoint_path, device, width_mult, dropout
        )
        return predictor

    @classmethod
    def from_onnx(
        cls,
        onnx_path: Union[str, Path],
        providers: Optional[list] = None,
    ) -> 'AUPredictor':
        """
        Load from ONNX model.

        Args:
            onnx_path: Path to .onnx file
            providers: ONNX Runtime execution providers (None for auto)

        Returns:
            AUPredictor instance
        """
        predictor = cls()
        predictor.backend = 'onnx'
        predictor._predictor = _ONNXAUPredictor(onnx_path, providers)
        return predictor

    @classmethod
    def from_coreml(
        cls,
        mlmodel_path: Union[str, Path],
    ) -> 'AUPredictor':
        """
        Load from CoreML model.

        Args:
            mlmodel_path: Path to .mlpackage or .mlmodel

        Returns:
            AUPredictor instance
        """
        predictor = cls()
        predictor.backend = 'coreml'
        predictor._predictor = _CoreMLAUPredictor(mlmodel_path)
        return predictor

    @classmethod
    def from_auto(
        cls,
        model_dir: Union[str, Path],
        prefer_coreml: bool = True,
    ) -> 'AUPredictor':
        """
        Automatically select best available model.

        Checks for CoreML, ONNX, and PyTorch checkpoints in order.

        Args:
            model_dir: Directory containing model files
            prefer_coreml: Prefer CoreML on Apple Silicon

        Returns:
            AUPredictor instance
        """
        model_dir = Path(model_dir)

        # Check for CoreML (best for Apple Silicon)
        if prefer_coreml:
            coreml_path = model_dir / 'au_prediction.mlpackage'
            if coreml_path.exists():
                try:
                    return cls.from_coreml(coreml_path)
                except ImportError:
                    pass

        # Check for ONNX
        onnx_path = model_dir / 'au_prediction.onnx'
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
            f"No AU model found in {model_dir}. "
            "Expected au_prediction.mlpackage, au_prediction.onnx, or checkpoint_best.pt"
        )

    def predict(
        self,
        aligned_face: np.ndarray,
        return_dict: bool = True,
    ) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Predict AU intensities from aligned face image.

        Args:
            aligned_face: Aligned face image (112, 112, 3) in BGR format (uint8)
            return_dict: If True, also return named AU dictionary

        Returns:
            Dictionary with:
                - 'au_intensities': (17,) AU intensities in [0, 5]
                - 'au_dict': Dict mapping AU names to intensities (if return_dict=True)
        """
        au_intensities = self._predictor.predict(aligned_face)

        result = {'au_intensities': au_intensities}

        if return_dict:
            result['au_dict'] = {
                name: float(au_intensities[i])
                for i, name in enumerate(self.au_names)
            }

        return result

    def predict_batch(
        self,
        aligned_faces: np.ndarray,
        return_dict: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[Dict[str, float]]]]:
        """
        Predict AU intensities for a batch of faces.

        Args:
            aligned_faces: Batch of aligned faces (N, 112, 112, 3) in BGR format
            return_dict: If True, also return list of named AU dictionaries

        Returns:
            Dictionary with:
                - 'au_intensities': (N, 17) AU intensities
                - 'au_dicts': List of AU name -> intensity dicts (if return_dict=True)
        """
        au_intensities = self._predictor.predict_batch(aligned_faces)

        result = {'au_intensities': au_intensities}

        if return_dict:
            result['au_dicts'] = [
                {name: float(au_intensities[i, j]) for j, name in enumerate(self.au_names)}
                for i in range(len(au_intensities))
            ]

        return result

    def get_au_names(self) -> List[str]:
        """Get list of AU names in order."""
        return list(self.au_names)


class _PyTorchAUPredictor:
    """PyTorch backend for AU inference."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        width_mult: float = 1.0,
        dropout: float = 0.3,
    ):
        import torch
        from .au_prediction_net import AUPredictionNet

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
        self.model = AUPredictionNet(
            width_mult=width_mult,
            dropout=dropout,
        )

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
        tensor = np.transpose(tensor, (2, 0, 1))

        # Add batch dimension
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        return tensor

    def _preprocess_batch(self, aligned_faces: np.ndarray) -> 'torch.Tensor':
        """Convert batch of BGR images to model input tensor."""
        import torch

        tensors = []
        for face in aligned_faces:
            rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = rgb.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            tensors.append(tensor)

        batch = torch.from_numpy(np.stack(tensors)).to(self.device)
        return batch

    def predict(self, aligned_face: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            input_tensor = self._preprocess(aligned_face)
            output = self.model.predict(input_tensor)  # Uses clamped predict
            return output[0].cpu().numpy()

    def predict_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            input_tensor = self._preprocess_batch(aligned_faces)
            output = self.model.predict(input_tensor)
            return output.cpu().numpy()


class _ONNXAUPredictor:
    """ONNX Runtime backend for AU inference."""

    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: Optional[list] = None,
    ):
        import onnxruntime as ort

        if providers is None:
            available = ort.get_available_providers()
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
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
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

    def predict(self, aligned_face: np.ndarray) -> np.ndarray:
        input_array = self._preprocess(aligned_face)
        outputs = self.session.run(None, {self.input_name: input_array})
        au_intensities = np.clip(outputs[0][0], 0, 5)
        return au_intensities

    def predict_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        input_array = self._preprocess_batch(aligned_faces)
        outputs = self.session.run(None, {self.input_name: input_array})
        au_intensities = np.clip(outputs[0], 0, 5)
        return au_intensities


class _CoreMLAUPredictor:
    """CoreML backend for AU inference (Apple Silicon optimized)."""

    def __init__(self, mlmodel_path: Union[str, Path]):
        import coremltools as ct

        self.model = ct.models.MLModel(str(mlmodel_path))

    def _preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Convert BGR image to model input array."""
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        array = rgb.astype(np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return array[np.newaxis, ...]

    def predict(self, aligned_face: np.ndarray) -> np.ndarray:
        input_array = self._preprocess(aligned_face)
        result = self.model.predict({'image': input_array})
        au_intensities = np.clip(np.array(result['au_intensities']).flatten(), 0, 5)
        return au_intensities

    def predict_batch(self, aligned_faces: np.ndarray) -> np.ndarray:
        # CoreML doesn't support dynamic batching well
        results = []
        for face in aligned_faces:
            results.append(self.predict(face))
        return np.stack(results)


# Convenience function
def load_au_predictor(
    model_path: Union[str, Path],
    backend: Optional[str] = None,
) -> AUPredictor:
    """
    Load an AU predictor from a model file or directory.

    Args:
        model_path: Path to model file (.pt, .onnx, .mlpackage) or directory
        backend: Force specific backend ('pytorch', 'onnx', 'coreml', or None for auto)

    Returns:
        AUPredictor instance
    """
    model_path = Path(model_path)

    # If directory, auto-select
    if model_path.is_dir():
        return AUPredictor.from_auto(model_path)

    # Determine from extension
    suffix = model_path.suffix.lower()

    if backend == 'pytorch' or suffix == '.pt':
        return AUPredictor.from_checkpoint(model_path)
    elif backend == 'onnx' or suffix == '.onnx':
        return AUPredictor.from_onnx(model_path)
    elif backend == 'coreml' or suffix in ('.mlpackage', '.mlmodel'):
        return AUPredictor.from_coreml(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python au_prediction_inference.py <model_path> [image_path]")
        sys.exit(1)

    model_path = sys.argv[1]
    predictor = load_au_predictor(model_path)

    print(f"Loaded AU predictor with backend: {predictor.backend}")
    print(f"AUs: {predictor.get_au_names()}")

    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            sys.exit(1)

        result = predictor.predict(image)
        print("\nAU Predictions:")
        for name, intensity in result['au_dict'].items():
            print(f"  {name}: {intensity:.2f}")
    else:
        # Test with random input
        dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        result = predictor.predict(dummy_face)
        print(f"\nAU intensities shape: {result['au_intensities'].shape}")
        print(f"AU intensities range: [{result['au_intensities'].min():.2f}, {result['au_intensities'].max():.2f}]")
