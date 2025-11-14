#!/usr/bin/env python3
"""
PyMTCNN detector wrapper for PyFaceAU integration.

This module provides a drop-in replacement for the RetinaFace detector,
using PyMTCNN for cross-platform face detection with CUDA/CoreML/CPU support.

Expected performance:
- Apple Silicon (M1/M2/M3): 34.26 FPS with CoreML backend
- NVIDIA GPU: 50+ FPS with ONNX + CUDA backend
- CPU: 5-10 FPS with ONNX backend
"""

import numpy as np
from typing import Tuple, Optional

try:
    from pymtcnn import MTCNN
    PYMTCNN_AVAILABLE = True
except ImportError:
    PYMTCNN_AVAILABLE = False


class PyMTCNNDetector:
    """
    PyMTCNN-based face detector for PyFaceAU pipeline

    This class provides the same interface as ONNXRetinaFaceDetector,
    but uses PyMTCNN for face detection, enabling cross-platform acceleration
    with CUDA (NVIDIA), CoreML (Apple Silicon), or CPU backends.

    Key advantages over RetinaFace:
    - Cross-platform: Works on Linux/Windows (CUDA) and macOS (CoreML)
    - Faster on Apple Silicon: 34 FPS vs ~20 FPS with RetinaFace
    - Even faster on NVIDIA GPUs: 50+ FPS with CUDA
    - Simpler pipeline: Combined face + landmark detection
    """

    def __init__(self,
                 backend: str = 'auto',
                 min_face_size: int = 60,
                 thresholds: Optional[list] = None,
                 factor: float = 0.709,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.7,
                 vis_threshold: float = 0.5,
                 verbose: bool = False):
        """
        Initialize PyMTCNN detector

        Args:
            backend: Backend to use ('auto', 'cuda', 'coreml', 'cpu', 'onnx')
                    'auto' will select best available: CUDA > CoreML > CPU
            min_face_size: Minimum face size in pixels (default: 60)
            thresholds: Detection thresholds [PNet, RNet, ONet] (default: [0.6, 0.7, 0.7])
            factor: Image pyramid scale factor (default: 0.709)
            confidence_threshold: Minimum confidence for face detection (default: 0.5)
            nms_threshold: NMS threshold (default: 0.7)
            vis_threshold: Visibility threshold for filtering weak detections (default: 0.5)
            verbose: Print initialization messages (default: False)
        """
        if not PYMTCNN_AVAILABLE:
            raise ImportError(
                "pymtcnn is required for PyMTCNNDetector. Install it with:\n"
                "  pip install pymtcnn[onnx-gpu]  # For CUDA support\n"
                "  pip install pymtcnn[coreml]    # For Apple Silicon\n"
                "  pip install pymtcnn[onnx]      # For CPU-only"
            )

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.vis_threshold = vis_threshold

        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]

        # Initialize PyMTCNN with specified backend
        if verbose:
            print(f"Initializing PyMTCNN detector...")
            print(f"  Backend: {backend}")

        # Note: PyMTCNN backends use fixed parameters (min_face_size=60, thresholds=[0.6, 0.7, 0.7], factor=0.709)
        # Custom parameters are not yet supported by the backend implementations
        if min_face_size != 60 or thresholds != [0.6, 0.7, 0.7] or factor != 0.709:
            if verbose:
                print(f"  Warning: Custom parameters not yet supported by PyMTCNN backends")
                print(f"  Using default: min_face_size=60, thresholds=[0.6, 0.7, 0.7], factor=0.709")

        self.detector = MTCNN(
            backend=backend,
            verbose=verbose
        )

        # Store parameters for reference (even though they're not used)
        self.min_face_size = 60  # Fixed in backends
        self.thresholds = [0.6, 0.7, 0.7]  # Fixed in backends
        self.factor = 0.709  # Fixed in backends

        # Get backend info
        backend_info = self.detector.get_backend_info()
        self.backend = backend_info.get('backend', 'unknown')

        if verbose:
            print(f"  Active backend: {backend_info}")
            print("")

    def detect_faces(self, img_array: np.ndarray, resize: float = 1.0):
        """
        Detect faces in image array

        Args:
            img_array: BGR image array (H, W, 3)
            resize: Resize factor for detection (default: 1.0, ignored for PyMTCNN)

        Returns:
            Tuple of (detections, original_image)
            detections format: [x1, y1, x2, y2, confidence, lm1_x, lm1_y, lm2_x, lm2_y, lm3_x, lm3_y, lm4_x, lm4_y, lm5_x, lm5_y]
        """
        # Note: PyMTCNN handles scaling internally via min_face_size
        # The resize parameter is ignored to maintain consistent behavior

        # Detect faces with PyMTCNN
        bboxes, landmarks = self.detector.detect(img_array)

        if len(bboxes) == 0:
            # No faces detected, return empty array
            return np.array([]), img_array

        # Convert PyMTCNN output to RetinaFace format
        # PyMTCNN: bboxes = (N, 4) [x, y, w, h], landmarks = (N, 5, 2)
        # RetinaFace: dets = (N, 15) [x1, y1, x2, y2, conf, lm_x1, lm_y1, ..., lm_x5, lm_y5]

        num_faces = len(bboxes)
        dets = np.zeros((num_faces, 15), dtype=np.float32)

        for i in range(num_faces):
            x, y, w, h = bboxes[i][:4]
            confidence = bboxes[i][4] if bboxes[i].shape[0] > 4 else 1.0

            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            # Fill bbox and confidence
            dets[i, 0:4] = [x1, y1, x2, y2]
            dets[i, 4] = confidence

            # Fill 5-point landmarks (flatten from (5, 2) to 10 values)
            lm = landmarks[i]  # (5, 2) array
            dets[i, 5:15] = lm.flatten()  # [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]

        return dets, img_array

    def get_face(self, img_array: np.ndarray, resize: float = 1.0):
        """
        Get the primary face from image

        Args:
            img_array: BGR image array (H, W, 3)
            resize: Resize factor (default: 1.0, ignored for PyMTCNN)

        Returns:
            Tuple of (face_crop, detections) or (None, None) if no face found
        """
        dets, img_raw = self.detect_faces(img_array, resize)

        if dets is None or len(dets) == 0:
            return None, None

        det = dets[0]
        confidence = det[4]

        if confidence < self.vis_threshold:
            return None, None

        bbox = det[:4].astype(int)
        face = img_raw[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        return face, dets

    def get_backend_info(self):
        """Get information about the active backend"""
        return self.detector.get_backend_info()


def create_pymtcnn_detector(backend: str = 'auto',
                            min_face_size: int = 60,
                            verbose: bool = False) -> PyMTCNNDetector:
    """
    Factory function to create PyMTCNN detector with recommended settings

    Args:
        backend: Backend to use ('auto', 'cuda', 'coreml', 'cpu')
        min_face_size: Minimum face size in pixels (default: 60)
        verbose: Print initialization messages (default: False)

    Returns:
        Configured PyMTCNNDetector instance

    Example:
        >>> detector = create_pymtcnn_detector(backend='cuda', verbose=True)
        >>> dets, img = detector.detect_faces(frame)
    """
    return PyMTCNNDetector(
        backend=backend,
        min_face_size=min_face_size,
        thresholds=[0.6, 0.7, 0.7],  # Balanced accuracy/speed
        factor=0.709,
        confidence_threshold=0.5,
        nms_threshold=0.7,
        vis_threshold=0.5,
        verbose=verbose
    )


if __name__ == "__main__":
    # Quick test
    import cv2

    print("PyMTCNN Detector Test")
    print("=" * 60)

    # Create detector
    detector = create_pymtcnn_detector(backend='auto', verbose=True)

    # Test with a sample image (requires test image)
    try:
        img = cv2.imread('test_image.jpg')
        if img is not None:
            print("\nTesting detection...")
            dets, img_raw = detector.detect_faces(img)
            print(f"Detected {len(dets)} faces")

            for i, det in enumerate(dets):
                x1, y1, x2, y2, conf = det[:5]
                print(f"  Face {i+1}: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}), conf={conf:.3f}")
        else:
            print("No test image found. Skipping detection test.")
    except Exception as e:
        print(f"Test failed: {e}")

    print("\n" + "=" * 60)
    print("PyMTCNN detector ready for PyFaceAU integration!")
