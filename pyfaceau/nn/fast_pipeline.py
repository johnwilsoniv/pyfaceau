"""
Fast Neural Network Pipeline for Face Analysis

Replaces the slow CLNF + HOG/SVM pipeline with neural networks:
- PyMTCNN: Face detection (unchanged, already fast)
- LandmarkPoseNet: 68 landmarks + pose (replaces pyCLNF)
- AUPredictionNet: 17 AU intensities (replaces HOG + SVM)

Target: 25-30 FPS (vs ~1 FPS original)

Usage:
    from pyfaceau.nn.fast_pipeline import FastAUPipeline

    pipeline = FastAUPipeline()
    results = pipeline.process_frame(frame)
    # results['landmarks'], results['aus'], results['pose']
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import cv2

# Try to import inference engines in order of preference
ONNX_AVAILABLE = False
COREML_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    pass


class LandmarkPosePredictor:
    """
    Fast landmark and pose prediction using ONNX/CoreML.

    Input: Aligned face image (112x112x3, RGB, uint8)
    Output: 68 landmarks, 6 pose params, 34 local params
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = 'auto',
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model file (.onnx or .mlpackage)
            backend: 'onnx', 'coreml', or 'auto'
        """
        self.backend = backend
        self.session = None

        if model_path is None:
            # Default path
            base = Path(__file__).parent.parent / 'weights' / 'nn'
            if backend == 'coreml' or (backend == 'auto' and COREML_AVAILABLE):
                model_path = base / 'landmark_pose.mlpackage'
            else:
                model_path = base / 'landmark_pose.onnx'

        self._load_model(str(model_path))

    def _load_model(self, model_path: str):
        """Load model from file."""
        if model_path.endswith('.mlpackage') and COREML_AVAILABLE:
            self.backend = 'coreml'
            self.model = ct.models.MLModel(model_path)
        elif model_path.endswith('.onnx') and ONNX_AVAILABLE:
            self.backend = 'onnx'
            # Use GPU if available, otherwise CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
        else:
            raise RuntimeError(f"Cannot load model: {model_path}. "
                             f"ONNX available: {ONNX_AVAILABLE}, "
                             f"CoreML available: {COREML_AVAILABLE}")

    def predict(self, aligned_face: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict landmarks and pose from aligned face.

        Args:
            aligned_face: (112, 112, 3) uint8 RGB image

        Returns:
            dict with 'landmarks' (68, 2), 'global_params' (6,), 'local_params' (34,)
        """
        # Preprocess: normalize to [0, 1] and convert to CHW
        image = aligned_face.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, 0)  # Add batch dim

        if self.backend == 'onnx':
            outputs = self.session.run(None, {'image': image})
            landmarks = outputs[0][0]  # (68, 2)
            global_params = outputs[1][0]  # (6,)
            local_params = outputs[2][0]  # (34,)
        else:  # coreml
            pred = self.model.predict({'image': image})
            landmarks = pred['landmarks'][0]
            global_params = pred['global_params'][0]
            local_params = pred['local_params'][0]

        return {
            'landmarks': landmarks,
            'global_params': global_params,
            'local_params': local_params,
        }


class AUPredictor:
    """
    Fast AU prediction using ONNX/CoreML.

    Input: Aligned face image (112x112x3, RGB, uint8)
    Output: 17 AU intensities
    """

    AU_NAMES = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = 'auto',
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model file (.onnx or .mlpackage)
            backend: 'onnx', 'coreml', or 'auto'
        """
        self.backend = backend
        self.session = None

        if model_path is None:
            base = Path(__file__).parent.parent / 'weights' / 'nn'
            if backend == 'coreml' or (backend == 'auto' and COREML_AVAILABLE):
                model_path = base / 'au_prediction.mlpackage'
            else:
                model_path = base / 'au_prediction.onnx'

        self._load_model(str(model_path))

    def _load_model(self, model_path: str):
        """Load model from file."""
        if model_path.endswith('.mlpackage') and COREML_AVAILABLE:
            self.backend = 'coreml'
            self.model = ct.models.MLModel(model_path)
        elif model_path.endswith('.onnx') and ONNX_AVAILABLE:
            self.backend = 'onnx'
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
        else:
            raise RuntimeError(f"Cannot load model: {model_path}")

    def predict(self, aligned_face: np.ndarray) -> Dict[str, float]:
        """
        Predict AU intensities from aligned face.

        Args:
            aligned_face: (112, 112, 3) uint8 RGB image

        Returns:
            dict mapping AU names to intensities
        """
        # Preprocess
        image = aligned_face.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)

        if self.backend == 'onnx':
            outputs = self.session.run(None, {'image': image})
            au_values = outputs[0][0]
        else:
            pred = self.model.predict({'image': image})
            au_values = pred['au_intensities'][0]

        return {name: float(val) for name, val in zip(self.AU_NAMES, au_values)}


class FastAUPipeline:
    """
    Complete fast pipeline for AU analysis.

    Combines:
    - PyMTCNN for face detection
    - LandmarkPoseNet for landmarks/pose
    - AUPredictionNet for AU intensities

    Usage:
        pipeline = FastAUPipeline()

        # Process single frame
        results = pipeline.process_frame(frame)

        # Process video
        for results in pipeline.process_video(video_path):
            print(results['aus'])
    """

    def __init__(
        self,
        landmark_model: Optional[str] = None,
        au_model: Optional[str] = None,
        backend: str = 'auto',
        mtcnn_backend: str = 'auto',
    ):
        """
        Initialize fast pipeline.

        Args:
            landmark_model: Path to landmark model
            au_model: Path to AU model
            backend: 'onnx', 'coreml', or 'auto' for NN models
            mtcnn_backend: Backend for MTCNN ('coreml', 'onnx', 'pytorch')
        """
        # Import here to avoid circular imports
        from pymtcnn import MTCNN

        print("Initializing FastAUPipeline...")

        # Face detector
        print("  Loading MTCNN...")
        self.detector = MTCNN(backend=mtcnn_backend)

        # Landmark predictor
        print("  Loading LandmarkPoseNet...")
        self.landmark_predictor = LandmarkPosePredictor(landmark_model, backend)

        # AU predictor
        print("  Loading AUPredictionNet...")
        self.au_predictor = AUPredictor(au_model, backend)

        # Face aligner (reuse from pyfaceau)
        from pyfaceau.alignment import FaceAligner
        self.aligner = FaceAligner()

        print("  Pipeline ready!")

    def process_frame(
        self,
        frame: np.ndarray,
        return_aligned: bool = False,
    ) -> Optional[Dict]:
        """
        Process a single frame.

        Args:
            frame: BGR image from OpenCV
            return_aligned: Whether to include aligned face in output

        Returns:
            dict with 'landmarks', 'pose', 'aus', 'bbox', optionally 'aligned_face'
            None if no face detected
        """
        # Convert BGR to RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        detections = self.detector.detect(rgb_frame)

        if len(detections) == 0:
            return None

        # Use first/best detection
        det = detections[0]
        bbox = det['box']  # [x, y, w, h]

        # Get aligned face for NN input
        # First, run landmark prediction on cropped face
        x, y, w, h = bbox

        # Expand bbox slightly for better alignment
        pad = int(0.2 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = rgb_frame[y1:y2, x1:x2]

        # Resize to 112x112 for NN
        aligned_face = cv2.resize(face_crop, (112, 112))

        # Predict landmarks and pose
        lm_results = self.landmark_predictor.predict(aligned_face)

        # Predict AUs
        au_results = self.au_predictor.predict(aligned_face)

        # Build output
        result = {
            'bbox': bbox,
            'landmarks': lm_results['landmarks'],
            'global_params': lm_results['global_params'],
            'local_params': lm_results['local_params'],
            'aus': au_results,
            'success': True,
        }

        if return_aligned:
            result['aligned_face'] = aligned_face

        return result

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
    ):
        """
        Process video file, yielding results for each frame.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None for all)
            skip_frames: Process every Nth frame (0 for all)

        Yields:
            dict with frame results
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            result = self.process_frame(frame)

            if result is not None:
                result['frame_idx'] = frame_idx
                result['timestamp'] = frame_idx / fps
                yield result

            frame_idx += 1

            if max_frames and frame_idx >= max_frames:
                break

        cap.release()
