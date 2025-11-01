#!/usr/bin/env python3
"""
Optimized RetinaFace detector using ONNX Runtime with CoreML acceleration.

This module provides a drop-in replacement for the PyTorch-based RetinaFace detector,
optimized for Apple Silicon using the Neural Engine via CoreML execution provider.

Expected performance: 5-10x speedup (from ~191ms to ~20-40ms per detection)
"""

import numpy as np
import torch
import onnxruntime as ort
from typing import Tuple

# Import RetinaFace post-processing utilities
from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
from openface.Pytorch_Retinaface.utils.box_utils import decode, decode_landm
from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from openface.Pytorch_Retinaface.data import cfg_mnet

# Import performance profiler
from performance_profiler import get_profiler


class ONNXRetinaFaceDetector:
    """
    ONNX-accelerated RetinaFace detector for Apple Silicon

    This class provides the same interface as OpenFace 3.0's FaceDetector,
    but uses ONNX Runtime with CoreML execution provider for massive speedup.
    """

    def __init__(self, onnx_model_path: str, use_coreml: bool = True,
                 confidence_threshold: float = 0.02,
                 nms_threshold: float = 0.4,
                 vis_threshold: float = 0.5):
        """
        Initialize ONNX RetinaFace detector

        Args:
            onnx_model_path: Path to converted ONNX model
            use_coreml: Whether to attempt CoreML execution provider (default: True)
            confidence_threshold: Minimum confidence for face detection (default: 0.02)
            nms_threshold: NMS threshold for duplicate suppression (default: 0.4)
            vis_threshold: Visibility threshold for filtering weak detections (default: 0.5)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.vis_threshold = vis_threshold
        self.cfg = cfg_mnet

        # Configure execution providers
        # NOTE: CoreML may not fully support all RetinaFace operations
        # We try CoreML first, but gracefully fall back to optimized CPU execution
        if use_coreml:
            providers = [
                ('CoreMLExecutionProvider', {
                    'MLComputeUnits': 'ALL',  # Use Neural Engine + GPU + CPU
                    'ModelFormat': 'MLProgram',  # Use latest CoreML format
                }),
                'CPUExecutionProvider'  # Fallback
            ]
        else:
            providers = ['CPUExecutionProvider']

        # Load ONNX model
        print(f"Loading ONNX RetinaFace model from: {onnx_model_path}")

        if use_coreml:
            print("")
            print("=" * 70)
            print("⏰ CoreML First-Time Compilation Notice:")
            print("   If this is the first time loading this model with CoreML,")
            print("   compilation may take 30-60 seconds (one-time only).")
            print("   Subsequent loads will be instant (model is cached).")
            print("   Please wait...")
            print("=" * 70)
            print("")

        # Configure session options to prevent thread conflicts
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # Single thread per operator
        sess_options.inter_op_num_threads = 1  # Sequential operator execution
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Suppress CoreML compilation warnings (they're expected for complex models)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.session = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=providers)

        # Check which providers are actually active
        active_providers = self.session.get_providers()

        if 'CoreMLExecutionProvider' in active_providers:
            print("Using CoreML Neural Engine acceleration for face detection")
            print("  Expected: 5-10x speedup")
            self.backend = 'coreml'
        else:
            print("Using ONNX Runtime with optimized CPU execution for face detection")
            print("  Expected: 2-4x speedup over PyTorch")
            print("  (CoreML not available for this model - some operations unsupported)")
            self.backend = 'onnx_cpu'

    def preprocess_image(self, img_array: np.ndarray, resize: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image for RetinaFace inference

        Args:
            img_array: BGR image array (H, W, 3)
            resize: Resize factor (default: 1.0 for no resize)

        Returns:
            Tuple of (preprocessed_tensor, original_image)
        """
        img_raw = img_array.copy()
        img = np.float32(img_raw)

        # Resize if needed
        if resize != 1.0:
            import cv2
            img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        # RetinaFace preprocessing: subtract ImageNet mean
        img -= np.array([104.0, 117.0, 123.0], dtype=np.float32)

        # Convert to NCHW format (batch, channels, height, width)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        return img, img_raw

    def detect_faces(self, img_array: np.ndarray, resize: float = 1.0):
        """
        Detect faces in image array

        Args:
            img_array: BGR image array (H, W, 3)
            resize: Resize factor for detection (default: 1.0)

        Returns:
            Tuple of (detections, original_image)
            detections format: [x1, y1, x2, y2, confidence, landmark_x1, landmark_y1, ...]
        """
        profiler = get_profiler()

        # Preprocess
        with profiler.time_block("preprocessing", f"RetinaFace_preprocess"):
            img, img_raw = self.preprocess_image(img_array, resize)

        # Run ONNX inference on Neural Engine (or optimized CPU)
        with profiler.time_block("model_inference", f"RetinaFace_{self.backend}"):
            outputs = self.session.run(None, {'input': img})

        # Unpack outputs: loc, conf, landms
        loc = outputs[0]  # Bounding box predictions
        conf = outputs[1]  # Confidence scores
        landms = outputs[2]  # 5-point landmarks

        # Post-processing (same as PyTorch version)
        # This part stays in Python as it's fast and complex to export
        with profiler.time_block("postprocessing", f"RetinaFace_postprocess"):
            im_height, im_width, _ = img_raw.shape

            # Convert outputs to torch tensors for compatibility with existing utilities
            loc = torch.from_numpy(loc)
            conf = torch.from_numpy(conf)
            landms = torch.from_numpy(landms)

            # Create scale tensor
            scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])

            # Generate prior boxes for decoding
            priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
            priors = priorbox.forward()
            prior_data = priors.data

            # Decode boxes
            boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
            boxes = boxes * scale / resize
            boxes = boxes.cpu().numpy()

            # Extract scores
            scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

            # Decode landmarks
            landms_decoded = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
            scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5)
            landms_decoded = landms_decoded * scale1 / resize
            landms_decoded = landms_decoded.cpu().numpy()

            # Filter by confidence threshold
            inds = np.where(scores > self.confidence_threshold)[0]
            boxes, landms_decoded, scores = boxes[inds], landms_decoded[inds], scores[inds]

            # Apply NMS
            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = py_cpu_nms(dets, self.nms_threshold)
            dets = dets[keep]
            landms_decoded = landms_decoded[keep]

            # Concatenate boxes and landmarks
            dets = np.concatenate((dets, landms_decoded), axis=1)

        return dets, img_raw

    def get_face(self, img_array: np.ndarray, resize: float = 1.0):
        """
        Get the primary face from image

        Args:
            img_array: BGR image array (H, W, 3)
            resize: Resize factor (default: 1.0)

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


class OptimizedFaceDetector:
    """
    Wrapper class that automatically selects ONNX or PyTorch implementation

    This class provides seamless fallback from ONNX (fast) to PyTorch (slow)
    based on model availability.
    """

    def __init__(self, model_path: str, onnx_model_path: str = None,
                 device: str = "cpu",
                 confidence_threshold: float = 0.02,
                 nms_threshold: float = 0.4,
                 vis_threshold: float = 0.5):
        """
        Initialize face detector with intelligent backend selection.

        Selection logic:
        - CUDA device: Use PyTorch (optimized for NVIDIA GPUs)
        - CPU device: Use ONNX (CoreML on Apple Silicon, optimized CPU on Intel)

        Args:
            model_path: Path to PyTorch model (.pth)
            onnx_model_path: Path to ONNX model (.onnx), defaults to same directory
            device: Device ('cpu' or 'cuda')
            confidence_threshold: Minimum confidence for detection
            nms_threshold: NMS threshold
            vis_threshold: Visibility threshold
        """
        from pathlib import Path

        # Store configuration
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.vis_threshold = vis_threshold
        self.cfg = cfg_mnet

        # Determine ONNX model path
        if onnx_model_path is None:
            model_dir = Path(model_path).parent
            onnx_model_path = model_dir / 'retinaface_mobilenet025_coreml.onnx'

        # CUDA: Use PyTorch directly (best for NVIDIA GPUs)
        if device == 'cuda':
            print("Using PyTorch RetinaFace detector (CUDA-accelerated)")
            from openface.face_detection import FaceDetector
            self.detector = FaceDetector(
                model_path=model_path,
                device=device,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold,
                vis_threshold=vis_threshold
            )
            self.backend = 'pytorch_cuda'
            self.model = self.detector.model
            return

        # CPU: Try ONNX first (CoreML on Apple Silicon, optimized CPU on Intel)
        if Path(onnx_model_path).exists():
            try:
                print("Using ONNX-accelerated RetinaFace detector")
                self.detector = ONNXRetinaFaceDetector(
                    str(onnx_model_path),
                    use_coreml=True,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                    vis_threshold=vis_threshold
                )
                self.backend = 'onnx'

                # Expose model wrapper for compatibility
                class ONNXModelWrapper:
                    def __init__(self, detector):
                        self.detector = detector

                    def __call__(self, img):
                        img_np = img.cpu().numpy()
                        outputs = self.detector.session.run(None, {'input': img_np})
                        return tuple(torch.from_numpy(o) for o in outputs)

                self.model = ONNXModelWrapper(self.detector)
                return
            except Exception as e:
                print(f"Failed to load ONNX model: {e}")
                print("Falling back to PyTorch CPU")

        # Fallback: PyTorch CPU
        print("Using PyTorch RetinaFace detector (CPU)")
        from openface.face_detection import FaceDetector
        self.detector = FaceDetector(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            vis_threshold=vis_threshold
        )
        self.backend = 'pytorch_cpu'
        self.model = self.detector.model

    def detect_faces(self, img_array: np.ndarray, resize: float = 1.0):
        """Detect faces using the selected backend"""
        return self.detector.detect_faces(img_array, resize)

    def get_face(self, img_array: np.ndarray, resize: float = 1.0):
        """Get primary face using the selected backend"""
        return self.detector.get_face(img_array, resize)


if __name__ == '__main__':
    print("ONNX RetinaFace Detector Module")
    print("=" * 60)
    print("This module provides CoreML-accelerated RetinaFace face detection.")
    print("")
    print("Usage:")
    print("  from onnx_retinaface_detector import OptimizedFaceDetector")
    print("  detector = OptimizedFaceDetector('weights/Alignment_RetinaFace.pth')")
    print("  dets, img = detector.detect_faces(image_array)")
    print("=" * 60)
