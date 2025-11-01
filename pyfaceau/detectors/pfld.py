#!/usr/bin/env python3
"""Cunjian PFLD landmark detector wrapper for AU extraction pipeline"""

import numpy as np
import cv2
import onnxruntime as ort


class CunjianPFLDDetector:
    """
    Wrapper for cunjian's PFLD 68-point landmark detector.

    Model: PFLD_ExternalData (112×112)
    Published accuracy: 3.97% NME on 300W Full Set
    Measured accuracy: 4.37% NME on our validation
    Size: 2.9MB
    Speed: 0.01s per face
    """

    def __init__(self, model_path, use_coreml=True):
        """Initialize the PFLD detector.

        Args:
            model_path: Path to the ONNX model file
            use_coreml: Whether to attempt CoreML acceleration on Apple Silicon (default: True)
        """
        # Configure execution providers for Apple Silicon Neural Engine acceleration
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

        # Suppress CoreML compilation warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.session = ort.InferenceSession(model_path, providers=providers)

        # Check which provider is active
        active_providers = self.session.get_providers()
        if 'CoreMLExecutionProvider' in active_providers:
            print("PFLD using CoreML Neural Engine acceleration (2-3x speedup)")
        else:
            print("Warning: PFLD using CPU execution (CoreML unavailable)")

        self.input_name = self.session.get_inputs()[0].name

        # Model expects 112x112 RGB input normalized to [0, 1]
        self.input_size = 112

    def detect_landmarks(self, frame, bbox):
        """Detect 68 facial landmarks.

        Args:
            frame: BGR image (HxWx3)
            bbox: Face bounding box [x_min, y_min, x_max, y_max]

        Returns:
            landmarks: (68, 2) array of (x, y) coordinates in original image space
            confidence: Dummy confidence (always 1.0 for this model)
        """
        x_min, y_min, x_max, y_max = bbox

        # Calculate square bbox with 10% padding (cunjian approach)
        w = x_max - x_min
        h = y_max - y_min
        size = int(max([w, h]) * 1.1)
        cx = int(x_min + w / 2)
        cy = int(y_min + h / 2)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        # Clip to image bounds and add padding if needed
        height, width = frame.shape[:2]
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop face
        cropped = frame[y1:y2, x1:x2]

        # Add border padding if face was at edge
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx),
                                          cv2.BORDER_CONSTANT, 0)

        # Preprocess: BGR -> RGB, resize to 112x112, normalize to [0, 1]
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(cropped_rgb, (self.input_size, self.input_size))
        face_normalized = face_resized.astype(np.float32) / 255.0
        face_input = np.transpose(face_normalized, (2, 0, 1))  # HWC -> CHW
        face_input = np.expand_dims(face_input, axis=0)  # Add batch dimension

        # Run inference
        output = self.session.run(None, {self.input_name: face_input})[0]

        # Output is (1, 136) -> reshape to (68, 2) normalized coordinates
        landmarks = output.reshape(-1, 2)

        # Reproject from normalized [0, 1] space to original image coordinates
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        landmarks_reprojected = np.zeros_like(landmarks)
        for i, point in enumerate(landmarks):
            landmarks_reprojected[i, 0] = point[0] * bbox_w + x1
            landmarks_reprojected[i, 1] = point[1] * bbox_h + y1

        # Return dummy confidence (PFLD doesn't output per-landmark confidence)
        confidence = 1.0

        return landmarks_reprojected, confidence

    def __repr__(self):
        return (f"CunjianPFLDDetector(input_size={self.input_size}, "
                f"landmarks=68, accuracy=4.37% NME)")
