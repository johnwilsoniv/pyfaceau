"""
Quality Filter for Training Data

Filters frames based on:
- Face detection confidence
- Landmark detection convergence
- Pose extremity
- Temporal jitter (frame-to-frame landmark movement)
- Face size relative to image
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class QualityThresholds:
    """Thresholds for quality filtering."""
    min_detection_confidence: float = 0.8
    max_pitch: float = 45.0  # degrees
    max_yaw: float = 45.0    # degrees
    max_roll: float = 45.0   # degrees
    max_jitter: float = 10.0  # pixels RMS
    min_face_size: float = 50.0  # pixels (bbox width)
    max_face_size: float = 500.0  # pixels
    min_landmark_spread: float = 20.0  # minimum landmark bounding box size


class QualityFilter:
    """
    Quality filter for training data frames.

    Computes a quality score [0, 1] for each frame and determines
    whether it should be included in training data.
    """

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize quality filter.

        Args:
            thresholds: Quality thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or QualityThresholds()
        self.prev_landmarks = None

    def reset(self):
        """Reset temporal state for new video."""
        self.prev_landmarks = None

    def compute_quality_score(
        self,
        landmarks: np.ndarray,
        global_params: np.ndarray,
        bbox: np.ndarray,
        detection_confidence: float = 1.0,
        convergence_success: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute quality score for a frame.

        Args:
            landmarks: 2D landmarks (68, 2)
            global_params: Pose parameters [scale, rx, ry, rz, tx, ty]
            bbox: Face bounding box [x1, y1, x2, y2]
            detection_confidence: Face detection confidence [0, 1]
            convergence_success: Whether landmark fitting converged

        Returns:
            Tuple of (overall_score, detailed_scores)
        """
        scores = {}

        # Detection confidence score
        scores['detection'] = min(detection_confidence / self.thresholds.min_detection_confidence, 1.0)

        # Convergence score - use 0.7 for non-convergence since CLNF may still
        # produce good landmarks even when not fully converged
        scores['convergence'] = 1.0 if convergence_success else 0.7

        # Pose score (penalize extreme poses)
        rx, ry, rz = global_params[1], global_params[2], global_params[3]
        pitch_deg = np.abs(np.degrees(rx))
        yaw_deg = np.abs(np.degrees(ry))
        roll_deg = np.abs(np.degrees(rz))

        pitch_score = max(0, 1.0 - pitch_deg / self.thresholds.max_pitch)
        yaw_score = max(0, 1.0 - yaw_deg / self.thresholds.max_yaw)
        roll_score = max(0, 1.0 - roll_deg / self.thresholds.max_roll)
        scores['pose'] = (pitch_score + yaw_score + roll_score) / 3.0

        # Face size score
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        face_size = (bbox_width + bbox_height) / 2.0

        if face_size < self.thresholds.min_face_size:
            scores['size'] = face_size / self.thresholds.min_face_size
        elif face_size > self.thresholds.max_face_size:
            scores['size'] = self.thresholds.max_face_size / face_size
        else:
            scores['size'] = 1.0

        # Landmark spread score (detect collapsed landmarks)
        lm_min = landmarks.min(axis=0)
        lm_max = landmarks.max(axis=0)
        lm_spread = np.mean(lm_max - lm_min)
        scores['spread'] = min(lm_spread / self.thresholds.min_landmark_spread, 1.0)

        # Temporal jitter score
        if self.prev_landmarks is not None:
            diff = landmarks - self.prev_landmarks
            jitter = np.sqrt(np.mean(diff ** 2))
            scores['jitter'] = max(0, 1.0 - jitter / self.thresholds.max_jitter)
        else:
            scores['jitter'] = 1.0

        # Update previous landmarks for next frame
        self.prev_landmarks = landmarks.copy()

        # Compute overall score (weighted geometric mean)
        weights = {
            'detection': 2.0,
            'convergence': 3.0,
            'pose': 1.5,
            'size': 1.0,
            'spread': 1.5,
            'jitter': 1.0,
        }

        total_weight = sum(weights.values())
        log_sum = sum(w * np.log(max(s, 1e-6)) for (k, s), w in zip(scores.items(), weights.values()))
        overall_score = np.exp(log_sum / total_weight)

        return overall_score, scores

    def should_include(
        self,
        landmarks: np.ndarray,
        global_params: np.ndarray,
        bbox: np.ndarray,
        detection_confidence: float = 1.0,
        convergence_success: bool = True,
        min_quality: float = 0.5
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determine if a frame should be included in training data.

        Args:
            landmarks: 2D landmarks (68, 2)
            global_params: Pose parameters
            bbox: Face bounding box
            detection_confidence: Face detection confidence
            convergence_success: Whether landmark fitting converged
            min_quality: Minimum quality score to include

        Returns:
            Tuple of (should_include, quality_score, detailed_scores)
        """
        quality_score, detailed_scores = self.compute_quality_score(
            landmarks, global_params, bbox, detection_confidence, convergence_success
        )

        # Hard filters (instant rejection) - track reason in detailed_scores
        # Note: We no longer hard-reject on convergence failure since CLNF may still
        # produce usable landmarks even when not fully converged. Convergence is
        # incorporated into the soft quality score instead.

        if detection_confidence < self.thresholds.min_detection_confidence:
            detailed_scores['reject_reason'] = f'low_confidence:{detection_confidence:.3f}<{self.thresholds.min_detection_confidence}'
            return False, quality_score, detailed_scores

        # Check pose limits
        rx, ry, rz = global_params[1], global_params[2], global_params[3]
        pitch_deg, yaw_deg, roll_deg = np.degrees(rx), np.degrees(ry), np.degrees(rz)
        if np.abs(pitch_deg) > self.thresholds.max_pitch:
            detailed_scores['reject_reason'] = f'extreme_pitch:{pitch_deg:.1f}'
            return False, quality_score, detailed_scores
        if np.abs(yaw_deg) > self.thresholds.max_yaw:
            detailed_scores['reject_reason'] = f'extreme_yaw:{yaw_deg:.1f}'
            return False, quality_score, detailed_scores
        if np.abs(roll_deg) > self.thresholds.max_roll:
            detailed_scores['reject_reason'] = f'extreme_roll:{roll_deg:.1f}'
            return False, quality_score, detailed_scores

        # Soft filter based on overall quality
        should_include = quality_score >= min_quality

        return should_include, quality_score, detailed_scores


class TemporalJitterFilter:
    """
    Specialized filter for temporal jitter detection.

    Uses a sliding window to detect frames with excessive
    landmark movement relative to neighbors.
    """

    def __init__(self, window_size: int = 5, max_jitter_ratio: float = 2.0):
        """
        Initialize jitter filter.

        Args:
            window_size: Size of sliding window
            max_jitter_ratio: Maximum ratio of frame jitter to window average
        """
        self.window_size = window_size
        self.max_jitter_ratio = max_jitter_ratio
        self.landmark_history: List[np.ndarray] = []
        self.jitter_history: List[float] = []

    def reset(self):
        """Reset for new video."""
        self.landmark_history = []
        self.jitter_history = []

    def add_frame(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """
        Add a frame and check for jitter.

        Args:
            landmarks: 2D landmarks (68, 2)

        Returns:
            Tuple of (is_valid, jitter_value)
        """
        # Compute jitter from previous frame
        if len(self.landmark_history) > 0:
            diff = landmarks - self.landmark_history[-1]
            jitter = np.sqrt(np.mean(diff ** 2))
        else:
            jitter = 0.0

        self.landmark_history.append(landmarks.copy())
        self.jitter_history.append(jitter)

        # Keep window size
        if len(self.landmark_history) > self.window_size:
            self.landmark_history.pop(0)
            self.jitter_history.pop(0)

        # Check if current frame is outlier
        if len(self.jitter_history) < 3:
            return True, jitter

        avg_jitter = np.mean(self.jitter_history[:-1])  # Exclude current
        if avg_jitter > 0 and jitter > self.max_jitter_ratio * avg_jitter:
            return False, jitter

        return True, jitter

    def filter_sequence(
        self,
        landmarks_sequence: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Filter a sequence of landmarks.

        Args:
            landmarks_sequence: Array of landmarks (N, 68, 2)

        Returns:
            Tuple of (valid_mask, jitter_values)
        """
        self.reset()

        valid_mask = np.ones(len(landmarks_sequence), dtype=bool)
        jitter_values = []

        for i, landmarks in enumerate(landmarks_sequence):
            is_valid, jitter = self.add_frame(landmarks)
            valid_mask[i] = is_valid
            jitter_values.append(jitter)

        return valid_mask, jitter_values
