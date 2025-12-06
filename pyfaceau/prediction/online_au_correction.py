#!/usr/bin/env python3
"""
Online AU Correction - Python port of C++ CorrectOnlineAUs

This module implements the online AU correction mechanism from OpenFace's FaceAnalyser.
The correction drags AU predictions toward 0, assuming the bottom 10% of predictions
correspond to neutral expressions.

C++ Reference: FaceAnalyser.cpp, lines 942-1003 (CorrectOnlineAUs) and 1190-1245 (UpdatePredictionTrack)

Key parameters (matching C++):
- num_bins: 200 histogram bins
- min_val: -3.0 (histogram range minimum)
- max_val: 5.0 (histogram range maximum)
- ratio: 0.10 (10th percentile cutoff)
- min_frames: 10 (minimum frames before correction starts)
"""

import numpy as np
from typing import Dict, List, Optional


class OnlineAUCorrection:
    """
    Online AU correction that tracks per-AU histograms and computes 10th percentile
    correction offsets that are subtracted from raw predictions.

    This mirrors C++ FaceAnalyser::CorrectOnlineAUs exactly.
    """

    def __init__(
        self,
        au_names: List[str],
        num_bins: int = 200,
        min_val: float = -3.0,
        max_val: float = 5.0,
        ratio: float = 0.10,
        min_frames: int = 10,
        clip_values: bool = True
    ):
        """
        Initialize online AU correction.

        Args:
            au_names: List of AU names (e.g., ['AU01_r', 'AU02_r', ...])
            num_bins: Number of histogram bins (C++ default: 200)
            min_val: Minimum value for histogram range (C++ default: -3)
            max_val: Maximum value for histogram range (C++ default: 5)
            ratio: Percentile ratio for correction (C++ default: 0.10 = 10th percentile)
            min_frames: Minimum frames before correction is applied (C++ default: 10)
            clip_values: Whether to clip corrected values to [0, 5] (C++ default: True)
        """
        self.au_names = list(au_names)
        self.num_aus = len(au_names)
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.ratio = ratio
        self.min_frames = min_frames
        self.clip_values = clip_values

        # Compute histogram length
        self.length = abs(max_val - min_val)

        # Prediction correction histogram: shape (num_aus, num_bins)
        # Matches: cv::Mat_<int>((int)predictions.size(), num_bins, (int)0)
        self.prediction_corr_histogram = np.zeros((self.num_aus, num_bins), dtype=np.int32)

        # Frame count
        self.prediction_correction_count = 0

        # Current correction offsets (computed from histogram)
        self.correction = np.zeros(self.num_aus, dtype=np.float64)

    def reset(self):
        """Reset correction state (for new video/person)."""
        self.prediction_corr_histogram.fill(0)
        self.prediction_correction_count = 0
        self.correction.fill(0.0)

    def update_prediction_track(self, predictions: Dict[str, float]) -> None:
        """
        Update the histogram with new predictions.

        This mirrors C++ UpdatePredictionTrack function (FaceAnalyser.cpp:1190-1245).

        Args:
            predictions: Dictionary mapping AU names to raw prediction values
        """
        # Update histogram for each AU
        for i, au_name in enumerate(self.au_names):
            if au_name not in predictions:
                continue

            value = predictions[au_name]

            # Find the bin corresponding to this prediction
            # C++: int index = (int)((predictions[i].second - min_val)*((double)num_bins)/(length));
            index = int((value - self.min_val) * self.num_bins / self.length)

            # Clamp to valid range
            if index < 0:
                index = 0
            elif index > self.num_bins - 1:
                index = self.num_bins - 1

            self.prediction_corr_histogram[i, index] += 1

        # Update frame count
        self.prediction_correction_count += 1

        # Recompute correction if we have enough frames
        if self.prediction_correction_count >= self.min_frames:
            self._recompute_correction()

    def _recompute_correction(self) -> None:
        """
        Recompute correction offsets from histogram.

        This mirrors the correction recomputation in UpdatePredictionTrack (FaceAnalyser.cpp:1226-1243).
        """
        # Cutoff point: 10% of total frames
        cutoff_point = int(self.ratio * self.prediction_correction_count)

        # For each AU, find the value at the cutoff percentile
        for i in range(self.num_aus):
            cumulative_sum = 0
            for j in range(self.num_bins):
                cumulative_sum += self.prediction_corr_histogram[i, j]
                if cumulative_sum > cutoff_point:
                    # C++: double corr = min_val + j * (length/num_bins);
                    corr = self.min_val + j * (self.length / self.num_bins)
                    self.correction[i] = corr
                    break

    def correct(
        self,
        predictions: Dict[str, float],
        update_track: bool = True,
        dyn_shift: bool = True
    ) -> Dict[str, float]:
        """
        Apply online correction to AU predictions.

        This mirrors C++ CorrectOnlineAUs (FaceAnalyser.cpp:942-1003).

        Args:
            predictions: Dictionary mapping AU names to raw prediction values
            update_track: Whether to update histogram with these predictions (default: True)
            dyn_shift: Whether to apply the correction shift (default: True)

        Returns:
            Dictionary with corrected AU predictions
        """
        # Update tracking histogram
        if update_track:
            self.update_prediction_track(predictions)

        # Create output predictions
        corrected = dict(predictions)

        # Apply correction shift
        if dyn_shift:
            for i, au_name in enumerate(self.au_names):
                if au_name in corrected:
                    # C++: predictions[i].second = predictions[i].second - correction[i];
                    corrected[au_name] = corrected[au_name] - self.correction[i]

        # Clip values to [0, 5]
        if self.clip_values:
            for au_name in corrected:
                if corrected[au_name] < 0:
                    corrected[au_name] = 0.0
                if corrected[au_name] > 5:
                    corrected[au_name] = 5.0

        return corrected

    def get_correction(self, au_name: str) -> float:
        """Get current correction offset for a specific AU."""
        if au_name in self.au_names:
            idx = self.au_names.index(au_name)
            return self.correction[idx]
        return 0.0

    def get_all_corrections(self) -> Dict[str, float]:
        """Get all current correction offsets."""
        return {name: self.correction[i] for i, name in enumerate(self.au_names)}

    def get_frame_count(self) -> int:
        """Get number of frames processed."""
        return self.prediction_correction_count

    def is_correction_active(self) -> bool:
        """Check if correction has enough frames to be active."""
        return self.prediction_correction_count >= self.min_frames


def test_online_correction():
    """Test the online AU correction implementation."""
    print("=" * 60)
    print("Testing OnlineAUCorrection")
    print("=" * 60)

    # Create corrector for a few test AUs
    au_names = ['AU01_r', 'AU02_r', 'AU25_r']
    corrector = OnlineAUCorrection(au_names)

    # Simulate some frames with biased predictions
    # (predictions consistently higher than they should be)
    np.random.seed(42)
    n_frames = 50

    print(f"\nSimulating {n_frames} frames with biased predictions...")

    # Generate predictions with known bias
    # True baseline: ~0.2 for neutral, predictions are ~0.2 + bias
    bias = {'AU01_r': 0.3, 'AU02_r': 0.2, 'AU25_r': 0.5}

    for i in range(n_frames):
        # Simulate neutral expression most of the time, occasional expression
        is_expression = np.random.random() < 0.3

        predictions = {}
        for au_name in au_names:
            if is_expression:
                # Expression: higher value
                predictions[au_name] = 1.5 + np.random.random() * 0.5 + bias[au_name]
            else:
                # Neutral: should be ~0, but biased
                predictions[au_name] = 0.1 + np.random.random() * 0.1 + bias[au_name]

        # Apply correction
        corrected = corrector.correct(predictions)

        if i < 5 or i >= n_frames - 3:
            print(f"\nFrame {i+1}:")
            print(f"  Raw:       {predictions}")
            print(f"  Corrected: {corrected}")
            print(f"  Correction active: {corrector.is_correction_active()}")

    print(f"\n\nFinal correction offsets after {n_frames} frames:")
    print(f"  Corrections: {corrector.get_all_corrections()}")
    print(f"  True biases: {bias}")

    # Verify corrections are close to true biases
    corrections = corrector.get_all_corrections()
    print(f"\nCorrection accuracy:")
    for au_name in au_names:
        diff = abs(corrections[au_name] - bias[au_name])
        status = "GOOD" if diff < 0.15 else "NEEDS WORK"
        print(f"  {au_name}: correction={corrections[au_name]:.3f}, true_bias={bias[au_name]:.3f}, diff={diff:.3f} [{status}]")


if __name__ == "__main__":
    test_online_correction()
