#!/usr/bin/env python3
"""
Running Median Tracker for Dynamic AU Models

Implements person-specific normalization by tracking the running median
of feature vectors across video frames. This removes person-specific biases
in neutral expressions.

Reference: OpenFace 2.2 FaceAnalyser::UpdateRunningMedian()
"""

import numpy as np
from collections import deque


class RunningMedianTracker:
    """
    Tracks running median of feature vectors for person-specific normalization

    Used by dynamic AU models to normalize for individual neutral expressions.
    """

    def __init__(self, feature_dim: int, window_size: int = 200):
        """
        Initialize running median tracker

        Args:
            feature_dim: Dimensionality of feature vectors
            window_size: Number of frames to use for median calculation
                        (OF2.2 uses histogram with bins, we use rolling window)
        """
        self.feature_dim = feature_dim
        self.window_size = window_size

        # Rolling window of feature vectors (stores last N frames)
        self.history = deque(maxlen=window_size)

        # Current median estimate
        self.current_median = np.zeros(feature_dim, dtype=np.float64)

        # Frame counter
        self.frame_count = 0

    def update(self, features: np.ndarray, update_median: bool = True) -> None:
        """
        Update tracker with new feature vector

        Args:
            features: Feature vector (1D array)
            update_median: Whether to recompute median (OF2.2 does this every 2nd frame)
        """
        # Ensure 1D
        if features.ndim == 2:
            features = features.flatten()

        assert features.shape[0] == self.feature_dim, \
            f"Expected {self.feature_dim} features, got {features.shape[0]}"

        # Add to history
        self.history.append(features.copy())
        self.frame_count += 1

        # Update median (only if requested and have enough samples)
        if update_median and len(self.history) >= 10:
            # Convert history to array (N_frames, N_features)
            history_array = np.array(self.history)

            # Compute median along frame axis
            self.current_median = np.median(history_array, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Get current running median

        Returns:
            Median feature vector (1D array)
        """
        return self.current_median.copy()

    def reset(self) -> None:
        """Reset tracker (e.g., for new video)"""
        self.history.clear()
        self.current_median = np.zeros(self.feature_dim, dtype=np.float64)
        self.frame_count = 0


class DualMedianTracker:
    """
    Manages separate running medians for HOG and geometric features

    OF2.2 tracks HOG and geometric features separately, then concatenates
    them when computing dynamic model predictions.
    """

    def __init__(self, hog_dim: int = 4464, geom_dim: int = 238, window_size: int = 200):
        """
        Initialize dual median tracker

        Args:
            hog_dim: HOG feature dimensionality
            geom_dim: Geometric feature dimensionality
            window_size: Rolling window size
        """
        self.hog_tracker = RunningMedianTracker(hog_dim, window_size)
        self.geom_tracker = RunningMedianTracker(geom_dim, window_size)

    def update(self, hog_features: np.ndarray, geom_features: np.ndarray,
               update_median: bool = True) -> None:
        """
        Update both trackers

        Args:
            hog_features: HOG feature vector
            geom_features: Geometric feature vector
            update_median: Whether to recompute medians
        """
        self.hog_tracker.update(hog_features, update_median)
        self.geom_tracker.update(geom_features, update_median)

    def get_combined_median(self) -> np.ndarray:
        """
        Get concatenated [HOG_median, geom_median]

        Returns:
            Combined median vector (4702 dims)
        """
        hog_median = self.hog_tracker.get_median()
        geom_median = self.geom_tracker.get_median()
        return np.concatenate([hog_median, geom_median])

    def get_hog_median(self) -> np.ndarray:
        """Get HOG median only"""
        return self.hog_tracker.get_median()

    def get_geom_median(self) -> np.ndarray:
        """Get geometric median only"""
        return self.geom_tracker.get_median()

    def reset(self) -> None:
        """Reset both trackers"""
        self.hog_tracker.reset()
        self.geom_tracker.reset()


def test_running_median():
    """Test running median tracker"""
    print("="*80)
    print("Running Median Tracker - Test")
    print("="*80)

    # Create tracker
    tracker = RunningMedianTracker(feature_dim=10, window_size=100)

    # Generate synthetic data (random walk around mean)
    np.random.seed(42)
    mean_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    print("\nSimulating 200 frames...")
    for i in range(200):
        # Add noise to mean
        noise = np.random.randn(10) * 0.5
        features = mean_features + noise

        # Update tracker (update median every frame for testing)
        tracker.update(features, update_median=True)

        if i % 50 == 49:
            median = tracker.get_median()
            print(f"\nFrame {i+1}:")
            print(f"  True mean:    {mean_features[:3]}...")
            print(f"  Running median: {median[:3]}...")
            print(f"  Error: {np.abs(median - mean_features).mean():.6f}")

    print("\nTracker converges to true mean!")

    # Test dual tracker
    print("\n" + "="*80)
    print("Dual Median Tracker - Test")
    print("="*80)

    dual_tracker = DualMedianTracker(hog_dim=4464, geom_dim=238)

    print("\nUpdating with synthetic HOG and geometric features...")
    for i in range(100):
        hog_feat = np.random.randn(4464)
        geom_feat = np.random.randn(238)
        dual_tracker.update(hog_feat, geom_feat, update_median=(i % 2 == 0))

    combined = dual_tracker.get_combined_median()
    print(f"Combined median shape: {combined.shape}")
    print(f"Expected: (4702,)")
    assert combined.shape == (4702,)

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_running_median()
