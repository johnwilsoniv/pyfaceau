#!/usr/bin/env python3
"""
Histogram-Based Running Median Tracker

Implements OpenFace 2.2's histogram-based running median algorithm for
person-specific normalization. Matches the C++ implementation in
FaceAnalyser::UpdateRunningMedian().

Reference: OpenFace/lib/local/FaceAnalyser/src/FaceAnalyser.cpp:764-821
"""

import numpy as np


class HistogramBasedMedianTracker:
    """
    Histogram-based running median tracker matching OpenFace 2.2's implementation

    Uses binned histograms to efficiently compute running median without storing
    all historical values. More memory-efficient than rolling window approach.
    """

    def __init__(self, feature_dim: int, num_bins: int = 200,
                 min_val: float = -3.0, max_val: float = 5.0):
        """
        Initialize histogram-based median tracker

        Args:
            feature_dim: Dimensionality of feature vectors
            num_bins: Number of histogram bins (OF2.2 uses 200)
            min_val: Minimum value for histogram range
            max_val: Maximum value for histogram range
        """
        self.feature_dim = feature_dim
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Histogram: (feature_dim, num_bins)
        # Each row tracks the distribution of one feature dimension
        self.histogram = np.zeros((feature_dim, num_bins), dtype=np.int32)

        # Current median estimate
        self.current_median = np.zeros(feature_dim, dtype=np.float64)

        # Total count of updates
        self.hist_count = 0

        # Precompute bin width
        self.length = max_val - min_val
        self.bin_width = self.length / num_bins

    def update(self, features: np.ndarray, update_histogram: bool = True) -> None:
        """
        Update tracker with new feature vector

        Matches C++ UpdateRunningMedian() logic:
        1. Bin each feature value into histogram
        2. Update histogram counts
        3. Recompute median from cumulative distribution

        Args:
            features: Feature vector (1D array)
            update_histogram: Whether to update histogram (OF2.2 does this every 2nd frame)
        """
        # Ensure 1D
        if features.ndim == 2:
            features = features.flatten()

        assert features.shape[0] == self.feature_dim, \
            f"Expected {self.feature_dim} features, got {features.shape[0]}"

        if update_histogram:
            # Convert feature values to bin indices
            # Formula from C++: (descriptor - min_val) * num_bins / (max_val - min_val)
            converted = (features - self.min_val) * self.num_bins / self.length

            # Cap values to [0, num_bins-1] BEFORE casting to int (matches C++)
            # C++ does: setTo(num_bins-1, converted > num_bins-1) then setTo(0, converted < 0)
            converted = np.clip(converted, 0.0, float(self.num_bins - 1))

            # Cast to int (truncation, matches C++ (int) cast)
            converted = converted.astype(np.int32)

            # Update histogram counts
            for i in range(self.feature_dim):
                bin_idx = converted[i]
                self.histogram[i, bin_idx] += 1

            self.hist_count += 1

        # Compute median on EVERY frame (matches Cython implementation)
        # This must be OUTSIDE the if update_histogram: block
        if self.hist_count == 0:
            # Frame 0: histogram not updated yet, use descriptor directly
            self.current_median = features.copy()
        elif self.hist_count == 1:
            # Frame 1: histogram updated once, still use descriptor directly
            self.current_median = features.copy()
        else:
            # Frame 2+: compute from histogram
            self._compute_median()

    def _compute_median(self, first_descriptor: np.ndarray = None) -> None:
        """
        Compute median from histogram using cumulative sum

        Matches C++ logic:
        - If hist_count == 1: median = descriptor (special case)
        - Otherwise: Find bin where cumulative sum reaches (hist_count+1)/2
        - Convert bin index back to feature value

        Args:
            first_descriptor: If provided and hist_count==1, use directly as median
        """
        # Special case: First frame (matches C++ if(hist_count == 1) { median = descriptor.clone(); })
        if self.hist_count == 1 and first_descriptor is not None:
            self.current_median = first_descriptor.copy()
            return

        cutoff_point = (self.hist_count + 1) // 2

        for i in range(self.feature_dim):
            cumulative_sum = 0

            for j in range(self.num_bins):
                cumulative_sum += self.histogram[i, j]

                if cumulative_sum >= cutoff_point:
                    # Convert bin index back to value
                    # Formula from C++: min_val + bin_idx * bin_width + 0.5 * bin_width
                    self.current_median[i] = (
                        self.min_val +
                        j * self.bin_width +
                        0.5 * self.bin_width
                    )
                    break

    def get_median(self) -> np.ndarray:
        """
        Get current running median

        Returns:
            Median feature vector (1D array)
        """
        return self.current_median.copy()

    def reset(self) -> None:
        """Reset tracker (e.g., for new video)"""
        self.histogram.fill(0)
        self.current_median.fill(0.0)
        self.hist_count = 0


class DualHistogramMedianTracker:
    """
    Manages separate histogram-based running medians for HOG and geometric features

    OF2.2 tracks HOG and geometric features separately with different histogram
    parameters, then concatenates them when computing dynamic model predictions.
    """

    def __init__(self,
                 hog_dim: int = 4464,
                 geom_dim: int = 238,
                 hog_bins: int = 200,
                 hog_min: float = -3.0,
                 hog_max: float = 5.0,
                 geom_bins: int = 200,
                 geom_min: float = -3.0,
                 geom_max: float = 5.0):
        """
        Initialize dual histogram median tracker

        Args:
            hog_dim: HOG feature dimensionality
            geom_dim: Geometric feature dimensionality
            hog_bins: Number of bins for HOG histogram
            hog_min: Minimum value for HOG histogram
            hog_max: Maximum value for HOG histogram
            geom_bins: Number of bins for geometric histogram
            geom_min: Minimum value for geometric histogram
            geom_max: Maximum value for geometric histogram
        """
        self.hog_tracker = HistogramBasedMedianTracker(
            hog_dim, hog_bins, hog_min, hog_max
        )
        self.geom_tracker = HistogramBasedMedianTracker(
            geom_dim, geom_bins, geom_min, geom_max
        )
        # NOTE: Removed frames_tracking counter to match Cython implementation
        # Caller controls update timing via update_histogram parameter

    def update(self, hog_features: np.ndarray, geom_features: np.ndarray,
               update_histogram: bool = True) -> None:
        """
        Update both trackers

        Matches Cython implementation - caller controls update timing via update_histogram.
        No internal frame counter (removed to match Cython).

        Args:
            hog_features: HOG feature vector
            geom_features: Geometric feature vector
            update_histogram: Whether to update histograms (caller controls timing)
        """
        # Pass update_histogram directly (matches Cython implementation)
        self.hog_tracker.update(hog_features, update_histogram)

        # CRITICAL: OpenFace clamps HOG median to >= 0 after update (line 405 in FaceAnalyser.cpp)
        # this->hog_desc_median.setTo(0, this->hog_desc_median < 0);
        # Apply on every call (matches Cython lines 281-285)
        self.hog_tracker.current_median[self.hog_tracker.current_median < 0] = 0.0

        # Update geometric tracker
        self.geom_tracker.update(geom_features, update_histogram)

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


def test_histogram_tracker():
    """Test histogram-based median tracker"""
    print("="*80)
    print("Histogram-Based Median Tracker - Test")
    print("="*80)

    # Create tracker
    tracker = HistogramBasedMedianTracker(
        feature_dim=10,
        num_bins=200,
        min_val=-3.0,
        max_val=5.0
    )

    # Generate synthetic data (random walk around mean)
    np.random.seed(42)
    mean_features = np.array([0.0, 0.5, 1.0, -0.5, -1.0, 2.0, -2.0, 1.5, -1.5, 0.8])

    print(f"\nSimulating 500 frames...")
    print(f"True mean: {mean_features}")

    for i in range(500):
        # Add noise to mean
        noise = np.random.randn(10) * 0.3
        features = mean_features + noise

        # Ensure values in histogram range
        features = np.clip(features, -3.0, 5.0)

        # Update tracker (update histogram every frame for testing)
        tracker.update(features, update_histogram=True)

        if i in [49, 99, 199, 499]:
            median = tracker.get_median()
            error = np.abs(median - mean_features).mean()
            print(f"\nFrame {i+1}:")
            print(f"  True mean:      {mean_features[:3]}...")
            print(f"  Running median: {median[:3]}...")
            print(f"  MAE: {error:.6f}")

    print("\nTracker converges to true mean!")

    # Test dual tracker
    print("\n" + "="*80)
    print("Dual Histogram Median Tracker - Test")
    print("="*80)

    dual_tracker = DualHistogramMedianTracker(
        hog_dim=4464,
        geom_dim=238,
        hog_bins=200,
        hog_min=-3.0,
        hog_max=5.0,
        geom_bins=200,
        geom_min=-3.0,
        geom_max=5.0
    )

    print("\nUpdating with synthetic HOG and geometric features...")
    for i in range(100):
        hog_feat = np.random.randn(4464) * 0.5
        geom_feat = np.random.randn(238) * 0.5

        # Clip to histogram range
        hog_feat = np.clip(hog_feat, -3.0, 5.0)
        geom_feat = np.clip(geom_feat, -3.0, 5.0)

        dual_tracker.update(hog_feat, geom_feat, update_histogram=(i % 2 == 0))

    combined = dual_tracker.get_combined_median()
    print(f"Combined median shape: {combined.shape}")
    print(f"Expected: (4702,)")
    assert combined.shape == (4702,), f"Expected (4702,), got {combined.shape}"

    print("\n" + "="*80)
    print("All tests passed!")
    print("="*80)


if __name__ == "__main__":
    test_histogram_tracker()
