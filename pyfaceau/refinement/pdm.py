#!/usr/bin/env python3
"""
Point Distribution Model (PDM) for facial landmark shape constraints.

This module implements a PCA-based shape model similar to OpenFace 2.2 C++
for constraining facial landmarks to plausible shapes. This is particularly
useful for challenging cases like:
- Surgical markings on the face
- Severe facial asymmetry (paralysis)
- Poor lighting or occlusions

Based on the "In-the-wild Aligned PDM" from the OpenFace project.
"""

import numpy as np
from pathlib import Path


class PDM:
    """
    Point Distribution Model using PCA for shape constraints.

    The PDM learns the statistical distribution of facial landmark positions
    and can project arbitrary landmark positions to the nearest plausible shape.

    File format (OpenFace .txt format):
        Line 1: # comments
        Line 2: n_points (e.g., 204 = 68 landmarks * 3 dimensions)
        Line 3: n_modes (number of PCA components, e.g., 1)
        Line 4: n_components (number of principal components, e.g., 6)
        Lines 5+: Mean shape values (one per line)
        Following: Eigenvalues (one per line)
        Following: Eigenvectors (flattened, one value per line)
    """

    def __init__(self, pdm_file_path):
        """
        Load PDM from OpenFace format file.

        Args:
            pdm_file_path: Path to PDM .txt file (e.g., In-the-wild_aligned_PDM_68.txt)
        """
        self.pdm_file = Path(pdm_file_path)
        if not self.pdm_file.exists():
            raise FileNotFoundError(f"PDM file not found: {pdm_file_path}")

        # Parse PDM file
        self._load_pdm()

    def _load_pdm(self):
        """Parse OpenFace PDM format file."""
        with open(self.pdm_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        # Helper function to get next non-comment, non-empty line
        def get_next_value(idx):
            while idx < len(lines):
                line = lines[idx]
                # Skip comments and empty lines
                if line.startswith('#') or len(line) == 0:
                    idx += 1
                    continue
                return line, idx
            raise ValueError("Unexpected end of PDM file")

        # Section 1: Read mean shape header
        idx = 0
        value, idx = get_next_value(idx)
        self.n_points = int(value)  # Total values (68 landmarks * 3 = 204)
        idx += 1

        value, idx = get_next_value(idx)
        n_modes_mean = int(value)  # Number of modes for mean (usually 1)
        idx += 1

        value, idx = get_next_value(idx)
        self.n_components = int(value)  # Number of PCA components
        idx += 1

        # Calculate number of landmarks
        self.n_landmarks = self.n_points // 3

        # Read mean shape (n_points values, one per line)
        mean_values = []
        for _ in range(self.n_points):
            value, idx = get_next_value(idx)
            mean_values.append(float(value))
            idx += 1
        self.mean_shape = np.array(mean_values)

        # Section 2: Read eigenvectors header
        value, idx = get_next_value(idx)  # n_points again
        idx += 1
        value, idx = get_next_value(idx)  # n_modes (total number of eigenvector modes, e.g. 34)
        n_eigen_modes = int(value)
        idx += 1
        value, idx = get_next_value(idx)  # n_components (number we'll use, e.g. 6)
        idx += 1

        # Read eigenvectors (n_points rows, each with n_eigen_modes space-separated values)
        # Format: Each row is one point's coefficients across all modes
        # We only use the first n_components columns
        eigenvector_matrix = np.zeros((self.n_points, n_eigen_modes))
        for i in range(self.n_points):
            value, idx = get_next_value(idx)
            values = [float(v) for v in value.split()]
            if len(values) != n_eigen_modes:
                raise ValueError(f"Expected {n_eigen_modes} values in eigenvector row {i}, got {len(values)}")
            eigenvector_matrix[i, :] = values
            idx += 1

        # Extract only the first n_components columns (the principal components we want)
        self.eigenvectors = eigenvector_matrix[:, :self.n_components]

        # Section 3: Read eigenvalues header
        value, idx = get_next_value(idx)  # n_modes for eigenvalues (usually 1)
        idx += 1
        value, idx = get_next_value(idx)  # n_eigen_modes again
        idx += 1
        value, idx = get_next_value(idx)  # n_components again
        idx += 1

        # Read eigenvalues (space-separated on one line)
        value, idx = get_next_value(idx)
        eigenvalues_all = [float(v) for v in value.split()]

        # Use only first n_components eigenvalues
        self.eigenvalues = np.array(eigenvalues_all[:self.n_components])

        # Store standard deviations (sqrt of eigenvalues)
        self.std_devs = np.sqrt(self.eigenvalues)

    def get_mean_shape_2d(self):
        """
        Get mean facial shape as 2D landmarks (68, 2).

        Returns:
            np.ndarray: Mean landmark positions (68, 2) in pixel coordinates
        """
        # Extract X and Y coordinates (ignore Z for 2D use)
        # OpenFace format: [x1, x2, ..., x68, y1, y2, ..., y68, z1, z2, ..., z68]
        x_coords = self.mean_shape[0:self.n_landmarks]
        y_coords = self.mean_shape[self.n_landmarks:2*self.n_landmarks]

        return np.column_stack([x_coords, y_coords])

    def project_to_shape_space(self, landmarks_2d, n_std_devs=3.0):
        """
        Project arbitrary 2D landmarks to nearest plausible shape using PDM constraints.

        This constrains landmarks to stay within n_std_devs standard deviations
        of the learned shape distribution, similar to OpenFace 2.2 C++ CLNF.

        Args:
            landmarks_2d: (68, 2) array of landmark positions
            n_std_devs: Maximum standard deviations allowed (default: 3.0)

        Returns:
            np.ndarray: (68, 2) constrained landmarks
        """
        if landmarks_2d.shape != (self.n_landmarks, 2):
            raise ValueError(f"Expected landmarks shape (68, 2), got {landmarks_2d.shape}")

        # Convert 2D landmarks to PDM format (X, Y, Z=0)
        # OpenFace format: [x1...x68, y1...y68, z1...z68]
        landmarks_3d = np.zeros(self.n_points)
        landmarks_3d[0:self.n_landmarks] = landmarks_2d[:, 0]  # X coordinates
        landmarks_3d[self.n_landmarks:2*self.n_landmarks] = landmarks_2d[:, 1]  # Y coordinates
        # Z coordinates remain 0 for 2D

        # Center the shape (subtract mean)
        centered_shape = landmarks_3d - self.mean_shape

        # Project onto principal components to get shape parameters
        # shape_params = eigenvectors^T * centered_shape
        shape_params = self.eigenvectors.T @ centered_shape

        # Clamp shape parameters to n_std_devs standard deviations
        # This enforces the PDM constraint
        for i in range(self.n_components):
            max_val = n_std_devs * self.std_devs[i]
            shape_params[i] = np.clip(shape_params[i], -max_val, max_val)

        # Reconstruct shape from clamped parameters
        # constrained_shape = mean + eigenvectors * shape_params
        constrained_shape = self.mean_shape + self.eigenvectors @ shape_params

        # Convert back to 2D landmarks
        x_coords = constrained_shape[0:self.n_landmarks]
        y_coords = constrained_shape[self.n_landmarks:2*self.n_landmarks]

        return np.column_stack([x_coords, y_coords])

    def get_shape_parameters(self, landmarks_2d):
        """
        Get PCA shape parameters for given landmarks.

        Args:
            landmarks_2d: (68, 2) array of landmark positions

        Returns:
            np.ndarray: Shape parameters (n_components,)
        """
        # Convert to 3D format
        landmarks_3d = np.zeros(self.n_points)
        landmarks_3d[0:self.n_landmarks] = landmarks_2d[:, 0]
        landmarks_3d[self.n_landmarks:2*self.n_landmarks] = landmarks_2d[:, 1]

        # Center and project
        centered_shape = landmarks_3d - self.mean_shape
        shape_params = self.eigenvectors.T @ centered_shape

        return shape_params

    def reconstruct_from_parameters(self, shape_params):
        """
        Reconstruct 2D landmarks from shape parameters.

        Args:
            shape_params: (n_components,) array of shape parameters

        Returns:
            np.ndarray: (68, 2) reconstructed landmarks
        """
        # Reconstruct 3D shape
        reconstructed_shape = self.mean_shape + self.eigenvectors @ shape_params

        # Convert to 2D
        x_coords = reconstructed_shape[0:self.n_landmarks]
        y_coords = reconstructed_shape[self.n_landmarks:2*self.n_landmarks]

        return np.column_stack([x_coords, y_coords])


def check_landmark_quality(landmarks):
    """
    Check if landmarks are poorly distributed (clustered, asymmetric).

    Poor landmark quality indicators:
    - Clustering: landmarks concentrated in small region
    - Poor spatial distribution: uneven coverage of face

    Args:
        landmarks: (68, 2) array of landmarks

    Returns:
        tuple: (is_poor_quality: bool, reason: str, clustering_ratio: float)
    """
    if landmarks is None or len(landmarks) != 68:
        return True, "invalid_landmarks", 1.0

    # Calculate bounding box of landmarks
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Check 1: Clustering - landmarks should span reasonable area
    # Split face into left/right halves
    x_center = (x_min + x_max) / 2
    left_count = np.sum(x_coords < x_center)
    right_count = np.sum(x_coords >= x_center)

    # If more than 75% of landmarks on one side, it's clustered
    clustering_ratio = max(left_count, right_count) / 68.0
    if clustering_ratio > 0.75:
        return True, f"clustering_{clustering_ratio:.2f}", clustering_ratio

    # Check 2: Poor spatial distribution
    # Calculate standard deviation of landmark positions
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)

    # If std is too low, landmarks are not well distributed
    expected_x_std = bbox_width * 0.25  # Expect at least 25% of bbox width
    expected_y_std = bbox_height * 0.25  # Expect at least 25% of bbox height

    if x_std < expected_x_std or y_std < expected_y_std:
        return True, "poor_distribution", clustering_ratio

    # Quality is acceptable
    return False, "ok", clustering_ratio
