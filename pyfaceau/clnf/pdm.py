#!/usr/bin/env python3
"""
Point Distribution Model (PDM) for shape-constrained landmark fitting.

This module provides a statistical shape model using PCA to represent
plausible facial landmark configurations.
"""

import numpy as np
from pathlib import Path


class PointDistributionModel:
    """
    Point Distribution Model using PCA for shape representation.

    The PDM represents facial landmark shapes as:
        shape = mean_shape + eigenvectors @ params

    where params are the PCA coefficients that control shape variation.
    """

    def __init__(self, model_path):
        """
        Load PDM from OpenFace model file.

        Args:
            model_path: Path to PDM .txt file
        """
        self.model_path = Path(model_path)

        # Load model data
        with open(self.model_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        idx = 0

        # Read mean shape header
        n_dims = int(lines[idx])  # 204 = 68 landmarks × 3 (x, y, z)
        idx += 1
        n_means = int(lines[idx])  # Usually 1
        idx += 1
        n_modes_header = int(lines[idx])  # Header value (may differ from actual)
        idx += 1

        # Read mean shape (204 values)
        self.mean_shape = np.array([float(lines[idx + i]) for i in range(n_dims)], dtype=np.float32)
        idx += n_dims

        # Read eigenvectors header (skip comment line if present)
        n_dims_check = int(lines[idx])
        idx += 1
        n_values_per_dim = int(lines[idx])  # Number of modes
        idx += 1
        n_modes_header2 = int(lines[idx])  # Another header value
        idx += 1

        # Read eigenvectors (204 rows × n_modes columns)
        # Each row contains all mode values for one dimension
        eigenvectors = []
        for i in range(n_dims):
            row_values = [float(x) for x in lines[idx].split()]
            eigenvectors.append(row_values)
            idx += 1

        self.eigenvectors = np.array(eigenvectors, dtype=np.float32)  # Shape: (204, n_modes)
        self.n_modes = self.eigenvectors.shape[1]

        # Read eigenvalues (variances) header
        n_means_check = int(lines[idx])
        idx += 1
        n_eigenvalues = int(lines[idx])
        idx += 1
        n_modes_header3 = int(lines[idx])
        idx += 1

        # Read eigenvalues
        eigenvalue_values = [float(x) for x in lines[idx].split()]
        self.eigenvalues = np.array(eigenvalue_values, dtype=np.float32)  # Shape: (n_modes,)

        # Compute number of landmarks
        self.n_landmarks = n_dims // 3  # Should be 68

        print(f"Loaded PDM: {self.n_landmarks} landmarks, {self.n_modes} modes")
        print(f"  Mean shape: {self.mean_shape.shape}")
        print(f"  Eigenvectors: {self.eigenvectors.shape}")
        print(f"  Eigenvalues: {self.eigenvalues.shape}")

    def params_to_shape(self, params):
        """
        Convert PDM parameters to 3D landmark positions.

        Args:
            params: PDM parameters (n_modes,) controlling shape variation

        Returns:
            shape: 3D landmarks (n_landmarks, 3) as [x, y, z]
        """
        if params.shape[0] != self.n_modes:
            raise ValueError(f"Expected {self.n_modes} parameters, got {params.shape[0]}")

        # shape = mean + eigenvectors @ params
        shape_flat = self.mean_shape + self.eigenvectors @ params

        # Reshape from (204,) to (68, 3)
        shape = shape_flat.reshape(self.n_landmarks, 3)

        return shape

    def shape_to_params(self, shape):
        """
        Project 3D landmark shape to PDM parameter space.

        Args:
            shape: 3D landmarks (n_landmarks, 3) as [x, y, z]

        Returns:
            params: PDM parameters (n_modes,)
        """
        # Flatten shape to (204,)
        shape_flat = shape.reshape(-1)

        # Project: params = eigenvectors^T @ (shape - mean)
        params = self.eigenvectors.T @ (shape_flat - self.mean_shape)

        return params

    def clamp_params(self, params, n_std=3.0):
        """
        Clamp PDM parameters to plausible range based on training data variance.

        Args:
            params: PDM parameters (n_modes,)
            n_std: Number of standard deviations to allow (default: 3.0)

        Returns:
            clamped_params: Parameters clamped to [-n_std*sqrt(eigenvalue), +n_std*sqrt(eigenvalue)]
        """
        # Compute limits: n_std * sqrt(eigenvalue) for each mode
        limits = n_std * np.sqrt(self.eigenvalues)

        # Clamp each parameter
        clamped = np.clip(params, -limits, limits)

        return clamped

    def params_to_landmarks_2d(self, params, scale, translation):
        """
        Convert PDM parameters to 2D landmark positions (x, y only).

        Args:
            params: PDM parameters (n_modes,)
            scale: Scale factor for 3D -> 2D projection
            translation: Translation (tx, ty) for centering

        Returns:
            landmarks: 2D landmarks (n_landmarks, 2) as [x, y]
        """
        # Get 3D shape
        shape_3d = self.params_to_shape(params)

        # Project to 2D (orthographic projection: just use x, y)
        landmarks_2d = shape_3d[:, :2] * scale + translation

        return landmarks_2d

    def landmarks_to_params_2d(self, landmarks_2d, depth_prior=None):
        """
        Estimate PDM parameters from 2D landmarks.

        Args:
            landmarks_2d: 2D landmarks (n_landmarks, 2) as [x, y]
            depth_prior: Optional prior for Z coordinates (n_landmarks,)

        Returns:
            params: Estimated PDM parameters (n_modes,)
            scale: Estimated scale factor
            translation: Estimated translation (2,)
        """
        # Use mean shape's z coordinates if no depth prior
        if depth_prior is None:
            mean_shape_3d = self.mean_shape.reshape(self.n_landmarks, 3)
            depth_prior = mean_shape_3d[:, 2]

        # Construct 3D shape by combining 2D landmarks with depth prior
        # First, estimate scale and translation by aligning to mean shape
        mean_shape_2d = self.mean_shape.reshape(self.n_landmarks, 3)[:, :2]

        # Center both shapes
        landmarks_centered = landmarks_2d - np.mean(landmarks_2d, axis=0)
        mean_centered = mean_shape_2d - np.mean(mean_shape_2d, axis=0)

        # Estimate scale (ratio of norms)
        scale = np.linalg.norm(landmarks_centered) / (np.linalg.norm(mean_centered) + 1e-8)

        # Estimate translation
        translation = np.mean(landmarks_2d, axis=0) - scale * np.mean(mean_shape_2d, axis=0)

        # Reconstruct 3D shape
        landmarks_3d_xy = (landmarks_2d - translation) / scale
        shape_3d = np.column_stack([landmarks_3d_xy, depth_prior])

        # Project to parameter space
        params = self.shape_to_params(shape_3d)

        return params, scale, translation
