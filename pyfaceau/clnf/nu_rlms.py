#!/usr/bin/env python3
"""
Non-Uniform Regularized Landmark Mean Shift (NU-RLMS) optimization.

This module implements the iterative landmark refinement algorithm used in
CLNF, which combines patch expert responses with shape model constraints.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


class NURLMSOptimizer:
    """
    Non-Uniform Regularized Landmark Mean Shift optimizer.

    This optimizer iteratively refines landmark positions by:
    1. Computing patch expert responses
    2. Finding peak responses using mean-shift with KDE
    3. Updating PDM parameters via regularized least squares
    4. Enforcing shape constraints

    The key insight is that we optimize in PDM parameter space rather than
    directly adjusting landmarks, which naturally enforces plausibility.
    """

    def __init__(self, pdm, patch_experts, max_iterations=10, convergence_threshold=0.01):
        """
        Initialize NU-RLMS optimizer.

        Args:
            pdm: PointDistributionModel for shape constraints
            patch_experts: CENPatchExperts for response computation
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Convergence threshold (average landmark movement in pixels)
        """
        self.pdm = pdm
        self.patch_experts = patch_experts
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(self, image, initial_landmarks, scale_idx=2, regularization=0.5):
        """
        Optimize landmark positions using NU-RLMS.

        Args:
            image: Grayscale image (H, W) as float32 [0, 255]
            initial_landmarks: Initial 2D landmark positions (68, 2)
            scale_idx: CEN patch expert scale index (0-3, default 2 for 0.50 scale)
            regularization: Regularization weight for shape constraints (default 0.5)

        Returns:
            refined_landmarks: Optimized 2D landmarks (68, 2)
            converged: Whether optimization converged
            num_iterations: Number of iterations performed
        """
        # Initialize with current landmarks
        landmarks = initial_landmarks.copy()

        # Estimate initial PDM parameters, scale, and translation from 2D landmarks
        params, scale, translation = self.pdm.landmarks_to_params_2d(landmarks)

        # Clamp parameters to plausible range
        params = self.pdm.clamp_params(params, n_std=3.0)

        # Optimization loop
        converged = False
        for iteration in range(self.max_iterations):
            # 1. Compute CEN response maps for current landmarks
            responses, extraction_bounds = self.patch_experts.response(image, landmarks, scale_idx)

            # 2. Find target positions using mean-shift on response maps
            target_landmarks = self._mean_shift_targets(landmarks, responses, extraction_bounds, scale_idx)

            # 3. Compute residual: difference between target and current positions
            residual = (target_landmarks - landmarks).flatten()

            # 4. Check convergence
            avg_movement = np.sqrt(np.mean(residual ** 2))
            if avg_movement < self.convergence_threshold:
                converged = True
                break

            # 5. Compute Jacobian of 2D landmarks w.r.t. PDM parameters
            jacobian = self._compute_jacobian(params, scale, translation)

            # 6. Solve regularized least squares for parameter update
            # minimize: ||J * delta_params - residual||^2 + lambda * ||delta_params||^2
            delta_params = self._solve_regularized_ls(jacobian, residual, regularization)

            # 7. Update parameters
            params = params + delta_params

            # 8. Clamp parameters to plausible range
            params = self.pdm.clamp_params(params, n_std=3.0)

            # 9. Convert parameters back to 2D landmarks
            landmarks = self.pdm.params_to_landmarks_2d(params, scale, translation)

        return landmarks, converged, iteration + 1

    def _mean_shift_targets(self, landmarks, responses, extraction_bounds, scale_idx):
        """
        Find target landmark positions using mean-shift on response maps.

        For each landmark, we perform mean-shift to find the peak of its
        response map, weighted by response values (kernel density estimation).

        Args:
            landmarks: Current 2D landmarks (68, 2)
            responses: List of 68 response maps from patch experts
            extraction_bounds: List of 68 tuples (x1, y1, x2, y2) with actual extraction bounds
            scale_idx: Scale index used for response computation

        Returns:
            target_landmarks: Target positions (68, 2) from mean-shift
        """
        target_landmarks = np.zeros_like(landmarks)

        experts_at_scale = self.patch_experts.patch_experts[scale_idx]

        for lm_idx in range(len(landmarks)):
            response = responses[lm_idx]
            expert = experts_at_scale[lm_idx]
            x1, y1, x2, y2 = extraction_bounds[lm_idx]

            if response.size == 0 or response.max() == 0:
                # No valid response, keep current position
                target_landmarks[lm_idx] = landmarks[lm_idx]
                continue

            # Apply Gaussian smoothing to response map for better peak finding
            response_smooth = gaussian_filter(response, sigma=1.0)

            # Normalize response to [0, 1] for stability
            response_norm = (response_smooth - response_smooth.min())
            response_max = response_norm.max()
            if response_max > 0:
                response_norm = response_norm / response_max
            else:
                # All zeros after smoothing
                target_landmarks[lm_idx] = landmarks[lm_idx]
                continue

            # Create coordinate grids for response map
            h, w = response.shape
            y_grid, x_grid = np.mgrid[0:h, 0:w]

            # Compute weighted mean position (mean-shift with uniform kernel)
            # Weight each pixel by its response value
            total_weight = np.sum(response_norm)
            if total_weight > 0:
                # Mean position in response map coordinates
                mean_x = np.sum(x_grid * response_norm) / total_weight
                mean_y = np.sum(y_grid * response_norm) / total_weight

                # Convert from response map coordinates to image coordinates
                # Response map (0, 0) corresponds to the ACTUAL extraction start (x1, y1)
                # This accounts for boundary clamping that may have occurred
                target_x = x1 + mean_x
                target_y = y1 + mean_y

                target_landmarks[lm_idx] = [target_x, target_y]
            else:
                # Fallback: keep current position
                target_landmarks[lm_idx] = landmarks[lm_idx]

        return target_landmarks

    def _compute_jacobian(self, params, scale, translation):
        """
        Compute Jacobian of 2D landmarks w.r.t. PDM parameters.

        The Jacobian J has shape (136, n_modes) where 136 = 68 landmarks × 2 (x, y).
        J[i, j] = d(landmark_i) / d(param_j)

        Args:
            params: Current PDM parameters (n_modes,)
            scale: Current scale factor
            translation: Current translation (2,)

        Returns:
            jacobian: Jacobian matrix (136, n_modes)
        """
        n_modes = self.pdm.n_modes
        n_landmarks = self.pdm.n_landmarks

        # Jacobian: d(landmarks_2d) / d(params)
        # landmarks_2d = (shape_3d[:, :2]) * scale + translation
        # shape_3d = mean_shape_3d + eigenvectors_3d @ params
        #
        # So: d(landmarks_2d) / d(params) = eigenvectors_2d * scale
        #
        # where eigenvectors_2d = eigenvectors_3d[:, :2].reshape(n_landmarks, 2, n_modes)

        # Extract 2D eigenvectors (x, y only) from 3D eigenvectors
        # eigenvectors is (204, n_modes) = (68*3, n_modes)
        # We want (68*2, n_modes) for x, y coordinates only

        jacobian = np.zeros((n_landmarks * 2, n_modes), dtype=np.float32)

        for mode_idx in range(n_modes):
            # Get eigenvector for this mode: (204,)
            eigenvector_3d = self.pdm.eigenvectors[:, mode_idx]

            # Reshape to (68, 3) and extract x, y columns
            eigenvector_3d_reshaped = eigenvector_3d.reshape(n_landmarks, 3)
            eigenvector_2d = eigenvector_3d_reshaped[:, :2]  # (68, 2)

            # Scale by the scale factor
            eigenvector_2d_scaled = eigenvector_2d * scale

            # Flatten to (136,) and store in jacobian
            jacobian[:, mode_idx] = eigenvector_2d_scaled.flatten()

        return jacobian

    def _solve_regularized_ls(self, jacobian, residual, regularization):
        """
        Solve regularized least squares for parameter update.

        minimize: ||J * delta_params - residual||^2 + lambda * ||delta_params||^2

        Solution: delta_params = (J^T J + lambda * I)^{-1} J^T residual

        Args:
            jacobian: Jacobian matrix (136, n_modes)
            residual: Residual vector (136,)
            regularization: Regularization weight lambda

        Returns:
            delta_params: Parameter update (n_modes,)
        """
        # Normal equation: (J^T J + lambda * I) delta_params = J^T residual
        JtJ = jacobian.T @ jacobian
        Jt_residual = jacobian.T @ residual

        # Add regularization to diagonal
        regularized_matrix = JtJ + regularization * np.eye(JtJ.shape[0])

        # Solve using Cholesky decomposition (faster for positive definite matrices)
        try:
            delta_params = np.linalg.solve(regularized_matrix, Jt_residual)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            delta_params = np.linalg.lstsq(regularized_matrix, Jt_residual, rcond=None)[0]

        return delta_params.astype(np.float32)
