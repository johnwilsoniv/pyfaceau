#!/usr/bin/env python3
"""
CLNF (Constrained Local Neural Fields) landmark detector.

This module integrates CEN patch experts, PDM shape model, and NU-RLMS
optimization to provide robust landmark detection that handles challenging
cases like surgical markings and severe facial paralysis.
"""

import numpy as np
from pathlib import Path
from .cen_patch_experts import CENPatchExperts
from .pdm import PointDistributionModel
from .nu_rlms import NURLMSOptimizer


class CLNFDetector:
    """
    CLNF landmark detector with shape-constrained optimization.

    This detector refines initial landmark estimates using:
    - CEN patch expert responses (likelihood of landmark at each position)
    - PDM shape constraints (plausible facial configurations)
    - NU-RLMS optimization (iterative refinement in parameter space)
    """

    def __init__(self, model_dir, max_iterations=10, convergence_threshold=0.01):
        """
        Initialize CLNF detector.

        Args:
            model_dir: Directory containing CLNF model files
            max_iterations: Maximum iterations for NU-RLMS optimization
            convergence_threshold: Convergence threshold in pixels
        """
        self.model_dir = Path(model_dir)

        # Load PDM
        # Try both direct path and pdms subdirectory
        pdm_path = self.model_dir / "In-the-wild_aligned_PDM_68.txt"
        if not pdm_path.exists():
            pdm_path = self.model_dir / "pdms" / "In-the-wild_aligned_PDM_68.txt"
        print(f"Loading PDM from {pdm_path}...")
        self.pdm = PointDistributionModel(pdm_path)

        # Load CEN patch experts
        print("Loading CEN patch experts...")
        self.patch_experts = CENPatchExperts(self.model_dir)

        # Create optimizer
        self.optimizer = NURLMSOptimizer(
            self.pdm,
            self.patch_experts,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )

        print(f"CLNF detector initialized with {max_iterations} max iterations")

    def refine_landmarks(self, image, initial_landmarks, scale_idx=2,
                        regularization=0.5, multi_scale=False):
        """
        Refine landmark positions using CLNF optimization.

        Args:
            image: BGR or grayscale image
            initial_landmarks: Initial 68-point landmarks (68, 2)
            scale_idx: CEN scale index (0-3, default 2 for 0.50 scale)
            regularization: Shape regularization weight (default 0.5)
            multi_scale: Whether to use multi-scale refinement (default False)

        Returns:
            refined_landmarks: Refined 68-point landmarks (68, 2)
            converged: Whether optimization converged
            num_iterations: Total number of iterations
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = image.astype(np.float32)

        if multi_scale:
            # Multi-scale optimization: coarse to fine
            # Start with coarsest scale (0 = 0.25), progress to finest (3 = 1.00)
            landmarks = initial_landmarks.copy()
            total_iterations = 0

            for scale in [0, 1, 2, 3]:
                # Adjust regularization: stronger at coarse scales
                scale_regularization = regularization * (2.0 ** (3 - scale))

                # Optimize at this scale
                landmarks, converged, iters = self.optimizer.optimize(
                    gray, landmarks, scale_idx=scale,
                    regularization=scale_regularization
                )
                total_iterations += iters

                if converged:
                    break

            return landmarks, converged, total_iterations
        else:
            # Single-scale optimization
            return self.optimizer.optimize(
                gray, initial_landmarks, scale_idx=scale_idx,
                regularization=regularization
            )

    def detect_with_refinement(self, image, initial_detector, bbox):
        """
        Detect landmarks using initial detector, then refine with CLNF.

        Args:
            image: BGR image
            initial_detector: Initial landmark detector (e.g., PFLD)
            bbox: Face bounding box [x_min, y_min, x_max, y_max]

        Returns:
            refined_landmarks: Refined 68-point landmarks (68, 2)
            initial_landmarks: Initial landmarks before refinement (68, 2)
            converged: Whether CLNF optimization converged
        """
        # Get initial landmarks from fast detector
        initial_landmarks, _ = initial_detector.detect_landmarks(image, bbox)

        # Refine with CLNF (using multi-scale for best quality)
        refined_landmarks, converged, _ = self.refine_landmarks(
            image, initial_landmarks, multi_scale=True
        )

        return refined_landmarks, initial_landmarks, converged
