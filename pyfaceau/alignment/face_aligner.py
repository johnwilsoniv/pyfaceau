#!/usr/bin/env python3
"""
OpenFace 2.2 Face Alignment - Pure Python Implementation

This module provides face alignment to a canonical 112×112 reference frame
using similarity transform (scale + rotation + translation) based on 68 facial landmarks.

Replicates the OpenFace 2.2 C++ alignment algorithm from:
- Face_utils.cpp::AlignFace() (lines 109-146)
- RotationHelpers.h::AlignShapesWithScale() (lines 195-242)
- RotationHelpers.h::AlignShapesKabsch2D() (lines 168-191)

Usage:
    aligner = OpenFace22FaceAligner("pdm_68_multi_pie.txt")
    aligned_face = aligner.align_face(image, landmarks_68, pose_tx, pose_ty)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
from pyfaceau.features.pdm import PDMParser


class OpenFace22FaceAligner:
    """
    Pure Python implementation of OpenFace 2.2 face alignment

    Aligns faces from 68 landmarks to 112×112 canonical reference frame
    using similarity transform (scale + rotation + translation).
    """

    # Rigid landmark indices (0-indexed) from OpenFace C++
    # These correspond to rigid facial structures: forehead, nose bridge, eye corners
    # Excludes soft tissue like lips and cheeks
    # NOTE: Includes 8 eye landmarks (36,39,40,41,42,45,46,47) which affect rotation
    # Testing shows removing eyes improves STABILITY but ruins MAGNITUDE (31° vs 5°)
    RIGID_INDICES = [1, 2, 3, 4, 12, 13, 14, 15, 27, 28, 29, 31, 32, 33, 34, 35, 36, 39, 40, 41, 42, 45, 46, 47]

    def __init__(self, pdm_file: str, sim_scale: float = 0.7, output_size: Tuple[int, int] = (112, 112), y_offset: float = 0.0):
        """
        Initialize face aligner with PDM reference shape

        Args:
            pdm_file: Path to PDM model file (e.g., "pdm_68_multi_pie.txt")
            sim_scale: Scaling factor for reference shape (default: 0.7 for AU analysis)
            output_size: Output aligned face size in pixels (default: 112×112)
            y_offset: Y-axis offset for centering (negative shifts face UP, default: 0.0)
                      Note: Non-zero values can disrupt HOG feature alignment with C++ models.
        """
        self.sim_scale = sim_scale
        self.output_width, self.output_height = output_size
        self.y_offset = y_offset

        # Load PDM and extract mean shape
        print(f"Loading PDM from: {pdm_file}")
        pdm = PDMParser(pdm_file)

        # Preprocess mean shape: 204 values (68 landmarks × 3D) → 68 landmarks × 2D
        # OpenFace C++ logic (Face_utils.cpp:112-119):
        # 1. Scale mean shape by sim_scale
        # 2. Extract X and Y coordinates (grouped format)
        # 3. Stack to (68, 2) format
        #
        # CRITICAL FIX: PDM stores as GROUPED format:
        #   [x0, x1, ..., x67, y0, y1, ..., y67, z0, z1, ..., z67]
        # NOT interleaved: [x0, y0, x1, y1, ...]
        # So we must: take first 68 as X, next 68 as Y, stack them
        mean_shape_scaled = pdm.mean_shape.flatten() * sim_scale  # (204,)
        x_coords = mean_shape_scaled[:68]    # First 68 = all X values
        y_coords = mean_shape_scaled[68:136] # Next 68 = all Y values
        self.reference_shape = np.column_stack([x_coords, y_coords])  # (68, 2)

        print(f"Face aligner initialized")
        print(f"  Sim scale: {sim_scale}")
        print(f"  Output size: {output_size}")
        print(f"  Reference shape: {self.reference_shape.shape}")
        print(f"  Rigid points: {len(self.RIGID_INDICES)}")

    def align_face(self, image: np.ndarray, landmarks_68: np.ndarray,
                   pose_tx: float, pose_ty: float, p_rz: float = 0.0,
                   apply_mask: bool = False, triangulation=None,
                   mask_style: str = 'detected') -> np.ndarray:
        """
        Align face to canonical 112×112 reference frame

        Args:
            image: Input BGR image (any size)
            landmarks_68: 68 facial landmarks as (68, 2) array or (136,) flat array
            pose_tx: Pose translation X (from OpenFace pose estimation)
            pose_ty: Pose translation Y (from OpenFace pose estimation)
            p_rz: Pose rotation Z in radians (from OpenFace params_global[3])
            apply_mask: If True, mask out regions outside the face (like OpenFace C++)
            triangulation: TriangulationParser object (required if apply_mask=True)
            mask_style: 'detected' uses warped detected landmarks (C++ OpenFace style),
                       'reference' uses reference shape (legacy behavior)

        Returns:
            aligned_face: 112×112 aligned face image (BGR format)
        """
        # Ensure landmarks are (68, 2) shape
        if landmarks_68.shape == (136,):
            landmarks_68 = landmarks_68.reshape(68, 2)
        elif landmarks_68.shape != (68, 2):
            raise ValueError(f"landmarks_68 must be (68, 2) or (136,), got {landmarks_68.shape}")

        # Extract rigid points from both source and destination
        source_rigid = self._extract_rigid_points(landmarks_68)
        dest_rigid = self._extract_rigid_points(self.reference_shape)

        # Match C++ exactly: use AlignShapesWithScale to compute BOTH scale and rotation
        # via Kabsch algorithm. This does NOT use p_rz - the rotation comes from
        # finding the optimal alignment between source and destination rigid points.
        # C++ code: Face_utils.cpp line 127:
        #   cv::Matx22f scale_rot_matrix = Utilities::AlignShapesWithScale(source_landmarks, destination_landmarks);
        scale_rot_matrix = self._align_shapes_with_scale(source_rigid, dest_rigid)

        # Build 2×3 affine warp matrix using pose translation
        warp_matrix = self._build_warp_matrix(scale_rot_matrix, pose_tx, pose_ty)

        # Apply affine transformation
        aligned_face = cv2.warpAffine(
            image,
            warp_matrix,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR
        )

        # Apply face mask if requested
        if apply_mask:
            if triangulation is None:
                raise ValueError("triangulation required when apply_mask=True")

            if mask_style == 'detected':
                # C++ OpenFace style: transform detected landmarks by warp matrix
                # This adapts the mask per-frame based on actual face shape
                # Reference: Face_utils.cpp::AlignFaceMask() lines 186-209
                warp_2d = scale_rot_matrix
                translation = np.array([warp_matrix[0, 2], warp_matrix[1, 2]])
                aligned_landmarks = landmarks_68 @ warp_2d.T + translation
            else:
                # Legacy style: use reference shape (consistent mask across frames)
                center = np.array([self.output_width / 2, self.output_height / 2])
                aligned_landmarks = self.reference_shape + center
                # Apply correction shift for reference shape centering
                aligned_landmarks[:, 0] += 5.0
                aligned_landmarks[:, 1] += 3.0

            # Adjust eyebrow landmarks upward to include forehead (like C++)
            # Indices 17-26 are eyebrows, 0 and 16 are jaw corners
            forehead_offset = (30 / 0.7) * self.sim_scale
            for idx in [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
                aligned_landmarks[idx, 1] -= forehead_offset

            # Create mask
            mask = triangulation.create_face_mask(
                aligned_landmarks,
                self.output_width,
                self.output_height
            )

            # Apply mask to each channel
            aligned_face = cv2.bitwise_and(aligned_face, aligned_face, mask=mask)

        return aligned_face

    def align_face_with_matrix(self, image: np.ndarray, landmarks_68: np.ndarray,
                                pose_tx: float, pose_ty: float, p_rz: float = 0.0,
                                apply_mask: bool = False, triangulation=None,
                                mask_style: str = 'detected') -> tuple:
        """
        Align face and return both the aligned image and the warp matrix.

        This is the same as align_face() but also returns the 2x3 affine transform
        matrix used for alignment, which can be used to transform landmarks from
        original frame coordinates to aligned face coordinates.

        Returns:
            tuple: (aligned_face, warp_matrix)
                - aligned_face: 112×112 aligned face image (BGR format)
                - warp_matrix: (2, 3) affine transform matrix
        """
        # Ensure landmarks are (68, 2) shape
        if landmarks_68.shape == (136,):
            landmarks_68 = landmarks_68.reshape(68, 2)
        elif landmarks_68.shape != (68, 2):
            raise ValueError(f"landmarks_68 must be (68, 2) or (136,), got {landmarks_68.shape}")

        # Extract rigid points from both source and destination
        source_rigid = self._extract_rigid_points(landmarks_68)
        dest_rigid = self._extract_rigid_points(self.reference_shape)

        # Compute scale-rotation matrix using Kabsch algorithm
        scale_rot_matrix = self._align_shapes_with_scale(source_rigid, dest_rigid)

        # Build 2×3 affine warp matrix using pose translation
        warp_matrix = self._build_warp_matrix(scale_rot_matrix, pose_tx, pose_ty)

        # Apply affine transformation
        aligned_face = cv2.warpAffine(
            image,
            warp_matrix,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LINEAR
        )

        # Apply face mask if requested
        if apply_mask:
            if triangulation is None:
                raise ValueError("triangulation required when apply_mask=True")

            if mask_style == 'detected':
                # C++ OpenFace style: transform detected landmarks by warp matrix
                warp_2d = scale_rot_matrix
                translation = np.array([warp_matrix[0, 2], warp_matrix[1, 2]])
                aligned_landmarks = landmarks_68 @ warp_2d.T + translation
            else:
                # Legacy style: use reference shape
                center = np.array([self.output_width / 2, self.output_height / 2])
                aligned_landmarks = self.reference_shape + center
                aligned_landmarks[:, 0] += 5.0
                aligned_landmarks[:, 1] += 3.0

            # Adjust eyebrow landmarks upward to include forehead
            forehead_offset = (30 / 0.7) * self.sim_scale
            for idx in [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
                aligned_landmarks[idx, 1] -= forehead_offset

            # Create mask
            mask = triangulation.create_face_mask(
                aligned_landmarks,
                self.output_width,
                self.output_height
            )

            # Apply mask to each channel
            aligned_face = cv2.bitwise_and(aligned_face, aligned_face, mask=mask)

        return aligned_face, warp_matrix

    def _transform_landmarks(self, landmarks: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
        """
        Transform landmarks using affine warp matrix

        Args:
            landmarks: (68, 2) landmark coordinates in original image
            warp_matrix: (2, 3) affine transformation matrix

        Returns:
            (68, 2) transformed landmark coordinates
        """
        # Convert to homogeneous coordinates
        ones = np.ones((landmarks.shape[0], 1))
        landmarks_hom = np.concatenate([landmarks, ones], axis=1)  # (68, 3)

        # Apply transform
        transformed = (warp_matrix @ landmarks_hom.T).T  # (68, 2)

        return transformed

    def _compute_scale_only(self, src: np.ndarray, dst: np.ndarray) -> float:
        """
        Compute scale factor between two point sets (no rotation)

        Args:
            src: (N, 2) source points
            dst: (N, 2) destination points

        Returns:
            Scale factor
        """
        n = src.shape[0]

        # Mean normalize
        src_centered = src - src.mean(axis=0)
        dst_centered = dst - dst.mean(axis=0)

        # Compute RMS scale
        s_src = np.sqrt(np.sum(src_centered ** 2) / n)
        s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)

        return s_dst / s_src

    def _align_shapes_with_scale(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Compute scale AND rotation using Kabsch algorithm (like C++ AlignShapesWithScale)

        This is the correct approach used by OpenFace C++:
        1. Mean-normalize both point sets
        2. Compute RMS scale for each
        3. Normalize to unit scale
        4. Use Kabsch/SVD to find optimal rotation
        5. Return scale * rotation matrix

        Args:
            src: (N, 2) source points (detected landmarks)
            dst: (N, 2) destination points (reference shape)

        Returns:
            (2, 2) scale-rotation matrix
        """
        n = src.shape[0]

        # 1. Mean normalize both
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        # 2. Compute RMS scale for each
        s_src = np.sqrt(np.sum(src_centered ** 2) / n)
        s_dst = np.sqrt(np.sum(dst_centered ** 2) / n)

        # 3. Normalize to unit scale
        src_normed = src_centered / s_src
        dst_normed = dst_centered / s_dst

        # 4. Kabsch algorithm (SVD) to find optimal rotation
        H = src_normed.T @ dst_normed
        U, S, Vt = np.linalg.svd(H)

        # Handle reflection (ensure proper rotation) - check BEFORE computing R
        d = np.linalg.det(Vt.T @ U.T)
        corr = np.eye(2)
        if d < 0:
            corr[1, 1] = -1

        R = Vt.T @ corr @ U.T

        # Note: NOT transposing R - testing if direct Kabsch matches C++

        # 5. Combine scale and rotation
        scale = s_dst / s_src
        scale_rot = scale * R

        return scale_rot.astype(np.float32)

    def _build_warp_matrix(self, scale_rot: np.ndarray, pose_tx: float, pose_ty: float) -> np.ndarray:
        """
        Build 2×3 affine warp matrix from 2×2 scale-rotation matrix and pose translation

        Implementation matches Face_utils.cpp::AlignFace (lines 127-143)

        Critical details:
        1. Copy scale-rotation matrix directly (NO transpose)
        2. Transform pose translation THROUGH scale-rotation matrix
        3. Simple centering with out_width/2, out_height/2 (NO empirical shifts)

        Args:
            scale_rot: (2, 2) similarity transform matrix (scale × rotation)
            pose_tx: Pose translation X (from params_global[4])
            pose_ty: Pose translation Y (from params_global[5])

        Returns:
            (2, 3) affine warp matrix for cv2.warpAffine
        """
        # Initialize 2×3 warp matrix
        warp_matrix = np.zeros((2, 3), dtype=np.float32)

        # Copy scale-rotation to first 2×2 block (NO transpose)
        warp_matrix[:2, :2] = scale_rot

        # Transform pose translation through scale-rotation
        # C++ code (line 138-139):
        #   cv::Vec2f T(tx, ty);
        #   T = scale_rot_matrix * T;
        T = np.array([pose_tx, pose_ty], dtype=np.float32)
        T_transformed = scale_rot @ T

        # Translation for centering in output image
        # C++ code (lines 142-143):
        #   warp_matrix(0,2) = -T(0) + out_width/2;
        #   warp_matrix(1,2) = -T(1) + out_height/2;
        # We add y_offset to shift the face up slightly (negative = up)
        # to account for small differences between Python CalcParams and C++ CLNF fitting

        warp_matrix[0, 2] = -T_transformed[0] + self.output_width / 2
        warp_matrix[1, 2] = -T_transformed[1] + self.output_height / 2 + self.y_offset

        return warp_matrix

    def _build_warp_matrix_centroid(self, scale_rot: np.ndarray, src_centroid: np.ndarray, dst_centroid: np.ndarray) -> np.ndarray:
        """
        Build 2×3 affine warp matrix using source and destination centroids

        This is the corrected version that uses rigid point centroids instead of
        pose translation parameters, which gives better alignment with C++ OpenFace.

        Args:
            scale_rot: (2, 2) similarity transform matrix (scale × rotation)
            src_centroid: (2,) centroid of source rigid points
            dst_centroid: (2,) centroid of destination rigid points

        Returns:
            (2, 3) affine warp matrix for cv2.warpAffine
        """
        # Initialize 2×3 warp matrix
        warp_matrix = np.zeros((2, 3), dtype=np.float32)

        # Copy scale-rotation to first 2×2 block
        warp_matrix[:2, :2] = scale_rot

        # Transform source centroid through scale-rotation
        T_src = scale_rot @ src_centroid

        # Translation: map src_centroid to dst_centroid, then center in output
        # dst_centroid is in PDM space (centered around 0), so add output_center
        warp_matrix[0, 2] = dst_centroid[0] - T_src[0] + self.output_width / 2
        warp_matrix[1, 2] = dst_centroid[1] - T_src[1] + self.output_height / 2

        return warp_matrix

    def _extract_rigid_points(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract 24 rigid points from 68 landmarks

        Rigid points correspond to stable facial structures (forehead, nose, eye corners)
        that don't deform much during facial expressions.

        Args:
            landmarks: (68, 2) array of facial landmarks

        Returns:
            (24, 2) array of rigid landmarks
        """
        return landmarks[self.RIGID_INDICES]

    def _align_shapes_kabsch_2d(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Compute 2D rotation matrix using Kabsch algorithm

        This assumes src and dst are already mean-normalized.

        Implementation based on RotationHelpers.h::AlignShapesKabsch2D (lines 168-191)

        Args:
            src: (N, 2) mean-normalized source points
            dst: (N, 2) mean-normalized destination points

        Returns:
            (2, 2) rotation matrix
        """
        # SVD decomposition: src^T × dst
        # OpenFace C++ uses: cv::SVD svd(align_from.t() * align_to)
        U, S, Vt = np.linalg.svd(src.T @ dst)

        # Check for reflection vs rotation
        # OpenFace C++ uses determinant check to prevent reflections
        d = np.linalg.det(Vt.T @ U.T)

        # Correction matrix to prevent reflection
        corr = np.eye(2)
        if d > 0:
            corr[1, 1] = 1
        else:
            corr[1, 1] = -1

        # Rotation matrix: R = V^T × corr × U^T
        # OpenFace C++ uses: R = svd.vt.t() * corr * svd.u.t()
        # But we need to transpose to match C++ behavior
        # Testing showed R.T gives correct rotation direction (+18° vs -18°)
        R = Vt.T @ corr @ U.T

        return R.T  # Transpose to match C++ rotation direction

    def _align_shapes_with_scale_and_rotation(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Compute similarity transform (scale + rotation via Kabsch) between two point sets

        This matches C++ AlignShapesWithScale in RotationHelpers.h lines 195-241.

        CRITICAL: p_rz is NOT used for alignment! C++ computes rotation from landmarks
        using Kabsch algorithm.

        Algorithm (matching C++):
        1. Mean-normalize both src and dst
        2. Compute RMS scale for each
        3. Normalize by scale
        4. Compute rotation via Kabsch2D
        5. Return: (s_dst / s_src) × R_kabsch

        Args:
            src: (N, 2) source points (detected landmarks)
            dst: (N, 2) destination points (reference shape)

        Returns:
            (2, 2) similarity transform matrix (scale × rotation)
        """
        n = src.shape[0]

        # 1. Mean normalize both point sets
        mean_src_x = np.mean(src[:, 0])
        mean_src_y = np.mean(src[:, 1])
        mean_dst_x = np.mean(dst[:, 0])
        mean_dst_y = np.mean(dst[:, 1])

        src_mean_normed = src.copy()
        src_mean_normed[:, 0] -= mean_src_x
        src_mean_normed[:, 1] -= mean_src_y

        dst_mean_normed = dst.copy()
        dst_mean_normed[:, 0] -= mean_dst_x
        dst_mean_normed[:, 1] -= mean_dst_y

        # 2. Compute RMS scale for each point set
        # C++ RotationHelpers.h line 221-222
        src_sq = src_mean_normed ** 2
        dst_sq = dst_mean_normed ** 2

        s_src = np.sqrt(np.sum(src_sq) / n)
        s_dst = np.sqrt(np.sum(dst_sq) / n)

        # 3. Normalize by scale (C++ line 224-225)
        src_norm = src_mean_normed / s_src
        dst_norm = dst_mean_normed / s_dst

        # 4. Get rotation via Kabsch2D (C++ line 230)
        R = self._align_shapes_kabsch_2d(src_norm, dst_norm)

        # 5. Return scale * rotation (C++ line 233)
        scale = s_dst / s_src
        return scale * R
