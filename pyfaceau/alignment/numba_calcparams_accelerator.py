#!/usr/bin/env python3
"""
Numba JIT-Accelerated Functions for CalcParams

Provides 2-5x speedup for pose estimation by compiling hot loops to machine code.
Maintains 100% accuracy (identical math, just compiled).

Performance targets:
- CalcParams: 42.5ms → 14-21ms (2-3x speedup)
- Overall FPS: 17.99 → 30-40 FPS

Author: Optimized for Apple Silicon MacBook
Date: 2025-11-01
"""

import numpy as np
import numba


@numba.jit(nopython=True, cache=True, fastmath=False)
def euler_to_rotation_matrix_jit(rx, ry, rz):
    """
    JIT-compiled Euler to rotation matrix conversion

    Args:
        rx, ry, rz: Euler angles in radians

    Returns:
        3x3 rotation matrix (float32)
    """
    s1, s2, s3 = np.sin(rx), np.sin(ry), np.sin(rz)
    c1, c2, c3 = np.cos(rx), np.cos(ry), np.cos(rz)

    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = c2 * c3
    R[0, 1] = -c2 * s3
    R[0, 2] = s2
    R[1, 0] = c1 * s3 + c3 * s1 * s2
    R[1, 1] = c1 * c3 - s1 * s2 * s3
    R[1, 2] = -c2 * s1
    R[2, 0] = s1 * s3 - c1 * c3 * s2
    R[2, 1] = c3 * s1 + c1 * s2 * s3
    R[2, 2] = c1 * c2

    return R


@numba.jit(nopython=True, cache=True, fastmath=False)
def project_shape_to_2d_jit(shape_3d, R, s, tx, ty, n_vis):
    """
    JIT-compiled 2D projection of 3D shape

    Replaces the slow Python loop in calc_params (lines 521-525)

    Args:
        shape_3d: (3, n_vis) 3D coordinates [X, Y, Z] rows
        R: (3, 3) rotation matrix
        s: scale
        tx, ty: translation
        n_vis: number of visible landmarks

    Returns:
        (n_vis*2,) 2D projected landmarks [X0...Xn, Y0...Yn]
    """
    curr_shape_2d = np.empty(n_vis * 2, dtype=np.float32)

    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]

    for i in range(n_vis):
        X = shape_3d[0, i]
        Y = shape_3d[1, i]
        Z = shape_3d[2, i]

        curr_shape_2d[i] = s * (r11*X + r12*Y + r13*Z) + tx
        curr_shape_2d[i + n_vis] = s * (r21*X + r22*Y + r23*Z) + ty

    return curr_shape_2d


@numba.jit(nopython=True, cache=True, fastmath=False)
def compute_jacobian_jit(shape_3d, R, s, princ_comp_vis, n_vis, m):
    """
    JIT-compiled Jacobian computation - the main bottleneck!

    Replaces compute_jacobian() nested loops (lines 271-313)

    This function computes partial derivatives of 2D projection w.r.t.:
    - 6 global params: scale, rx, ry, rz, tx, ty
    - m local params: PCA coefficients (34)

    Args:
        shape_3d: (3, n_vis) 3D shape [X, Y, Z] rows
        R: (3, 3) rotation matrix
        s: scale
        princ_comp_vis: (n_vis*3, m) principal components for visible landmarks
        n_vis: number of visible landmarks
        m: number of PCA modes (34)

    Returns:
        J: (n_vis*2, 6+m) Jacobian matrix
    """
    J = np.zeros((n_vis * 2, 6 + m), dtype=np.float32)

    # Extract rotation matrix elements
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]  # Not used in 2D projection but needed for derivatives

    # Iterate over each landmark
    for i in range(n_vis):
        X = shape_3d[0, i]
        Y = shape_3d[1, i]
        Z = shape_3d[2, i]

        row_x = i
        row_y = i + n_vis

        # --- Global parameter derivatives ---

        # Scaling term (column 0)
        J[row_x, 0] = X * r11 + Y * r12 + Z * r13
        J[row_y, 0] = X * r21 + Y * r22 + Z * r23

        # Rotation terms (columns 1-3: rx, ry, rz)
        J[row_x, 1] = s * (Y * r13 - Z * r12)
        J[row_y, 1] = s * (Y * r23 - Z * r22)

        J[row_x, 2] = -s * (X * r13 - Z * r11)
        J[row_y, 2] = -s * (X * r23 - Z * r21)

        J[row_x, 3] = s * (X * r12 - Y * r11)
        J[row_y, 3] = s * (X * r22 - Y * r21)

        # Translation terms (columns 4-5: tx, ty)
        J[row_x, 4] = 1.0
        J[row_y, 4] = 0.0

        J[row_x, 5] = 0.0
        J[row_y, 5] = 1.0

        # --- Local parameter derivatives (columns 6 to 6+m-1) ---
        for j in range(m):
            # Get principal component for this mode at this landmark
            Vx = princ_comp_vis[i, j]
            Vy = princ_comp_vis[i + n_vis, j]
            Vz = princ_comp_vis[i + 2*n_vis, j]

            # Derivative of 2D projection w.r.t. local parameter j
            J[row_x, 6 + j] = s * (r11*Vx + r12*Vy + r13*Vz)
            J[row_y, 6 + j] = s * (r21*Vx + r22*Vy + r23*Vz)

    return J


@numba.jit(nopython=True, cache=True, fastmath=False)
def apply_jacobian_weighting_jit(J, weight_diag, n_vis):
    """
    JIT-compiled Jacobian weighting application

    Args:
        J: (n_vis*2, 6+m) Jacobian matrix
        weight_diag: (n_vis*2,) diagonal of weight matrix
        n_vis: number of visible landmarks

    Returns:
        J_weighted: (n_vis*2, 6+m) weighted Jacobian
    """
    J_weighted = J.copy()

    for i in range(n_vis):
        w_x = weight_diag[i]
        w_y = weight_diag[i + n_vis]

        # Weight all columns for this landmark's rows
        for j in range(J.shape[1]):
            J_weighted[i, j] *= w_x
            J_weighted[i + n_vis, j] *= w_y

    return J_weighted


def compute_jacobian_accelerated(params_local, params_global, princ_comp_vis, mean_shape_vis, weight_matrix, n_vis, m):
    """
    Python wrapper for JIT-compiled Jacobian computation

    This function provides the same interface as CalcParams.compute_jacobian()
    but uses Numba JIT for 2-5x speedup.

    Args:
        params_local: (34,) local parameters
        params_global: (6,) global parameters [scale, rx, ry, rz, tx, ty]
        princ_comp_vis: (n_vis*3, 34) principal components for visible landmarks
        mean_shape_vis: (n_vis*3, 1) mean shape for visible landmarks
        weight_matrix: (n_vis*2, n_vis*2) diagonal weight matrix
        n_vis: number of visible landmarks (68 or less)
        m: number of PCA modes (34)

    Returns:
        J: (n_vis*2, 6+m) Jacobian matrix
        J_w_t: (6+m, n_vis*2) weighted Jacobian transpose
    """
    # Extract global parameters
    s = params_global[0]
    euler = params_global[1:4]

    # Get 3D shape from current local parameters
    params_local_col = params_local.reshape(-1, 1)
    shape_3d_flat = mean_shape_vis.flatten() + (princ_comp_vis @ params_local_col).flatten()
    shape_3d = shape_3d_flat.reshape(3, n_vis).astype(np.float32)

    # Get rotation matrix using JIT function
    R = euler_to_rotation_matrix_jit(euler[0], euler[1], euler[2])

    # Compute Jacobian using JIT function
    J = compute_jacobian_jit(shape_3d, R, s, princ_comp_vis, n_vis, m)

    # Apply weighting
    weight_diag = np.diag(weight_matrix).astype(np.float32)
    if not np.allclose(weight_diag, 1.0):
        J_weighted = apply_jacobian_weighting_jit(J, weight_diag, n_vis)
        J_w_t = J_weighted.T
    else:
        J_w_t = J.T

    return J, J_w_t


# Warmup JIT compilation on import (prevents first-call overhead)
def _warmup_jit():
    """Pre-compile JIT functions to avoid first-call overhead"""
    # Dummy data for compilation
    dummy_R = euler_to_rotation_matrix_jit(0.1, 0.2, 0.3)
    dummy_shape = np.random.randn(3, 68).astype(np.float32)
    dummy_proj = project_shape_to_2d_jit(dummy_shape, dummy_R, 1.0, 0.0, 0.0, 68)
    dummy_princ = np.random.randn(68*3, 34).astype(np.float32)
    dummy_J = compute_jacobian_jit(dummy_shape, dummy_R, 1.0, dummy_princ, 68, 34)
    dummy_weight = np.ones(68*2, dtype=np.float32)
    dummy_J_weighted = apply_jacobian_weighting_jit(dummy_J, dummy_weight, 68)


# Warmup on import
_warmup_jit()
print("Numba CalcParams accelerator loaded - targeting 2-5x speedup")
