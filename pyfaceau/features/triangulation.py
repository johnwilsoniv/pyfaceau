#!/usr/bin/env python3
"""
Parse OpenFace triangulation file for face masking
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _point_in_triangle(px, py, v0x, v0y, v1x, v1y, v2x, v2y):
    """
    Check if point (px, py) is inside triangle using barycentric coordinates.
    Matches C++ PAW::findTriangle implementation.
    """
    denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y)
    if abs(denom) < 1e-10:
        return False

    a = ((v1y - v2y) * (px - v2x) + (v2x - v1x) * (py - v2y)) / denom
    b = ((v2y - v0y) * (px - v2x) + (v0x - v2x) * (py - v2y)) / denom
    c = 1.0 - a - b

    return a >= 0.0 and b >= 0.0 and c >= 0.0


@njit(parallel=True, cache=True)
def _create_mask_cpp_style(landmarks, triangles, width, height):
    """
    Create mask using point-in-triangle testing like C++ PAW.
    Uses Numba for performance.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    for y in prange(height):
        for x in range(width):
            for t in range(len(triangles)):
                tri = triangles[t]
                v0 = landmarks[tri[0]]
                v1 = landmarks[tri[1]]
                v2 = landmarks[tri[2]]

                if _point_in_triangle(x, y, v0[0], v0[1], v1[0], v1[1], v2[0], v2[1]):
                    mask[y, x] = 255
                    break

    return mask


class TriangulationParser:
    """Parser for OpenFace tris_68.txt triangulation data"""

    def __init__(self, tris_file: str):
        """
        Load triangulation from OpenFace format file

        Args:
            tris_file: Path to tris_68.txt file
        """
        with open(tris_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        # Parse file format:
        # Line 0: Total number of triangle lines (111)
        # Line 1: Number of triangulation sets (3)
        # Line 2: Dimension (4)
        # Line 3+: Triangle definitions (vertex indices)

        total_triangles = int(lines[0])

        # Read all triangles starting from line 3
        triangles = []
        for i in range(3, len(lines)):  # Start from line 3, skip header
            tri = list(map(int, lines[i].split()))
            if len(tri) == 3:  # Valid triangle
                triangles.append(tri)

        self.triangles = np.array(triangles, dtype=np.int32)

        print(f"Loaded {len(self.triangles)} triangles from {tris_file}")

    def create_face_mask(self, landmarks: np.ndarray, img_width: int, img_height: int,
                         use_cpp_style: bool = True) -> np.ndarray:
        """
        Create binary mask for face region using triangulation

        Args:
            landmarks: (68, 2) array of facial landmark coordinates
            img_width: Mask width in pixels
            img_height: Mask height in pixels
            use_cpp_style: If True, use point-in-triangle testing like C++ PAW.
                          If False, use cv2.fillConvexPoly (legacy behavior).

        Returns:
            (height, width) binary mask (0=background, 255=face)
        """
        if use_cpp_style:
            # Use C++ PAW-style point-in-triangle testing
            # This matches OpenFace exactly at the pixel level
            landmarks_float = landmarks.astype(np.float64)
            return _create_mask_cpp_style(landmarks_float, self.triangles, img_width, img_height)
        else:
            # Legacy fillPoly-based approach (may have edge differences)
            import cv2
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            for tri in self.triangles:
                pts = landmarks[tri].astype(np.int32)
                cv2.fillConvexPoly(mask, pts, 255)
            return mask
