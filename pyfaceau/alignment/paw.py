"""
Piecewise Affine Warp (PAW) for face alignment.

Based on OpenFace implementation:
- lib/local/LandmarkDetector/src/PAW.cpp
- Active Appearance Models Revisited (Matthews & Baker, IJCV 2004)

This implementation matches the C++ PAW algorithm for pixel-perfect alignment.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class PAW:
    """
    Piecewise Affine Warp using triangulation.

    Warps faces by applying independent affine transforms to each triangle,
    allowing for complex non-affine deformations.
    """

    def __init__(self, destination_landmarks: np.ndarray, triangulation: np.ndarray,
                 min_x: Optional[float] = None, min_y: Optional[float] = None,
                 max_x: Optional[float] = None, max_y: Optional[float] = None):
        """
        Initialize PAW with destination shape and triangulation.

        Args:
            destination_landmarks: (2*N,) array with [x0...xN, y0...yN] format
            triangulation: (M, 3) array of triangle vertex indices
            min_x, min_y, max_x, max_y: Optional bounds for output image
        """
        self.destination_landmarks = destination_landmarks.copy()
        self.triangulation = triangulation.copy()

        num_points = len(destination_landmarks) // 2
        num_tris = len(triangulation)

        # Extract x and y coordinates
        xs = destination_landmarks[:num_points]
        ys = destination_landmarks[num_points:]

        # Pre-compute alpha and beta coefficients for each triangle
        self.alpha = np.zeros((num_tris, 3), dtype=np.float32)
        self.beta = np.zeros((num_tris, 3), dtype=np.float32)

        # Store triangle bounding boxes for optimization
        self.triangle_bounds = []

        for tri_idx in range(num_tris):
            j, k, l = triangulation[tri_idx]

            # Compute coefficients (from PAW.cpp lines 83-96)
            c1 = ys[l] - ys[j]
            c2 = xs[l] - xs[j]
            c4 = ys[k] - ys[j]
            c3 = xs[k] - xs[j]
            c5 = c3 * c1 - c2 * c4

            if abs(c5) < 1e-10:
                # Degenerate triangle, skip
                continue

            self.alpha[tri_idx, 0] = (ys[j] * c2 - xs[j] * c1) / c5
            self.alpha[tri_idx, 1] = c1 / c5
            self.alpha[tri_idx, 2] = -c2 / c5

            self.beta[tri_idx, 0] = (xs[j] * c4 - ys[j] * c3) / c5
            self.beta[tri_idx, 1] = -c4 / c5
            self.beta[tri_idx, 2] = c3 / c5

            # Store triangle vertices and bounding box for point-in-triangle tests
            tri_xs = [xs[j], xs[k], xs[l]]
            tri_ys = [ys[j], ys[k], ys[l]]
            self.triangle_bounds.append({
                'vertices': [(tri_xs[i], tri_ys[i]) for i in range(3)],
                'min_x': min(tri_xs),
                'max_x': max(tri_xs),
                'min_y': min(tri_ys),
                'max_y': max(tri_ys)
            })

        # Determine output image bounds
        if min_x is None:
            min_x = float(np.min(xs))
            min_y = float(np.min(ys))
            max_x = float(np.max(xs))
            max_y = float(np.max(ys))

        self.min_x = min_x
        self.min_y = min_y

        width = int(max_x - min_x + 1.5)
        height = int(max_y - min_y + 1.5)

        # Create pixel mask and triangle ID map
        self.pixel_mask = np.zeros((height, width), dtype=np.uint8)
        self.triangle_id = np.full((height, width), -1, dtype=np.int32)

        # Determine which triangle each pixel belongs to
        curr_tri = -1
        for y in range(height):
            for x in range(width):
                px = x + min_x
                py = y + min_y
                curr_tri = self._find_triangle(px, py, curr_tri)
                if curr_tri != -1:
                    self.triangle_id[y, x] = curr_tri
                    self.pixel_mask[y, x] = 1

        # Pre-allocate arrays
        self.coefficients = np.zeros((num_tris, 6), dtype=np.float32)
        self.map_x = np.zeros((height, width), dtype=np.float32)
        self.map_y = np.zeros((height, width), dtype=np.float32)

    def warp(self, image: np.ndarray, source_landmarks: np.ndarray) -> np.ndarray:
        """
        Warp image using source landmarks to destination landmarks.

        Args:
            image: Source image to warp
            source_landmarks: (2*N,) array with [x0...xN, y0...yN] format

        Returns:
            Warped image matching destination shape
        """
        # Compute warp coefficients from source landmarks
        self._calc_coeff(source_landmarks)

        # Compute pixel mapping (where to sample from)
        self._warp_region()

        # Apply warp using OpenCV remap with bilinear interpolation
        warped = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)

        return warped

    def _calc_coeff(self, source_landmarks: np.ndarray):
        """
        Calculate warping coefficients from source landmarks.
        Matches PAW::CalcCoeff() in PAW.cpp lines 338-370.
        """
        num_points = len(source_landmarks) // 2

        for tri_idx in range(len(self.triangulation)):
            i, j, k = self.triangulation[tri_idx]

            # Extract source coordinates for triangle vertices
            c1 = source_landmarks[i]
            c2 = source_landmarks[j] - c1
            c3 = source_landmarks[k] - c1
            c4 = source_landmarks[i + num_points]
            c5 = source_landmarks[j + num_points] - c4
            c6 = source_landmarks[k + num_points] - c4

            # Get precomputed alpha and beta
            alpha = self.alpha[tri_idx]
            beta = self.beta[tri_idx]

            # Compute 6 coefficients for affine transform
            self.coefficients[tri_idx, 0] = c1 + c2 * alpha[0] + c3 * beta[0]
            self.coefficients[tri_idx, 1] = c2 * alpha[1] + c3 * beta[1]
            self.coefficients[tri_idx, 2] = c2 * alpha[2] + c3 * beta[2]
            self.coefficients[tri_idx, 3] = c4 + c5 * alpha[0] + c6 * beta[0]
            self.coefficients[tri_idx, 4] = c5 * alpha[1] + c6 * beta[1]
            self.coefficients[tri_idx, 5] = c5 * alpha[2] + c6 * beta[2]

    def _warp_region(self):
        """
        Compute source pixel coordinates for each destination pixel.
        Matches PAW::WarpRegion() in PAW.cpp lines 374-436.
        """
        height, width = self.pixel_mask.shape

        for y in range(height):
            yi = float(y) + self.min_y

            for x in range(width):
                xi = float(x) + self.min_x

                if self.pixel_mask[y, x] == 0:
                    # Outside face region
                    self.map_x[y, x] = -1
                    self.map_y[y, x] = -1
                else:
                    # Get triangle for this pixel
                    tri_idx = self.triangle_id[y, x]
                    coeff = self.coefficients[tri_idx]

                    # Apply affine transform: x_src = coeff[0] + coeff[1]*xi + coeff[2]*yi
                    self.map_x[y, x] = coeff[0] + coeff[1] * xi + coeff[2] * yi
                    self.map_y[y, x] = coeff[3] + coeff[4] * xi + coeff[5] * yi

    @staticmethod
    def _same_side(x0: float, y0: float, x1: float, y1: float,
                    x2: float, y2: float, x3: float, y3: float) -> bool:
        """
        Check if point (x0,y0) is on same side of line (x2,y2)-(x3,y3) as point (x1,y1).
        Matches PAW::sameSide() in PAW.cpp lines 443-451.
        """
        x = (x3 - x2) * (y0 - y2) - (x0 - x2) * (y3 - y2)
        y = (x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2)
        return x * y >= 0

    @staticmethod
    def _point_in_triangle(x0: float, y0: float, x1: float, y1: float,
                           x2: float, y2: float, x3: float, y3: float) -> bool:
        """
        Check if point (x0,y0) is inside triangle (x1,y1)-(x2,y2)-(x3,y3).
        Matches PAW::pointInTriangle() in PAW.cpp lines 454-461.
        """
        same_1 = PAW._same_side(x0, y0, x1, y1, x2, y2, x3, y3)
        same_2 = PAW._same_side(x0, y0, x2, y2, x1, y1, x3, y3)
        same_3 = PAW._same_side(x0, y0, x3, y3, x1, y1, x2, y2)
        return same_1 and same_2 and same_3

    def _find_triangle(self, x: float, y: float, guess: int = -1) -> int:
        """
        Find which triangle contains point (x, y).
        Matches PAW::findTriangle() in PAW.cpp lines 465-515.

        Args:
            x, y: Point coordinates
            guess: Previous triangle index for optimization

        Returns:
            Triangle index or -1 if point is outside all triangles
        """
        # Try guess first for speed
        if guess != -1:
            bounds = self.triangle_bounds[guess]
            vertices = bounds['vertices']
            if self._point_in_triangle(x, y, vertices[0][0], vertices[0][1],
                                      vertices[1][0], vertices[1][1],
                                      vertices[2][0], vertices[2][1]):
                return guess

        # Search all triangles
        for tri_idx, bounds in enumerate(self.triangle_bounds):
            # Quick bounding box check
            if (x < bounds['min_x'] or x > bounds['max_x'] or
                y < bounds['min_y'] or y > bounds['max_y']):
                continue

            # Precise point-in-triangle test
            vertices = bounds['vertices']
            if self._point_in_triangle(x, y, vertices[0][0], vertices[0][1],
                                      vertices[1][0], vertices[1][1],
                                      vertices[2][0], vertices[2][1]):
                return tri_idx

        return -1


def load_triangulation(filepath: str) -> np.ndarray:
    """
    Load triangulation file in OpenFace format.

    Format:
        Line 1: Number of triangles
        Line 2: Number of columns (always 3)
        Lines 3+: Triangle vertex indices (3 per line)

    Args:
        filepath: Path to triangulation file (e.g., tris_68_full.txt)

    Returns:
        (M, 3) array of triangle vertex indices
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_tris = int(lines[0].strip())
    num_cols = int(lines[1].strip())

    assert num_cols == 3, f"Expected 3 columns, got {num_cols}"

    triangulation = np.zeros((num_tris, 3), dtype=np.int32)
    for i in range(num_tris):
        parts = lines[i + 2].strip().split()
        triangulation[i] = [int(parts[j]) for j in range(3)]

    return triangulation
