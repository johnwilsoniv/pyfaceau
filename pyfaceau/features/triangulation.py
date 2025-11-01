#!/usr/bin/env python3
"""
Parse OpenFace triangulation file for face masking
"""

import numpy as np

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

    def create_face_mask(self, landmarks: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Create binary mask for face region using triangulation

        Args:
            landmarks: (68, 2) array of facial landmark coordinates
            img_width: Mask width in pixels
            img_height: Mask height in pixels

        Returns:
            (height, width) binary mask (0=background, 255=face)
        """
        import cv2

        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Fill each triangle
        for tri in self.triangles:
            # Get the three vertices for this triangle
            pts = landmarks[tri].astype(np.int32)

            # Fill the triangle
            cv2.fillConvexPoly(mask, pts, 255)

        return mask
