#!/usr/bin/env python3
"""
OpenFace 2.2 PDM (Point Distribution Model) Parser

Parses text-based PDM files containing mean landmark positions and
principal components for facial shape reconstruction.

Format:
1. Mean values section: (204,) - average 3D landmark positions
2. Principal components section: (204, 34) - eigenvectors for reconstruction

Usage:
    pdm = PDMParser("In-the-wild_aligned_PDM_68.txt")
    reconstructed_landmarks = pdm.reconstruct_from_params(pdm_params)
"""

import numpy as np
from pathlib import Path
from typing import Tuple


class PDMParser:
    """Parser for OpenFace 2.2 PDM files"""

    def __init__(self, pdm_file_path: str):
        """
        Initialize parser and load PDM

        Args:
            pdm_file_path: Path to PDM .txt file
        """
        self.pdm_file_path = Path(pdm_file_path)
        if not self.pdm_file_path.exists():
            raise FileNotFoundError(f"PDM file not found: {pdm_file_path}")

        # Load PDM components
        self.mean_shape, self.princ_comp, self.eigen_values = self._parse_pdm()

        print(f"Loaded PDM from {self.pdm_file_path.name}")
        print(f"  Mean shape: {self.mean_shape.shape}")
        print(f"  Principal components: {self.princ_comp.shape}")
        print(f"  Eigenvalues: {self.eigen_values.shape}")

    def _parse_pdm(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse PDM file

        Returns:
            Tuple of (mean_shape, principal_components, eigen_values)
            - mean_shape: (204, 1) array of mean landmark positions
            - principal_components: (204, 34) matrix of eigenvectors
            - eigen_values: (34,) array of eigenvalues (variances)
        """
        with open(self.pdm_file_path, 'r') as f:
            lines = f.readlines()

        # Parse mean values section
        mean_shape = self._parse_matrix_section(lines, 0)

        # Find principal components section
        # Look for the comment line that starts the PC section
        pc_start = None
        for i, line in enumerate(lines):
            if 'principal components' in line.lower() and 'eigenvectors' in line.lower():
                pc_start = i
                break

        if pc_start is None:
            raise ValueError("Principal components section not found in PDM file")

        # Parse principal components section
        princ_comp = self._parse_matrix_section(lines, pc_start)

        # Find eigenvalues section
        # Look for the comment line that starts the eigenvalues section
        eigen_start = None
        for i, line in enumerate(lines):
            if 'eigenvalues' in line.lower() and 'variances' in line.lower():
                eigen_start = i
                break

        if eigen_start is None:
            raise ValueError("Eigenvalues section not found in PDM file")

        # Parse eigenvalues section
        eigen_values = self._parse_matrix_section(lines, eigen_start)
        eigen_values = eigen_values.flatten()  # Convert to 1D array

        return mean_shape, princ_comp, eigen_values

    def _parse_matrix_section(self, lines, start_idx):
        """
        Parse a matrix section starting at given line index

        Args:
            lines: All lines from file
            start_idx: Index of comment line before matrix

        Returns:
            numpy array with the matrix data
        """
        # Skip comment line
        idx = start_idx + 1

        # Read dimensions
        rows = int(lines[idx].strip())
        idx += 1

        cols = int(lines[idx].strip())
        idx += 1

        # Skip dtype line
        idx += 1

        # Read matrix data
        data = []
        values_read = 0
        total_values = rows * cols

        while values_read < total_values and idx < len(lines):
            line = lines[idx].strip()
            if line and not line.startswith('#'):
                # Split on whitespace and convert to float
                values = [float(x) for x in line.split()]
                data.extend(values)
                values_read += len(values)
            idx += 1

        if len(data) != total_values:
            raise ValueError(
                f"Expected {total_values} values, got {len(data)}"
            )

        # Reshape to matrix
        matrix = np.array(data, dtype=np.float64).reshape(rows, cols)

        return matrix

    def reconstruct_from_params(self, pdm_params: np.ndarray) -> np.ndarray:
        """
        Reconstruct 3D landmarks from PDM parameters

        This matches OpenFace 2.2's reconstruction:
        reconstructed_landmarks = princ_comp × pdm_params

        Args:
            pdm_params: (34,) array of PDM parameters from CSV (p_0...p_33)

        Returns:
            (204,) array of reconstructed 3D landmarks
            Format: [X_0...X_67, Y_0...Y_67, Z_0...Z_67]
        """
        if pdm_params.ndim == 2:
            pdm_params = pdm_params.flatten()

        if pdm_params.shape[0] != self.princ_comp.shape[1]:
            raise ValueError(
                f"Expected {self.princ_comp.shape[1]} PDM params, "
                f"got {pdm_params.shape[0]}"
            )

        # Reconstruct: princ_comp (204, 34) × params (34,) = (204,)
        reconstructed = np.dot(self.princ_comp, pdm_params)

        return reconstructed

    def extract_geometric_features(self, pdm_params: np.ndarray) -> np.ndarray:
        """
        Extract geometric features for AU prediction

        Matches OpenFace 2.2's geom_descriptor_frame construction:
        1. Reconstruct landmarks from PDM
        2. Concatenate: [reconstructed_landmarks, pdm_params]

        Args:
            pdm_params: (34,) array of PDM parameters

        Returns:
            Geometric feature vector matching OF2.2 format
        """
        # Reconstruct landmarks
        reconstructed_landmarks = self.reconstruct_from_params(pdm_params)

        # Concatenate: [landmarks, params]
        geom_features = np.concatenate([reconstructed_landmarks, pdm_params])

        return geom_features


def test_pdm_parser():
    """Test PDM parser"""
    import pandas as pd

    print("="*80)
    print("PDM Parser - Test")
    print("="*80)

    # Path to PDM file
    pdm_file = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt"

    # Load PDM
    print("\nLoading PDM...")
    pdm = PDMParser(pdm_file)

    # Load CSV to get PDM params
    csv_file = "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/of22_validation/IMG_0942_left_mirrored.csv"
    print(f"\nLoading CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    # Extract PDM params from first frame
    pdm_cols = [f'p_{i}' for i in range(34)]
    pdm_params = df.iloc[0][pdm_cols].values

    print(f"\nPDM params from frame 0: {pdm_params.shape}")
    print(f"  First 5 values: {pdm_params[:5]}")

    # Reconstruct landmarks
    print("\nReconstructing landmarks...")
    reconstructed = pdm.reconstruct_from_params(pdm_params)
    print(f"Reconstructed landmarks: {reconstructed.shape}")
    print(f"  First 5 values: {reconstructed[:5]}")

    # Check range
    print(f"\nReconstructed landmark statistics:")
    print(f"  Min: {reconstructed.min():.2f}")
    print(f"  Max: {reconstructed.max():.2f}")
    print(f"  Mean: {reconstructed.mean():.2f}")
    print(f"  Std: {reconstructed.std():.2f}")

    # Compare to raw landmarks
    X_cols = [f'X_{i}' for i in range(68)]
    Y_cols = [f'Y_{i}' for i in range(68)]
    Z_cols = [f'Z_{i}' for i in range(68)]

    raw_landmarks = np.concatenate([
        df.iloc[0][X_cols].values,
        df.iloc[0][Y_cols].values,
        df.iloc[0][Z_cols].values
    ])

    print(f"\nRaw landmark statistics (from CSV):")
    print(f"  Min: {raw_landmarks.min():.2f}")
    print(f"  Max: {raw_landmarks.max():.2f}")
    print(f"  Mean: {raw_landmarks.mean():.2f}")
    print(f"  Std: {raw_landmarks.std():.2f}")

    # Extract geometric features
    print("\nExtracting geometric features...")
    geom_features = pdm.extract_geometric_features(pdm_params)
    print(f"Geometric features: {geom_features.shape}")
    print(f"  Expected: (238,) = 204 (landmarks) + 34 (params)")

    # Check if reconstructed landmarks fit in histogram range [-60, 60]
    in_range = np.sum((reconstructed >= -60) & (reconstructed <= 60))
    total = reconstructed.size
    pct = 100.0 * in_range / total

    print(f"\nHistogram range check [-60, 60]:")
    print(f"  Values in range: {in_range}/{total} ({pct:.2f}%)")
    print(f"  Values below -60: {np.sum(reconstructed < -60)}")
    print(f"  Values above 60: {np.sum(reconstructed > 60)}")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)


if __name__ == "__main__":
    test_pdm_parser()
