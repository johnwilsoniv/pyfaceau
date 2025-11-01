"""
SVR Patch Expert implementation for CLNF landmark refinement

This module loads and uses SVR patch experts from OpenFace to refine landmark positions.
The implementation is based on OpenFace 2.2's SVR_patch_expert.cpp.
"""

import numpy as np
from typing import Dict, List, Optional


class SVRPatchExpert:
    """
    Single SVR patch expert for one landmark.

    Similar to OpenFace's SVR_patch_expert class but in Python.
    Each patch expert is a trained SVR model that evaluates local patches
    around a landmark position to determine optimal placement.
    """

    def __init__(self):
        self.type = 0  # 0 = raw pixels, 1 = gradient features
        self.confidence = 0.0
        self.scaling = 1.0  # Logistic regression slope
        self.bias = 0.0  # Logistic regression bias
        self.weights = None  # SVR weights matrix (similar to AU SVR!)

    def __repr__(self):
        patch_type = "raw" if self.type == 0 else "gradient"
        if self.weights is not None:
            return f"SVRPatchExpert(type={patch_type}, weights_shape={self.weights.shape}, confidence={self.confidence:.3f})"
        return f"SVRPatchExpert(type={patch_type}, not_loaded)"


class SVRPatchExpertLoader:
    """
    Loader for OpenFace SVR patch expert models.

    Parses the svr_patches_*.txt format and loads patch experts for specified landmarks.
    File format matches OpenFace 2.2's SVR patch expert format.
    """

    # OpenCV type constants (from OpenCV headers)
    CV_8UC1 = 0
    CV_32FC1 = 5
    CV_64FC1 = 6
    CV_32SC1 = 4

    def __init__(self, filepath: str):
        """
        Initialize loader with patch expert model file.

        Args:
            filepath: Path to svr_patches_*.txt file
        """
        self.filepath = filepath
        self.scale = None
        self.num_views = None
        self.view_centers = []
        self.visibility_indices = []

    def load(self, target_landmarks: Optional[List[int]] = None) -> Dict[int, SVRPatchExpert]:
        """
        Load patch experts from file.

        Args:
            target_landmarks: List of landmark indices to load (0-67).
                            If None, loads all 68 landmarks.
                            Example: [17, 18, 19, 20, 21, 22, 26, 48, 54]

        Returns:
            Dictionary mapping landmark index -> SVRPatchExpert
        """
        with open(self.filepath, 'r') as f:
            # Parse header
            self._parse_header(f)

            # Load patch experts for each landmark
            patch_experts = {}
            for landmark_idx in range(68):
                if target_landmarks is None or landmark_idx in target_landmarks:
                    expert = self._load_single_patch_expert(f, landmark_idx)
                    patch_experts[landmark_idx] = expert
                else:
                    # Skip this patch expert to save memory
                    self._skip_patch_expert(f)

            return patch_experts

    def _parse_header(self, f):
        """Parse file header (scale, views, visibility)"""
        # Skip comment line "# scaling factor of training"
        f.readline()

        # Read scaling factor
        self.scale = float(f.readline().strip())

        # Skip comment "# number of views"
        f.readline()

        # Read number of views
        self.num_views = int(f.readline().strip())

        # Skip comment "# centers of the views"
        f.readline()

        # Read view centers
        # Each view has 6 lines: 3 metadata lines + 3 rotation values
        for _ in range(self.num_views):
            f.readline()  # 3
            f.readline()  # 1
            f.readline()  # 6
            f.readline()  # rx value
            f.readline()  # ry value
            f.readline()  # rz value

        # Skip visibility comment "# visibility indices per view"
        f.readline()

        # Read visibility indices (one block per view)
        for _ in range(self.num_views):
            # Each visibility block has:
            # - num_landmarks line (68)
            # - format line (1)
            # - format line (4)
            # - 68 visibility values (all on separate lines)
            num_landmarks = int(f.readline().strip())

            # Skip format lines
            f.readline()  # "1"
            f.readline()  # "4"

            # Skip all visibility values
            for _ in range(num_landmarks):
                f.readline()

        # Skip comment "# Patches themselves (1 line patches of a vertex)"
        f.readline()

    def _load_single_patch_expert(self, f, landmark_idx: int) -> SVRPatchExpert:
        """
        Load a single patch expert from file.

        Implements Multi_SVR_patch_expert::Read() and SVR_patch_expert::Read() from OpenFace.
        All data is on a single line per view!

        We only load view 0 (frontal) and skip the other 6 views.
        """
        expert = SVRPatchExpert()

        # For each view (we'll just use the first/frontal view for now)
        for view_idx in range(self.num_views):
            # Read entire patch expert line
            line = f.readline().strip().split()

            # Only process view 0 (frontal)
            if view_idx != 0:
                continue

            idx = 0
            read_type = int(line[idx])
            idx += 1

            # Handle Multi_SVR_patch_expert format (read_type=3)
            if read_type == 3:
                # Read Multi_SVR header
                width = int(line[idx])
                idx += 1
                height = int(line[idx])
                idx += 1
                num_modalities_declared = int(line[idx])
                idx += 1

                # Note: num_modalities_declared may say 2, but we only have data for 1 modality on this line
                # We'll just read the one SVR expert that's present
                svr_read_type = int(line[idx])
                idx += 1

                if svr_read_type != 2:
                    raise ValueError(f"Expected SVR read_type=2, got {svr_read_type}")

                # Read SVR patch expert data
                expert.type = int(line[idx])
                idx += 1
                expert.confidence = float(line[idx])
                idx += 1
                expert.scaling = float(line[idx])
                idx += 1
                expert.bias = float(line[idx])
                idx += 1

                # Read matrix dimensions and type
                rows = int(line[idx])
                idx += 1
                cols = int(line[idx])
                idx += 1
                mat_type = int(line[idx])
                idx += 1

                # Determine dtype
                if mat_type == self.CV_32FC1:
                    dtype = np.float32
                elif mat_type == self.CV_64FC1:
                    dtype = np.float64
                else:
                    raise ValueError(f"Unsupported matrix type: {mat_type}")

                # Read matrix values
                total_values = rows * cols
                values = [dtype(line[idx + i]) for i in range(total_values)]
                expert.weights = np.array(values, dtype=dtype).reshape(rows, cols)
                # Transpose weights (OpenFace does this for Matlab compatibility)
                expert.weights = expert.weights.T

            elif read_type == 2:
                # Simple SVR_patch_expert format (not wrapped in Multi_SVR)
                # Read type, confidence, scaling, bias
                expert.type = int(line[idx])
                idx += 1
                expert.confidence = float(line[idx])
                idx += 1
                expert.scaling = float(line[idx])
                idx += 1
                expert.bias = float(line[idx])
                idx += 1

                # Read matrix dimensions and type
                rows = int(line[idx])
                idx += 1
                cols = int(line[idx])
                idx += 1
                mat_type = int(line[idx])
                idx += 1

                # Determine dtype
                if mat_type == self.CV_32FC1:
                    dtype = np.float32
                elif mat_type == self.CV_64FC1:
                    dtype = np.float64
                else:
                    raise ValueError(f"Unsupported matrix type: {mat_type}")

                # Read matrix values
                total_values = rows * cols
                values = [dtype(line[idx + i]) for i in range(total_values)]
                expert.weights = np.array(values, dtype=dtype).reshape(rows, cols)

                # Transpose weights (OpenFace does this for Matlab compatibility)
                expert.weights = expert.weights.T
            else:
                raise ValueError(f"Unsupported read_type: {read_type}")

        return expert

    def _skip_patch_expert(self, f):
        """Skip a patch expert without loading (to save memory)"""
        for view_idx in range(self.num_views):
            # Each view is a single line, just skip it
            f.readline()

    def _read_mat(self, f) -> np.ndarray:
        """
        Read matrix in OpenFace format.

        Format:
            rows cols type
            value1 value2 value3 ...

        This matches OpenFace's ReadMat() function.
        """
        # Read dimensions and type
        line = f.readline().strip().split()
        rows = int(line[0])
        cols = int(line[1])
        mat_type = int(line[2])

        # Determine numpy dtype from OpenCV type
        if mat_type == self.CV_32FC1:
            dtype = np.float32
        elif mat_type == self.CV_64FC1:
            dtype = np.float64
        elif mat_type == self.CV_32SC1:
            dtype = np.int32
        elif mat_type == self.CV_8UC1:
            dtype = np.uint8
        else:
            raise ValueError(f"Unsupported matrix type: {mat_type}")

        # Read all values (may span multiple lines)
        total_values = rows * cols
        values = []
        while len(values) < total_values:
            line = f.readline().strip()
            if line:
                values.extend([dtype(v) for v in line.split()])

        # Create matrix and reshape
        mat = np.array(values, dtype=dtype).reshape(rows, cols)
        return mat

    def _skip_mat(self, f):
        """Skip matrix without reading values"""
        # Read dimensions
        line = f.readline().strip().split()
        rows = int(line[0])
        cols = int(line[1])
        total_values = rows * cols

        # Skip all values
        values_read = 0
        while values_read < total_values:
            line = f.readline().strip()
            if line:
                values_read += len(line.split())


def test_loader():
    """Quick test of the patch expert loader"""
    import os

    weights_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')
    patch_expert_file = os.path.join(weights_dir, 'svr_patches_0.25_general.txt')

    if not os.path.exists(patch_expert_file):
        print(f"Patch expert file not found: {patch_expert_file}")
        return

    print("Testing SVR Patch Expert Loader...")
    print(f"Loading from: {patch_expert_file}")

    # Load only critical landmarks
    critical_landmarks = [17, 18, 19, 20, 21, 22, 26, 48, 54]

    loader = SVRPatchExpertLoader(patch_expert_file)
    experts = loader.load(target_landmarks=critical_landmarks)

    print(f"\nLoaded {len(experts)} patch experts")
    print(f"Scale: {loader.scale}")
    print(f"Num views: {loader.num_views}")

    print("\nPatch expert details:")
    for idx, expert in experts.items():
        print(f"  Landmark {idx:2d}: {expert}")

    # Test a specific expert
    if 48 in experts:
        expert = experts[48]
        print(f"\nDetailed info for landmark 48 (left lip corner):")
        print(f"  Type: {'raw pixels' if expert.type == 0 else 'gradient'}")
        print(f"  Confidence: {expert.confidence:.4f}")
        print(f"  Scaling: {expert.scaling:.4f}")
        print(f"  Bias: {expert.bias:.4f}")
        print(f"  Weights shape: {expert.weights.shape}")
        print(f"  Weights dtype: {expert.weights.dtype}")
        print(f"  Weights range: [{expert.weights.min():.4f}, {expert.weights.max():.4f}]")

    print("\nPatch expert loader test complete!")


if __name__ == '__main__':
    test_loader()
