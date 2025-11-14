#!/usr/bin/env python3
"""
CEN (Convolutional Expert Network) patch expert loader and inference.

Loads and runs patch expert models from OpenFace 2.2's .dat format.
"""

import numpy as np
import cv2
from pathlib import Path
import struct

# Try to import numba for JIT compilation (optional)
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])


class CENPatchExpert:
    """
    Single CEN patch expert for one landmark at one scale.

    A patch expert is a small neural network that evaluates how likely
    a landmark is at each position in a local patch.
    """

    def __init__(self):
        self.width_support = 0
        self.height_support = 0
        self.weights = []  # List of weight matrices for each layer
        self.biases = []   # List of bias vectors for each layer
        self.activation_function = []  # Activation type for each layer
        self.confidence = 0.0
        self.is_empty = False

    @classmethod
    def from_stream(cls, stream):
        """
        Load a CEN patch expert from binary stream.

        Args:
            stream: Binary file stream positioned at patch expert data

        Returns:
            CENPatchExpert instance
        """
        expert = cls()

        # Read header
        read_type = struct.unpack('i', stream.read(4))[0]
        if read_type != 6:
            raise ValueError(f"Invalid CEN patch expert type: {read_type}, expected 6")

        # Read dimensions and layer count
        expert.width_support = struct.unpack('i', stream.read(4))[0]
        expert.height_support = struct.unpack('i', stream.read(4))[0]
        num_layers = struct.unpack('i', stream.read(4))[0]

        # Empty patch (landmark invisible at this orientation)
        if num_layers == 0:
            expert.confidence = struct.unpack('d', stream.read(8))[0]
            expert.is_empty = True
            return expert

        # Read layers
        for i in range(num_layers):
            # Activation function type
            neuron_type = struct.unpack('i', stream.read(4))[0]
            expert.activation_function.append(neuron_type)

            # Read bias matrix
            bias = read_mat_bin(stream)
            expert.biases.append(bias)

            # Read weight matrix
            weight = read_mat_bin(stream)
            expert.weights.append(weight)

        # Read confidence
        expert.confidence = struct.unpack('d', stream.read(8))[0]

        return expert

    def response(self, area_of_interest):
        """
        Compute patch expert response map for an image patch.

        Args:
            area_of_interest: Grayscale image patch (H, W) as float32

        Returns:
            response: Response map (response_height, response_width) as float32
        """
        if self.is_empty:
            # Return zero response for empty patches
            response_height = max(1, area_of_interest.shape[0] - self.height_support + 1)
            response_width = max(1, area_of_interest.shape[1] - self.width_support + 1)
            return np.zeros((response_height, response_width), dtype=np.float32)

        # Apply contrast normalization (row-wise)
        normalized = contrast_norm(area_of_interest)

        # Convert to column format with bias (im2col)
        input_col = im2col_bias(normalized, self.width_support, self.height_support)

        # Forward pass through neural network layers
        layer_output = input_col
        for i in range(len(self.weights)):
            # Linear: output = input * weight^T + bias
            layer_output = layer_output @ self.weights[i].T + self.biases[i]

            # Apply activation function
            if self.activation_function[i] == 0:
                # Sigmoid (clamp extreme values to prevent overflow)
                layer_output = np.clip(layer_output, -88, 88)
                layer_output = 1.0 / (1.0 + np.exp(-layer_output))
            elif self.activation_function[i] == 1:
                # Tanh
                layer_output = np.tanh(layer_output)
            elif self.activation_function[i] == 2:
                # ReLU
                layer_output = np.maximum(0, layer_output)
            # else: linear (no activation)

        # Reshape output to 2D response map
        response_height = area_of_interest.shape[0] - self.height_support + 1
        response_width = area_of_interest.shape[1] - self.width_support + 1

        # Output is (num_patches,) -> reshape to (height, width)
        # Patches are now in row-major order (natural numpy order)
        response = layer_output.reshape(response_height, response_width)

        return response.astype(np.float32)


class CENPatchExperts:
    """
    Collection of CEN patch experts for all landmarks and scales.

    Manages multi-scale patch experts from OpenFace 2.2 .dat files.
    """

    def __init__(self, model_dir):
        """
        Load CEN patch experts from model directory.

        Args:
            model_dir: Path to directory containing cen_patches_*.dat files
        """
        self.model_dir = Path(model_dir)
        self.patch_scaling = [0.25, 0.35, 0.50, 1.00]  # Scales available
        self.num_landmarks = 68

        # patch_experts[scale][landmark]
        self.patch_experts = []

        # Load all scale levels
        print(f"Loading CEN patch experts (410 MB, ~5-10 seconds)...")
        for idx, scale in enumerate(self.patch_scaling):
            scale_file = self.model_dir / f"cen_patches_{scale:.2f}_of.dat"
            if not scale_file.exists():
                # Try patch_experts subdirectory
                scale_file = self.model_dir / "patch_experts" / f"cen_patches_{scale:.2f}_of.dat"
            if not scale_file.exists():
                raise FileNotFoundError(f"CEN model not found: {scale_file}")

            print(f"  [{idx+1}/4] Loading scale {scale}...")
            experts_at_scale = self._load_scale(scale_file)
            self.patch_experts.append(experts_at_scale)
            print(f"      ✓ {len(experts_at_scale)} patch experts loaded")

    def _load_scale(self, dat_file):
        """
        Load all patch experts at one scale level.

        Args:
            dat_file: Path to .dat file

        Returns:
            List of CENPatchExpert instances (frontal view only)
        """
        with open(dat_file, 'rb') as f:
            # File structure (confirmed from OpenFace C++ source):
            # 1. Header: patch_scale (double) + num_views (int)
            # 2. View centers: num_views × (x, y, z) as doubles
            # 3. Visibility matrices: num_views × cv::Mat<int>
            # 4. Patch experts: num_views × num_landmarks × CEN_patch_expert

            # Read file header
            patch_scale = struct.unpack('d', f.read(8))[0]
            num_views = struct.unpack('i', f.read(4))[0]

            # Read view centers (3D orientation for each view)
            # Each view center is immediately followed by an empty matrix
            view_centers = []
            for _ in range(num_views):
                x = struct.unpack('d', f.read(8))[0]
                y = struct.unpack('d', f.read(8))[0]
                z = struct.unpack('d', f.read(8))[0]
                view_centers.append((x, y, z))

                # Read empty matrix immediately after this view center
                empty_mat = read_mat_bin(f)  # Should be 0×0 matrix

            # Read visibility matrices (one per view)
            # These are cv::Mat<int> storing which landmarks are visible in each view
            visibilities = []
            for _ in range(num_views):
                vis_mat = read_mat_bin(f)
                visibilities.append(vis_mat)

            # Read mirror metadata (for facial symmetry)
            mirror_inds = read_mat_bin(f)  # 1×68 matrix
            mirror_views = read_mat_bin(f)  # 1×7 matrix

            # Now read patch experts for all views
            all_experts = []
            for view_idx in range(num_views):
                experts_for_view = []
                for lm_idx in range(self.num_landmarks):
                    try:
                        expert = CENPatchExpert.from_stream(f)
                        experts_for_view.append(expert)
                    except Exception as e:
                        print(f"Error loading expert (view={view_idx}, landmark={lm_idx}): {e}")
                        raise
                all_experts.append(experts_for_view)

        # Return only frontal view (view 0) for now
        # TODO: Add multi-view support for profile faces
        return all_experts[0]

    def response(self, image, landmarks, scale_idx):
        """
        Compute patch expert responses for all landmarks at given scale.

        Args:
            image: Grayscale image (H, W) as float32 [0, 255]
            landmarks: Current landmark positions (68, 2) as float32
            scale_idx: Scale index (0-3)

        Returns:
            responses: List of 68 response maps (one per landmark)
            extraction_bounds: List of 68 tuples (x1, y1, x2, y2) with actual extraction bounds
        """
        if scale_idx < 0 or scale_idx >= len(self.patch_experts):
            raise ValueError(f"Invalid scale index: {scale_idx}")

        experts_at_scale = self.patch_experts[scale_idx]
        responses = []
        extraction_bounds = []

        # For each landmark, extract patch and compute response
        for lm_idx in range(self.num_landmarks):
            expert = experts_at_scale[lm_idx]

            # Extract a SEARCH AREA around the landmark
            # The search area should be larger than the support window to allow
            # the patch expert to evaluate multiple positions
            # Search radius: 2.0x support (OpenFace default for robust refinement)
            search_radius = int(max(expert.width_support, expert.height_support) * 2.0)

            lm_x, lm_y = landmarks[lm_idx]
            x1 = max(0, int(lm_x - search_radius))
            y1 = max(0, int(lm_y - search_radius))
            x2 = min(image.shape[1], int(lm_x + search_radius))
            y2 = min(image.shape[0], int(lm_y + search_radius))

            # Extract and compute response
            patch = image[y1:y2, x1:x2]
            if patch.size > 0 and patch.shape[0] > expert.height_support and patch.shape[1] > expert.width_support:
                response = expert.response(patch)
            else:
                response = np.zeros((1, 1), dtype=np.float32)

            responses.append(response)
            extraction_bounds.append((x1, y1, x2, y2))

        return responses, extraction_bounds


def read_mat_bin(stream):
    """
    Read OpenCV matrix from binary stream (OpenFace format).

    Args:
        stream: Binary file stream

    Returns:
        numpy array with matrix data
    """
    # Read dimensions and type
    rows = struct.unpack('i', stream.read(4))[0]
    cols = struct.unpack('i', stream.read(4))[0]
    cv_type = struct.unpack('i', stream.read(4))[0]

    # Handle empty matrices (0×0 or 0×N or N×0)
    if rows == 0 or cols == 0:
        return np.array([], dtype=np.float32).reshape(rows, cols) if rows >= 0 and cols >= 0 else np.array([])

    # Map OpenCV type to numpy dtype
    # OpenCV type codes: CV_8U=0, CV_8S=1, CV_16U=2, CV_16S=3,
    #                    CV_32S=4, CV_32F=5, CV_64F=6
    if cv_type == 0:  # CV_8U
        dtype = np.uint8
    elif cv_type == 1:  # CV_8S
        dtype = np.int8
    elif cv_type == 2:  # CV_16U
        dtype = np.uint16
    elif cv_type == 3:  # CV_16S
        dtype = np.int16
    elif cv_type == 4:  # CV_32S
        dtype = np.int32
    elif cv_type == 5:  # CV_32F
        dtype = np.float32
    elif cv_type == 6:  # CV_64F
        dtype = np.float64
    else:
        raise ValueError(f"Unsupported OpenCV matrix type: {cv_type} (rows={rows}, cols={cols})")

    # Read data
    size = rows * cols
    data = np.frombuffer(stream.read(size * np.dtype(dtype).itemsize), dtype=dtype)

    # Reshape to matrix (OpenCV uses row-major order like NumPy)
    matrix = data.reshape(rows, cols)

    # For weight/bias matrices, convert to float32; for visibility, keep as-is
    if cv_type in [5, 6]:  # Float types
        return matrix.astype(np.float32)
    else:  # Integer types
        return matrix


@njit(fastmath=True, cache=True)
def _contrast_norm_numba(input_patch, output):
    """Numba-optimized contrast normalization (5-10x faster)."""
    for y in range(input_patch.shape[0]):
        # Skip first column, compute mean of rest
        row_sum = 0.0
        cols = input_patch.shape[1] - 1
        for x in range(1, input_patch.shape[1]):
            row_sum += input_patch[y, x]
        mean = row_sum / cols

        # Compute standard deviation
        sum_sq = 0.0
        for x in range(1, input_patch.shape[1]):
            diff = input_patch[y, x] - mean
            sum_sq += diff * diff

        norm = np.sqrt(sum_sq)
        if norm < 1e-10:
            norm = 1.0

        # Normalize (skip first column)
        output[y, 0] = input_patch[y, 0]  # Keep first column
        for x in range(1, input_patch.shape[1]):
            output[y, x] = (input_patch[y, x] - mean) / norm

    return output


def contrast_norm(input_patch):
    """
    Apply row-wise contrast normalization.

    Uses Numba JIT if available for 5-10x speedup.

    Args:
        input_patch: Image patch (H, W) as float32

    Returns:
        normalized: Contrast-normalized patch
    """
    output = np.empty_like(input_patch, dtype=np.float32)

    if NUMBA_AVAILABLE:
        return _contrast_norm_numba(input_patch, output)
    else:
        # Fallback: NumPy vectorized version
        output = input_patch.copy()
        for y in range(input_patch.shape[0]):
            row = input_patch[y, 1:]  # Skip first column
            mean = np.mean(row)
            norm = np.std(row)
            if norm < 1e-10:
                norm = 1.0
            output[y, 1:] = (input_patch[y, 1:] - mean) / norm
        return output


def im2col_bias(input_patch, width, height):
    """
    Convert image to column format with bias for convolutional processing.

    Uses vectorized stride tricks for 10-20x speedup over loop-based version.

    Args:
        input_patch: Image patch (m, n) as float32
        width: Sliding window width
        height: Sliding window height

    Returns:
        output: Matrix (num_windows, width*height+1) with bias column
    """
    m, n = input_patch.shape
    y_blocks = m - height + 1
    x_blocks = n - width + 1
    num_windows = y_blocks * x_blocks

    # Use stride tricks to create view of all sliding windows (zero-copy!)
    from numpy.lib.stride_tricks import as_strided

    # Create 4D array: (y_blocks, x_blocks, height, width)
    windows = as_strided(
        input_patch,
        shape=(y_blocks, x_blocks, height, width),
        strides=(input_patch.strides[0], input_patch.strides[1],
                input_patch.strides[0], input_patch.strides[1])
    )

    # Reshape windows - keep natural row-major order
    # Original: windows shape is (y_blocks, x_blocks, height, width)
    # Flatten to (num_windows, width*height) in row-major order (natural numpy order)
    windows_flat = windows.reshape(num_windows, height * width)

    # Add bias column
    output = np.ones((num_windows, height * width + 1), dtype=np.float32)
    output[:, 1:] = windows_flat

    return output
