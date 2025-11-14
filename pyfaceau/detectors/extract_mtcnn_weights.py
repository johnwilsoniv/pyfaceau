"""
OpenFace MTCNN Weight Extractor

Extracts weights from OpenFace 2.2's custom binary format (.dat files) and converts
them to PyTorch state_dict format for use with PyTorch-based MTCNN implementation.

Binary format documentation:
- File starts with uint32 num_layers
- Each layer has uint32 type (0=conv, 1=maxpool, 2=fc, 3=prelu, 4=sigmoid)
- Followed by layer-specific data

Weight ordering is PyTorch-compatible:
- Conv: (out_channels, in_channels, kernel_h, kernel_w)
- FC: (out_features, in_features) - NOTE: transposed from storage
- PReLU: (num_channels,)
"""

import struct
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple


def read_matrix_bin(f) -> np.ndarray:
    """Read OpenCV matrix from binary format"""
    rows = struct.unpack('<I', f.read(4))[0]
    cols = struct.unpack('<I', f.read(4))[0]
    dtype = struct.unpack('<I', f.read(4))[0]

    # dtype 5 is CV_32F (float32)
    if dtype != 5:
        raise ValueError(f"Unsupported dtype: {dtype} (expected 5 for CV_32F)")

    elem_size = 4
    data = np.frombuffer(f.read(rows * cols * elem_size), dtype=np.float32)
    return data.reshape(rows, cols)


def extract_network_weights(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Extract weights from a single MTCNN network .dat file

    Returns:
        Dictionary mapping layer names to PyTorch tensors
    """
    weights = {}

    with open(filepath, 'rb') as f:
        num_layers = struct.unpack('<I', f.read(4))[0]

        conv_idx = 0
        prelu_idx = 0
        fc_idx = 0

        for layer_idx in range(num_layers):
            layer_type = struct.unpack('<I', f.read(4))[0]

            if layer_type == 0:  # Convolutional
                num_in_maps = struct.unpack('<I', f.read(4))[0]
                num_kernels = struct.unpack('<I', f.read(4))[0]

                # Read biases
                biases = np.array(struct.unpack(f'<{num_kernels}f', f.read(4 * num_kernels)))

                # Read kernels - stored as [in_map][kernel][rows][cols]
                # Need to convert to PyTorch format: [out_channels, in_channels, h, w]
                kernels = None
                for in_map in range(num_in_maps):
                    for kernel in range(num_kernels):
                        matrix = read_matrix_bin(f)
                        if kernels is None:
                            # Initialize with correct shape on first read
                            k_h, k_w = matrix.shape
                            kernels = np.zeros((num_kernels, num_in_maps, k_h, k_w), dtype=np.float32)
                        kernels[kernel, in_map] = matrix

                # Store as PyTorch tensors
                weights[f'conv{conv_idx + 1}.weight'] = torch.from_numpy(kernels)
                weights[f'conv{conv_idx + 1}.bias'] = torch.from_numpy(biases)
                conv_idx += 1

            elif layer_type == 1:  # Max pooling
                # Just read parameters, don't store (defined in model architecture)
                kernel_x = struct.unpack('<I', f.read(4))[0]
                kernel_y = struct.unpack('<I', f.read(4))[0]
                stride_x = struct.unpack('<I', f.read(4))[0]
                stride_y = struct.unpack('<I', f.read(4))[0]

            elif layer_type == 2:  # Fully connected
                # Read bias matrix
                bias = read_matrix_bin(f)

                # Read weight matrix - stored as (in_features, out_features)
                # PyTorch expects (out_features, in_features), so we transpose
                weight = read_matrix_bin(f)

                # Store as PyTorch tensors (copy to make writable)
                weights[f'fc{fc_idx + 1}.weight'] = torch.from_numpy(weight.T.copy())  # Transpose!
                weights[f'fc{fc_idx + 1}.bias'] = torch.from_numpy(bias.flatten().copy())
                fc_idx += 1

            elif layer_type == 3:  # PReLU
                prelu_weight = read_matrix_bin(f)
                weights[f'prelu{prelu_idx + 1}.weight'] = torch.from_numpy(prelu_weight.flatten())
                prelu_idx += 1

            elif layer_type == 4:  # Sigmoid
                # No parameters to read
                pass

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

    return weights


def extract_all_mtcnn_weights(model_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Extract weights from all three MTCNN networks

    Args:
        model_dir: Path to directory containing PNet.dat, RNet.dat, ONet.dat

    Returns:
        Tuple of (pnet_weights, rnet_weights, onet_weights)
    """
    model_path = Path(model_dir)

    print("Extracting PNet weights...")
    pnet_weights = extract_network_weights(model_path / "PNet.dat")
    print(f"  Found {len(pnet_weights)} parameter tensors")

    print("\nExtracting RNet weights...")
    rnet_weights = extract_network_weights(model_path / "RNet.dat")
    print(f"  Found {len(rnet_weights)} parameter tensors")

    print("\nExtracting ONet weights...")
    onet_weights = extract_network_weights(model_path / "ONet.dat")
    print(f"  Found {len(onet_weights)} parameter tensors")

    return pnet_weights, rnet_weights, onet_weights


def save_weights(pnet_weights: Dict, rnet_weights: Dict, onet_weights: Dict, output_path: str):
    """Save extracted weights as PyTorch state dict"""
    state_dict = {
        'pnet': pnet_weights,
        'rnet': rnet_weights,
        'onet': onet_weights
    }

    torch.save(state_dict, output_path)
    print(f"\nSaved weights to: {output_path}")


def print_weight_summary(weights: Dict, network_name: str):
    """Print summary of extracted weights"""
    print(f"\n{network_name} Weight Summary:")
    print("=" * 60)

    for name, tensor in weights.items():
        shape_str = str(tuple(tensor.shape))
        print(f"{name:20s} shape={shape_str:20s} dtype={tensor.dtype}")


if __name__ == "__main__":
    # Path to OpenFace MTCNN models
    model_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/LandmarkDetector/model/mtcnn_detector/"

    # Output path
    output_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/pyfaceau/detectors/openface_mtcnn_weights.pth"

    # Extract weights
    pnet_weights, rnet_weights, onet_weights = extract_all_mtcnn_weights(model_dir)

    # Print summaries
    print_weight_summary(pnet_weights, "PNet")
    print_weight_summary(rnet_weights, "RNet")
    print_weight_summary(onet_weights, "ONet")

    # Save
    save_weights(pnet_weights, rnet_weights, onet_weights, output_path)

    # Verify saved file
    print("\nVerifying saved file...")
    loaded = torch.load(output_path)
    print(f"Successfully loaded file with keys: {list(loaded.keys())}")
    print(f"PNet has {len(loaded['pnet'])} parameters")
    print(f"RNet has {len(loaded['rnet'])} parameters")
    print(f"ONet has {len(loaded['onet'])} parameters")
