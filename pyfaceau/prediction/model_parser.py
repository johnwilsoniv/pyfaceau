#!/usr/bin/env python3
"""
OpenFace 2.2 SVR Model Parser

Parses OpenFace 2.2's binary .dat files containing Linear SVR models for AU prediction.

Binary Format (per model):
1. means matrix: (rows, cols, dtype, data[rows*cols*8])
2. support_vectors matrix: (rows, cols, dtype, data[rows*cols*8])
3. bias: float64 (8 bytes)

Usage:
    parser = OF22ModelParser(models_dir)
    models = parser.load_all_models()
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import struct


class OF22ModelParser:
    """
    Parser for OpenFace 2.2 Linear SVR models stored in binary .dat format
    """

    def __init__(self, models_dir: str):
        """
        Initialize parser with path to OF2.2 models directory

        Args:
            models_dir: Path to AU_predictors directory (will use svr_combined by default)
        """
        self.models_dir = Path(models_dir)
        if not self.models_dir.exists():
            raise ValueError(f"Models directory not found: {models_dir}")

        # All 17 AUs that OF2.2 supports with intensity estimation (SVR)
        # AU28 is excluded - it only has occurrence detection (SVM), not intensity
        # Based on AU_all_best.txt configuration
        self.available_aus = [
            'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
            'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
            'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
        ]

        # Mapping of AU to recommended model type from AU_all_best.txt
        self.recommended_model_types = {
            'AU01_r': 'dynamic',
            'AU02_r': 'dynamic',
            'AU04_r': 'static',
            'AU05_r': 'dynamic',
            'AU06_r': 'static',
            'AU07_r': 'static',
            'AU09_r': 'dynamic',
            'AU10_r': 'static',
            'AU12_r': 'static',
            'AU14_r': 'static',
            'AU15_r': 'dynamic',
            'AU17_r': 'dynamic',
            'AU20_r': 'dynamic',
            'AU23_r': 'dynamic',
            'AU25_r': 'dynamic',
            'AU26_r': 'dynamic',
            'AU45_r': 'dynamic'
        }

    def parse_svr_model(self, dat_file_path: Path, is_dynamic: bool = True) -> Dict:
        """
        Parse a single SVR model from .dat file

        Binary format for DYNAMIC models:
        1. int32: marker (always 1)
        2. float64: cutoff for person-specific calibration
        3. means matrix: (rows, cols, dtype, data)
        4. support_vectors matrix: (rows, cols, dtype, data)
        5. bias: float64

        Binary format for STATIC models:
        1. means matrix: (rows, cols, dtype, data)
        2. support_vectors matrix: (rows, cols, dtype, data)
        3. bias: float64

        Args:
            dat_file_path: Path to .dat file
            is_dynamic: True for dynamic models (has cutoff), False for static models

        Returns:
            dict with keys: 'cutoff', 'means', 'support_vectors', 'bias'
        """
        if not dat_file_path.exists():
            raise FileNotFoundError(f"Model file not found: {dat_file_path}")

        with open(dat_file_path, 'rb') as f:
            # All models start with a marker (int32)
            # marker = 1: dynamic model (has cutoff)
            # marker = 0: static model (no cutoff)
            marker = struct.unpack('<i', f.read(4))[0]

            if marker == 1:
                # Dynamic model - read cutoff value
                cutoff = struct.unpack('<d', f.read(8))[0]  # float64
                if not is_dynamic:
                    print(f"Warning: File has dynamic marker but loaded as static")
            elif marker == 0:
                # Static model - no cutoff
                cutoff = 0.0
                if is_dynamic:
                    print(f"Warning: File has static marker but loaded as dynamic")
            else:
                raise ValueError(f"Invalid marker: expected 0 or 1, got {marker}")

            # Read means matrix header
            means_rows = struct.unpack('<i', f.read(4))[0]
            means_cols = struct.unpack('<i', f.read(4))[0]
            means_dtype = struct.unpack('<i', f.read(4))[0]  # OpenCV type code

            # Read means data (may be empty if cols=0)
            if means_rows > 0 and means_cols > 0:
                means_data = np.frombuffer(
                    f.read(means_rows * means_cols * 8),
                    dtype=np.float64
                )
                means = means_data.reshape((means_rows, means_cols))
            else:
                # Empty means matrix - no centering needed
                means = np.array([], dtype=np.float64).reshape(means_rows, means_cols)

            # Read support vectors matrix header
            sv_rows = struct.unpack('<i', f.read(4))[0]
            sv_cols = struct.unpack('<i', f.read(4))[0]
            sv_dtype = struct.unpack('<i', f.read(4))[0]

            # Read support vectors data
            sv_data = np.frombuffer(
                f.read(sv_rows * sv_cols * 8),
                dtype=np.float64
            )
            support_vectors = sv_data.reshape((sv_rows, sv_cols))

            # Read bias
            bias = struct.unpack('<d', f.read(8))[0]  # double = float64

        return {
            'cutoff': cutoff,
            'means': means,
            'support_vectors': support_vectors,
            'bias': bias,
            'file': str(dat_file_path)
        }

    def load_au_model(self, au_name: str, use_dynamic: bool = None, use_combined: bool = True) -> Dict:
        """
        Load a specific AU model

        Args:
            au_name: AU name (e.g., 'AU01_r', 'AU12_r')
            use_dynamic: If True, load dynamic model; if False, load static model;
                        if None, use recommended type from AU_all_best.txt
            use_combined: If True, prefer svr_combined models (default)

        Returns:
            Parsed model dict
        """
        # Convert AU name to file format (e.g., 'AU01_r' -> 'AU_1')
        au_num = au_name.replace('AU', '').replace('_r', '').lstrip('0')

        # Determine model type (dynamic vs static)
        if use_dynamic is None:
            # Use recommended type from configuration
            model_type_str = self.recommended_model_types.get(au_name, 'dynamic')
        else:
            model_type_str = 'dynamic' if use_dynamic else 'static'

        # Try different file naming patterns in svr_combined
        base_dir = self.models_dir if not use_combined else (self.models_dir / 'svr_combined' if (self.models_dir / 'svr_combined').exists() else self.models_dir)

        # Possible filename patterns (svr_combined uses inconsistent naming)
        possible_files = [
            f"AU_{au_num}_{model_type_str}_intensity_comb.dat",
            f"AU_{au_num}_{model_type_str}_intensity.dat",
        ]

        dat_path = None
        for filename in possible_files:
            candidate = base_dir / filename
            if candidate.exists():
                dat_path = candidate
                break

        if dat_path is None:
            raise FileNotFoundError(
                f"Model not found for {au_name}. Tried: {possible_files} in {base_dir}"
            )

        # Parse the model with correct format (dynamic vs static)
        is_dynamic = (model_type_str == 'dynamic')
        model = self.parse_svr_model(dat_path, is_dynamic=is_dynamic)
        model['au_name'] = au_name
        model['model_type'] = model_type_str
        model['filename'] = dat_path.name

        return model

    def load_all_models(self, use_recommended: bool = True, use_combined: bool = True, verbose: bool = False) -> Dict[str, Dict]:
        """
        Load all available AU models

        Args:
            use_recommended: If True, use recommended static/dynamic type for each AU
                            from AU_all_best.txt configuration
            use_combined: If True, load from svr_combined (higher quality, trained on
                         combined datasets)
            verbose: If True, print detailed loading information

        Returns:
            Dictionary mapping AU names to model dicts
        """
        models = {}
        failed_aus = []

        for au_name in self.available_aus:
            try:
                use_dynamic = None if use_recommended else True
                model = self.load_au_model(au_name, use_dynamic=use_dynamic, use_combined=use_combined)
                models[au_name] = model
                if verbose:
                    print(f"Loaded {au_name} ({model['model_type']}): cutoff={model['cutoff']:.2f}, "
                          f"means={model['means'].shape}, SV={model['support_vectors'].shape}, "
                          f"bias={model['bias']:.4f} [{model['filename']}]")
            except FileNotFoundError as e:
                failed_aus.append(au_name)
                if verbose:
                    print(f"✗ Failed to load {au_name}: {e}")

        if verbose:
            print(f"\nLoaded {len(models)}/{len(self.available_aus)} AU models")
            if failed_aus:
                print(f"Failed AUs: {failed_aus}")

        return models

    def predict_au(self, fhog_features: np.ndarray, model: Dict) -> float:
        """
        Predict AU intensity using Linear SVR

        Prediction formula (from OF2.2 SVR_static_lin_regressors.cpp:98):
        - preds = (fhog_descriptor - means) * support_vectors + bias

        Note: The 'cutoff' value in dynamic models is for person-specific
        calibration and is NOT used in the linear prediction itself.

        Args:
            fhog_features: FHOG feature vector (1D array)
            model: Parsed SVR model dict

        Returns:
            Predicted AU intensity (float, typically 0-5 range)
        """
        # Ensure features are 2D (1, n_features)
        if fhog_features.ndim == 1:
            fhog_features = fhog_features.reshape(1, -1)

        # Check dimensions against support vectors
        # SV shape is (n_features, 1), so check against dimension 0
        expected_dims = model['support_vectors'].shape[0]
        if fhog_features.shape[1] != expected_dims:
            raise ValueError(
                f"Feature dimension mismatch: got {fhog_features.shape[1]}, "
                f"expected {expected_dims}"
            )

        # Center features (if means matrix exists and is non-empty)
        if model['means'].size > 0:
            centered = fhog_features - model['means']
        else:
            centered = fhog_features

        # Apply support vectors (matrix multiplication)
        # centered: (1, 4702), support_vectors: (4702, 1) -> result: (1, 1)
        prediction = np.dot(centered, model['support_vectors']) + model['bias']

        return float(prediction[0, 0])


def main():
    """Test the model parser"""
    import sys

    models_dir = "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors"

    print("=" * 80)
    print("OpenFace 2.2 SVR Model Parser - Test (All 17 AUs)")
    print("=" * 80)
    print(f"\nModels directory: {models_dir}\n")

    # Test parser
    parser = OF22ModelParser(models_dir)

    # Load all models (using recommended static/dynamic types and svr_combined)
    print("\nLoading all AU models (using AU_all_best.txt configuration)...")
    print("-" * 80)
    models = parser.load_all_models(use_recommended=True, use_combined=True)

    # Summary
    print("\n" + "=" * 80)
    print("Model Loading Summary")
    print("=" * 80)
    print(f"Successfully loaded: {len(models)} AU models")
    print(f"AU names: {list(models.keys())}")

    # Test prediction with dummy features
    if models:
        print("\n" + "=" * 80)
        print("Testing Prediction with Dummy Features")
        print("=" * 80)

        # Get first model
        au_name = list(models.keys())[0]
        model = models[au_name]

        # Create dummy FHOG features (all zeros)
        n_features = model['means'].shape[1]
        dummy_features = np.zeros(n_features, dtype=np.float64)

        print(f"\nTesting {au_name}:")
        print(f"  Feature dimensions: {n_features}")
        print(f"  Dummy features: all zeros")

        prediction = parser.predict_au(dummy_features, model)
        print(f"  Prediction: {prediction:.4f}")
        print(f"  (Expected: close to bias={model['bias']:.4f} since features are zero)")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
