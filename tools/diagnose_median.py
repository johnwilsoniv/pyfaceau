#!/usr/bin/env python3
"""
Frame-by-frame diagnostic to identify where running median diverges
between Python implementation and expected C++ behavior
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

from pyfaceau.detectors.retinaface_detector import RetinaFaceDetector
from pyfaceau.detectors.pfld import PFLDLandmarkDetector
from pyfaceau.alignment.calc_params import CalcParams
from pyfaceau.alignment.face_aligner import FaceAligner
from pyfaceau.features.hog_extractor import extract_hog_features
from pyfaceau.features.geom_extractor import extract_geom_features
from pyfaceau.features.histogram_median_tracker import DualHistogramMedianTracker
from pyfaceau.prediction.au_predictor import AUPredictor


def load_cpp_reference(csv_path: str, max_frames: int = 100) -> pd.DataFrame:
    """Load C++ OpenFace reference outputs"""
    print(f"Loading C++ reference from: {csv_path}")
    df = pd.read_csv(csv_path)

    if max_frames:
        df = df.head(max_frames)

    print(f"  Loaded {len(df)} frames")
    return df


def extract_landmarks_from_cpp(cpp_row) -> np.ndarray:
    """Extract 68 landmarks from C++ CSV row"""
    x_cols = [f'x_{i}' for i in range(68)]
    y_cols = [f'y_{i}' for i in range(68)]

    landmarks = np.zeros((68, 2), dtype=np.float64)
    landmarks[:, 0] = cpp_row[x_cols].values
    landmarks[:, 1] = cpp_row[y_cols].values

    return landmarks


def diagnose_running_median(
    video_path: str,
    cpp_csv_path: str,
    max_frames: int = 100,
    output_dir: str = 'median_debug'
):
    """
    Process video frame-by-frame and log running median values
    to diagnose where divergence from C++ occurs
    """

    print("="*80)
    print("RUNNING MEDIAN DIAGNOSTIC")
    print("="*80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load C++ reference
    cpp_df = load_cpp_reference(cpp_csv_path, max_frames)

    # Initialize components
    print("\nInitializing components...")

    calc_params = CalcParams(pdm_file='weights/In-the-wild_aligned_PDM_68.txt')
    face_aligner = FaceAligner(pdm_file='weights/In-the-wild_aligned_PDM_68.txt')

    # Running median tracker (with C++ OpenFace parameters)
    median_tracker = DualHistogramMedianTracker(
        hog_dim=4464,
        geom_dim=238,
        hog_bins=1000,
        hog_min=-0.005,
        hog_max=1.0,
        geom_bins=10000,
        geom_min=-60.0,
        geom_max=60.0
    )

    # AU predictor
    au_predictor = AUPredictor('weights/AU_predictors')

    print("✓ Components initialized")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Storage for diagnostics
    diagnostics = []

    # Process frames
    print(f"\nProcessing {max_frames} frames...")
    print("-"*80)

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠ End of video at frame {frame_idx}")
            break

        # Get C++ CLNF landmarks (identical to reference)
        cpp_row = cpp_df.iloc[frame_idx]
        landmarks_68 = extract_landmarks_from_cpp(cpp_row)

        # CalcParams (global + local parameters)
        params_result = calc_params.process(landmarks_68)
        params_global = params_result['params_global']
        params_local = params_result['params_local']

        # Face alignment
        aligned_face = face_aligner.align_face(frame, landmarks_68)

        # Extract features
        hog_features = extract_hog_features(aligned_face, visualize=False)
        geom_features = extract_geom_features(params_global, params_local)

        # Update running median (first 3000 frames only during first pass)
        update_histogram = (frame_idx < 3000)
        median_tracker.update(hog_features, geom_features, update_histogram)

        # Get current medians
        hog_median = median_tracker.hog_tracker.current_median.copy()
        geom_median = median_tracker.geom_tracker.current_median.copy()

        # Apply running median correction (dynamic AUs only)
        hog_corrected = hog_features - hog_median
        geom_corrected = geom_features - geom_median

        # Predict AUs (both static and dynamic)
        static_aus = au_predictor.predict_frame_aus(
            hog_features, geom_features, use_running_median=False
        )
        dynamic_aus = au_predictor.predict_frame_aus(
            hog_features, geom_features, use_running_median=True,
            hog_median=hog_median, geom_median=geom_median
        )

        # Log diagnostics
        diag = {
            'frame': frame_idx,
            'update_histogram': update_histogram,
            'hog_median_mean': np.mean(hog_median),
            'hog_median_std': np.std(hog_median),
            'hog_median_min': np.min(hog_median),
            'hog_median_max': np.max(hog_median),
            'geom_median_mean': np.mean(geom_median),
            'geom_median_std': np.std(geom_median),
            'geom_median_min': np.min(geom_median),
            'geom_median_max': np.max(geom_median),
            'hog_corrected_mean': np.mean(hog_corrected),
            'hog_corrected_std': np.std(hog_corrected),
            'geom_corrected_mean': np.mean(geom_corrected),
            'geom_corrected_std': np.std(geom_corrected),
        }

        # Add AU predictions
        for au_name, au_value in static_aus.items():
            diag[f'static_{au_name}'] = au_value
        for au_name, au_value in dynamic_aus.items():
            diag[f'dynamic_{au_name}'] = au_value

        # Add C++ reference values
        diag['cpp_AU01_r'] = cpp_row['AU01_r']
        diag['cpp_AU02_r'] = cpp_row['AU02_r']
        diag['cpp_AU04_r'] = cpp_row['AU04_r']
        diag['cpp_AU05_r'] = cpp_row['AU05_r']

        diagnostics.append(diag)

        if (frame_idx + 1) % 10 == 0:
            print(f"  Frame {frame_idx+1}/{max_frames}: "
                  f"HOG median mean={diag['hog_median_mean']:.4f}, "
                  f"geom median mean={diag['geom_median_mean']:.4f}")

    cap.release()

    # Convert to DataFrame
    diag_df = pd.DataFrame(diagnostics)

    # Save diagnostics
    output_csv = output_dir / 'running_median_diagnostics.csv'
    diag_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved diagnostics to: {output_csv}")

    # Compute correlations with C++ reference
    print("\n" + "="*80)
    print("AU CORRELATIONS (Python vs C++ Reference)")
    print("="*80)

    print("\nStatic AUs (no running median):")
    for au_name in ['AU04_r', 'AU06_r', 'AU07_r', 'AU10_r', 'AU12_r', 'AU14_r']:
        if f'static_{au_name}' in diag_df.columns and f'cpp_{au_name}' in diag_df.columns:
            corr = diag_df[f'static_{au_name}'].corr(diag_df[f'cpp_{au_name}'])
            print(f"  {au_name}: r = {corr:.4f}")

    print("\nDynamic AUs (with running median):")
    for au_name in ['AU01_r', 'AU02_r', 'AU05_r']:
        if f'dynamic_{au_name}' in diag_df.columns and f'cpp_{au_name}' in diag_df.columns:
            corr = diag_df[f'dynamic_{au_name}'].corr(diag_df[f'cpp_{au_name}'])
            print(f"  {au_name}: r = {corr:.4f}")

    # Analyze median evolution
    print("\n" + "="*80)
    print("RUNNING MEDIAN EVOLUTION")
    print("="*80)

    print(f"\nFrame 0 (first frame):")
    print(f"  HOG median: mean={diag_df.loc[0, 'hog_median_mean']:.6f}, "
          f"std={diag_df.loc[0, 'hog_median_std']:.6f}")
    print(f"  Geom median: mean={diag_df.loc[0, 'geom_median_mean']:.6f}, "
          f"std={diag_df.loc[0, 'geom_median_std']:.6f}")

    print(f"\nFrame 50:")
    print(f"  HOG median: mean={diag_df.loc[50, 'hog_median_mean']:.6f}, "
          f"std={diag_df.loc[50, 'hog_median_std']:.6f}")
    print(f"  Geom median: mean={diag_df.loc[50, 'geom_median_mean']:.6f}, "
          f"std={diag_df.loc[50, 'geom_median_std']:.6f}")

    if len(diag_df) > 99:
        print(f"\nFrame 99:")
        print(f"  HOG median: mean={diag_df.loc[99, 'hog_median_mean']:.6f}, "
              f"std={diag_df.loc[99, 'hog_median_std']:.6f}")
        print(f"  Geom median: mean={diag_df.loc[99, 'geom_median_mean']:.6f}, "
              f"std={diag_df.loc[99, 'geom_median_std']:.6f}")

    # Check for HOG median clamping
    print("\n" + "="*80)
    print("HOG MEDIAN CLAMPING CHECK")
    print("="*80)

    # Count how many HOG median values are exactly 0.0 (clamped)
    frames_with_negative_before_clamp = (diag_df['hog_median_min'] < 0).sum()
    print(f"\nFrames where HOG median min < 0: {frames_with_negative_before_clamp}")
    print(f"(These should be clamped to 0 in our implementation)")

    return diag_df


if __name__ == '__main__':
    video_path = '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0434.MOV'
    cpp_csv_path = 'cpp_openface_reference.csv'

    diag_df = diagnose_running_median(
        video_path=video_path,
        cpp_csv_path=cpp_csv_path,
        max_frames=100
    )

    print("\n" + "="*80)
    print("✓ Diagnostic complete!")
    print("  Review 'median_debug/running_median_diagnostics.csv' for details")
    print("="*80)
