#!/usr/bin/env python3
"""
Test PyMTCNN + PFLD vs C++ OpenFace landmarks

Compares:
- Python: PyMTCNN (bbox) → PFLD (68 landmarks)
- C++ Gold Standard: OpenFace FeatureExtraction → CSV landmarks

Metrics: Pixel error (Euclidean distance) per landmark
"""

import sys
import cv2
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from typing import Tuple, List, Dict

# Add pymtcnn to path
pymtcnn_path = Path(__file__).parent.parent / "pymtcnn"
sys.path.insert(0, str(pymtcnn_path))
sys.path.insert(0, str(Path(__file__).parent))

# Import PyMTCNN from standalone package
from pymtcnn import MTCNN

# Load PFLD detector directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pfld",
    Path(__file__).parent / "pyfaceau" / "detectors" / "pfld.py"
)
pfld_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfld_module)
CunjianPFLDDetector = pfld_module.CunjianPFLDDetector


class LandmarkComparison:
    """Compare PyMTCNN+PFLD vs C++ OpenFace landmarks"""

    def __init__(self, video_path: str, num_frames: int = 10):
        """
        Initialize comparison test.

        Args:
            video_path: Path to test video
            num_frames: Number of frames to extract and test
        """
        self.video_path = Path(video_path)
        self.num_frames = num_frames
        self.output_dir = Path("pfld_vs_openface_test")
        self.output_dir.mkdir(exist_ok=True)

        # Paths for frame extraction and results
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.cpp_output_dir = self.output_dir / "cpp_output"
        self.cpp_output_dir.mkdir(exist_ok=True)

        # OpenFace executable path
        self.openface_exe = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")

        # Initialize Python detectors
        print("Initializing PyMTCNN + PFLD...")
        weights_dir = Path(__file__).parent / "weights"

        # PyMTCNN
        self.pymtcnn = MTCNN(backend='auto', verbose=True)
        print(f"  PyMTCNN: initialized with backend={self.pymtcnn.backend_name}")

        # PFLD
        pfld_model = weights_dir / "pfld_cunjian.onnx"
        if not pfld_model.exists():
            raise FileNotFoundError(f"PFLD model not found: {pfld_model}")
        self.pfld = CunjianPFLDDetector(str(pfld_model), use_coreml=True)
        print(f"  PFLD: {self.pfld}")

    def extract_frames(self) -> List[Path]:
        """Extract evenly spaced frames from video and rotate them properly"""
        print(f"\nExtracting {self.num_frames} frames from video...")

        cap = cv2.VideoCapture(str(self.video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame indices
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frame_paths = []
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # Rotate frame 90 degrees clockwise to correct orientation
                # (video is rotated 90 degrees counter-clockwise)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                frame_path = self.frames_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(frame_path)
                print(f"  Frame {i+1}/{self.num_frames}: index {frame_idx} -> {frame_path.name} (rotated)")

        cap.release()
        print(f"✓ Extracted {len(frame_paths)} frames (rotated 90° CW)")
        return frame_paths

    def run_python_pipeline(self, frame_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run PyMTCNN + PFLD on a frame.

        Returns:
            bbox: [x_min, y_min, x_max, y_max]
            landmarks: (68, 2) array of (x, y) coordinates
        """
        img = cv2.imread(str(frame_path))

        # PyMTCNN detection
        # MTCNN.detect() returns (bboxes, landmarks) where bboxes are [x, y, w, h]
        bboxes, landmarks_mtcnn = self.pymtcnn.detect(img)

        if len(bboxes) == 0:
            raise RuntimeError(f"No face detected in {frame_path}")

        # Use first/best detection
        # Convert from [x, y, w, h] to [x1, y1, x2, y2] for PFLD
        bbox_xywh = bboxes[0]
        bbox = np.array([
            bbox_xywh[0],
            bbox_xywh[1],
            bbox_xywh[0] + bbox_xywh[2],
            bbox_xywh[1] + bbox_xywh[3]
        ])

        # PFLD landmark detection
        landmarks, conf = self.pfld.detect_landmarks(img, bbox)

        return bbox, landmarks

    def run_cpp_openface(self, frame_path: Path) -> Path:
        """
        Run C++ OpenFace FeatureExtraction on a frame.

        Returns:
            Path to output CSV file with landmarks
        """
        csv_output = self.cpp_output_dir / f"{frame_path.stem}.csv"

        # Run OpenFace with timeout
        cmd = [
            str(self.openface_exe),
            "-f", str(frame_path),
            "-out_dir", str(self.cpp_output_dir),
            "-2Dfp",  # Output 2D landmarks
            "-pose",  # Output pose
            "-aus"    # Output AUs
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"  Warning: OpenFace returned {result.returncode}")
                print(f"  stderr: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print(f"  Warning: OpenFace timed out on {frame_path}")
            return None

        return csv_output

    def parse_cpp_landmarks(self, csv_path: Path) -> np.ndarray:
        """
        Parse C++ OpenFace CSV to extract 68 landmarks.

        Returns:
            landmarks: (68, 2) array of (x, y) coordinates
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"C++ CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Extract x_0 to x_67 and y_0 to y_67
        landmarks = np.zeros((68, 2))
        for i in range(68):
            x_col = f'x_{i}'
            y_col = f'y_{i}'

            if x_col not in df.columns or y_col not in df.columns:
                raise ValueError(f"Landmark columns not found: {x_col}, {y_col}")

            landmarks[i, 0] = df[x_col].iloc[0]
            landmarks[i, 1] = df[y_col].iloc[0]

        return landmarks

    def compute_errors(self, python_lm: np.ndarray, cpp_lm: np.ndarray) -> Dict:
        """
        Compute pixel errors between Python and C++ landmarks.

        Returns:
            Dictionary with error metrics
        """
        # Euclidean distance per landmark
        errors = np.linalg.norm(python_lm - cpp_lm, axis=1)

        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'median_error': np.median(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'per_landmark_errors': errors
        }

    def run_test(self):
        """Run full comparison test"""
        print("\n" + "="*80)
        print("PFLD vs OpenFace Landmark Comparison")
        print("="*80)
        print(f"Video: {self.video_path.name}")
        print(f"Frames: {self.num_frames}")
        print()

        # Extract frames
        frame_paths = self.extract_frames()

        # Run comparison on each frame
        results = []

        for i, frame_path in enumerate(frame_paths):
            print(f"\nProcessing frame {i+1}/{len(frame_paths)}: {frame_path.name}")

            try:
                # Python pipeline
                print("  Running PyMTCNN + PFLD...")
                bbox, python_landmarks = self.run_python_pipeline(frame_path)
                print(f"    Bbox: {bbox}")
                print(f"    Landmarks: {python_landmarks.shape}")

                # C++ OpenFace
                print("  Running C++ OpenFace...")
                csv_path = self.run_cpp_openface(frame_path)

                if csv_path is None or not csv_path.exists():
                    print(f"    ✗ Skipping (OpenFace failed)")
                    continue

                cpp_landmarks = self.parse_cpp_landmarks(csv_path)
                print(f"    Landmarks: {cpp_landmarks.shape}")

                # Compute errors
                errors = self.compute_errors(python_landmarks, cpp_landmarks)

                print(f"  Pixel Error:")
                print(f"    Mean:   {errors['mean_error']:.2f} px")
                print(f"    Median: {errors['median_error']:.2f} px")
                print(f"    Max:    {errors['max_error']:.2f} px")
                print(f"    Std:    {errors['std_error']:.2f} px")

                results.append({
                    'frame': frame_path.name,
                    'frame_idx': i,
                    'bbox': bbox,
                    'python_landmarks': python_landmarks,
                    'cpp_landmarks': cpp_landmarks,
                    'errors': errors
                })

            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Summary statistics
        self.print_summary(results)

        return results

    def print_summary(self, results: List[Dict]):
        """Print summary statistics across all frames"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        if not results:
            print("No successful comparisons!")
            return

        # Aggregate errors
        all_mean_errors = [r['errors']['mean_error'] for r in results]
        all_max_errors = [r['errors']['max_error'] for r in results]

        # Collect all per-landmark errors
        all_per_landmark = np.array([r['errors']['per_landmark_errors'] for r in results])

        print(f"\nFrames processed: {len(results)}/{self.num_frames}")
        print(f"\nOverall Mean Error: {np.mean(all_mean_errors):.2f} ± {np.std(all_mean_errors):.2f} px")
        print(f"Overall Max Error:  {np.mean(all_max_errors):.2f} ± {np.std(all_max_errors):.2f} px")

        # Per-landmark statistics (averaged across frames)
        mean_per_landmark = np.mean(all_per_landmark, axis=0)

        print(f"\nWorst 10 landmarks (averaged across frames):")
        worst_indices = np.argsort(mean_per_landmark)[-10:][::-1]
        for idx in worst_indices:
            print(f"  Landmark {idx:2d}: {mean_per_landmark[idx]:.2f} px")

        print(f"\nBest 10 landmarks (averaged across frames):")
        best_indices = np.argsort(mean_per_landmark)[:10]
        for idx in best_indices:
            print(f"  Landmark {idx:2d}: {mean_per_landmark[idx]:.2f} px")

        print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare PyMTCNN+PFLD vs C++ OpenFace landmarks")
    parser.add_argument(
        "--video",
        type=str,
        default="/Users/johnwilsoniv/Documents/SplitFace Open3/Patient Data/Paralysis Cohort/20240731_181857000_iOS.MOV",
        help="Path to test video"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of frames to test"
    )

    args = parser.parse_args()

    # Run test
    comparison = LandmarkComparison(args.video, args.frames)
    results = comparison.run_test()
