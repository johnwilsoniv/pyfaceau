#!/usr/bin/env python3
"""
PyMTCNN + PyFaceAU Integration Example

This example demonstrates how to use PyMTCNN for cross-platform face detection
combined with PyFaceAU for Action Unit extraction.

Benefits of PyMTCNN + PyFaceAU:
- Cross-platform: Works on Linux/Windows (CUDA) and macOS (CoreML)
- Faster on Apple Silicon: 34 FPS (PyMTCNN) vs ~20 FPS (RetinaFace)
- Even faster on NVIDIA GPUs: 50+ FPS with CUDA
- Simplified pipeline: No separate landmark detection step needed initially

Performance Comparison:
┌─────────────────────┬──────────────┬────────────────┐
│ Platform            │ RetinaFace   │ PyMTCNN        │
├─────────────────────┼──────────────┼────────────────┤
│ Apple Silicon (M3)  │ ~20 FPS      │ 34 FPS         │
│ NVIDIA GPU (CUDA)   │ ~20 FPS      │ 50+ FPS        │
│ CPU                 │ ~5 FPS       │ 5-10 FPS       │
└─────────────────────┴──────────────┴────────────────┘

Usage:
    python pymtcnn_integration_example.py --video input.mp4 --output results.csv
"""

import argparse
import cv2
import pandas as pd
from pathlib import Path
import sys
import time

# Check for PyMTCNN availability
try:
    from pyfaceau.detectors import PyMTCNNDetector, PYMTCNN_AVAILABLE
except ImportError:
    print("Error: pyfaceau not found. Please install it:")
    print("  cd /path/to/pyfaceau && pip install -e .")
    sys.exit(1)

if not PYMTCNN_AVAILABLE:
    print("Error: pymtcnn not installed. Install with:")
    print("  pip install pymtcnn[onnx-gpu]  # For CUDA support")
    print("  pip install pymtcnn[coreml]    # For Apple Silicon")
    print("  pip install pymtcnn[onnx]      # For CPU-only")
    sys.exit(1)

# Import PyFaceAU components
from pyfaceau.detectors import CunjianPFLDDetector
from pyfaceau.alignment import CalcParams, OpenFace22FaceAligner
from pyfaceau.features import PDMParser, TriangulationParser
from pyfaceau.prediction import OF22ModelParser
import pyfhog


class PyMTCNN_PyFaceAU_Pipeline:
    """
    Complete AU extraction pipeline using PyMTCNN for face detection

    Pipeline:
        PyMTCNN (face detection + 5-point landmarks)
            ↓
        PFLD (68-point landmark refinement)
            ↓
        CalcParams (3D pose estimation)
            ↓
        OpenFace22FaceAligner (face alignment)
            ↓
        PyFHOG (HOG feature extraction)
            ↓
        SVR Models (AU prediction)
    """

    def __init__(self,
                 pymtcnn_backend: str = 'auto',
                 pfld_model: str = None,
                 pdm_file: str = None,
                 au_models_dir: str = None,
                 triangulation_file: str = None,
                 verbose: bool = True):
        """
        Initialize PyMTCNN + PyFaceAU pipeline

        Args:
            pymtcnn_backend: PyMTCNN backend ('auto', 'cuda', 'coreml', 'cpu')
            pfld_model: Path to PFLD ONNX model for 68-point landmarks
            pdm_file: Path to PDM shape model
            au_models_dir: Directory containing AU SVR models
            triangulation_file: Path to triangulation file for masking
            verbose: Print progress messages
        """
        self.verbose = verbose

        if verbose:
            print("=" * 80)
            print("Initializing PyMTCNN + PyFaceAU Pipeline")
            print("=" * 80)

        # Initialize PyMTCNN for face detection
        if verbose:
            print("\n1. Initializing PyMTCNN face detector...")

        self.face_detector = PyMTCNNDetector(
            backend=pymtcnn_backend,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            verbose=verbose
        )

        backend_info = self.face_detector.get_backend_info()
        if verbose:
            print(f"   Active backend: {backend_info}")

        # Initialize PFLD for 68-point landmark refinement
        if pfld_model and verbose:
            print("\n2. Initializing PFLD landmark detector...")
            self.landmark_detector = CunjianPFLDDetector(pfld_model)
        elif pfld_model:
            self.landmark_detector = CunjianPFLDDetector(pfld_model)
        else:
            self.landmark_detector = None

        # Initialize CalcParams for 3D pose estimation
        if pdm_file and verbose:
            print("\n3. Initializing CalcParams for 3D pose estimation...")
            self.calc_params = CalcParams(pdm_file)
            self.pdm_parser = PDMParser(pdm_file)
        elif pdm_file:
            self.calc_params = CalcParams(pdm_file)
            self.pdm_parser = PDMParser(pdm_file)
        else:
            self.calc_params = None
            self.pdm_parser = None

        # Initialize face aligner
        if pdm_file and triangulation_file and verbose:
            print("\n4. Initializing OpenFace 2.2 face aligner...")
            self.face_aligner = OpenFace22FaceAligner(pdm_file, triangulation_file)
            self.triangulation = TriangulationParser(triangulation_file)
        elif pdm_file and triangulation_file:
            self.face_aligner = OpenFace22FaceAligner(pdm_file, triangulation_file)
            self.triangulation = TriangulationParser(triangulation_file)
        else:
            self.face_aligner = None
            self.triangulation = None

        # Initialize AU models
        if au_models_dir and verbose:
            print("\n5. Loading AU prediction models...")
            self.au_predictor = OF22ModelParser(au_models_dir)
            if verbose:
                print(f"   Loaded {len(self.au_predictor.models)} AU models")
        elif au_models_dir:
            self.au_predictor = OF22ModelParser(au_models_dir)
        else:
            self.au_predictor = None

        if verbose:
            print("\n" + "=" * 80)
            print("Pipeline initialized successfully!")
            print("=" * 80)

    def process_frame(self, frame):
        """
        Process a single frame

        Args:
            frame: BGR image array (H, W, 3)

        Returns:
            Dictionary with AU results or None if detection failed
        """
        # Step 1: Detect face with PyMTCNN
        dets, _ = self.face_detector.detect_faces(frame)

        if len(dets) == 0:
            if self.verbose:
                print("  No face detected!")
            return None

        # Get primary face
        det = dets[0]
        bbox = det[:4].astype(int)
        confidence = det[4]

        if confidence < 0.5:
            if self.verbose:
                print(f"  Low confidence: {confidence:.3f}")
            return None

        # Step 2: Get 68-point landmarks with PFLD
        if self.landmark_detector:
            landmarks, lm_conf = self.landmark_detector.detect_landmarks(frame, bbox)
            if landmarks is None:
                if self.verbose:
                    print("  Landmark detection failed!")
                return None
        else:
            if self.verbose:
                print("  Warning: No landmark detector available")
            return None

        # Step 3: 3D pose estimation
        if self.calc_params:
            h, w = frame.shape[:2]
            params_local, params_global, detected_landmarks = self.calc_params.estimate_pose(
                landmarks, w, h
            )

            # Extract pose parameters
            tx = params_global[4]
            ty = params_global[5]
            rz = params_global[3]  # Z-rotation for alignment
        else:
            if self.verbose:
                print("  Warning: No pose estimator available")
            return None

        # Step 4: Face alignment
        if self.face_aligner:
            aligned_face = self.face_aligner.align_face(frame, landmarks, tx, ty, rz)
        else:
            if self.verbose:
                print("  Warning: No face aligner available")
            return None

        # Step 5: Extract HOG features
        hog_features, hog_vis = pyfhog.extract_hog_features(
            aligned_face,
            cell_size=8,
            num_bins=9,
            visualize=False
        )

        # Step 6: Extract geometric features
        geom_features = self.pdm_parser.extract_geometric_features(params_local)

        # Step 7: Predict AUs
        if self.au_predictor:
            au_results = self.au_predictor.predict(hog_features, geom_features)
        else:
            au_results = {}

        return {
            'success': True,
            'bbox': bbox,
            'confidence': confidence,
            'landmarks': landmarks,
            'pose': {'tx': tx, 'ty': ty, 'rz': rz},
            **au_results
        }

    def process_video(self, video_path: str, output_csv: str = None):
        """
        Process entire video

        Args:
            video_path: Path to input video
            output_csv: Path to output CSV file (optional)

        Returns:
            List of frame results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\nProcessing video: {Path(video_path).name}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")

        results = []
        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            if result:
                result['frame'] = frame_idx
            else:
                result = {'frame': frame_idx, 'success': False}

            results.append(result)
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_idx}/{total_frames} frames ({fps_current:.2f} FPS)")

        cap.release()

        # Save to CSV if requested
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"\nResults saved to: {output_csv}")

        return results


def main():
    parser = argparse.ArgumentParser(description='PyMTCNN + PyFaceAU Integration Example')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='pymtcnn_au_results.csv', help='Output CSV path')
    parser.add_argument('--backend', type=str, default='auto', choices=['auto', 'cuda', 'coreml', 'cpu'],
                       help='PyMTCNN backend (default: auto)')
    parser.add_argument('--pfld-model', type=str, default='weights/pfld_cunjian.onnx',
                       help='PFLD model path')
    parser.add_argument('--pdm-file', type=str, default='weights/In-the-wild_aligned_PDM_68.txt',
                       help='PDM file path')
    parser.add_argument('--au-models-dir', type=str, default='weights/AU_predictors',
                       help='AU models directory')
    parser.add_argument('--triangulation-file', type=str, default='weights/tris_68_full.txt',
                       help='Triangulation file path')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PyMTCNN_PyFaceAU_Pipeline(
        pymtcnn_backend=args.backend,
        pfld_model=args.pfld_model,
        pdm_file=args.pdm_file,
        au_models_dir=args.au_models_dir,
        triangulation_file=args.triangulation_file,
        verbose=True
    )

    # Process video
    results = pipeline.process_video(args.video, args.output)

    # Print summary
    successful_frames = sum(1 for r in results if r.get('success', False))
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print(f"  Total frames: {len(results)}")
    print(f"  Successful: {successful_frames}")
    print(f"  Failed: {len(results) - successful_frames}")
    print("=" * 80)


if __name__ == "__main__":
    main()
