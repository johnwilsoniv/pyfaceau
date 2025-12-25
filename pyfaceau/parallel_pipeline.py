#!/usr/bin/env python3
"""
Parallel AU Extraction Pipeline - High Performance

This module extends the base pipeline with multiprocessing support to achieve
30-50 FPS by processing multiple frames simultaneously.

Key Features:
- Process 4-8 frames in parallel (scales with CPU cores)
- Maintains frame ordering for sequential running median updates
- Compatible with existing pipeline components
- Target: 30-50 FPS (6-10x speedup over sequential processing)

Usage:
    from pyauface.parallel_pipeline import ParallelAUPipeline

    pipeline = ParallelAUPipeline(
        pfld_model='weights/pfld.onnx',
        pdm_file='weights/pdm.txt',
        au_models_dir='weights/AU_predictors',
        triangulation_file='weights/tris.txt',
        mtcnn_backend='auto',  # or 'cuda', 'coreml', 'cpu'
        num_workers=6  # Number of parallel workers
    )

    results = pipeline.process_video('input.mp4', 'output.csv')
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
from multiprocessing import Pool, Manager
import queue
import time

# Import pipeline components
from pyfaceau.pipeline import FullPythonAUPipeline
import pyfhog


class ParallelAUPipeline:
    """
    High-performance parallel AU extraction pipeline

    Processes multiple video frames simultaneously using multiprocessing
    to achieve 30-50 FPS throughput (6-10x faster than sequential).

    Architecture:
    1. Main process: Reads frames from video
    2. Worker pool: Processes frames in parallel (face detection, landmarks, alignment, features)
    3. Main process: Updates running median sequentially, predicts AUs
    """

    def __init__(
        self,
        pfld_model: str,
        pdm_file: str,
        au_models_dir: str,
        triangulation_file: str,
        mtcnn_backend: str = 'auto',
        num_workers: int = 6,
        batch_size: int = 30,
        use_calc_params: bool = True,
        track_faces: bool = True,
        use_batched_predictor: bool = True,
        verbose: bool = True
    ):
        """
        Initialize parallel pipeline

        Args:
            pfld_model: Path to PFLD ONNX model
            pdm_file: Path to PDM shape model
            au_models_dir: Directory containing AU SVR models
            triangulation_file: Path to triangulation file
            mtcnn_backend: PyMTCNN backend ('auto', 'cuda', 'coreml', 'cpu') (default: 'auto')
            num_workers: Number of parallel worker processes (default: 6)
            batch_size: Frames to process per batch (default: 30)
            use_calc_params: Use full CalcParams for pose estimation
            track_faces: Enable face tracking (skip detection on most frames)
            use_batched_predictor: Use optimized batched AU predictor (2-5x faster)
            verbose: Print progress messages
        """
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose

        # Store initialization parameters
        self.init_params = {
            'mtcnn_backend': mtcnn_backend,
            'pfld_model': pfld_model,
            'pdm_file': pdm_file,
            'au_models_dir': au_models_dir,
            'triangulation_file': triangulation_file,
            'use_calc_params': use_calc_params,
            'track_faces': track_faces,
            'use_batched_predictor': use_batched_predictor,
            'verbose': False  # Disable worker verbosity
        }

        # Initialize main process pipeline (for AU prediction)
        if self.verbose:
            print(f"Initializing parallel pipeline with {num_workers} workers...")
            print(f"Target FPS: {num_workers * 4.6:.1f} FPS (theoretical)")
            print("")

        self.main_pipeline = FullPythonAUPipeline(
            pfld_model=pfld_model,
            pdm_file=pdm_file,
            au_models_dir=au_models_dir,
            triangulation_file=triangulation_file,
            mtcnn_backend=mtcnn_backend,
            use_calc_params=use_calc_params,
            track_faces=track_faces,
            use_batched_predictor=use_batched_predictor,
            verbose=verbose
        )

        # Initialize components in main process
        self.main_pipeline._initialize_components()

    def process_video(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process video with parallel frame processing

        Args:
            video_path: Path to input video
            output_csv: Optional path to save results
            max_frames: Optional limit on frames to process

        Returns:
            DataFrame with AU predictions for all frames
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.verbose:
            print(f"Processing video: {video_path.name}")
            print("=" * 80)
            print("")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        if self.verbose:
            print(f"Video info:")
            print(f"  FPS: {fps:.2f}")
            print(f"  Total frames: {total_frames}")
            print(f"  Workers: {self.num_workers}")
            print(f"  Batch size: {self.batch_size}")
            print("")

        # Process video in batches
        results = []
        frame_idx = 0
        start_time = time.time()

        try:
            # Create worker pool
            with Pool(processes=self.num_workers, initializer=_init_worker, initargs=(self.init_params,)) as pool:
                while frame_idx < total_frames:
                    # Read batch of frames
                    batch_frames = []
                    batch_indices = []

                    for _ in range(self.batch_size):
                        if frame_idx >= total_frames:
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break

                        batch_frames.append(frame)
                        batch_indices.append(frame_idx)
                        frame_idx += 1

                    if not batch_frames:
                        break

                    # Process batch in parallel
                    batch_start = time.time()

                    # Map frames to workers
                    frame_data = [(idx, frame, fps) for idx, frame in zip(batch_indices, batch_frames)]
                    feature_results = pool.map(_process_frame_worker, frame_data)

                    batch_time = time.time() - batch_start

                    # Process results sequentially (running median + AU prediction)
                    for idx, frame_features in zip(batch_indices, feature_results):
                        if frame_features is None:
                            # Frame processing failed
                            results.append({
                                'frame': idx,
                                'timestamp': idx / fps,
                                'success': False
                            })
                            continue

                        # Extract features
                        hog_features = frame_features['hog_features']
                        geom_features = frame_features['geom_features']

                        # Update running median (must be sequential)
                        # C++ increments frames_tracking BEFORE check, update on even frames
                        update_histogram = (idx % 2 == 0)
                        self.main_pipeline.running_median.update(
                            hog_features,
                            geom_features,
                            update_histogram=update_histogram
                        )
                        running_median = self.main_pipeline.running_median.get_combined_median()

                        # Predict AUs
                        au_results = self.main_pipeline._predict_aus(
                            hog_features,
                            geom_features,
                            running_median
                        )

                        # Store result
                        result = {
                            'frame': idx,
                            'timestamp': idx / fps,
                            'success': True
                        }
                        result.update(au_results)
                        results.append(result)

                    # Progress update
                    if self.verbose:
                        elapsed = time.time() - start_time
                        current_fps = frame_idx / elapsed if elapsed > 0 else 0
                        batch_fps = len(batch_frames) / batch_time if batch_time > 0 else 0
                        eta = (total_frames - frame_idx) / current_fps if current_fps > 0 else 0

                        print(f"Progress: {frame_idx}/{total_frames} frames "
                              f"({frame_idx/total_frames*100:.1f}%) - "
                              f"Batch FPS: {batch_fps:.1f}, "
                              f"Overall FPS: {current_fps:.1f}, "
                              f"ETA: {eta:.1f}s")

        finally:
            cap.release()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Statistics
        total_time = time.time() - start_time
        total_processed = df['success'].sum()
        overall_fps = total_processed / total_time if total_time > 0 else 0

        if self.verbose:
            print("")
            print("=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)
            print(f"Total frames: {len(df)}")
            print(f"Successful: {total_processed}")
            print(f"Failed: {len(df) - total_processed}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Overall FPS: {overall_fps:.2f}")
            print(f"Speedup vs sequential (4.6 FPS): {overall_fps/4.6:.2f}x")
            print("")

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            if self.verbose:
                print(f"Results saved to: {output_csv}")
                print("")

        return df


# Global worker pipeline (initialized once per worker)
_worker_pipeline = None


def _init_worker(init_params):
    """Initialize worker process with its own pipeline instance"""
    global _worker_pipeline

    # Disable CoreML in workers (use CPU for multiprocessing)
    _worker_pipeline = FullPythonAUPipeline(
        retinaface_model=init_params['retinaface_model'],
        pfld_model=init_params['pfld_model'],
        pdm_file=init_params['pdm_file'],
        au_models_dir=init_params['au_models_dir'],
        triangulation_file=init_params['triangulation_file'],
        use_calc_params=init_params['use_calc_params'],
        use_coreml=False,  # CPU only for workers
        track_faces=False,  # Disable tracking in workers (each frame independent)
        use_batched_predictor=init_params.get('use_batched_predictor', True),
        verbose=False
    )

    # Initialize components
    _worker_pipeline._initialize_components()


def _process_frame_worker(frame_data):
    """
    Worker function to process a single frame

    Extracts features (detection, landmarks, alignment, HOG, geometric)
    but does NOT update running median or predict AUs (done in main process)

    Args:
        frame_data: Tuple of (frame_idx, frame, fps)

    Returns:
        Dictionary with extracted features or None if failed
    """
    global _worker_pipeline

    frame_idx, frame, fps = frame_data

    try:
        # Step 1: Detect face
        detections, _ = _worker_pipeline.face_detector.detect_faces(frame)
        if len(detections) == 0:
            return None

        det = detections[0]
        bbox = det[:4].astype(int)

        # Step 2: Detect landmarks
        landmarks_68, _ = _worker_pipeline.landmark_detector.detect_landmarks(frame, bbox)

        # Step 3: Estimate 3D pose
        if _worker_pipeline.use_calc_params and _worker_pipeline.calc_params:
            params_global, params_local = _worker_pipeline.calc_params.calc_params(
                landmarks_68.flatten()
            )
            scale = params_global[0]
            rx, ry, rz = params_global[1:4]
            tx, ty = params_global[4:6]
        else:
            # Simplified pose
            tx = (bbox[0] + bbox[2]) / 2
            ty = (bbox[1] + bbox[3]) / 2
            rz = 0.0
            params_local = np.zeros(34)

        # Step 4: Align face
        aligned_face = _worker_pipeline.face_aligner.align_face(
            image=frame,
            landmarks_68=landmarks_68,
            pose_tx=tx,
            pose_ty=ty,
            p_rz=rz,
            apply_mask=True,
            triangulation=_worker_pipeline.triangulation
        )

        # Step 5: Extract HOG features
        hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
        hog_features = hog_features.flatten().astype(np.float32)

        # Step 6: Extract geometric features
        geom_features = _worker_pipeline.pdm_parser.extract_geometric_features(params_local)
        geom_features = geom_features.astype(np.float32)

        return {
            'hog_features': hog_features,
            'geom_features': geom_features
        }

    except Exception as e:
        # Frame processing failed
        return None


def main():
    """Command-line interface for parallel AU pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel AU Extraction Pipeline (30-50 FPS)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output CSV file (default: <video>_aus.csv)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--workers', type=int, default=6, help='Number of parallel workers (default: 6)')
    parser.add_argument('--batch-size', type=int, default=30, help='Frames per batch (default: 30)')

    # Model paths
    parser.add_argument('--retinaface', default='weights/retinaface_mobilenet025_coreml.onnx')
    parser.add_argument('--pfld', default='weights/pfld_cunjian.onnx')
    parser.add_argument('--pdm', default='weights/In-the-wild_aligned_PDM_68.txt')
    parser.add_argument('--au-models', default='weights/AU_predictors')
    parser.add_argument('--triangulation', default='weights/tris_68_full.txt')

    args = parser.parse_args()

    # Set default output path
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_parallel_aus.csv")

    # Initialize parallel pipeline
    try:
        pipeline = ParallelAUPipeline(
            retinaface_model=args.retinaface,
            pfld_model=args.pfld,
            pdm_file=args.pdm,
            au_models_dir=args.au_models,
            triangulation_file=args.triangulation,
            num_workers=args.workers,
            batch_size=args.batch_size,
            verbose=True
        )
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return 1

    # Process video
    try:
        df = pipeline.process_video(
            video_path=args.video,
            output_csv=args.output,
            max_frames=args.max_frames
        )

        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)
        print(f"Processed {len(df)} frames")
        print(f"Results saved to: {args.output}")
        print("")

        return 0

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
