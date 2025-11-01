#!/usr/bin/env python3
"""
Detailed Performance Benchmark - PyfaceAU Pipeline

Processes 200 frames and provides comprehensive timing analysis for each component:
- Face Detection (RetinaFace)
- Landmark Detection (PFLD)
- Pose Estimation (CalcParams)
- Face Alignment
- HOG Extraction (PyFHOG)
- Geometric Features
- Running Median Update
- AU Prediction (17 SVR models)

Also identifies hardware acceleration opportunities.

Usage:
    python benchmark_detailed.py --video test_video.mp4
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Add pyfaceau to path
sys.path.insert(0, str(Path(__file__).parent))

from pyfaceau.pipeline import FullPythonAUPipeline
import pyfhog


class DetailedBenchmark:
    """Comprehensive performance benchmarking with per-component timing"""

    def __init__(self, pipeline: FullPythonAUPipeline):
        self.pipeline = pipeline
        self.timings = defaultdict(list)
        self.frame_count = 0

    def benchmark_frame(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> Dict:
        """
        Process a single frame with detailed timing for each component

        Returns:
            Dictionary with frame results and timing breakdowns
        """
        result = {
            'frame': frame_idx,
            'timestamp': timestamp,
            'success': False
        }

        component_times = {}

        try:
            # ===== STEP 1: Face Detection (with tracking) =====
            t0 = time.perf_counter()
            bbox = None
            need_detection = True

            # Try using cached bbox (face tracking)
            if self.pipeline.track_faces and self.pipeline.cached_bbox is not None:
                bbox = self.pipeline.cached_bbox
                need_detection = False
                self.pipeline.frames_since_detection += 1

            # Run face detection if needed
            if need_detection or bbox is None:
                detections, _ = self.pipeline.face_detector.detect_faces(frame)

                if len(detections) == 0:
                    self.pipeline.cached_bbox = None
                    component_times['face_detection'] = (time.perf_counter() - t0) * 1000
                    return result, component_times

                det = detections[0]
                bbox = det[:4].astype(int)

                # Cache bbox for next frame
                if self.pipeline.track_faces:
                    self.pipeline.cached_bbox = bbox
                    self.pipeline.frames_since_detection = 0

            component_times['face_detection'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 2: Landmark Detection =====
            t0 = time.perf_counter()
            landmarks_68, _ = self.pipeline.landmark_detector.detect_landmarks(frame, bbox)
            component_times['landmark_detection'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 3: Pose Estimation (CalcParams) =====
            t0 = time.perf_counter()
            if self.pipeline.use_calc_params and self.pipeline.calc_params:
                params_global, params_local = self.pipeline.calc_params.calc_params(
                    landmarks_68.flatten()
                )
                scale = params_global[0]
                rx, ry, rz = params_global[1:4]
                tx, ty = params_global[4:6]
            else:
                tx = (bbox[0] + bbox[2]) / 2
                ty = (bbox[1] + bbox[3]) / 2
                rz = 0.0
                params_local = np.zeros(34)
            component_times['pose_estimation'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 4: Face Alignment =====
            t0 = time.perf_counter()
            aligned_face = self.pipeline.face_aligner.align_face(
                image=frame,
                landmarks_68=landmarks_68,
                pose_tx=tx,
                pose_ty=ty,
                p_rz=rz,
                apply_mask=True,
                triangulation=self.pipeline.triangulation
            )
            component_times['face_alignment'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 5: HOG Extraction =====
            t0 = time.perf_counter()
            hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
            hog_features = hog_features.flatten().astype(np.float32)
            component_times['hog_extraction'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 6: Geometric Features =====
            t0 = time.perf_counter()
            geom_features = self.pipeline.pdm_parser.extract_geometric_features(params_local)
            geom_features = geom_features.astype(np.float32)
            component_times['geometric_features'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 7: Running Median Update =====
            t0 = time.perf_counter()
            update_histogram = (frame_idx % 2 == 1)
            self.pipeline.running_median.update(hog_features, geom_features, update_histogram=update_histogram)
            running_median = self.pipeline.running_median.get_combined_median()
            component_times['running_median'] = (time.perf_counter() - t0) * 1000

            # ===== STEP 8: AU Prediction =====
            t0 = time.perf_counter()
            au_results = self.pipeline._predict_aus(hog_features, geom_features, running_median)
            component_times['au_prediction'] = (time.perf_counter() - t0) * 1000

            # Update result
            result.update(au_results)
            result['success'] = True

        except Exception as e:
            print(f"Frame {frame_idx} failed: {e}")

        return result, component_times

    def run_benchmark(self, video_path: str, max_frames: int = 200) -> pd.DataFrame:
        """
        Run full benchmark on video

        Args:
            video_path: Path to test video
            max_frames: Number of frames to process (default: 200)

        Returns:
            DataFrame with results and timing statistics
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print("=" * 80)
        print("DETAILED PERFORMANCE BENCHMARK - PyfaceAU")
        print("=" * 80)
        print(f"Video: {video_path.name}")
        print(f"Frames to process: {max_frames}")
        print("")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = min(max_frames, total_frames)

        print(f"Video info:")
        print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print("")
        print("Processing...")
        print("")

        # Process frames
        results = []
        frame_idx = 0
        overall_start = time.time()

        try:
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_idx / fps

                # Benchmark this frame
                frame_result, component_times = self.benchmark_frame(frame, frame_idx, timestamp)
                results.append(frame_result)

                # Store timing data
                for component, timing in component_times.items():
                    self.timings[component].append(timing)

                # Progress update
                if (frame_idx + 1) % 50 == 0:
                    elapsed = time.time() - overall_start
                    current_fps = (frame_idx + 1) / elapsed
                    print(f"  Processed {frame_idx + 1}/{max_frames} frames - {current_fps:.2f} FPS")

                frame_idx += 1

        finally:
            cap.release()

        overall_time = time.time() - overall_start
        df = pd.DataFrame(results)

        # Calculate statistics
        self._print_statistics(df, overall_time)

        return df

    def _print_statistics(self, df: pd.DataFrame, overall_time: float):
        """Print detailed performance statistics"""
        successful_frames = df['success'].sum()
        failed_frames = len(df) - successful_frames
        overall_fps = successful_frames / overall_time if overall_time > 0 else 0

        print("")
        print("=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        print("")

        # Overall metrics
        print("Overall Performance:")
        print(f"  Total frames: {len(df)}")
        print(f"  Successful: {successful_frames}")
        print(f"  Failed: {failed_frames}")
        print(f"  Success rate: {successful_frames/len(df)*100:.1f}%")
        print(f"  Total time: {overall_time:.2f}s")
        print(f"  Overall FPS: {overall_fps:.2f}")
        print(f"  Avg time per frame: {overall_time/len(df)*1000:.1f}ms")
        print("")

        # Component breakdown
        print("Per-Component Timing (milliseconds):")
        print("-" * 80)
        print(f"{'Component':<25} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'%'}")
        print("-" * 80)

        # Calculate total time for percentage
        total_component_time = sum(np.mean(times) for times in self.timings.values())

        # Sort by mean time (descending)
        sorted_components = sorted(
            self.timings.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )

        for component, times in sorted_components:
            mean_time = np.mean(times)
            median_time = np.median(times)
            min_time = np.min(times)
            max_time = np.max(times)
            percentage = (mean_time / total_component_time * 100) if total_component_time > 0 else 0

            print(f"{component:<25} {mean_time:<10.2f} {median_time:<10.2f} {min_time:<10.2f} {max_time:<10.2f} {percentage:.1f}%")

        print("-" * 80)
        print(f"{'TOTAL':<25} {total_component_time:<10.2f}")
        print("")

        # Bottleneck analysis
        print("Bottleneck Analysis:")
        print("-" * 80)

        bottlenecks = []
        for component, times in sorted_components:
            mean_time = np.mean(times)
            percentage = (mean_time / total_component_time * 100) if total_component_time > 0 else 0

            if percentage > 20:
                status = "🔴 CRITICAL BOTTLENECK"
                bottlenecks.append((component, mean_time, percentage, "critical"))
            elif percentage > 10:
                status = "🟡 MAJOR BOTTLENECK"
                bottlenecks.append((component, mean_time, percentage, "major"))
            elif percentage > 5:
                status = "⚠️  MINOR BOTTLENECK"
                bottlenecks.append((component, mean_time, percentage, "minor"))
            else:
                status = "✅ OPTIMIZED"

            print(f"{component:<25} {mean_time:>7.1f}ms ({percentage:>5.1f}%) - {status}")

        print("")

        # Hardware acceleration recommendations
        self._print_acceleration_recommendations(bottlenecks)

    def _print_acceleration_recommendations(self, bottlenecks: List):
        """Print hardware acceleration recommendations based on bottlenecks"""
        print("=" * 80)
        print("HARDWARE ACCELERATION OPPORTUNITIES")
        print("=" * 80)
        print("")

        recommendations = []

        for component, mean_time, percentage, severity in bottlenecks:
            if component == 'pose_estimation':
                recommendations.append({
                    'component': 'Pose Estimation (CalcParams)',
                    'current': f'{mean_time:.1f}ms',
                    'methods': [
                        '1. Numba JIT compilation (2-5x faster)',
                        '2. GPU acceleration with CuPy (3-10x faster)',
                        '3. JAX GPU implementation (5-15x faster)',
                        '4. Reduce iterations (20 vs 50)',
                        '5. Warmstart from previous frame'
                    ],
                    'expected_speedup': '2-15x',
                    'priority': '🔥 HIGH' if severity == 'critical' else '⚠️ MEDIUM'
                })

            elif component == 'hog_extraction':
                recommendations.append({
                    'component': 'HOG Extraction',
                    'current': f'{mean_time:.1f}ms',
                    'methods': [
                        '1. GPU-accelerated HOG (CUDA)',
                        '2. Increase cell_size (8→12, fewer computations)',
                        '3. Reduce aligned face size (112x112→96x96)',
                        '4. OpenCV GPU HOG descriptor'
                    ],
                    'expected_speedup': '2-5x',
                    'priority': '🔥 HIGH' if severity == 'critical' else '⚠️ MEDIUM'
                })

            elif component == 'au_prediction':
                recommendations.append({
                    'component': 'AU Prediction (17 SVRs)',
                    'current': f'{mean_time:.1f}ms',
                    'methods': [
                        '1. Batch predictions (stack all SVs, single matmul)',
                        '2. GPU matrix multiplication (CuPy/JAX)',
                        '3. Numba JIT for prediction loop',
                        '4. Pre-cache mean centering'
                    ],
                    'expected_speedup': '2-10x',
                    'priority': '⚠️ MEDIUM'
                })

            elif component == 'face_detection':
                recommendations.append({
                    'component': 'Face Detection',
                    'current': f'{mean_time:.1f}ms',
                    'methods': [
                        '1. Enable CoreML (2-3x faster on Mac)',
                        '2. Face tracking (skip detection 90% of frames)',
                        '3. Lower input resolution (0.5x scale)',
                        '4. Switch to YOLO-Face (10-20ms)',
                        '5. Use GPU ONNX provider'
                    ],
                    'expected_speedup': '2-10x',
                    'priority': '🔥 HIGH' if severity == 'critical' else '⚠️ MEDIUM'
                })

            elif component == 'landmark_detection':
                recommendations.append({
                    'component': 'Landmark Detection',
                    'current': f'{mean_time:.1f}ms',
                    'methods': [
                        '1. GPU ONNX execution provider',
                        '2. CoreML acceleration (Mac)',
                        '3. TensorRT (NVIDIA GPUs)',
                        '4. Reduce input size if possible'
                    ],
                    'expected_speedup': '2-5x',
                    'priority': '⚠️ MEDIUM'
                })

        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{rec['priority']} - {rec['component']}")
                print(f"  Current: {rec['current']} ({rec['expected_speedup']} speedup potential)")
                print("")
                print("  Optimization Methods:")
                for method in rec['methods']:
                    print(f"    {method}")
                print("")

        # General recommendations
        print("=" * 80)
        print("GENERAL ACCELERATION STRATEGIES")
        print("=" * 80)
        print("")
        print("🚀 GPU Acceleration:")
        print("  - Install CuPy: pip install cupy-cuda11x")
        print("  - Install JAX GPU: pip install jax[cuda]")
        print("  - ONNX GPU: pip install onnxruntime-gpu")
        print("")
        print("⚡ JIT Compilation:")
        print("  - Install Numba: pip install numba")
        print("  - Add @jit decorator to CalcParams hotspots")
        print("")
        print("🔄 Parallelization:")
        print("  - Use ParallelAUPipeline for multi-core processing")
        print("  - 6-10x speedup with 6-10 workers")
        print("")
        print("🎯 Algorithmic Optimizations:")
        print("  - Enable face tracking (already implemented)")
        print("  - Reduce CalcParams iterations")
        print("  - Increase HOG cell size")
        print("  - Batch SVR predictions")
        print("")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed performance benchmark for PyfaceAU pipeline"
    )

    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--max-frames', type=int, default=200, help='Frames to process (default: 200)')
    parser.add_argument('--output', help='Save results to CSV (optional)')

    # Model paths
    parser.add_argument('--retinaface', default='weights/retinaface_mobilenet025_coreml.onnx')
    parser.add_argument('--pfld', default='weights/pfld_cunjian.onnx')
    parser.add_argument('--pdm', default='weights/In-the-wild_aligned_PDM_68.txt')
    parser.add_argument('--au-models', default='weights/AU_predictors')
    parser.add_argument('--triangulation', default='weights/tris_68_full.txt')

    args = parser.parse_args()

    # Verify video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        return 1

    # Initialize pipeline
    try:
        print("Initializing pipeline...")
        pipeline = FullPythonAUPipeline(
            retinaface_model=args.retinaface,
            pfld_model=args.pfld,
            pdm_file=args.pdm,
            au_models_dir=args.au_models,
            triangulation_file=args.triangulation,
            use_calc_params=True,
            use_coreml=False,  # Use CPU for consistent benchmarking
            track_faces=True,
            verbose=False
        )
        pipeline._initialize_components()
        print("✓ Pipeline initialized")
        print("")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run benchmark
    try:
        benchmark = DetailedBenchmark(pipeline)
        df = benchmark.run_benchmark(
            video_path=str(video_path),
            max_frames=args.max_frames
        )

        # Save results if requested
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"✓ Results saved to: {args.output}")
            print("")

        print("=" * 80)
        print("✅ BENCHMARK COMPLETE")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
