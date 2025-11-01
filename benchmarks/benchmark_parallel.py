#!/usr/bin/env python3
"""
Benchmark Parallel AU Pipeline

Tests the parallel processing pipeline to verify 30-50 FPS target performance.

Usage:
    python benchmark_parallel.py --video test_video.mp4 --workers 6

Results will show:
- Sequential baseline: ~4.6 FPS
- Parallel with N workers: target 27-50 FPS
"""

import sys
import time
import argparse
from pathlib import Path

# Add pyauface to path
sys.path.insert(0, str(Path(__file__).parent))

from pyauface.parallel_pipeline import ParallelAUPipeline


def benchmark_parallel(video_path, num_workers=6, max_frames=100):
    """
    Benchmark the parallel pipeline

    Args:
        video_path: Path to test video
        num_workers: Number of parallel workers
        max_frames: Number of frames to process
    """
    print("=" * 80)
    print("PARALLEL PIPELINE BENCHMARK")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Workers: {num_workers}")
    print(f"Test frames: {max_frames}")
    print("")
    print(f"Theoretical speedup: {num_workers}x")
    print(f"Expected FPS: {4.6 * num_workers:.1f} FPS")
    print(f"Target: 30 FPS (minimum), 50 FPS (stretch)")
    print("")
    print("=" * 80)
    print("")

    # Initialize pipeline
    print("Initializing parallel pipeline...")
    pipeline = ParallelAUPipeline(
        retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
        pfld_model='weights/pfld_cunjian.onnx',
        pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
        au_models_dir='weights/AU_predictors',
        triangulation_file='weights/tris_68_full.txt',
        num_workers=num_workers,
        batch_size=30,
        verbose=True
    )

    # Process video
    print("Starting processing...")
    print("")

    start_time = time.time()
    df = pipeline.process_video(
        video_path=video_path,
        max_frames=max_frames
    )
    total_time = time.time() - start_time

    # Calculate metrics
    successful_frames = df['success'].sum()
    failed_frames = len(df) - successful_frames
    overall_fps = successful_frames / total_time if total_time > 0 else 0
    speedup = overall_fps / 4.6

    # Results
    print("")
    print("=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total frames: {len(df)}")
    print(f"Successful: {successful_frames}")
    print(f"Failed: {failed_frames}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Per frame: {total_time/len(df)*1000:.1f}ms")
    print("")
    print(f"Overall FPS: {overall_fps:.2f} FPS")
    print(f"Speedup vs sequential: {speedup:.2f}x")
    print("")

    # Goal assessment
    if overall_fps >= 50:
        print("✅ STRETCH GOAL ACHIEVED! (50+ FPS)")
    elif overall_fps >= 30:
        print("✅ MINIMUM GOAL ACHIEVED! (30+ FPS)")
    elif overall_fps >= 20:
        print("⚠️ GOOD PROGRESS (20+ FPS) - Continue optimizing")
    else:
        print("❌ BELOW TARGET (< 20 FPS) - More optimization needed")

    print("")
    print("Comparison:")
    print(f"  Baseline (sequential): 4.6 FPS, 217ms/frame")
    print(f"  This run ({num_workers} workers): {overall_fps:.1f} FPS, {total_time/len(df)*1000:.0f}ms/frame")
    print(f"  Improvement: {speedup:.1f}x faster")
    print("")
    print("=" * 80)

    return overall_fps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark parallel AU pipeline performance"
    )

    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--workers', type=int, default=6, help='Number of workers (default: 6)')
    parser.add_argument('--max-frames', type=int, default=100, help='Frames to process (default: 100)')

    args = parser.parse_args()

    # Verify video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video not found: {video_path}")
        return 1

    # Run benchmark
    try:
        fps = benchmark_parallel(
            video_path=str(video_path),
            num_workers=args.workers,
            max_frames=args.max_frames
        )

        return 0

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
