#!/usr/bin/env python3
"""
Generate C++ OpenFace ground truth for all videos.

This runs the C++ OpenFace FeatureExtraction binary on all videos
to produce CSV files with landmarks, params, and AUs.

Usage:
    python tools/generate_cpp_groundtruth.py [--start INDEX] [--count N]
"""

import argparse
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# Configuration
OPENFACE_BIN = Path("/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/build/bin/FeatureExtraction")
VIDEO_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S Data/Paralysis Cohort")
OUTPUT_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/cpp_groundtruth")


def get_all_videos():
    """Get list of all video files."""
    videos = []
    for ext in ['*.MOV', '*.mov', '*.mp4', '*.MP4']:
        videos.extend(VIDEO_DIR.glob(ext))
    return sorted(videos)


def run_openface(video_path: Path, output_dir: Path) -> tuple[bool, str]:
    """
    Run OpenFace on a single video.

    Returns:
        (success: bool, message: str)
    """
    video_name = video_path.stem
    output_csv = output_dir / f"{video_name}.csv"

    # Skip if already processed
    if output_csv.exists():
        return True, f"Skipped {video_name} (already exists)"

    cmd = [
        str(OPENFACE_BIN),
        "-f", str(video_path),
        "-out_dir", str(output_dir),
        "-2Dfp",  # Output 2D landmarks
        "-3Dfp",  # Output 3D landmarks
        "-pdmparams",  # Output PDM parameters
        "-pose",  # Output head pose
        "-aus",  # Output action units
        "-gaze",  # Output gaze
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per video
        )

        if result.returncode == 0 and output_csv.exists():
            return True, f"Processed {video_name}"
        else:
            return False, f"Failed {video_name}: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, f"Timeout {video_name}"
    except Exception as e:
        return False, f"Error {video_name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Generate C++ OpenFace ground truth")
    parser.add_argument("--start", type=int, default=0, help="Starting video index")
    parser.add_argument("--count", type=int, default=None, help="Number of videos to process")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1 for sequential)")
    parser.add_argument("--list", action="store_true", help="List videos and exit")
    args = parser.parse_args()

    # Verify OpenFace exists
    if not OPENFACE_BIN.exists():
        print(f"ERROR: OpenFace binary not found at {OPENFACE_BIN}")
        sys.exit(1)

    # Get videos
    videos = get_all_videos()
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")

    if args.list:
        for i, v in enumerate(videos):
            print(f"  {i}: {v.name}")
        return

    # Select range
    end_idx = len(videos) if args.count is None else min(args.start + args.count, len(videos))
    videos = videos[args.start:end_idx]
    print(f"Processing videos {args.start} to {end_idx - 1} ({len(videos)} total)")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check what's already done
    existing = set(f.stem for f in OUTPUT_DIR.glob("*.csv"))
    remaining = [v for v in videos if v.stem not in existing]
    print(f"Already processed: {len(videos) - len(remaining)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All videos already processed!")
        return

    # Process videos
    success_count = 0
    fail_count = 0
    start_time = time.time()

    if args.workers == 1:
        # Sequential processing with progress
        for i, video in enumerate(remaining):
            print(f"\n[{i+1}/{len(remaining)}] Processing {video.name}...")
            success, msg = run_openface(video, OUTPUT_DIR)
            print(f"  {msg}")
            if success:
                success_count += 1
            else:
                fail_count += 1
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_openface, v, OUTPUT_DIR): v for v in remaining}

            for future in as_completed(futures):
                video = futures[future]
                success, msg = future.result()
                print(msg)
                if success:
                    success_count += 1
                else:
                    fail_count += 1

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Completed in {elapsed/60:.1f} minutes")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
