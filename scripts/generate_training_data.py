#!/usr/bin/env python3
"""
Generate Training Data for Neural Network Training

This script processes video files and extracts training data using the
existing pyCLNF pipeline. The output is saved as an HDF5 file containing:
- Face images (112x112)
- HOG features
- 68 landmarks
- Pose parameters (6 global params)
- PDM parameters (34 local params)
- 17 AU intensities

Usage:
    # Process a single video
    python generate_training_data.py video.mp4 -o output.h5

    # Process multiple videos from a directory
    python generate_training_data.py /path/to/videos/ -o dataset.h5 --pattern "*.MOV"

    # Process with frame limit
    python generate_training_data.py videos/ -o output.h5 --max-frames 1000

Example:
    cd pyfaceau
    PYTHONPATH="..:../pyclnf:../pymtcnn:../pyfhog" python scripts/generate_training_data.py \\
        "/path/to/Patient Data/Normal Cohort/" \\
        -o training_data.h5 \\
        --pattern "*.MOV" \\
        --max-frames-per-video 2000
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'input',
        type=str,
        help="Input video file or directory containing videos"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help="Output HDF5 file path"
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default="*.mp4",
        help="Glob pattern for video files (default: *.mp4)"
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help="Maximum total frames to process"
    )
    parser.add_argument(
        '--max-frames-per-video',
        type=int,
        default=None,
        help="Maximum frames per video"
    )
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=0,
        help="Skip N frames between samples (default: 0, process all)"
    )
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.5,
        help="Minimum quality score to include frame (default: 0.5)"
    )
    parser.add_argument(
        '--pdm-path',
        type=str,
        default="pyfaceau/weights/In-the-wild_aligned_PDM_68.txt",
        help="Path to PDM file"
    )
    parser.add_argument(
        '--au-models-dir',
        type=str,
        default="pyfaceau/weights/AU_predictors",
        help="Path to AU models directory"
    )
    parser.add_argument(
        '--triangulation-path',
        type=str,
        default="pyfaceau/weights/tris_68_full.txt",
        help="Path to triangulation file"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Import here to allow --help without dependencies
    from pyfaceau.data import TrainingDataGenerator, TrainingDataWriter
    from pyfaceau.data.training_data_generator import GeneratorConfig

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Collect video files
    if input_path.is_file():
        video_files = [input_path]
    elif input_path.is_dir():
        video_files = sorted(input_path.glob(args.pattern))
        if not video_files:
            print(f"No videos found matching pattern '{args.pattern}' in {input_path}")
            sys.exit(1)
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s)")

    # Create config
    config = GeneratorConfig(
        pdm_path=args.pdm_path,
        au_models_dir=args.au_models_dir,
        triangulation_path=args.triangulation_path,
        min_quality=args.min_quality,
        skip_frames=args.skip_frames,
        verbose=not args.quiet,
    )

    # Create generator
    generator = TrainingDataGenerator(config)

    # Process videos
    if len(video_files) == 1:
        stats = generator.process_video(
            video_files[0],
            output_path,
            max_frames=args.max_frames
        )
    else:
        stats = generator.process_multiple_videos(
            video_files,
            output_path,
            max_frames_per_video=args.max_frames_per_video
        )

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
