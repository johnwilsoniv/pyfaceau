"""
OpenFace-compatible AU extraction processor.

This module provides a drop-in replacement for OpenFace 3.0
with the same API for easy integration into existing workflows
like S1 Face Mirror.
"""

import cv2
import csv
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from .pipeline import FullPythonAUPipeline
from .download_weights import ensure_weights
from .models import model_paths


def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError in GUI subprocess contexts."""
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, IOError):
        pass  # Stdout disconnected (e.g., GUI subprocess terminated)


class OpenFaceProcessor:
    """
    OpenFace 2.2-compatible AU extraction processor.

    Drop-in replacement for OpenFace 3.0 with the pyfaceau pipeline.
    Designed for seamless integration with S1 Face Mirror and other
    OpenFace-based applications.

    Features:
    - 17 Action Units (AU01-AU45)
    - r > 0.92 correlation with OpenFace 2.2 C++
    - CLNF landmark refinement
    - Real-time capable (72 fps)
    - 100% Python (no compilation)

    Example:
        ```python
        processor = OpenFaceProcessor(
            weights_dir='weights/',
            use_clnf_refinement=True
        )

        processor.process_video(
            'input.mp4',
            'output.csv',
            progress_callback=my_callback
        )
        ```
    """

    def __init__(
        self,
        device: Optional[str] = None,
        weights_dir: Optional[str] = None,
        use_clnf_refinement: bool = True,
        num_threads: int = 6,
        verbose: bool = False,
        auto_download_weights: bool = True,
        **kwargs
    ):
        """
        Initialize OpenFace AU extraction processor.

        Args:
            device: Unused (kept for API compatibility). PyFaceAU auto-detects.
            weights_dir: Folder with the OpenFace model files. An explicit folder
                        always wins. If None: the PYFACEAU_WEIGHTS_DIR environment
                        variable (if it points to a complete folder), otherwise the
                        shared OpenFace model folder installed by
                        `pyfaceau-download-models` (see pyfaceau.models).
            use_clnf_refinement: Enable CLNF landmark refinement (default: True)
            num_threads: Unused (kept for API compatibility)
            verbose: Enable verbose logging (default: False)
            auto_download_weights: Ignored since 1.4.0 (kept for compatibility).
                Model files are never downloaded silently: run
                `pyfaceau-download-models` once, or set
                OPENFACE_MODELS_ACCEPT_LICENSE=1 to accept the OpenFace license.
            **kwargs: Additional arguments (ignored for compatibility)

        Raises:
            ModelsNotInstalledError (a FileNotFoundError): no weights_dir was
                given and the OpenFace model files are not installed yet; the
                message explains how to install them.
        """
        self.verbose = verbose

        # Determine weights directory
        if weights_dir is not None:
            weights_dir = Path(weights_dir)
            if not weights_dir.exists():
                raise FileNotFoundError(
                    f"Specified weights directory not found: {weights_dir}\n"
                    f"Please ensure the weights are downloaded to this location."
                )
        else:
            # Shared OpenFace model folder; never downloads without license acceptance
            weights_dir = ensure_weights()

        weights_dir = Path(weights_dir)

        if self.verbose:
            safe_print("Initializing PyFaceAU (OpenFace 2.2 Python replacement)...")
            safe_print(f"  Weights directory: {weights_dir}")

        # Initialize the PyFaceAU pipeline (OpenFace-compatible: PyMTCNN → CLNF → AU)
        self.pipeline = FullPythonAUPipeline(
            **model_paths(weights_dir),
            mtcnn_backend='auto',  # PyMTCNN for face detection
            use_batched_predictor=True,
            verbose=verbose
        )

        if self.verbose:
            safe_print(f"  PyFaceAU initialized")
            safe_print(f"  CLNF refinement: {'Enabled' if use_clnf_refinement else 'Disabled'}")
            safe_print(f"  Expected accuracy: r > 0.92 (OpenFace 2.2 correlation)")
            safe_print()

    def process_video(
        self,
        video_path: str,
        output_csv_path: str,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> int:
        """
        Process video and extract AUs.

        Compatible with S1 Face Mirror integration and other OpenFace-based
        applications.

        Args:
            video_path: Path to input video file
            output_csv_path: Path to output CSV file
            progress_callback: Optional callback function(current, total, fps)
                             for progress updates

        Returns:
            Number of frames successfully processed
        """
        video_path = Path(video_path)
        output_csv_path = Path(output_csv_path)

        if self.verbose:
            safe_print(f"Processing: {video_path.name}")

        # Ensure output directory exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Process video through pipeline
        try:
            df = self.pipeline.process_video(
                video_path=str(video_path),
                output_csv=str(output_csv_path),
                max_frames=None,
                progress_callback=progress_callback
            )

            success_count = df['success'].sum()

            if self.verbose:
                total_frames = len(df)
                safe_print(f"  Processed {success_count}/{total_frames} frames successfully")
                if success_count < total_frames:
                    failed = total_frames - success_count
                    safe_print(f"  {failed} frames failed (no face detected)")
                safe_print(f"  Output: {output_csv_path}")

            return int(success_count)

        except Exception as e:
            if self.verbose:
                safe_print(f"  Error processing video: {e}")
            raise

    def clear_cache(self):
        """
        Clear cached data and reset all per-video state to free memory and
        prevent inter-video contamination.

        Clears:
        - stored_features list (can be up to 56 MB for long videos)
        - running median histograms
        - face tracking cache
        - CLNF temporal state
        - OnlineAUCorrection prediction histogram + correction offsets
        - MPS/CUDA GPU memory

        This should be called between videos to prevent memory accumulation
        AND to prevent inter-video state contamination. Two historical
        reset bugs are fixed:

        - v1.3.13 added the OnlineAUCorrection reset; without it the per-AU
          prediction histogram saturated after ~40 videos and clamped
          subsequent predictions to zero (silent AU=0 bug in long batches).

        - v1.3.14 fixed the CLNF reset, which had been silently failing
          since v1.3.8 due to a wrong attribute lookup
          (`pipeline.clnf` instead of `pipeline.landmark_detector`). With
          the bug, CLNF temporal state carried over from one video to the
          next; landmark refinement on the new video's first frame was
          seeded with the previous video's last-frame landmarks, often
          driving CLNF into a wrong local minimum and corrupting AU
          intensities for the second-and-later video in any sequence.
        """
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            # Clear stored features (two-pass processing cache)
            if hasattr(self.pipeline, 'stored_features'):
                self.pipeline.stored_features.clear()

            # Reset running median tracker
            if hasattr(self.pipeline, 'running_median') and self.pipeline.running_median is not None:
                self.pipeline.running_median.reset()

            # Reset OnlineAUCorrection histogram + per-AU correction offsets.
            # WITHOUT this reset, the per-AU prediction_corr_histogram
            # accumulates across every frame of every video processed by
            # this OpenFaceProcessor instance. After ~160k frames
            # (~40 typical videos x 2 sides), `_recompute_correction()`
            # derives correction offsets large enough that subsequent
            # `correct()` calls clamp every AU prediction toward zero --
            # a silent failure mode (`success=True` is still reported per
            # frame). Adding the reset here eliminates the bug for all
            # callers without requiring per-call-site workarounds.
            if (hasattr(self.pipeline, 'online_au_correction')
                    and self.pipeline.online_au_correction is not None
                    and hasattr(self.pipeline.online_au_correction, 'reset')):
                self.pipeline.online_au_correction.reset()

            # Clear face tracking cache
            if hasattr(self.pipeline, 'cached_bbox'):
                self.pipeline.cached_bbox = None
                self.pipeline.detection_failures = 0
                self.pipeline.frames_since_detection = 0

            # Reset CLNF temporal state and clear GPU caches.
            # Bugfix (v1.3.14): the pipeline exposes the CLNF instance as
            # `landmark_detector` (set in pipeline.py:268, populated at
            # pipeline.py:349). The earlier `self.pipeline.clnf` lookup was
            # always None/missing, so this whole block silently no-op'd via
            # the hasattr guard and CLNF temporal state was NEVER reset
            # between videos in v1.3.8..v1.3.13. That caused the next video's
            # first-frame landmarks to be seeded with the previous video's
            # last-frame landmarks; when the two faces don't align, CLNF
            # fails to converge and downstream HOG/geom features (and AU
            # intensities) are corrupted - severely for the second-and-later
            # video in any sequence. See PYFACEAU_CLNF_RESET_BUG.md for the
            # full diagnosis and pilot11g_proper_reset_canary_PTNE.py for
            # empirical verification (broken: mean r vs C++ = -0.06; fixed:
            # mean r = 0.99).
            if hasattr(self.pipeline, 'landmark_detector') and self.pipeline.landmark_detector is not None:
                if hasattr(self.pipeline.landmark_detector, 'reset_temporal_state'):
                    self.pipeline.landmark_detector.reset_temporal_state()
                # Clear GPU memory caches (added in pyclnf 0.3.3)
                if hasattr(self.pipeline.landmark_detector, 'clear_gpu_cache'):
                    self.pipeline.landmark_detector.clear_gpu_cache()

        # Release GPU memory (MPS for Apple Silicon, CUDA for NVIDIA)
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore if torch not available

        # Force garbage collection
        import gc
        gc.collect()


def process_videos(
    directory_path: str,
    specific_files: Optional[list] = None,
    output_dir: Optional[str] = None,
    **processor_kwargs
) -> int:
    """
    Process multiple video files using OpenFaceProcessor.

    This function provides batch processing capability compatible with
    S1 Face Mirror workflows.

    Args:
        directory_path: Path to directory containing video files
        specific_files: List of specific files to process (optional)
        output_dir: Output directory for CSV files (optional)
        **processor_kwargs: Additional arguments passed to OpenFaceProcessor

    Returns:
        Number of files successfully processed

    Example:
        ```python
        # Process all mirrored videos in a directory
        count = process_videos(
            directory_path='/path/to/mirrored/videos',
            output_dir='/path/to/output',
            use_clnf_refinement=True
        )
        safe_print(f"Processed {count} videos")
        ```
    """
    directory_path = Path(directory_path)

    # Check if directory exists
    if not directory_path.is_dir():
        safe_print(f"Error: Directory '{directory_path}' does not exist.")
        return 0

    # Determine output directory
    if output_dir is None:
        # Default: S1O Processed Files/Combined Data/
        s1o_base = directory_path.parent.parent / 'S1O Processed Files'
        output_dir = s1o_base / 'Combined Data'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_print(f"Output directory: {output_dir}")

    # Initialize processor
    processor = OpenFaceProcessor(**processor_kwargs)

    # Counter for processed files
    processed_count = 0

    # Define which files to process
    files_to_process = []

    if specific_files:
        # Process only the specific files
        files_to_process = [Path(f) for f in specific_files]
        safe_print(f"Processing {len(files_to_process)} specific files from current session.")
    else:
        # Process all eligible files in the directory
        files_to_process = list(directory_path.iterdir())
        safe_print(f"Processing all eligible files in {directory_path}")

    # Process each file
    for file_path in files_to_process:
        # Skip if not a file or doesn't exist
        if not file_path.is_file():
            safe_print(f"Warning: {file_path} does not exist or is not a file. Skipping.")
            continue

        filename = file_path.name

        # Skip files with 'debug' in the filename
        if 'debug' in filename:
            safe_print(f"Skipping debug file: {filename}")
            continue

        # Process file with 'mirrored' in the filename
        if 'mirrored' in filename:
            # Generate output CSV filename
            # Example: "video_left_mirrored.mp4" -> "video_left_mirrored.csv"
            csv_filename = file_path.stem + '.csv'
            output_csv_path = output_dir / csv_filename

            try:
                # Process video and extract AUs
                frame_count = processor.process_video(file_path, output_csv_path)

                if frame_count > 0:
                    processed_count += 1
                    safe_print(f"Successfully processed: {filename}\n")
                else:
                    safe_print(f"Failed to process: {filename}\n")

            except Exception as e:
                safe_print(f"Error processing {filename}: {e}\n")

    safe_print(f"\nProcessing complete. {processed_count} files were processed.")

    return processed_count


def main(argv: Optional[list] = None) -> int:
    """The `pyfaceau` command: `pyfaceau VIDEO [-o OUTPUT.csv]`.

    Runs the default pipeline (OpenFaceProcessor) on one video and writes one
    CSV row per frame: frame, timestamp, success and the 17 AU intensities.
    Returns 0 on success, 1 if the video could not be processed (or no face
    was found in any frame), 2 for a usage error.
    """
    import argparse
    import sys
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="pyfaceau",
        description="Measure 17 facial action units (AUs) in a video with pyfaceau's "
                    "default pipeline and save one CSV row per frame.",
        epilog="Before the first run, install the OpenFace model files once with "
               "`pyfaceau-download-models`.",
    )
    parser.add_argument("video", help="the video file to process")
    parser.add_argument("-o", "--output", metavar="OUTPUT.csv",
                        help="the CSV file to write (default: the video's name with "
                             ".csv, in the current folder)")
    parser.add_argument("--version", action="version", version=f"pyfaceau {__version__}")
    args = parser.parse_args(argv)

    video = Path(args.video)
    if not video.is_file():
        parser.error(f"video not found: {video}")
    output = Path(args.output) if args.output else Path.cwd() / f"{video.stem}.csv"

    def say(text, end="\n"):
        try:
            print(text, end=end, file=sys.stderr, flush=True)
        except (BrokenPipeError, OSError):
            pass

    capture = cv2.VideoCapture(str(video))
    readable = capture.isOpened() and capture.read()[0]
    capture.release()
    if not readable:
        say(f"pyfaceau: cannot read any frame from {video}. Is it a video file?")
        return 1

    show_progress = sys.stderr.isatty()

    def progress(done, total, fps):
        if show_progress:
            say(f"\r  frame {done} of {total} ({fps:.1f} frames per second)", end="")

    try:
        processor = OpenFaceProcessor(verbose=False)
        say(f"Processing {video.name} ...")
        faces = processor.process_video(str(video), str(output), progress_callback=progress)
    except Exception as e:  # ModelsNotInstalledError explains how to install the files
        if show_progress:
            say("")
        say(f"pyfaceau: {e}")
        return 1
    if show_progress:
        say("")
    if faces == 0:
        say(f"No face was found in {video.name}. The CSV was still written: {output}")
        return 1
    say(f"Done: a face was found in {faces} frames. AU values saved to {output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
