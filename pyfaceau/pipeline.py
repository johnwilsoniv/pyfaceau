#!/usr/bin/env python3
"""
Full Python AU Extraction Pipeline - End-to-End

This script integrates all Python components into a complete AU extraction pipeline:
1. Face Detection (PyMTCNN with CUDA/CoreML/CPU support)
2. Landmark Detection (Cunjian PFLD)
3. 3D Pose Estimation (from pyclnf CLNF optimization params)
4. Face Alignment (OpenFace 2.2 algorithm)
5. HOG Feature Extraction (PyFHOG)
6. Geometric Feature Extraction (PDM)
7. Running Median Tracking (Cython-optimized)
8. AU Prediction (SVR models)

No C++ OpenFace binary required - 100% Python!

Usage:
    python full_python_au_pipeline.py --video input.mp4 --output results.csv
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import argparse
import sys
import time
import subprocess
import json

# Import configuration
from pyfaceau.config import CLNF_CONFIG, RUNNING_MEDIAN_CONFIG, AU_CONFIG

# Import all pipeline components
from pyfaceau.detectors.pymtcnn_detector import PyMTCNNDetector, PYMTCNN_AVAILABLE
# Note: CLNF import is done lazily in _initialize_components() to avoid circular import
from pyfaceau.alignment.calc_params import CalcParams
from pyfaceau.features.pdm import PDMParser
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.features.triangulation import TriangulationParser
from pyfaceau.prediction.model_parser import OF22ModelParser
from pyfaceau.refinement.targeted_refiner import TargetedCLNFRefiner

# Import Cython-optimized running median (with fallback)
try:
    from pyfaceau.cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
    USING_CYTHON = True
except ImportError:
    from pyfaceau.features.histogram_median_tracker import DualHistogramMedianTracker
    USING_CYTHON = False

# Import optimized batched AU predictor
try:
    from pyfaceau.prediction.batched_au_predictor import BatchedAUPredictor
    USING_BATCHED_PREDICTOR = True
except ImportError:
    USING_BATCHED_PREDICTOR = False

# NNCLNF (neural network replacement for CLNF) is defunct and archived
# Use pyclnf with GPU acceleration instead for best accuracy + speed

# Import online AU correction (C++ CorrectOnlineAUs equivalent)
from pyfaceau.prediction.online_au_correction import OnlineAUCorrection

# Import PyFHOG for HOG extraction
# Try different paths where pyfhog might be installed
try:
    import pyfhog
except ImportError:
    # Try parent directory (for development)
    import sys
    pyfhog_src_path = Path(__file__).parent.parent / 'pyfhog' / 'src'
    if pyfhog_src_path.exists():
        sys.path.insert(0, str(pyfhog_src_path))
        import pyfhog
    else:
        safe_print("Error: pyfhog not found. Please install it:")
        safe_print("   cd ../pyfhog && pip install -e .")
        sys.exit(1)


def safe_print(*args, **kwargs):
    """Print wrapper that handles BrokenPipeError in GUI subprocess contexts."""
    try:
        print(*args, **kwargs)
    except (BrokenPipeError, IOError):
        pass  # Stdout disconnected (e.g., GUI subprocess terminated)


def get_video_rotation(video_path: str) -> int:
    """
    Get video rotation from metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Rotation angle in degrees (0, 90, -90, 180, 270)
    """
    try:
        # Try JSON metadata first (most comprehensive)
        cmd = f'ffprobe -v quiet -print_format json -show_streams "{video_path}"'
        output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()

        metadata = json.loads(output)
        for stream in metadata.get('streams', []):
            # Check rotate tag
            rotation = stream.get('tags', {}).get('rotate')
            if rotation is None:
                rotation = stream.get('rotation')

            if rotation is not None:
                try:
                    return int(rotation)
                except ValueError:
                    pass

            # Check displaymatrix in side data (iOS videos)
            if 'side_data_list' in stream:
                for side_data in stream['side_data_list']:
                    # Check for 'rotation' key directly (newer ffprobe versions)
                    if 'rotation' in side_data:
                        try:
                            return int(side_data['rotation'])
                        except (ValueError, TypeError):
                            pass
                    # Fallback to string parsing for older formats
                    side_str = str(side_data).lower()
                    if 'displaymatrix' in side_str:
                        if 'rotation of -90' in side_str or "'rotation': -90" in side_str:
                            return -90
                        elif 'rotation of 90' in side_str or "'rotation': 90" in side_str:
                            return 90
                        elif 'rotation of 180' in side_str or "'rotation': 180" in side_str:
                            return 180
                        elif 'rotation of -180' in side_str or "'rotation': -180" in side_str:
                            return 180
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        pass

    # Try specific ffprobe commands
    specific_commands = [
        f'ffprobe -v error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1 "{video_path}"',
        f'ffprobe -v error -select_streams v:0 -show_entries stream=rotate -of default=nw=1:nk=1 "{video_path}"',
    ]

    for cmd in specific_commands:
        try:
            output = subprocess.check_output(cmd, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
            if output:
                return int(output)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass

    return 0


def apply_frame_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """
    Apply rotation to a video frame based on metadata rotation angle.

    The rotation metadata indicates how the video was recorded (device orientation).
    To display correctly, we rotate in the OPPOSITE direction.

    Args:
        frame: Input frame (BGR)
        rotation: Rotation angle from metadata (how video was recorded)

    Returns:
        Rotated frame (corrected to upright)
    """
    # Rotate OPPOSITE to the recorded rotation to correct orientation
    if rotation == 90 or rotation == -270:
        # Video recorded at +90, rotate -90 (counterclockwise) to correct
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == -90 or rotation == 270:
        # Video recorded at -90, rotate +90 (clockwise) to correct
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180 or rotation == -180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        return frame


class FullPythonAUPipeline:
    """
    Complete Python AU extraction pipeline

    Integrates face detection, landmark detection, pose estimation,
    alignment, feature extraction, and AU prediction into a single
    end-to-end pipeline.
    """

    def __init__(
        self,
        pdm_file: str,
        au_models_dir: str,
        triangulation_file: str,
        patch_expert_file: str,
        mtcnn_backend: str = 'auto',
        use_calc_params: bool = True,
        track_faces: bool = True,
        use_batched_predictor: bool = True,
        use_nnclnf: str = 'pyclnf',  # NNCLNF is defunct, always use pyclnf
        max_clnf_iterations: int = CLNF_CONFIG['max_iterations'],
        clnf_convergence_threshold: float = CLNF_CONFIG['convergence_threshold'],
        debug_mode: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the full Python AU pipeline (OpenFace-compatible)

        Architecture: PyMTCNN → pyclnf CLNF → AU Prediction
        (matches OpenFace C++ 2.2 pipeline)

        Args:
            pdm_file: Path to PDM shape model
            au_models_dir: Directory containing AU SVR models
            triangulation_file: Path to triangulation file for masking
            patch_expert_file: Path to CLNF patch expert file
            mtcnn_backend: PyMTCNN backend ('auto', 'cuda', 'coreml', 'cpu') (default: 'auto')
            use_calc_params: DEPRECATED - pyclnf params are now used instead (default: True)
            track_faces: Use face tracking between frames (default: True)
            use_batched_predictor: Use optimized batched AU predictor (default: True)
            use_nnclnf: DEPRECATED - always uses pyclnf (NNCLNF is defunct)
            max_clnf_iterations: Maximum CLNF optimization iterations (default: 10)
            clnf_convergence_threshold: CLNF convergence threshold in pixels (default: 0.01)
            debug_mode: Enable debug mode for diagnostics (default: False)
            verbose: Print progress messages (default: True)
        """
        import threading

        self.verbose = verbose
        self.debug_mode = debug_mode
        self.use_calc_params = use_calc_params
        self.track_faces = track_faces
        self.use_batched_predictor = use_batched_predictor and USING_BATCHED_PREDICTOR

        # Face tracking: cache bbox and only re-detect on failure (3x speedup!)
        self.cached_bbox = None
        self.detection_failures = 0
        self.frames_since_detection = 0

        # Store initialization parameters (lazy initialization)
        self._init_params = {
            'mtcnn_backend': mtcnn_backend,
            'pdm_file': pdm_file,
            'au_models_dir': au_models_dir,
            'triangulation_file': triangulation_file,
            'patch_expert_file': patch_expert_file,
            'use_nnclnf': use_nnclnf,
            'max_clnf_iterations': max_clnf_iterations,
            'clnf_convergence_threshold': clnf_convergence_threshold,
        }

        # Components will be initialized on first use (in worker thread if CoreML)
        self._components_initialized = False
        self._initialization_lock = threading.Lock()

        # Component placeholders
        self.face_detector = None
        self.landmark_detector = None
        self.clnf_refiner = None
        self.pdm_parser = None
        self.calc_params = None
        self.face_aligner = None
        self.triangulation = None
        self.au_models = None
        self.batched_au_predictor = None
        self.running_median = None
        self.online_au_correction = None  # C++ CorrectOnlineAUs equivalent

        # Two-pass processing: Store features for early frames
        self.stored_features = []  # List of (frame_idx, hog_features, geom_features)
        self.max_stored_frames = AU_CONFIG['max_stored_frames']  # OpenFace default

        # Note: Actual initialization happens in _initialize_components()
        # This is called lazily on first use (in worker thread if CoreML enabled)

    def _initialize_components(self):
        """
        Initialize all pipeline components (called lazily on first use).
        This allows CoreML to be initialized in worker thread.
        """
        with self._initialization_lock:
            if self._components_initialized:
                return  # Already initialized

            import threading

            if self.verbose:
                thread_name = threading.current_thread().name
                is_main = threading.current_thread() == threading.main_thread()
                safe_print("=" * 80)
                safe_print("INITIALIZING COMPONENTS")
                safe_print(f"Thread: {thread_name} (main={is_main})")
                safe_print("=" * 80)
                safe_print("")

            # Get initialization parameters
            mtcnn_backend = self._init_params['mtcnn_backend']
            pdm_file = self._init_params['pdm_file']
            au_models_dir = self._init_params['au_models_dir']
            triangulation_file = self._init_params['triangulation_file']
            patch_expert_file = self._init_params['patch_expert_file']
            max_clnf_iterations = self._init_params['max_clnf_iterations']
            clnf_convergence_threshold = self._init_params['clnf_convergence_threshold']

            # Component 1: Face Detection (PyMTCNN with multi-backend support)
            if self.verbose:
                safe_print("[1/8] Loading face detector (PyMTCNN)...")
                safe_print(f"  Backend: {mtcnn_backend}")

            if not PYMTCNN_AVAILABLE:
                raise ImportError(
                    "PyMTCNN is required. Install with:\n"
                    "  pip install pymtcnn[onnx-gpu]  # For CUDA\n"
                    "  pip install pymtcnn[coreml]    # For Apple Silicon\n"
                    "  pip install pymtcnn[onnx]      # For CPU"
                )

            self.face_detector = PyMTCNNDetector(
                backend=mtcnn_backend,
                confidence_threshold=0.5,
                nms_threshold=0.7,
                verbose=self.verbose
            )
            if self.verbose:
                backend_info = self.face_detector.get_backend_info()
                safe_print(f"  Active backend: {backend_info}")
                safe_print("Face detector loaded\n")

            # Component 2: Landmark Detection (pyclnf CLNF with GPU acceleration)
            if self.verbose:
                safe_print("[2/8] Loading CLNF landmark detector (pyclnf)...")
                safe_print(f"  Max iterations: {max_clnf_iterations}")
                safe_print(f"  Convergence threshold: {clnf_convergence_threshold} pixels")
                safe_print(f"  GPU enabled: {CLNF_CONFIG.get('use_gpu', False)}")

            # Lazy import to avoid circular import (pyfaceau ↔ pyclnf)
            from pyclnf import CLNF

            self.landmark_detector = CLNF(
                # Use default model_dir - pyclnf finds its own models from PyPI installation
                max_iterations=max_clnf_iterations,
                convergence_threshold=clnf_convergence_threshold,
                detector=CLNF_CONFIG['detector'],  # Disable built-in PyMTCNN (pyfaceau handles detection)
                use_eye_refinement=CLNF_CONFIG['use_eye_refinement'],  # Enable hierarchical eye model refinement
                convergence_profile=CLNF_CONFIG['convergence_profile'],  # Enable video mode with template tracking + scale adaptation
                sigma=CLNF_CONFIG['sigma'],  # KDE kernel sigma matching C++ CECLM
                use_gpu=CLNF_CONFIG.get('use_gpu', False),  # GPU acceleration (10-20x speedup)
                gpu_device=CLNF_CONFIG.get('gpu_device', 'auto'),  # GPU device selection
                use_validator=CLNF_CONFIG.get('use_validator', True)  # Detection validator (disabled for speed)
            )

            if self.verbose:
                safe_print(f"CLNF detector loaded\n")

            # Component 3: PDM Parser (moved before CLNF to support PDM enforcement)
            if self.verbose:
                safe_print("[3/8] Loading PDM shape model...")
            self.pdm_parser = PDMParser(pdm_file)
            if self.verbose:
                safe_print(f"PDM loaded: {self.pdm_parser.mean_shape.shape[0]//3} landmarks\n")

            # Note: CalcParams is no longer used for geometric features
            # pyclnf's optimized params are used instead (see GEOMETRIC_FEATURES_BUG.md)
            # CalcParams is kept for backwards compatibility but not initialized by default
            self.calc_params = None

            # Component 4: Face Aligner
            if self.verbose:
                safe_print("[4/8] Initializing face aligner...")
            self.face_aligner = OpenFace22FaceAligner(
                pdm_file=pdm_file,
                sim_scale=0.7,
                output_size=(112, 112)
            )
            if self.verbose:
                safe_print("Face aligner initialized\n")

            # Note: CLNF landmark detector is already initialized above (Component 2)
            # No separate refiner needed - CLNF does full detection from PDM mean shape

            # Component 5: Triangulation
            if self.verbose:
                safe_print("[5/8] Loading triangulation...")
            self.triangulation = TriangulationParser(triangulation_file)
            if self.verbose:
                safe_print(f"Triangulation loaded: {len(self.triangulation.triangles)} triangles\n")

            # Component 6: AU Models
            if self.verbose:
                safe_print("[6/8] Loading AU SVR models...")
            model_parser = OF22ModelParser(au_models_dir)
            self.au_models = model_parser.load_all_models(
                use_recommended=True,
                use_combined=True,
                verbose=self.verbose
            )
            if self.verbose:
                safe_print(f"Loaded {len(self.au_models)} AU models")

            # Initialize batched predictor if enabled
            if self.use_batched_predictor:
                self.batched_au_predictor = BatchedAUPredictor(self.au_models)
                if self.verbose:
                    safe_print(f"Batched AU predictor enabled (2-5x faster)")
            if self.verbose:
                safe_print("")

            # Component 7: Running Median Tracker
            if self.verbose:
                safe_print("[7/8] Initializing running median tracker...")
            # Use locked configuration from config.py (matches C++ OpenFace)
            self.running_median = DualHistogramMedianTracker(**RUNNING_MEDIAN_CONFIG)
            if self.verbose:
                if USING_CYTHON:
                    safe_print("Running median tracker initialized (Cython-optimized, 260x faster)\n")
                else:
                    safe_print("Running median tracker initialized (Python version)\n")

            # Component 8: Online AU Correction (C++ CorrectOnlineAUs equivalent)
            if self.verbose:
                safe_print("[8/9] Initializing online AU correction...")
            # Get AU names from loaded models
            au_names = list(self.au_models.keys())
            self.online_au_correction = OnlineAUCorrection(
                au_names=au_names,
                num_bins=200,      # C++ default
                min_val=-3.0,      # C++ default
                max_val=5.0,       # C++ default
                ratio=0.10,        # 10th percentile
                min_frames=10,     # C++ default
                clip_values=True
            )
            if self.verbose:
                safe_print(f"Online AU correction initialized for {len(au_names)} AUs\n")

            # Component 9: PyFHOG
            if self.verbose:
                safe_print("[9/9] PyFHOG ready for HOG extraction")
                safe_print("")
                safe_print("All components initialized successfully")
                safe_print("=" * 80)
                safe_print("")

            self._components_initialized = True

    @property
    def using_nnclnf(self) -> bool:
        """DEPRECATED: NNCLNF is defunct, always returns False."""
        return False

    @property
    def landmark_detector_name(self) -> str:
        """Get the name of the landmark detector being used."""
        if not self._components_initialized:
            return "not initialized"
        return "pyclnf"

    def _initialize_landmarks_from_bbox(self, bbox):
        """
        Initialize 2D landmarks from PDM mean shape scaled to fit bbox.

        This provides the initial landmark positions for CLNF optimization,
        following the OpenFace approach of starting from the mean shape.

        CRITICAL: Applies CLNF bbox correction to match C++ OpenFace behavior
        (from OpenFace C++ lines 386-396). This hardcoded correction adjusts
        the MTCNN bbox to CLNF's expected format.

        Args:
            bbox: Face bounding box [x_min, y_min, x_max, y_max]

        Returns:
            landmarks_2d: Initial 68-point landmarks (68, 2) as [x, y]
        """
        # Apply CLNF bbox correction (matching C++ OpenFace lines 386-396)
        # This converts MTCNN bbox format to what CLNF expects
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox
        bbox_width = bbox_x_max - bbox_x_min
        bbox_height = bbox_y_max - bbox_y_min

        # C++ correction coefficients
        corrected_x = bbox_x_min + bbox_width * -0.0075
        corrected_y = bbox_y_min + bbox_height * 0.2459
        corrected_width = bbox_width * 1.0323
        corrected_height = bbox_height * 0.7751

        # Use corrected bbox for initialization
        bbox_x_min = corrected_x
        bbox_y_min = corrected_y
        bbox_width = corrected_width
        bbox_height = corrected_height
        bbox_x_max = bbox_x_min + bbox_width
        bbox_y_max = bbox_y_min + bbox_height

        # Get PDM mean shape (204 values = 68 landmarks × 3 for x, y, z)
        # PDM format: [x0, y0, x1, y1, ..., x67, y67, z0, z1, ..., z67]
        # First 136 values are all X,Y coordinates
        mean_shape_2d = self.landmark_detector.pdm.mean_shape[:136].reshape(68, 2)

        # Compute bbox properties (using corrected bbox)
        bbox_center_x = (bbox_x_min + bbox_x_max) / 2
        bbox_center_y = (bbox_y_min + bbox_y_max) / 2

        # Compute mean shape properties
        mean_x_min = mean_shape_2d[:, 0].min()
        mean_x_max = mean_shape_2d[:, 0].max()
        mean_y_min = mean_shape_2d[:, 1].min()
        mean_y_max = mean_shape_2d[:, 1].max()
        mean_width = mean_x_max - mean_x_min
        mean_height = mean_y_max - mean_y_min
        mean_center_x = (mean_x_min + mean_x_max) / 2
        mean_center_y = (mean_y_min + mean_y_max) / 2

        # Compute scale to fit bbox (use smaller dimension for proper aspect ratio)
        scale_x = bbox_width / mean_width
        scale_y = bbox_height / mean_height
        scale = min(scale_x, scale_y)

        # Scale and translate mean shape to fit bbox
        landmarks_2d = mean_shape_2d.copy()
        landmarks_2d[:, 0] = (landmarks_2d[:, 0] - mean_center_x) * scale + bbox_center_x
        landmarks_2d[:, 1] = (landmarks_2d[:, 1] - mean_center_y) * scale + bbox_center_y

        return landmarks_2d

    def process_video(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> pd.DataFrame:
        """
        Process a video and extract AUs for all frames

        Args:
            video_path: Path to input video
            output_csv: Optional path to save CSV results
            max_frames: Optional limit on frames to process (for testing)
            progress_callback: Optional callback function(current, total, fps)
                             for progress updates to GUI

        Returns:
            DataFrame with columns: frame, timestamp, success, AU01_r, AU02_r, ...
        """
        # Reset stored features for new video processing
        self.stored_features = []

        # Use direct processing implementation
        return self._process_video_impl(video_path, output_csv, max_frames, progress_callback)


    def _process_video_impl(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> pd.DataFrame:
        """Internal implementation of video processing"""

        # Ensure components are initialized (lazy initialization)
        self._initialize_components()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        if self.verbose:
            safe_print(f"Processing video: {video_path.name}")
            safe_print("=" * 80)
            safe_print("")

        # Detect video rotation from metadata (important for mobile videos)
        rotation = get_video_rotation(str(video_path))
        if self.verbose and rotation != 0:
            safe_print(f"Detected video rotation: {rotation}°")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames:
            total_frames = min(total_frames, max_frames)

        if self.verbose:
            safe_print(f"Video info:")
            safe_print(f"  FPS: {fps:.2f}")
            safe_print(f"  Total frames: {total_frames}")
            safe_print(f"  Duration: {total_frames/fps:.2f} seconds")
            if rotation != 0:
                safe_print(f"  Rotation: {rotation}° (will be corrected)")
            safe_print("")

        # Results storage
        results = []
        frame_idx = 0

        # Statistics
        total_processed = 0
        total_failed = 0
        processing_start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

                # Apply rotation correction if needed
                if rotation != 0:
                    frame = apply_frame_rotation(frame, rotation)

                timestamp = frame_idx / fps

                # Process frame
                frame_result = self._process_frame(frame, frame_idx, timestamp)
                results.append(frame_result)

                if frame_result['success']:
                    total_processed += 1
                else:
                    total_failed += 1

                # Progress update
                if self.verbose and (frame_idx + 1) % 10 == 0:
                    progress = (frame_idx + 1) / total_frames * 100
                    safe_print(f"Progress: {frame_idx + 1}/{total_frames} frames ({progress:.1f}%) - "
                          f"Success: {total_processed}, Failed: {total_failed}", flush=True)

                # GUI progress callback (called every frame for smooth updates)
                if progress_callback is not None:
                    try:
                        # Calculate actual processing FPS (not video fps)
                        elapsed = time.time() - processing_start_time
                        actual_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0
                        progress_callback(frame_idx + 1, total_frames, actual_fps)
                    except Exception:
                        pass  # Don't let callback errors stop processing

                frame_idx += 1

        finally:
            cap.release()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Apply post-processing (cutoff adjustment, temporal smoothing)
        # This is CRITICAL for dynamic AU accuracy!
        if self.verbose:
            safe_print("\nApplying post-processing (cutoff adjustment, temporal smoothing)...")
        df = self.finalize_predictions(df)

        if self.verbose:
            safe_print("")
            safe_print("=" * 80)
            safe_print("PROCESSING COMPLETE")
            safe_print("=" * 80)
            safe_print(f"Total frames processed: {total_processed}")
            safe_print(f"Failed frames: {total_failed}")
            safe_print(f"Success rate: {total_processed/(total_processed+total_failed)*100:.1f}%")
            safe_print("")

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            if self.verbose:
                safe_print(f"Results saved to: {output_csv}")
                safe_print("")

        return df

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        return_debug: bool = False
    ) -> Dict:
        """
        Process a single frame through the complete pipeline

        Face Tracking Strategy (when enabled):
        - Frame 0: Run PyMTCNN detection, cache bbox
        - Frame 1+: Try cached bbox first
          - If landmark/alignment succeeds → keep using cached bbox
          - If landmark/alignment fails → re-run PyMTCNN, update cache

        This provides ~3x speedup by skipping expensive face detection!

        Args:
            frame: BGR image
            frame_idx: Frame index
            timestamp: Frame timestamp in seconds
            return_debug: If True, return debug info with component outputs

        Returns:
            Dictionary with frame results (success, AUs, etc.)
            If return_debug=True, also includes 'debug_info' key
        """
        # Ensure components are initialized (lazy initialization)
        self._initialize_components()

        result = {
            'frame': frame_idx,
            'timestamp': timestamp,
            'success': False
        }

        # Initialize debug info if requested
        debug_info = {} if (return_debug or self.debug_mode) else None

        try:
            bbox = None
            need_detection = True

            # Step 1: Face Detection (with tracking optimization)
            t0 = time.time() if debug_info is not None else None

            if self.track_faces and self.cached_bbox is not None:
                # Try using cached bbox (skip expensive PyMTCNN!)
                if self.verbose and frame_idx < 3:
                    safe_print(f"[Frame {frame_idx}] Step 1: Using cached bbox (tracking mode)")
                bbox = self.cached_bbox
                need_detection = False
                self.frames_since_detection += 1

            if need_detection or bbox is None:
                # First frame OR previous tracking failed - run PyMTCNN
                if self.verbose and frame_idx < 3:
                    safe_print(f"[Frame {frame_idx}] Step 1: Detecting face with {self.face_detector.backend}...")
                detections, _ = self.face_detector.detect_faces(frame)
                if self.verbose and frame_idx < 3:
                    safe_print(f"[Frame {frame_idx}] Step 1: Found {len(detections)} faces")

                if len(detections) == 0:
                    # No face detected - clear cache
                    self.cached_bbox = None
                    self.detection_failures += 1
                    if debug_info is not None:
                        debug_info['face_detection'] = {
                            'num_faces': 0,
                            'bbox': None,
                            'time_ms': (time.time() - t0) * 1000 if t0 else 0
                        }
                        result['debug_info'] = debug_info
                    return result

                # Use primary face (highest confidence)
                det = detections[0]
                bbox = det[:4].astype(int)  # [x1, y1, x2, y2]

                # Cache bbox for next frame
                if self.track_faces:
                    self.cached_bbox = bbox
                    self.frames_since_detection = 0

            if debug_info is not None:
                debug_info['face_detection'] = {
                    'num_faces': 1,
                    'bbox': bbox.copy(),
                    'cached': not need_detection,
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Step 2: Detect landmarks using CLNF (OpenFace approach)
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 2: Detecting landmarks with CLNF...")

            try:
                # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] for pyclnf
                bbox_x, bbox_y = bbox[0], bbox[1]
                bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                bbox_pyclnf = (bbox_x, bbox_y, bbox_w, bbox_h)

                # Detect landmarks with CLNF optimization
                # Pass detector_type='pymtcnn' so fit() applies the MTCNN bbox correction
                # CRITICAL: Use return_params=True to get params_local for geometric features
                # (Using CalcParams separately produces wrong scale - see GEOMETRIC_FEATURES_BUG.md)
                landmarks_68, info = self.landmark_detector.fit(frame, bbox_pyclnf, detector_type='pymtcnn', return_params=True)
                converged = info['converged']
                num_iterations = info['iterations']

                if self.verbose and frame_idx < 3:
                    safe_print(f"[Frame {frame_idx}] Step 2: Got {len(landmarks_68)} landmarks (CLNF converged: {converged}, iterations: {num_iterations})")

                if debug_info is not None:
                    debug_info['landmark_detection'] = {
                        'num_landmarks': len(landmarks_68),
                        'landmarks_68': landmarks_68.copy(),
                        'clnf_converged': converged,
                        'clnf_iterations': num_iterations,
                        'time_ms': (time.time() - t0) * 1000 if t0 else 0
                    }
            except Exception as e:
                # Landmark detection failed with cached bbox - re-run face detection
                if self.track_faces and not need_detection:
                    if self.verbose and frame_idx < 3:
                        safe_print(f"[Frame {frame_idx}] Step 2: Landmark detection failed with cached bbox, re-detecting face...")
                    self.detection_failures += 1
                    self.cached_bbox = None

                    # Re-run face detection
                    detections, _ = self.face_detector.detect_faces(frame)
                    if len(detections) == 0:
                        return result

                    det = detections[0]
                    bbox = det[:4].astype(int)  # [x1, y1, x2, y2]
                    self.cached_bbox = bbox
                    self.frames_since_detection = 0

                    # Retry landmark detection with new bbox
                    # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] for pyclnf
                    bbox_x, bbox_y = bbox[0], bbox[1]
                    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                    # Apply OpenFace MTCNN bbox calibration (matches C++ FaceDetectorMTCNN.cpp)
                    cal_x = bbox_x + bbox_w * (-0.0075)
                    cal_y = bbox_y + bbox_h * 0.2459
                    cal_w = bbox_w * 1.0323
                    cal_h = bbox_h * 0.7751
                    bbox_pyclnf = (cal_x, cal_y, cal_w, cal_h)

                    # CRITICAL: Use return_params=True to get params_local for geometric features
                    landmarks_68, info = self.landmark_detector.fit(frame, bbox_pyclnf, return_params=True)
                    converged = info['converged']
                    num_iterations = info['iterations']
                else:
                    # Not tracking or already re-detected - fail
                    raise

            # Step 3: Extract pose from pyclnf params
            # CRITICAL FIX: Use params from pyclnf optimization, NOT CalcParams!
            # CalcParams on raw landmarks produces wrong scale params_local (range -292 to 772)
            # pyclnf params_local has correct range (-29 to 32) matching C++ OpenFace
            # See GEOMETRIC_FEATURES_BUG.md for details
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 3: Extracting pose from pyclnf params...")

            if 'params' in info:
                # Use params from pyclnf CLNF optimization (CORRECT approach)
                clnf_params = info['params']
                params_global = clnf_params[:6]
                params_local = clnf_params[6:]

                # Extract pose parameters
                scale = params_global[0]
                rx, ry, rz = params_global[1:4]
                tx, ty = params_global[4:6]
            else:
                # Fallback: use bounding box for rough pose (shouldn't happen with return_params=True)
                scale = 1.0
                rx = ry = rz = 0.0
                tx = bbox[0] + bbox[2] / 2  # x + width/2
                ty = bbox[1] + bbox[3] / 2  # y + height/2
                params_local = np.zeros(34)

            if debug_info is not None:
                debug_info['pose_estimation'] = {
                    'scale': float(scale),
                    'rotation': [float(rx), float(ry), float(rz)],
                    'translation': [float(tx), float(ty)],
                    'params_local_shape': params_local.shape,
                    'params_local_range': [float(params_local.min()), float(params_local.max())],
                    'used_pyclnf_params': 'params' in info,
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Step 4: Align face
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 4: Aligning face...")
            aligned_face = self.face_aligner.align_face(
                image=frame,
                landmarks_68=landmarks_68,
                pose_tx=tx,
                pose_ty=ty,
                p_rz=rz,
                apply_mask=True,
                triangulation=self.triangulation
            )
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 4: Aligned face shape: {aligned_face.shape}")

            if debug_info is not None:
                debug_info['alignment'] = {
                    'aligned_face_shape': aligned_face.shape,
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Step 5: Extract HOG features
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 5: Extracting HOG features...")
            hog_features = pyfhog.extract_fhog_features(
                aligned_face,
                cell_size=8
            )
            # pyfhog 0.1.4+ outputs in OpenFace-compatible format (no transpose needed)
            # The HOG flattening order matches C++ OpenFace Face_utils.cpp line 265
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 5: HOG features shape: {hog_features.shape}")

            if debug_info is not None:
                debug_info['hog_extraction'] = {
                    'hog_shape': hog_features.shape,
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Step 6: Extract geometric features
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 6: Extracting geometric features...")
            geom_features = self.pdm_parser.extract_geometric_features(params_local)
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 6: Geometric features shape: {geom_features.shape}")

            # Ensure float32 for Cython compatibility
            hog_features = hog_features.astype(np.float32)
            geom_features = geom_features.astype(np.float32)

            if debug_info is not None:
                debug_info['geometric_extraction'] = {
                    'geom_shape': geom_features.shape,
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Step 7: Update running median
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 7: Updating running median...")
            # C++ increments frames_tracking BEFORE the check, so frame 0 → counter=1 → update
            # To match: update on frames 0, 2, 4, 6... (even frames)
            update_histogram = (frame_idx % 2 == 0)  # Match C++ timing
            self.running_median.update(hog_features, geom_features, update_histogram=update_histogram)
            running_median = self.running_median.get_combined_median()
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 7: Running median shape: {running_median.shape}")

            if debug_info is not None:
                debug_info['running_median'] = {
                    'median_shape': running_median.shape,
                    'update_histogram': update_histogram,
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Store features for two-pass processing (OpenFace reprocesses first 3000 frames)
            if frame_idx < self.max_stored_frames:
                self.stored_features.append((frame_idx, hog_features.copy(), geom_features.copy()))

            # Step 8: Predict AUs
            t0 = time.time() if debug_info is not None else None
            if self.verbose and frame_idx < 3:
                safe_print(f"[Frame {frame_idx}] Step 8: Predicting AUs...")
            au_results = self._predict_aus(
                hog_features,
                geom_features,
                running_median
            )

            if debug_info is not None:
                debug_info['au_prediction'] = {
                    'num_aus': len([k for k in au_results.keys() if k.startswith('AU')]),
                    'time_ms': (time.time() - t0) * 1000 if t0 else 0
                }

            # Step 9: Update prediction tracking (but DON'T apply online correction)
            # Fix 3: Online AU correction - C++ applies both online shift AND post-hoc cutoff
            # Configurable via AU_CONFIG['apply_online_dyn_shift']
            if self.online_au_correction is not None:
                # C++ applies 10% percentile shift during online processing (CorrectOnlineAUs)
                # AND applies per-model cutoffs post-hoc in ExtractAllPredictionsOfflineReg
                apply_shift = AU_CONFIG.get('apply_online_dyn_shift', False)
                au_results = self.online_au_correction.correct(
                    au_results,
                    update_track=True,
                    dyn_shift=apply_shift  # Apply online 10% shift if enabled
                )

            # Add AU predictions to result
            result.update(au_results)
            result['success'] = True

            # Add debug info to result if requested
            if debug_info is not None:
                result['debug_info'] = debug_info

        except Exception as e:
            if self.verbose:
                safe_print(f"Warning: Frame {frame_idx} failed: {e}")

        return result

    def _predict_aus(
        self,
        hog_features: np.ndarray,
        geom_features: np.ndarray,
        running_median: np.ndarray
    ) -> Dict[str, float]:
        """
        Predict AU intensities using SVR models

        Uses batched predictor if enabled (2-5x faster), otherwise falls back
        to sequential prediction.

        Args:
            hog_features: HOG feature vector (4464,)
            geom_features: Geometric feature vector (238,)
            running_median: Combined running median (4702,)

        Returns:
            Dictionary of AU predictions {AU_name: intensity}
        """
        # Use batched predictor if available (2-5x faster)
        if self.use_batched_predictor and self.batched_au_predictor is not None:
            return self.batched_au_predictor.predict(hog_features, geom_features, running_median)

        # Fallback to sequential prediction
        predictions = {}

        # Construct full feature vector
        full_vector = np.concatenate([hog_features, geom_features])

        for au_name, model in self.au_models.items():
            is_dynamic = (model['model_type'] == 'dynamic')

            # Center features
            if is_dynamic:
                centered = full_vector - model['means'].flatten() - running_median
            else:
                centered = full_vector - model['means'].flatten()

            # SVR prediction
            pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
            pred = float(pred[0, 0])

            # Clamp to [0, 5]
            pred = np.clip(pred, 0.0, 5.0)

            predictions[au_name] = pred

        return predictions

    def finalize_predictions(
        self,
        df: pd.DataFrame,
        max_init_frames: int = 3000
    ) -> pd.DataFrame:
        """
        Apply post-processing to AU predictions

        This includes:
        1. Two-pass processing (replace early frames with final median)
        2. Cutoff adjustment (person-specific calibration)
        3. Temporal smoothing (3-frame moving average)

        Args:
            df: DataFrame with raw AU predictions
            max_init_frames: Number of early frames to reprocess (default: 3000)

        Returns:
            DataFrame with finalized AU predictions
        """
        if self.verbose:
            safe_print("")
            safe_print("Applying post-processing...")
            safe_print("  [1/3] Two-pass median correction...")

        # Two-pass reprocessing: Re-predict AUs for early frames using final running median
        # This fixes systematic baseline offset from immature running median in early frames
        if len(self.stored_features) > 0:
            final_median = self.running_median.get_combined_median()

            if self.verbose:
                safe_print(f"    Re-predicting {len(self.stored_features)} early frames with final median...")

            # Re-predict AUs for stored frames
            for frame_idx, hog_features, geom_features in self.stored_features:
                # Re-predict AUs using final running median
                au_results = self._predict_aus(hog_features, geom_features, final_median)

                # Update DataFrame with re-predicted values
                for au_name, au_value in au_results.items():
                    df.loc[frame_idx, au_name] = au_value

            # Clear stored features to free memory
            self.stored_features = []

            if self.verbose:
                safe_print(f"    Two-pass correction complete")
        else:
            if self.verbose:
                safe_print("    (No stored features - skipping)")

        if self.verbose:
            safe_print("  [2/3] Cutoff adjustment...")

        # Apply cutoff adjustment for dynamic models
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        for au_col in au_cols:
            au_name = au_col
            if au_name not in self.au_models:
                continue

            model = self.au_models[au_name]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic:
                # Fix 1: Skip AU17 cutoff if configured (unusual weight distribution)
                # AU17 has lowest cutoff (20%) which is too aggressive
                if au_name == 'AU17_r' and AU_CONFIG.get('skip_au17_cutoff', False):
                    continue

                # Use the model's learned cutoff value (stored in the .dat file)
                # This is the percentile at which to compute the baseline offset
                # e.g., AU02 cutoff=0.75 means use 75th percentile as baseline
                # See SVR_dynamic_lin_regressors.cpp for how cutoff is used
                model_cutoff = model.get('cutoff', 0.0)

                # Check for AU-specific cutoff overrides (for AUs with systematic bias)
                cutoff_overrides = AU_CONFIG.get('cutoff_overrides', {})
                if au_name in cutoff_overrides:
                    model_cutoff = cutoff_overrides[au_name]

                # Skip if no valid cutoff (shouldn't happen for dynamic models)
                if model_cutoff <= 0 or model_cutoff >= 1.0:
                    continue

                # Match C++ - include ALL values from successful frames
                # C++ FaceAnalyser.cpp uses au_good which includes all values from frames
                # where successes[frame]==true, INCLUDING zeros
                au_values = df[au_col].values

                # Filter by success flag if available, otherwise use all non-NaN values
                if 'success' in df.columns:
                    valid_mask = df['success'].values == 1
                else:
                    valid_mask = ~np.isnan(au_values)
                valid_vals = au_values[valid_mask]

                # Need enough valid values to compute meaningful percentile
                if len(valid_vals) < 10:
                    continue

                sorted_vals = np.sort(valid_vals)
                cutoff_idx = int(len(sorted_vals) * model_cutoff)
                offset = sorted_vals[cutoff_idx]
                df[au_col] = np.clip(au_values - offset, 0.0, 5.0)

        if self.verbose:
            safe_print("  [3/3] Temporal smoothing...")

        # Apply 3-frame moving average
        for au_col in au_cols:
            smoothed = df[au_col].rolling(window=3, center=True, min_periods=1).mean()
            df[au_col] = smoothed

        if self.verbose:
            safe_print("Post-processing complete")

        return df


def main():
    """Command-line interface for full Python AU pipeline"""

    parser = argparse.ArgumentParser(
        description="Full Python AU Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python full_python_au_pipeline.py --video input.mp4 --output results.csv

  # Process first 100 frames only (for testing)
  python full_python_au_pipeline.py --video input.mp4 --max-frames 100

  # Use simplified pose estimation (faster, less accurate)
  python full_python_au_pipeline.py --video input.mp4 --simple-pose
        """
    )

    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', help='Output CSV file (default: <video>_aus.csv)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (for testing)')
    parser.add_argument('--simple-pose', action='store_true', help='Use simplified pose estimation')

    # Model paths (with defaults)
    parser.add_argument('--backend', default='auto',
                        choices=['auto', 'cuda', 'coreml', 'cpu', 'onnx'],
                        help='PyMTCNN backend (default: auto)')
    parser.add_argument('--pfld', default='weights/pfld_cunjian.onnx',
                        help='PFLD ONNX model path')
    parser.add_argument('--pdm', default='weights/In-the-wild_aligned_PDM_68.txt',
                        help='PDM shape model path')
    parser.add_argument('--au-models', default='weights/AU_predictors',
                        help='AU models directory')
    parser.add_argument('--triangulation', default='weights/tris_68_full.txt',
                        help='Triangulation file path')

    args = parser.parse_args()

    # Set default output path
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_python_aus.csv")

    # Initialize pipeline
    try:
        pipeline = FullPythonAUPipeline(
            pfld_model=args.pfld,
            pdm_file=args.pdm,
            au_models_dir=args.au_models,
            triangulation_file=args.triangulation,
            mtcnn_backend=args.backend,
            use_calc_params=not args.simple_pose,
            verbose=True
        )
    except Exception as e:
        safe_print(f"Failed to initialize pipeline: {e}")
        return 1

    # Process video
    try:
        df = pipeline.process_video(
            video_path=args.video,
            output_csv=args.output,
            max_frames=args.max_frames
        )

        # Apply post-processing
        df = pipeline.finalize_predictions(df)

        # Save final results
        df.to_csv(args.output, index=False)

        safe_print("=" * 80)
        safe_print("SUCCESS")
        safe_print("=" * 80)
        safe_print(f"Processed {len(df)} frames")
        safe_print(f"Results saved to: {args.output}")
        safe_print("")

        # Show AU statistics
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        if au_cols:
            safe_print("AU Statistics:")
            for au_col in sorted(au_cols):
                success_frames = df[df['success'] == True]
                if len(success_frames) > 0:
                    mean_val = success_frames[au_col].mean()
                    max_val = success_frames[au_col].max()
                    safe_print(f"  {au_col}: mean={mean_val:.3f}, max={max_val:.3f}")

        return 0

    except Exception as e:
        safe_print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
