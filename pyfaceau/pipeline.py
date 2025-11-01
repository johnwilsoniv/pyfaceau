#!/usr/bin/env python3
"""
Full Python AU Extraction Pipeline - End-to-End

This script integrates all Python components into a complete AU extraction pipeline:
1. Face Detection (RetinaFace ONNX)
2. Landmark Detection (Cunjian PFLD)
3. 3D Pose Estimation (CalcParams or simplified PnP)
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
from typing import Dict, List, Optional, Tuple
import argparse
import sys

# Import all pipeline components
from pyfaceau.detectors.retinaface import ONNXRetinaFaceDetector
from pyfaceau.detectors.pfld import CunjianPFLDDetector
from pyfaceau.alignment.calc_params import CalcParams
from pyfaceau.features.pdm import PDMParser
from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
from pyfaceau.features.triangulation import TriangulationParser
from pyfaceau.prediction.model_parser import OF22ModelParser
from pyfaceau.refinement.targeted_refiner import TargetedCLNFRefiner

# Import Cython-optimized running median (with fallback)
try:
    from cython_histogram_median import DualHistogramMedianTrackerCython as DualHistogramMedianTracker
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
        print("Error: pyfhog not found. Please install it:")
        print("   cd ../pyfhog && pip install -e .")
        sys.exit(1)


class FullPythonAUPipeline:
    """
    Complete Python AU extraction pipeline

    Integrates face detection, landmark detection, pose estimation,
    alignment, feature extraction, and AU prediction into a single
    end-to-end pipeline.
    """

    def __init__(
        self,
        retinaface_model: str,
        pfld_model: str,
        pdm_file: str,
        au_models_dir: str,
        triangulation_file: str,
        patch_expert_file: Optional[str] = None,
        use_calc_params: bool = True,
        use_coreml: bool = True,
        track_faces: bool = True,
        use_batched_predictor: bool = True,
        use_clnf_refinement: bool = True,
        enforce_clnf_pdm: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the full Python AU pipeline with lazy initialization for CoreML

        Args:
            retinaface_model: Path to RetinaFace ONNX model
            pfld_model: Path to PFLD ONNX model
            pdm_file: Path to PDM shape model
            au_models_dir: Directory containing AU SVR models
            triangulation_file: Path to triangulation file for masking
            patch_expert_file: Path to patch expert file for CLNF refinement (optional)
            use_calc_params: Use full CalcParams vs. simplified PnP (default: True)
            use_coreml: Enable CoreML Neural Engine acceleration (default: True)
            track_faces: Use face tracking (detect once, track between frames) (default: True)
            use_batched_predictor: Use optimized batched AU predictor (2-5x faster) (default: True)
            use_clnf_refinement: Enable CLNF landmark refinement (default: True)
            enforce_clnf_pdm: Enforce PDM constraints after CLNF refinement (default: False)
            verbose: Print progress messages (default: True)
        """
        import threading

        self.verbose = verbose
        self.use_calc_params = use_calc_params
        self.use_coreml = use_coreml
        self.track_faces = track_faces
        self.use_batched_predictor = use_batched_predictor and USING_BATCHED_PREDICTOR
        self.use_clnf_refinement = use_clnf_refinement
        self.enforce_clnf_pdm = enforce_clnf_pdm

        # Face tracking: cache bbox and only re-detect on failure (3x speedup!)
        self.cached_bbox = None
        self.detection_failures = 0
        self.frames_since_detection = 0

        # Store initialization parameters (lazy initialization for CoreML)
        self._init_params = {
            'retinaface_model': retinaface_model,
            'pfld_model': pfld_model,
            'pdm_file': pdm_file,
            'au_models_dir': au_models_dir,
            'triangulation_file': triangulation_file,
            'patch_expert_file': patch_expert_file,
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

        # Two-pass processing: Store features for early frames
        self.stored_features = []  # List of (frame_idx, hog_features, geom_features)
        self.max_stored_frames = 3000  # OpenFace default

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
                print("=" * 80)
                print("INITIALIZING COMPONENTS")
                print(f"Thread: {thread_name} (main={is_main})")
                print("=" * 80)
                print("")

            # Get initialization parameters
            retinaface_model = self._init_params['retinaface_model']
            pfld_model = self._init_params['pfld_model']
            pdm_file = self._init_params['pdm_file']
            au_models_dir = self._init_params['au_models_dir']
            triangulation_file = self._init_params['triangulation_file']

            # Component 1: Face Detection (with CoreML support)
            if self.verbose:
                print("[1/8] Loading face detector (RetinaFace ONNX)...")
                if self.use_coreml:
                    print("  CoreML: Enabled (initializing in worker thread)")
                else:
                    print("  CoreML: Disabled (CPU mode)")

            self.face_detector = ONNXRetinaFaceDetector(
                retinaface_model,
                use_coreml=self.use_coreml,
                confidence_threshold=0.5,
                nms_threshold=0.4
            )
            if self.verbose:
                print("Face detector loaded\n")

            # Component 2: Landmark Detection
            if self.verbose:
                print("[2/8] Loading landmark detector (PFLD)...")
            self.landmark_detector = CunjianPFLDDetector(pfld_model)
            if self.verbose:
                print(f"Landmark detector loaded: {self.landmark_detector}\n")

            # Component 3: PDM Parser (moved before CLNF to support PDM enforcement)
            if self.verbose:
                print("[3/8] Loading PDM shape model...")
            self.pdm_parser = PDMParser(pdm_file)
            if self.verbose:
                print(f"PDM loaded: {self.pdm_parser.mean_shape.shape[0]//3} landmarks\n")

            # Initialize CalcParams if using full pose estimation OR CLNF+PDM enforcement
            if self.use_calc_params or self.enforce_clnf_pdm:
                self.calc_params = CalcParams(self.pdm_parser)
            else:
                self.calc_params = None

            # Component 4: Face Aligner (needed for CLNF PDM enforcement)
            if self.verbose:
                print("[4/8] Initializing face aligner...")
            self.face_aligner = OpenFace22FaceAligner(
                pdm_file=pdm_file,
                sim_scale=0.7,
                output_size=(112, 112)
            )
            if self.verbose:
                print("Face aligner initialized\n")

            # Component 2.5: CLNF Refiner (optional, after PDM for constraint support)
            if self.use_clnf_refinement:
                if self.verbose:
                    print("[2.5/8] Loading CLNF landmark refiner...")
                patch_expert_file = self._init_params['patch_expert_file']
                if patch_expert_file is None:
                    patch_expert_file = 'weights/svr_patches_0.25_general.txt'

                # Pass CalcParams (has CalcParams/CalcShape methods) for PDM enforcement
                # Note: calc_params is initialized above if enforce_clnf_pdm is enabled
                pdm_for_clnf = self.calc_params if (self.enforce_clnf_pdm and self.calc_params) else None
                self.clnf_refiner = TargetedCLNFRefiner(
                    patch_expert_file,
                    search_window=3,
                    pdm=pdm_for_clnf,
                    enforce_pdm=self.enforce_clnf_pdm and (pdm_for_clnf is not None)
                )
                if self.verbose:
                    pdm_status = " + PDM constraints" if (self.enforce_clnf_pdm and pdm_for_clnf) else ""
                    print(f"CLNF refiner loaded with {len(self.clnf_refiner.patch_experts)} patch experts{pdm_status}\n")
            else:
                self.clnf_refiner = None

            # Component 5: Triangulation
            if self.verbose:
                print("[5/8] Loading triangulation...")
            self.triangulation = TriangulationParser(triangulation_file)
            if self.verbose:
                print(f"Triangulation loaded: {len(self.triangulation.triangles)} triangles\n")

            # Component 6: AU Models
            if self.verbose:
                print("[6/8] Loading AU SVR models...")
            model_parser = OF22ModelParser(au_models_dir)
            self.au_models = model_parser.load_all_models(
                use_recommended=True,
                use_combined=True
            )
            if self.verbose:
                print(f"Loaded {len(self.au_models)} AU models")

            # Initialize batched predictor if enabled
            if self.use_batched_predictor:
                self.batched_au_predictor = BatchedAUPredictor(self.au_models)
                if self.verbose:
                    print(f"Batched AU predictor enabled (2-5x faster)")
            if self.verbose:
                print("")

            # Component 7: Running Median Tracker
            if self.verbose:
                print("[7/8] Initializing running median tracker...")
            # Revert to original parameters while investigating C++ OpenFace histogram usage
            # TODO: Verify if C++ uses same histogram for HOG and geometric features
            self.running_median = DualHistogramMedianTracker(
                hog_dim=4464,
                geom_dim=238,
                hog_bins=1000,
                hog_min=-0.005,
                hog_max=1.0,
                geom_bins=10000,
                geom_min=-60.0,
                geom_max=60.0
            )
            if self.verbose:
                if USING_CYTHON:
                    print("Running median tracker initialized (Cython-optimized, 260x faster)\n")
                else:
                    print("Running median tracker initialized (Python version)\n")

            # Component 8: PyFHOG
            if self.verbose:
                print("[8/8] PyFHOG ready for HOG extraction")
                print("")
                print("All components initialized successfully")
                print("=" * 80)
                print("")

            self._components_initialized = True

    def process_video(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process a video and extract AUs for all frames

        ARCHITECTURE (CoreML mode):
        - Main thread: Opens VideoCapture (macOS NSRunLoop requirement), reads frames
        - Worker thread: Initializes CoreML, processes frames
        - Communication: Queues (main→worker: frames, worker→main: results)

        Args:
            video_path: Path to input video
            output_csv: Optional path to save CSV results
            max_frames: Optional limit on frames to process (for testing)

        Returns:
            DataFrame with columns: frame, timestamp, success, AU01_r, AU02_r, ...
        """
        import threading
        import queue

        # Reset stored features for new video processing
        self.stored_features = []

        # CoreML mode: Use queue-based architecture
        # Main thread handles VideoCapture, worker thread handles CoreML processing
        if self.use_coreml and threading.current_thread() == threading.main_thread():
            if self.verbose:
                print("[CoreML Mode] Queue-based processing:")
                print("  Main thread: VideoCapture (macOS NSRunLoop)")
                print("  Worker thread: CoreML + processing\n")

            # Validate path
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")

            # Open VideoCapture in MAIN thread (macOS requirement!)
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if max_frames:
                total_frames = min(total_frames, max_frames)

            if self.verbose:
                print(f"Processing video: {video_path.name}")
                print("=" * 80)
                print(f"Video info:")
                print(f"  FPS: {fps:.2f}")
                print(f"  Total frames: {total_frames}")
                print(f"  Duration: {total_frames/fps:.2f} seconds")
                print("")

            # Create queues for communication
            frame_queue = queue.Queue(maxsize=10)  # Limit buffering
            result_queue = queue.Queue()
            error_container = {'error': None}

            # Start worker thread for processing
            def worker():
                try:
                    self._process_frames_worker(frame_queue, result_queue, fps)
                except Exception as e:
                    error_container['error'] = e
                    import traceback
                    traceback.print_exc()

            worker_thread = threading.Thread(target=worker, daemon=False, name="CoreMLWorker")
            worker_thread.start()

            # Main thread: Read frames and send to worker
            frame_idx = 0
            try:
                if self.verbose:
                    print(f"[Main Thread] Reading frames from video...")
                while True:
                    ret, frame = cap.read()
                    if not ret or (max_frames and frame_idx >= max_frames):
                        break

                    timestamp = frame_idx / fps
                    if self.verbose and frame_idx < 3:
                        print(f"[Main Thread] Sending frame {frame_idx} to worker queue")
                    frame_queue.put((frame_idx, timestamp, frame))
                    frame_idx += 1

                if self.verbose:
                    print(f"[Main Thread] Finished reading {frame_idx} frames, sending termination signal")

            finally:
                # Signal worker to finish
                frame_queue.put(None)
                cap.release()
                if self.verbose:
                    print(f"[Main Thread] VideoCapture released, waiting for worker to finish...")

            # Wait for worker to complete
            worker_thread.join()
            if self.verbose:
                print(f"[Main Thread] Worker thread completed")

            # Check for errors
            if error_container['error']:
                raise error_container['error']

            # Collect results from worker
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())

            # Sort by frame index (in case of out-of-order processing)
            results.sort(key=lambda x: x['frame'])

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Apply post-processing (cutoff adjustment, temporal smoothing)
            # This is CRITICAL for dynamic AU accuracy!
            if self.verbose:
                print("\nApplying post-processing (cutoff adjustment, temporal smoothing)...")
            df = self.finalize_predictions(df)

            # Statistics
            total_processed = df['success'].sum()
            total_failed = len(df) - total_processed

            if self.verbose:
                print("")
                print("=" * 80)
                print("PROCESSING COMPLETE")
                print("=" * 80)
                print(f"Total frames processed: {total_processed}")
                print(f"Failed frames: {total_failed}")
                if len(df) > 0:
                    print(f"Success rate: {total_processed/len(df)*100:.1f}%")
                print("")

            # Save to CSV if requested
            if output_csv:
                df.to_csv(output_csv, index=False)
                if self.verbose:
                    print(f"Results saved to: {output_csv}")
                    print("")

            return df

        # Direct processing (CPU mode - no threading needed)
        return self._process_video_impl(video_path, output_csv, max_frames)

    def _process_frames_worker(self, frame_queue, result_queue, fps):
        """
        Worker thread method for processing frames (CoreML mode)

        This runs in a worker thread and:
        1. Initializes CoreML and all components (lazy init in worker thread)
        2. Receives frames from frame_queue
        3. Processes each frame
        4. Sends results to result_queue

        Args:
            frame_queue: Queue receiving (frame_idx, timestamp, frame) tuples from main thread
            result_queue: Queue for sending results back to main thread
            fps: Video framerate (for progress reporting)
        """
        import queue

        # Initialize all components in worker thread (CoreML loads here!)
        self._initialize_components()

        if self.verbose:
            print("[Worker Thread] Components initialized")
            if self.face_detector:
                print(f"[Worker Thread] Detector backend: {self.face_detector.backend}")
            print("[Worker Thread] Starting frame processing loop...")
            print("")

        # Process frames from queue
        total_processed = 0
        total_failed = 0

        while True:
            try:
                # Get frame from queue (blocks until available)
                if self.verbose and total_processed + total_failed < 3:
                    print(f"[Worker Thread] Waiting for frame from queue...")
                item = frame_queue.get(timeout=1.0)
                if self.verbose and total_processed + total_failed < 3:
                    print(f"[Worker Thread] Received item from queue")

                # Check for termination signal
                if item is None:
                    break

                frame_idx, timestamp, frame = item

                # Process frame
                frame_result = self._process_frame(frame, frame_idx, timestamp)
                result_queue.put(frame_result)

                # Update statistics
                if frame_result['success']:
                    total_processed += 1
                else:
                    total_failed += 1

                # Progress update
                if self.verbose and (frame_idx + 1) % 10 == 0:
                    print(f"[Worker] Processed {frame_idx + 1} frames - "
                          f"Success: {total_processed}, Failed: {total_failed}")

            except queue.Empty:
                # Timeout waiting for frame - continue
                continue
            except Exception as e:
                # Error processing frame - log and continue
                print(f"[Worker] Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                continue

        if self.verbose:
            print(f"\n[Worker Thread] Finished processing")
            print(f"  Total processed: {total_processed}")
            print(f"  Total failed: {total_failed}")

    def _process_video_impl(
        self,
        video_path: str,
        output_csv: Optional[str] = None,
        max_frames: Optional[int] = None
    ) -> pd.DataFrame:
        """Internal implementation of video processing"""

        # Ensure components are initialized (lazy initialization)
        self._initialize_components()

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
            print(f"  Duration: {total_frames/fps:.2f} seconds")
            print("")

        # Results storage
        results = []
        frame_idx = 0

        # Statistics
        total_processed = 0
        total_failed = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and frame_idx >= max_frames):
                    break

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
                    print(f"Progress: {frame_idx + 1}/{total_frames} frames ({progress:.1f}%) - "
                          f"Success: {total_processed}, Failed: {total_failed}")

                frame_idx += 1

        finally:
            cap.release()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Apply post-processing (cutoff adjustment, temporal smoothing)
        # This is CRITICAL for dynamic AU accuracy!
        if self.verbose:
            print("\nApplying post-processing (cutoff adjustment, temporal smoothing)...")
        df = self.finalize_predictions(df)

        if self.verbose:
            print("")
            print("=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)
            print(f"Total frames processed: {total_processed}")
            print(f"Failed frames: {total_failed}")
            print(f"Success rate: {total_processed/(total_processed+total_failed)*100:.1f}%")
            print("")

        # Save to CSV if requested
        if output_csv:
            df.to_csv(output_csv, index=False)
            if self.verbose:
                print(f"Results saved to: {output_csv}")
                print("")

        return df

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> Dict:
        """
        Process a single frame through the complete pipeline

        Face Tracking Strategy (when enabled):
        - Frame 0: Run RetinaFace detection, cache bbox
        - Frame 1+: Try cached bbox first
          - If landmark/alignment succeeds → keep using cached bbox
          - If landmark/alignment fails → re-run RetinaFace, update cache

        This provides ~3x speedup by skipping expensive face detection!

        Args:
            frame: BGR image
            frame_idx: Frame index
            timestamp: Frame timestamp in seconds

        Returns:
            Dictionary with frame results (success, AUs, etc.)
        """
        result = {
            'frame': frame_idx,
            'timestamp': timestamp,
            'success': False
        }

        try:
            bbox = None
            need_detection = True

            # Step 1: Face Detection (with tracking optimization)
            if self.track_faces and self.cached_bbox is not None:
                # Try using cached bbox (skip expensive RetinaFace!)
                if self.verbose and frame_idx < 3:
                    print(f"[Frame {frame_idx}] Step 1: Using cached bbox (tracking mode)")
                bbox = self.cached_bbox
                need_detection = False
                self.frames_since_detection += 1

            if need_detection or bbox is None:
                # First frame OR previous tracking failed - run RetinaFace
                if self.verbose and frame_idx < 3:
                    print(f"[Frame {frame_idx}] Step 1: Detecting face with {self.face_detector.backend}...")
                detections, _ = self.face_detector.detect_faces(frame)
                if self.verbose and frame_idx < 3:
                    print(f"[Frame {frame_idx}] Step 1: Found {len(detections)} faces")

                if len(detections) == 0:
                    # No face detected - clear cache
                    self.cached_bbox = None
                    self.detection_failures += 1
                    return result

                # Use primary face (highest confidence)
                det = detections[0]
                bbox = det[:4].astype(int)  # [x1, y1, x2, y2]

                # Cache bbox for next frame
                if self.track_faces:
                    self.cached_bbox = bbox
                    self.frames_since_detection = 0

            # Step 2: Detect landmarks
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 2: Detecting landmarks...")

            try:
                landmarks_68, _ = self.landmark_detector.detect_landmarks(frame, bbox)

                # Optional CLNF refinement for critical landmarks
                if self.use_clnf_refinement and self.clnf_refiner is not None:
                    landmarks_68 = self.clnf_refiner.refine_landmarks(frame, landmarks_68)

                if self.verbose and frame_idx < 3:
                    print(f"[Frame {frame_idx}] Step 2: Got {len(landmarks_68)} landmarks")
            except Exception as e:
                # Landmark detection failed with cached bbox - re-run face detection
                if self.track_faces and not need_detection:
                    if self.verbose and frame_idx < 3:
                        print(f"[Frame {frame_idx}] Step 2: Landmark detection failed with cached bbox, re-detecting face...")
                    self.detection_failures += 1
                    self.cached_bbox = None

                    # Re-run face detection
                    detections, _ = self.face_detector.detect_faces(frame)
                    if len(detections) == 0:
                        return result

                    det = detections[0]
                    bbox = det[:4].astype(int)
                    self.cached_bbox = bbox
                    self.frames_since_detection = 0

                    # Retry landmark detection with new bbox
                    landmarks_68, _ = self.landmark_detector.detect_landmarks(frame, bbox)

                    # Optional CLNF refinement for critical landmarks
                    if self.use_clnf_refinement and self.clnf_refiner is not None:
                        landmarks_68 = self.clnf_refiner.refine_landmarks(frame, landmarks_68)
                else:
                    # Not tracking or already re-detected - fail
                    raise

            # Step 3: Estimate 3D pose
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 3: Estimating 3D pose...")
            if self.use_calc_params and self.calc_params:
                # Full CalcParams optimization
                # Pass landmarks as (68, 2) array - CalcParams handles format conversion
                params_global, params_local = self.calc_params.calc_params(
                    landmarks_68
                )

                # Extract pose parameters
                scale = params_global[0]
                rx, ry, rz = params_global[1:4]
                tx, ty = params_global[4:6]
            else:
                # Simplified approach: use bounding box for rough pose
                # (This is a fallback - CalcParams is recommended)
                scale = 1.0
                rx = ry = rz = 0.0
                tx = (bbox[0] + bbox[2]) / 2
                ty = (bbox[1] + bbox[3]) / 2
                params_local = np.zeros(34)

            # Step 4: Align face
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 4: Aligning face...")
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
                print(f"[Frame {frame_idx}] Step 4: Aligned face shape: {aligned_face.shape}")

            # Step 5: Extract HOG features
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 5: Extracting HOG features...")
            hog_features = pyfhog.extract_fhog_features(
                aligned_face,
                cell_size=8
            )
            hog_features = hog_features.flatten()  # Should be 4464 dims
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 5: HOG features shape: {hog_features.shape}")

            # Step 6: Extract geometric features
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 6: Extracting geometric features...")
            geom_features = self.pdm_parser.extract_geometric_features(params_local)
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 6: Geometric features shape: {geom_features.shape}")

            # Ensure float32 for Cython compatibility
            hog_features = hog_features.astype(np.float32)
            geom_features = geom_features.astype(np.float32)

            # Step 7: Update running median
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 7: Updating running median...")
            update_histogram = (frame_idx % 2 == 1)  # Every 2nd frame
            self.running_median.update(hog_features, geom_features, update_histogram=update_histogram)
            running_median = self.running_median.get_combined_median()
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 7: Running median shape: {running_median.shape}")

            # Store features for two-pass processing (OpenFace reprocesses first 3000 frames)
            if frame_idx < self.max_stored_frames:
                self.stored_features.append((frame_idx, hog_features.copy(), geom_features.copy()))

            # Step 8: Predict AUs
            if self.verbose and frame_idx < 3:
                print(f"[Frame {frame_idx}] Step 8: Predicting AUs...")
            au_results = self._predict_aus(
                hog_features,
                geom_features,
                running_median
            )

            # Add AU predictions to result
            result.update(au_results)
            result['success'] = True

        except Exception as e:
            if self.verbose:
                print(f"Warning: Frame {frame_idx} failed: {e}")

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
            print("")
            print("Applying post-processing...")
            print("  [1/3] Two-pass median correction...")

        # Two-pass reprocessing: Re-predict AUs for early frames using final running median
        # This fixes systematic baseline offset from immature running median in early frames
        if len(self.stored_features) > 0:
            final_median = self.running_median.get_combined_median()

            if self.verbose:
                print(f"    Re-predicting {len(self.stored_features)} early frames with final median...")

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
                print(f"    Two-pass correction complete")
        else:
            if self.verbose:
                print("    (No stored features - skipping)")

        if self.verbose:
            print("  [2/3] Cutoff adjustment...")

        # Apply cutoff adjustment for dynamic models
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        for au_col in au_cols:
            au_name = au_col
            if au_name not in self.au_models:
                continue

            model = self.au_models[au_name]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic and model.get('cutoff', -1) != -1:
                cutoff = model['cutoff']
                au_values = df[au_col].values
                sorted_vals = np.sort(au_values)
                cutoff_idx = int(len(sorted_vals) * cutoff)
                offset = sorted_vals[cutoff_idx]
                df[au_col] = np.clip(au_values - offset, 0.0, 5.0)

        if self.verbose:
            print("  [3/3] Temporal smoothing...")

        # Apply 3-frame moving average
        for au_col in au_cols:
            smoothed = df[au_col].rolling(window=3, center=True, min_periods=1).mean()
            df[au_col] = smoothed

        if self.verbose:
            print("Post-processing complete")

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
    parser.add_argument('--retinaface', default='weights/retinaface_mobilenet025_coreml.onnx',
                        help='RetinaFace ONNX model path')
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
            retinaface_model=args.retinaface,
            pfld_model=args.pfld,
            pdm_file=args.pdm,
            au_models_dir=args.au_models,
            triangulation_file=args.triangulation,
            use_calc_params=not args.simple_pose,
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

        # Apply post-processing
        df = pipeline.finalize_predictions(df)

        # Save final results
        df.to_csv(args.output, index=False)

        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)
        print(f"Processed {len(df)} frames")
        print(f"Results saved to: {args.output}")
        print("")

        # Show AU statistics
        au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        if au_cols:
            print("AU Statistics:")
            for au_col in sorted(au_cols):
                success_frames = df[df['success'] == True]
                if len(success_frames) > 0:
                    mean_val = success_frames[au_col].mean()
                    max_val = success_frames[au_col].max()
                    print(f"  {au_col}: mean={mean_val:.3f}, max={max_val:.3f}")

        return 0

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
