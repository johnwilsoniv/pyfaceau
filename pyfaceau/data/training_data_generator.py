"""
Training Data Generator

Processes videos using the existing pyCLNF pipeline and extracts:
- Face crops (112x112)
- HOG features
- Landmarks
- Pose parameters (global_params)
- PDM parameters (local_params)
- AU intensities

This creates ground truth data from the current pipeline for training
neural network replacements.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union
from dataclasses import dataclass
import sys

from .hdf5_dataset import TrainingDataWriter, AU_NAMES
from .quality_filter import QualityFilter, QualityThresholds


@dataclass
class GeneratorConfig:
    """Configuration for training data generation."""
    # Paths
    pdm_path: str = "pyfaceau/weights/In-the-wild_aligned_PDM_68.txt"
    au_models_dir: str = "pyfaceau/weights/AU_predictors"
    triangulation_path: str = "pyfaceau/weights/tris_68_full.txt"

    # Processing
    output_size: Tuple[int, int] = (112, 112)
    sim_scale: float = 0.7

    # Quality filtering
    min_quality: float = 0.5
    skip_frames: int = 0  # Skip every N frames (0 = no skipping)

    # Output
    verbose: bool = True


class TrainingDataGenerator:
    """
    Generates training data from videos using the existing pipeline.

    Usage:
        generator = TrainingDataGenerator(config)
        generator.process_video('video.mp4', 'output.h5')

    Or for multiple videos:
        with TrainingDataWriter('dataset.h5') as writer:
            for video_path in video_paths:
                generator.process_video_to_writer(video_path, writer)
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize generator.

        Args:
            config: Generator configuration
        """
        self.config = config or GeneratorConfig()
        self._pipeline = None
        self._pyfhog = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize heavy dependencies."""
        if self._initialized:
            return

        # Import pyfhog
        try:
            import pyfhog
            self._pyfhog = pyfhog
        except ImportError:
            raise ImportError(
                "pyfhog is required for HOG feature extraction. "
                "Please ensure it's installed."
            )

        # Import pipeline components
        try:
            from pyfaceau.alignment.calc_params import CalcParams
            from pyfaceau.features.pdm import PDMParser
            from pyfaceau.alignment.face_aligner import OpenFace22FaceAligner
            from pyfaceau.prediction.model_parser import OF22ModelParser
            from pyfaceau.features.triangulation import TriangulationParser
        except ImportError as e:
            raise ImportError(f"Failed to import pyfaceau components: {e}")

        # Import face detector
        try:
            from pymtcnn import MTCNN
            self._face_detector = MTCNN()
        except ImportError:
            raise ImportError("pymtcnn is required for face detection.")

        # Import CLNF
        try:
            from pyclnf import CLNF
            self._clnf = CLNF()
        except ImportError:
            raise ImportError("pyclnf is required for landmark detection.")

        # Initialize components
        self._pdm_parser = PDMParser(self.config.pdm_path)
        self._calc_params = CalcParams(self._pdm_parser)
        self._face_aligner = OpenFace22FaceAligner(
            self.config.pdm_path,
            sim_scale=self.config.sim_scale,
            output_size=self.config.output_size
        )
        self._triangulation = TriangulationParser(self.config.triangulation_path)
        self._au_models = OF22ModelParser(self.config.au_models_dir).load_all_models(
            use_recommended=True, use_combined=True, verbose=False
        )

        self._initialized = True

    def _extract_hog(self, aligned_face: np.ndarray) -> np.ndarray:
        """Extract HOG features from aligned face."""
        hog = self._pyfhog.extract_fhog_features(aligned_face, cell_size=8)
        # pyfhog 0.1.4+ outputs in OpenFace-compatible format (4464,)
        return hog.astype(np.float32)

    def _crop_and_align_bbox(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ) -> tuple:
        """
        Crop face from image using bbox (matching NNCLNF inference alignment).

        This creates the SAME alignment that NNCLNF uses at inference time,
        ensuring training and inference see the same type of crops.

        Args:
            image: BGR image
            bbox: Face bounding box [x, y, width, height]

        Returns:
            aligned_face: 112x112 BGR face image
            warp_matrix: 2x3 affine transform matrix (for landmark transformation)
        """
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # Apply PyMTCNN bbox correction (matching NNCLNF)
        bbox_x = bbox_x + bbox_w * 0.0625
        bbox_y = bbox_y + bbox_h * 0.0625
        bbox_w = bbox_w * 0.875
        bbox_h = bbox_h * 0.875

        size = max(bbox_w, bbox_h)
        center_x = bbox_x + bbox_w / 2
        center_y = bbox_y + bbox_h / 2

        # Y offset correction (matching NNCLNF inference)
        center_y -= size * 0.12

        # Add padding (10% on each side)
        padding = 0.1
        size_with_pad = size * (1 + 2 * padding)

        x1 = center_x - size_with_pad / 2
        y1 = center_y - size_with_pad / 2
        x2 = center_x + size_with_pad / 2
        y2 = center_y + size_with_pad / 2

        src_pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
        ], dtype=np.float32)

        dst_pts = np.array([
            [0, 0],
            [112, 0],
            [112, 112],
        ], dtype=np.float32)

        M = cv2.getAffineTransform(src_pts, dst_pts)

        aligned_face = cv2.warpAffine(
            image, M, (112, 112),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return aligned_face, M

    def _predict_aus(self, hog_features: np.ndarray, geom_features: np.ndarray) -> np.ndarray:
        """Predict AU intensities from features."""
        full_vector = np.concatenate([hog_features, geom_features])

        # For training data, we use zero running median (static prediction)
        running_median = np.zeros_like(full_vector)

        au_values = []
        for au_name in AU_NAMES:
            if au_name not in self._au_models:
                au_values.append(0.0)
                continue

            model = self._au_models[au_name]
            is_dynamic = (model['model_type'] == 'dynamic')

            if is_dynamic:
                centered = full_vector - model['means'].flatten() - running_median
            else:
                centered = full_vector - model['means'].flatten()

            pred = np.dot(centered.reshape(1, -1), model['support_vectors']) + model['bias']
            au_values.append(float(np.clip(pred[0, 0], 0.0, 5.0)))

        return np.array(au_values, dtype=np.float32)

    def process_frame(
        self,
        frame: np.ndarray,
        prev_landmarks: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Process a single frame.

        Args:
            frame: BGR image
            prev_landmarks: Previous frame landmarks for temporal tracking

        Returns:
            Dictionary with extracted data, or None if processing failed
        """
        self._lazy_init()

        # Face detection
        try:
            # pymtcnn returns (bboxes, landmarks) not (bboxes, scores)
            bboxes, mtcnn_landmarks = self._face_detector.detect(frame)
            if bboxes is None or len(bboxes) == 0:
                if self.config.verbose:
                    print("    [DEBUG] No face detected")
                return None
            bbox = bboxes[0][:4].astype(np.float32)
            # MTCNN doesn't return confidence scores, use 1.0
            detection_confidence = 1.0
        except Exception as e:
            if self.config.verbose:
                import traceback
                print(f"    [DEBUG] Face detection error: {e}")
                traceback.print_exc()
            return None

        # Landmark detection with CLNF
        try:
            # CLNF API: fit(image, face_bbox, initial_params=None, return_params=False)
            # Note: We don't pass prev_landmarks as CLNF has its own initialization
            landmarks, info = self._clnf.fit(frame, bbox)
            landmarks = landmarks.astype(np.float32)
            convergence_success = info.get('converged', True)
        except Exception as e:
            if self.config.verbose:
                print(f"    [DEBUG] CLNF error: {e}")
            return None

        # Pose and PDM parameters
        try:
            global_params, local_params = self._calc_params.calc_params(landmarks)
            global_params = global_params.astype(np.float32)
            local_params = local_params.astype(np.float32)
        except Exception as e:
            if self.config.verbose:
                print(f"    [DEBUG] CalcParams error: {e}")
            return None

        # Face alignment - use bbox-based alignment (matching NNCLNF inference)
        # This ensures training and inference use the SAME alignment approach
        try:
            aligned_face, warp_matrix = self._crop_and_align_bbox(frame, bbox)
        except Exception as e:
            if self.config.verbose:
                print(f"    [DEBUG] Face alignment error: {e}")
            return None

        # HOG features
        try:
            hog_features = self._extract_hog(aligned_face)
        except Exception as e:
            if self.config.verbose:
                print(f"    [DEBUG] HOG extraction error: {e}")
            return None

        # Geometric features
        try:
            geom_features = self._pdm_parser.extract_geometric_features(local_params).astype(np.float32)
        except Exception as e:
            if self.config.verbose:
                print(f"    [DEBUG] Geometric features error: {e}")
            return None

        # AU prediction
        try:
            au_intensities = self._predict_aus(hog_features, geom_features)
        except Exception as e:
            if self.config.verbose:
                print(f"    [DEBUG] AU prediction error: {e}")
            return None

        # Convert BGR to RGB for neural network training (standard format)
        # HOG features were already computed from BGR image (which is fine)
        aligned_face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)

        return {
            'image': aligned_face_rgb,  # RGB format for NN training
            'hog_features': hog_features,
            'landmarks': landmarks,  # Original frame coordinates
            'global_params': global_params,
            'local_params': local_params,
            'au_intensities': au_intensities,
            'bbox': bbox,
            'warp_matrix': warp_matrix,  # Transform from original frame to aligned face
            'detection_confidence': detection_confidence,
            'convergence_success': convergence_success,
        }

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        max_frames: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process a video and save to HDF5.

        Args:
            video_path: Path to input video
            output_path: Path to output HDF5 file
            max_frames: Maximum frames to process (None = all)

        Returns:
            Statistics dictionary
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        quality_filter = QualityFilter()
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'filtered_frames': 0,
            'failed_frames': 0,
        }

        with TrainingDataWriter(output_path, expected_samples=total_frames) as writer:
            prev_landmarks = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if max_frames and frame_idx >= max_frames:
                    break

                stats['total_frames'] += 1

                # Skip frames if configured
                if self.config.skip_frames > 0 and frame_idx % (self.config.skip_frames + 1) != 0:
                    frame_idx += 1
                    continue

                # Process frame
                result = self.process_frame(frame, prev_landmarks)

                if result is None:
                    stats['failed_frames'] += 1
                    frame_idx += 1
                    continue

                prev_landmarks = result['landmarks']

                # Quality filtering
                should_include, quality_score, detailed_scores = quality_filter.should_include(
                    result['landmarks'],
                    result['global_params'],
                    result['bbox'],
                    result['detection_confidence'],
                    result['convergence_success'],
                    min_quality=self.config.min_quality
                )

                if not should_include:
                    if self.config.verbose and frame_idx < 5:
                        reject_reason = detailed_scores.get('reject_reason', f'low_quality:{quality_score:.3f}')
                        print(f"    [DEBUG] Frame {frame_idx} filtered: {reject_reason}")
                    stats['filtered_frames'] += 1
                    frame_idx += 1
                    continue

                # Add to dataset
                writer.add_sample(
                    image=result['image'],
                    hog_features=result['hog_features'],
                    landmarks=result['landmarks'],
                    global_params=result['global_params'],
                    local_params=result['local_params'],
                    au_intensities=result['au_intensities'],
                    bbox=result['bbox'],
                    warp_matrix=result['warp_matrix'],
                    video_name=video_path.name,
                    frame_index=frame_idx,
                    quality_score=quality_score
                )

                stats['processed_frames'] += 1
                frame_idx += 1

                if self.config.verbose and frame_idx % 100 == 0:
                    print(f"  Processed {frame_idx}/{total_frames} frames "
                          f"({stats['processed_frames']} saved, "
                          f"{stats['filtered_frames']} filtered, "
                          f"{stats['failed_frames']} failed)")

        cap.release()

        if self.config.verbose:
            print(f"\nStats for {video_path.name}:")
            print(f"  Total frames read: {stats['total_frames']}")
            print(f"  Failed (detection/CLNF): {stats['failed_frames']}")
            print(f"  Filtered (quality): {stats['filtered_frames']}")
            print(f"  Saved: {stats['processed_frames']}")

        return stats

    def process_video_to_writer(
        self,
        video_path: Union[str, Path],
        writer: TrainingDataWriter,
        max_frames: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Process a video and add to existing writer.

        Args:
            video_path: Path to input video
            writer: TrainingDataWriter instance
            max_frames: Maximum frames to process

        Returns:
            Statistics dictionary
        """
        video_path = Path(video_path)

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        quality_filter = QualityFilter()
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'filtered_frames': 0,
            'failed_frames': 0,
        }

        prev_landmarks = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_idx >= max_frames:
                break

            stats['total_frames'] += 1

            # Skip frames if configured
            if self.config.skip_frames > 0 and frame_idx % (self.config.skip_frames + 1) != 0:
                frame_idx += 1
                continue

            # Process frame
            result = self.process_frame(frame, prev_landmarks)

            if result is None:
                stats['failed_frames'] += 1
                frame_idx += 1
                continue

            prev_landmarks = result['landmarks']

            # Quality filtering
            should_include, quality_score, _ = quality_filter.should_include(
                result['landmarks'],
                result['global_params'],
                result['bbox'],
                result['detection_confidence'],
                result['convergence_success'],
                min_quality=self.config.min_quality
            )

            if not should_include:
                stats['filtered_frames'] += 1
                frame_idx += 1
                continue

            # Add to dataset
            writer.add_sample(
                image=result['image'],
                hog_features=result['hog_features'],
                landmarks=result['landmarks'],
                global_params=result['global_params'],
                local_params=result['local_params'],
                au_intensities=result['au_intensities'],
                bbox=result['bbox'],
                warp_matrix=result['warp_matrix'],
                video_name=video_path.name,
                frame_index=frame_idx,
                quality_score=quality_score
            )

            stats['processed_frames'] += 1
            frame_idx += 1

            if self.config.verbose and frame_idx % 100 == 0:
                print(f"  {video_path.name}: {frame_idx}/{total_frames} frames")

        cap.release()
        quality_filter.reset()

        return stats

    def process_multiple_videos(
        self,
        video_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        max_frames_per_video: Optional[int] = None
    ) -> Dict[str, Dict[str, int]]:
        """
        Process multiple videos into a single HDF5 file.

        Args:
            video_paths: List of video paths
            output_path: Output HDF5 path
            max_frames_per_video: Maximum frames per video

        Returns:
            Dictionary mapping video names to their stats
        """
        output_path = Path(output_path)

        # Estimate total samples
        total_expected = 0
        for vp in video_paths:
            cap = cv2.VideoCapture(str(vp))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames_per_video:
                frames = min(frames, max_frames_per_video)
            total_expected += frames
            cap.release()

        all_stats = {}

        with TrainingDataWriter(output_path, expected_samples=total_expected) as writer:
            for i, video_path in enumerate(video_paths):
                video_path = Path(video_path)
                if self.config.verbose:
                    print(f"\n[{i+1}/{len(video_paths)}] Processing {video_path.name}...")

                stats = self.process_video_to_writer(
                    video_path, writer, max_frames_per_video
                )
                all_stats[video_path.name] = stats

        # Print summary
        if self.config.verbose:
            total_processed = sum(s['processed_frames'] for s in all_stats.values())
            total_filtered = sum(s['filtered_frames'] for s in all_stats.values())
            total_failed = sum(s['failed_frames'] for s in all_stats.values())
            print(f"\n{'='*60}")
            print(f"SUMMARY: {len(video_paths)} videos processed")
            print(f"  Total frames saved: {total_processed}")
            print(f"  Total frames filtered: {total_filtered}")
            print(f"  Total frames failed: {total_failed}")
            print(f"  Output: {output_path}")

        return all_stats
