"""
HDF5 Dataset for Neural Network Training

Provides efficient storage and retrieval of training data:
- Face images (112x112x3) uint8 in RGB format (standard for neural networks)
- HOG features (4464,) float32
- Landmarks (68, 2) float32
- Pose parameters (6,) float32 - global params [scale, rx, ry, rz, tx, ty]
- PDM parameters (34,) float32 - local params
- AU intensities (17,) float32
- Bounding boxes (4,) float32

NOTE: Images are stored in RGB format (converted from OpenCV's BGR)
for compatibility with PyTorch/TensorFlow neural network training.

Storage structure:
    training_data.h5
    ├── metadata/
    │   ├── video_names (N,) - string array
    │   ├── frame_indices (N,) - int32
    │   └── quality_scores (N,) - float32
    ├── images (N, 112, 112, 3) uint8
    ├── hog_features (N, 4464) float32
    ├── landmarks (N, 68, 2) float32
    ├── global_params (N, 6) float32
    ├── local_params (N, 34) float32
    ├── au_intensities (N, 17) float32
    └── bboxes (N, 4) float32
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import io
import cv2


# Constants
IMAGE_SIZE = (112, 112)
HOG_DIM = 4464
NUM_LANDMARKS = 68
NUM_GLOBAL_PARAMS = 6
NUM_LOCAL_PARAMS = 34
NUM_AUS = 17

AU_NAMES = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
    'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
]


class TrainingDataWriter:
    """
    Writer for creating HDF5 training datasets.

    Usage:
        with TrainingDataWriter('training_data.h5', expected_samples=100000) as writer:
            for frame_data in process_video(video_path):
                writer.add_sample(frame_data)
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        expected_samples: int = 100000,
        chunk_size: int = 1000,
        compression: str = 'gzip',
        compression_level: int = 4
    ):
        """
        Initialize HDF5 writer.

        Args:
            output_path: Path to output HDF5 file
            expected_samples: Expected number of samples (for preallocation)
            chunk_size: Chunk size for HDF5 datasets
            compression: Compression algorithm ('gzip', 'lzf', or None)
            compression_level: Compression level (1-9 for gzip)
        """
        self.output_path = Path(output_path)
        self.expected_samples = expected_samples
        self.chunk_size = chunk_size
        self.compression = compression
        self.compression_level = compression_level

        self.h5file = None
        self.current_index = 0
        self.video_names = []
        self.frame_indices = []
        self.quality_scores = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Open HDF5 file and create datasets."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.h5file = h5py.File(self.output_path, 'w')

        # Compression options
        comp_opts = {}
        if self.compression:
            comp_opts['compression'] = self.compression
            if self.compression == 'gzip':
                comp_opts['compression_opts'] = self.compression_level

        # Create datasets with chunking and compression
        self.h5file.create_dataset(
            'images',
            shape=(self.expected_samples, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            maxshape=(None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            dtype=np.uint8,
            chunks=(self.chunk_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            **comp_opts
        )

        self.h5file.create_dataset(
            'hog_features',
            shape=(self.expected_samples, HOG_DIM),
            maxshape=(None, HOG_DIM),
            dtype=np.float32,
            chunks=(self.chunk_size, HOG_DIM),
            **comp_opts
        )

        self.h5file.create_dataset(
            'landmarks',
            shape=(self.expected_samples, NUM_LANDMARKS, 2),
            maxshape=(None, NUM_LANDMARKS, 2),
            dtype=np.float32,
            chunks=(self.chunk_size, NUM_LANDMARKS, 2),
            **comp_opts
        )

        self.h5file.create_dataset(
            'global_params',
            shape=(self.expected_samples, NUM_GLOBAL_PARAMS),
            maxshape=(None, NUM_GLOBAL_PARAMS),
            dtype=np.float32,
            chunks=(self.chunk_size, NUM_GLOBAL_PARAMS),
            **comp_opts
        )

        self.h5file.create_dataset(
            'local_params',
            shape=(self.expected_samples, NUM_LOCAL_PARAMS),
            maxshape=(None, NUM_LOCAL_PARAMS),
            dtype=np.float32,
            chunks=(self.chunk_size, NUM_LOCAL_PARAMS),
            **comp_opts
        )

        self.h5file.create_dataset(
            'au_intensities',
            shape=(self.expected_samples, NUM_AUS),
            maxshape=(None, NUM_AUS),
            dtype=np.float32,
            chunks=(self.chunk_size, NUM_AUS),
            **comp_opts
        )

        self.h5file.create_dataset(
            'bboxes',
            shape=(self.expected_samples, 4),
            maxshape=(None, 4),
            dtype=np.float32,
            chunks=(self.chunk_size, 4),
            **comp_opts
        )

        # Warp matrix for transforming landmarks from original frame to aligned face
        # Shape: (2, 3) affine transform matrix per sample
        self.h5file.create_dataset(
            'warp_matrices',
            shape=(self.expected_samples, 2, 3),
            maxshape=(None, 2, 3),
            dtype=np.float32,
            chunks=(self.chunk_size, 2, 3),
            **comp_opts
        )

        # Store AU names and format info as attributes
        self.h5file.attrs['au_names'] = AU_NAMES
        self.h5file.attrs['image_size'] = IMAGE_SIZE
        self.h5file.attrs['color_format'] = 'RGB'  # Images stored in RGB format

    def add_sample(
        self,
        image: np.ndarray,
        hog_features: np.ndarray,
        landmarks: np.ndarray,
        global_params: np.ndarray,
        local_params: np.ndarray,
        au_intensities: np.ndarray,
        bbox: np.ndarray,
        warp_matrix: np.ndarray = None,
        video_name: str = '',
        frame_index: int = 0,
        quality_score: float = 1.0
    ):
        """
        Add a single sample to the dataset.

        Args:
            image: Face image (112, 112, 3) uint8
            hog_features: HOG features (4464,) float32
            landmarks: 2D landmarks (68, 2) float32 in ORIGINAL frame coordinates
            global_params: Pose parameters (6,) float32
            local_params: PDM shape parameters (34,) float32
            au_intensities: AU intensities (17,) float32
            bbox: Face bounding box (4,) float32
            warp_matrix: Affine transform (2, 3) from original frame to aligned face
            video_name: Source video name
            frame_index: Frame index in source video
            quality_score: Quality score (0-1)
        """
        if self.current_index >= self.expected_samples:
            # Resize datasets
            new_size = self.current_index + self.chunk_size
            for name in ['images', 'hog_features', 'landmarks', 'global_params',
                        'local_params', 'au_intensities', 'bboxes', 'warp_matrices']:
                self.h5file[name].resize(new_size, axis=0)
            self.expected_samples = new_size

        idx = self.current_index

        # Validate and store data
        self.h5file['images'][idx] = image.astype(np.uint8)
        self.h5file['hog_features'][idx] = hog_features.astype(np.float32).flatten()
        self.h5file['landmarks'][idx] = landmarks.astype(np.float32).reshape(NUM_LANDMARKS, 2)
        self.h5file['global_params'][idx] = global_params.astype(np.float32).flatten()[:NUM_GLOBAL_PARAMS]
        self.h5file['local_params'][idx] = local_params.astype(np.float32).flatten()[:NUM_LOCAL_PARAMS]
        self.h5file['au_intensities'][idx] = au_intensities.astype(np.float32).flatten()[:NUM_AUS]
        self.h5file['bboxes'][idx] = bbox.astype(np.float32).flatten()[:4]

        # Store warp matrix (identity if not provided for backwards compatibility)
        if warp_matrix is not None:
            self.h5file['warp_matrices'][idx] = warp_matrix.astype(np.float32).reshape(2, 3)
        else:
            # Identity transform as fallback
            self.h5file['warp_matrices'][idx] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Store metadata
        self.video_names.append(video_name)
        self.frame_indices.append(frame_index)
        self.quality_scores.append(quality_score)

        self.current_index += 1

    def add_batch(
        self,
        images: np.ndarray,
        hog_features: np.ndarray,
        landmarks: np.ndarray,
        global_params: np.ndarray,
        local_params: np.ndarray,
        au_intensities: np.ndarray,
        bboxes: np.ndarray,
        video_names: List[str] = None,
        frame_indices: List[int] = None,
        quality_scores: List[float] = None
    ):
        """Add a batch of samples efficiently."""
        batch_size = len(images)

        # Ensure capacity
        if self.current_index + batch_size > self.expected_samples:
            new_size = self.current_index + batch_size + self.chunk_size
            for name in ['images', 'hog_features', 'landmarks', 'global_params',
                        'local_params', 'au_intensities', 'bboxes']:
                self.h5file[name].resize(new_size, axis=0)
            self.expected_samples = new_size

        start_idx = self.current_index
        end_idx = start_idx + batch_size

        # Batch write
        self.h5file['images'][start_idx:end_idx] = images.astype(np.uint8)
        self.h5file['hog_features'][start_idx:end_idx] = hog_features.astype(np.float32)
        self.h5file['landmarks'][start_idx:end_idx] = landmarks.astype(np.float32)
        self.h5file['global_params'][start_idx:end_idx] = global_params.astype(np.float32)
        self.h5file['local_params'][start_idx:end_idx] = local_params.astype(np.float32)
        self.h5file['au_intensities'][start_idx:end_idx] = au_intensities.astype(np.float32)
        self.h5file['bboxes'][start_idx:end_idx] = bboxes.astype(np.float32)

        # Metadata
        if video_names:
            self.video_names.extend(video_names)
        else:
            self.video_names.extend([''] * batch_size)

        if frame_indices:
            self.frame_indices.extend(frame_indices)
        else:
            self.frame_indices.extend([0] * batch_size)

        if quality_scores:
            self.quality_scores.extend(quality_scores)
        else:
            self.quality_scores.extend([1.0] * batch_size)

        self.current_index += batch_size

    def close(self):
        """Finalize and close the HDF5 file."""
        if self.h5file is None:
            return

        # Truncate to actual size
        actual_size = self.current_index
        for name in ['images', 'hog_features', 'landmarks', 'global_params',
                    'local_params', 'au_intensities', 'bboxes', 'warp_matrices']:
            self.h5file[name].resize(actual_size, axis=0)

        # Create metadata group
        metadata = self.h5file.create_group('metadata')

        # Store video names as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        metadata.create_dataset('video_names', data=self.video_names, dtype=dt)
        metadata.create_dataset('frame_indices', data=np.array(self.frame_indices, dtype=np.int32))
        metadata.create_dataset('quality_scores', data=np.array(self.quality_scores, dtype=np.float32))

        # Store total count
        self.h5file.attrs['num_samples'] = actual_size

        self.h5file.close()
        self.h5file = None

        print(f"Saved {actual_size} samples to {self.output_path}")


class TrainingDataset:
    """
    Reader for HDF5 training datasets.

    Can be used directly or wrapped with PyTorch DataLoader.

    Usage:
        dataset = TrainingDataset('training_data.h5')
        for i in range(len(dataset)):
            sample = dataset[i]
            image = sample['image']
            landmarks = sample['landmarks']
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        load_images: bool = True,
        load_hog: bool = True,
        transform = None
    ):
        """
        Initialize dataset reader.

        Args:
            h5_path: Path to HDF5 file
            load_images: Whether to load images (can disable for speed)
            load_hog: Whether to load HOG features
            transform: Optional transform to apply to samples
        """
        self.h5_path = Path(h5_path)
        self.load_images = load_images
        self.load_hog = load_hog
        self.transform = transform

        # Open file and get metadata
        self.h5file = h5py.File(self.h5_path, 'r')
        self.num_samples = self.h5file.attrs['num_samples']
        self.au_names = list(self.h5file.attrs['au_names'])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample by index."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        sample = {
            'landmarks': self.h5file['landmarks'][idx],
            'global_params': self.h5file['global_params'][idx],
            'local_params': self.h5file['local_params'][idx],
            'au_intensities': self.h5file['au_intensities'][idx],
            'bbox': self.h5file['bboxes'][idx],
        }

        if self.load_images:
            sample['image'] = self.h5file['images'][idx]

        if self.load_hog:
            sample['hog_features'] = self.h5file['hog_features'][idx]

        # Add metadata
        sample['video_name'] = self.h5file['metadata/video_names'][idx]
        sample['frame_index'] = self.h5file['metadata/frame_indices'][idx]
        sample['quality_score'] = self.h5file['metadata/quality_scores'][idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Get a batch of samples by indices."""
        indices = np.array(indices)

        batch = {
            'landmarks': self.h5file['landmarks'][indices],
            'global_params': self.h5file['global_params'][indices],
            'local_params': self.h5file['local_params'][indices],
            'au_intensities': self.h5file['au_intensities'][indices],
            'bboxes': self.h5file['bboxes'][indices],
        }

        if self.load_images:
            batch['images'] = self.h5file['images'][indices]

        if self.load_hog:
            batch['hog_features'] = self.h5file['hog_features'][indices]

        return batch

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute dataset statistics for normalization."""
        stats = {}

        # Landmarks
        landmarks = self.h5file['landmarks'][:]
        stats['landmarks'] = {
            'mean': landmarks.mean(axis=0),
            'std': landmarks.std(axis=0),
            'min': landmarks.min(axis=0),
            'max': landmarks.max(axis=0),
        }

        # Global params
        global_params = self.h5file['global_params'][:]
        stats['global_params'] = {
            'mean': global_params.mean(axis=0),
            'std': global_params.std(axis=0),
            'min': global_params.min(axis=0),
            'max': global_params.max(axis=0),
        }

        # Local params
        local_params = self.h5file['local_params'][:]
        stats['local_params'] = {
            'mean': local_params.mean(axis=0),
            'std': local_params.std(axis=0),
            'min': local_params.min(axis=0),
            'max': local_params.max(axis=0),
        }

        # AU intensities
        au_intensities = self.h5file['au_intensities'][:]
        stats['au_intensities'] = {
            'mean': au_intensities.mean(axis=0),
            'std': au_intensities.std(axis=0),
            'min': au_intensities.min(axis=0),
            'max': au_intensities.max(axis=0),
        }

        return stats

    def close(self):
        """Close the HDF5 file."""
        if self.h5file:
            self.h5file.close()
            self.h5file = None

    def __del__(self):
        self.close()


# PyTorch Dataset wrapper (optional, for training)
try:
    from torch.utils.data import Dataset as TorchDataset, Subset
    import torch

    class PyTorchTrainingDataset(TorchDataset):
        """
        PyTorch-compatible wrapper for HDF5 training data.

        Pre-loads all data into memory at initialization for fast access.
        This enables efficient multi-worker DataLoader usage since workers
        read from RAM arrays rather than compressed HDF5.

        Supports data augmentation for training:
        - Horizontal flip (with landmark mirroring)
        - Brightness/contrast adjustment
        - Color jitter
        - Small rotation (±5°)
        """

        # Landmark indices for horizontal flip mirroring (68-point model)
        # Maps left side landmarks to right side and vice versa
        FLIP_INDICES = np.array([
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  # Jaw (0-16)
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  # Eyebrows (17-26)
            27, 28, 29, 30,  # Nose bridge (27-30)
            35, 34, 33, 32, 31,  # Nose bottom (31-35)
            45, 44, 43, 42, 47, 46,  # Right eye (36-41) -> Left eye
            39, 38, 37, 36, 41, 40,  # Left eye (42-47) -> Right eye
            54, 53, 52, 51, 50, 49, 48,  # Outer lip top (48-54)
            59, 58, 57, 56, 55,  # Outer lip bottom (55-59)
            64, 63, 62, 61, 60,  # Inner lip top (60-64)
            67, 66, 65,  # Inner lip bottom (65-67)
        ])

        def __init__(
            self,
            h5_path: Union[str, Path],
            load_images: bool = True,
            load_hog: bool = False,
            augment: bool = False,
            augment_prob: float = 0.5
        ):
            import time
            start_time = time.time()
            print(f"[Dataset] Loading all data into memory from {h5_path}...", flush=True)

            self.load_images = load_images
            self.load_hog = load_hog
            self.augment = augment
            self.augment_prob = augment_prob

            # Load everything into memory at once
            with h5py.File(str(h5_path), 'r') as f:
                self.num_samples = f.attrs['num_samples']
                self.au_names = list(f.attrs['au_names'])

                print(f"[Dataset] Loading {self.num_samples} samples...", flush=True)

                # Load all arrays into memory (this decompresses once)
                print(f"[Dataset]   Loading landmarks...", flush=True)
                self.landmarks = f['landmarks'][:]
                print(f"[Dataset]   Loading global_params...", flush=True)
                self.global_params = f['global_params'][:]
                print(f"[Dataset]   Loading local_params...", flush=True)
                self.local_params = f['local_params'][:]
                print(f"[Dataset]   Loading au_intensities...", flush=True)
                self.au_intensities = f['au_intensities'][:]

                if load_images and 'images' in f:
                    print(f"[Dataset]   Loading images (this may take a moment)...", flush=True)
                    self.images = f['images'][:]
                else:
                    self.images = None

                if load_hog and 'hog_features' in f:
                    print(f"[Dataset]   Loading hog_features...", flush=True)
                    self.hog_features = f['hog_features'][:]
                else:
                    self.hog_features = None

                # Load warp matrices for landmark transformation
                if 'warp_matrices' in f:
                    print(f"[Dataset]   Loading warp_matrices...", flush=True)
                    self.warp_matrices = f['warp_matrices'][:]
                else:
                    print(f"[Dataset]   WARNING: warp_matrices not found, landmarks will be approximate", flush=True)
                    self.warp_matrices = None

                # Load video indices for video-stratified splitting
                # Support both formats: video_indices (int array) or metadata/video_names (string array)
                if 'video_indices' in f:
                    print(f"[Dataset]   Loading video_indices...", flush=True)
                    self.video_indices = f['video_indices'][:]
                    self.video_names = None  # Use indices instead
                    n_videos = len(np.unique(self.video_indices))
                    print(f"[Dataset]   Found {n_videos} unique videos", flush=True)
                elif 'metadata/video_names' in f:
                    print(f"[Dataset]   Loading video_names...", flush=True)
                    self.video_names = f['metadata/video_names'][:]
                    self.video_indices = None
                else:
                    print(f"[Dataset]   WARNING: video info not found", flush=True)
                    self.video_names = None
                    self.video_indices = None

            elapsed = time.time() - start_time
            print(f"[Dataset] All data loaded into memory in {elapsed:.1f}s", flush=True)

            # Report memory usage
            mem_mb = (
                self.landmarks.nbytes +
                self.global_params.nbytes +
                self.local_params.nbytes +
                self.au_intensities.nbytes +
                (self.images.nbytes if self.images is not None else 0) +
                (self.hog_features.nbytes if self.hog_features is not None else 0) +
                (self.warp_matrices.nbytes if self.warp_matrices is not None else 0)
            ) / (1024 * 1024)
            print(f"[Dataset] Memory usage: {mem_mb:.1f} MB", flush=True)

            if self.augment:
                print(f"[Dataset] Augmentation enabled (prob={augment_prob})", flush=True)

        def __len__(self):
            return self.num_samples

        def get_unique_videos(self) -> List[str]:
            """Get list of unique video names in the dataset."""
            if self.video_names is None:
                return []
            # Handle both bytes and string types
            videos = []
            for v in self.video_names:
                if isinstance(v, bytes):
                    videos.append(v.decode('utf-8'))
                else:
                    videos.append(str(v))
            return list(set(videos))

        def get_indices_by_video(self) -> Dict[str, List[int]]:
            """Get sample indices grouped by video."""
            # Use video_indices (int array) if available
            if self.video_indices is not None:
                video_to_indices = {}
                for idx, vid in enumerate(self.video_indices):
                    vid_str = str(int(vid))
                    if vid_str not in video_to_indices:
                        video_to_indices[vid_str] = []
                    video_to_indices[vid_str].append(idx)
                return video_to_indices

            # Fall back to video_names (string array)
            if self.video_names is not None:
                video_to_indices = {}
                for idx, v in enumerate(self.video_names):
                    if isinstance(v, bytes):
                        v = v.decode('utf-8')
                    else:
                        v = str(v)
                    if v not in video_to_indices:
                        video_to_indices[v] = []
                    video_to_indices[v].append(idx)
                return video_to_indices

            # No video info available
            return {'unknown': list(range(self.num_samples))}

        def _apply_augmentation(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Apply random augmentations to image and landmarks."""
            # Image is (C, H, W) float32 [0, 1], landmarks are (68, 2)

            # Horizontal flip (50% chance when augment is enabled)
            if np.random.random() < 0.5:
                image = np.ascontiguousarray(image[:, :, ::-1])  # Flip horizontally
                landmarks = landmarks.copy()
                landmarks[:, 0] = 112.0 - landmarks[:, 0]  # Flip x coordinates
                landmarks = landmarks[self.FLIP_INDICES]  # Reorder for symmetry

            # Brightness adjustment (±20%)
            if np.random.random() < 0.5:
                factor = 1.0 + np.random.uniform(-0.2, 0.2)
                image = np.clip(image * factor, 0, 1)

            # Contrast adjustment (±20%)
            if np.random.random() < 0.5:
                factor = 1.0 + np.random.uniform(-0.2, 0.2)
                mean = image.mean()
                image = np.clip((image - mean) * factor + mean, 0, 1)

            # Color jitter - adjust each channel independently (±10%)
            if np.random.random() < 0.3:
                for c in range(3):
                    factor = 1.0 + np.random.uniform(-0.1, 0.1)
                    image[c] = np.clip(image[c] * factor, 0, 1)

            # Small rotation (±5°) - rotate both image and landmarks
            if np.random.random() < 0.3:
                angle = np.random.uniform(-5, 5)
                # Rotation matrix around center (56, 56)
                center = 56.0
                cos_a = np.cos(np.radians(angle))
                sin_a = np.sin(np.radians(angle))

                # Rotate landmarks
                landmarks_centered = landmarks - center
                rotated = np.zeros_like(landmarks)
                rotated[:, 0] = landmarks_centered[:, 0] * cos_a - landmarks_centered[:, 1] * sin_a
                rotated[:, 1] = landmarks_centered[:, 0] * sin_a + landmarks_centered[:, 1] * cos_a
                landmarks = rotated + center

                # Rotate image (need to convert to HWC for cv2)
                image_hwc = np.transpose(image, (1, 2, 0))
                M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
                image_hwc = cv2.warpAffine(image_hwc, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
                image = np.transpose(image_hwc, (2, 0, 1))

            # Gaussian noise (small)
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0, 1)

            return image, landmarks

        def __getitem__(self, idx):
            # Fast RAM access - no HDF5 decompression needed
            landmarks = self.landmarks[idx].copy()  # (68, 2) in original frame coords

            # Transform landmarks from original frame to 112x112 aligned face space
            if self.warp_matrices is not None:
                # Use the stored warp matrix (exact transform used for alignment)
                warp_matrix = self.warp_matrices[idx]  # (2, 3) affine transform

                # Apply affine transform: landmarks_aligned = landmarks @ M[:2,:2].T + M[:,2]
                # Equivalent to: for each point p, p' = M @ [p, 1]
                scale_rot = warp_matrix[:, :2]  # (2, 2)
                translation = warp_matrix[:, 2]  # (2,)
                landmarks_aligned = landmarks @ scale_rot.T + translation
            else:
                # Fallback: approximate transform using landmark bounding box
                lm_min = landmarks.min(axis=0)
                lm_max = landmarks.max(axis=0)
                lm_center = (lm_min + lm_max) / 2
                lm_size = (lm_max - lm_min).max()
                target_size = 112 * 0.8
                scale = target_size / max(lm_size, 1.0)
                landmarks_aligned = (landmarks - lm_center) * scale + 56.0

            # Get image
            image = None
            if self.images is not None:
                # Normalize image to [0, 1] and convert to CHW
                image = self.images[idx].astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

                # Apply augmentation if enabled
                if self.augment and np.random.random() < self.augment_prob:
                    image, landmarks_aligned = self._apply_augmentation(image, landmarks_aligned)

            result = {
                'landmarks': torch.from_numpy(landmarks_aligned.astype(np.float32)),
                'global_params': torch.from_numpy(self.global_params[idx].copy()).float(),
                'local_params': torch.from_numpy(self.local_params[idx].copy()).float(),
                'au_intensities': torch.from_numpy(self.au_intensities[idx].copy()).float(),
            }

            if image is not None:
                result['image'] = torch.from_numpy(image)

            if self.hog_features is not None:
                result['hog_features'] = torch.from_numpy(self.hog_features[idx].copy()).float()

            # Also return the warp matrix for potential use in inverse transform
            if self.warp_matrices is not None:
                result['warp_matrix'] = torch.from_numpy(self.warp_matrices[idx].copy()).float()

            return result


    def create_video_stratified_split(
        dataset: PyTorchTrainingDataset,
        val_split: float = 0.1,
        seed: int = 42
    ) -> Tuple[Subset, Subset]:
        """
        Create train/val split stratified by video.

        Ensures that all frames from a video are either in train or val,
        preventing data leakage from similar sequential frames.

        Args:
            dataset: The full dataset
            val_split: Fraction of videos to use for validation
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_subset, val_subset)
        """
        np.random.seed(seed)

        video_indices = dataset.get_indices_by_video()
        videos = list(video_indices.keys())

        if len(videos) <= 1:
            # Fallback to random split if no video info
            print("[Dataset] WARNING: Only 1 video found, falling back to random split", flush=True)
            n_val = int(len(dataset) * val_split)
            indices = np.random.permutation(len(dataset))
            val_indices = indices[:n_val].tolist()
            train_indices = indices[n_val:].tolist()
        else:
            # Shuffle videos
            np.random.shuffle(videos)

            # Split videos
            n_val_videos = max(1, int(len(videos) * val_split))
            val_videos = set(videos[:n_val_videos])
            train_videos = set(videos[n_val_videos:])

            # Collect indices
            train_indices = []
            val_indices = []
            for video, indices in video_indices.items():
                if video in val_videos:
                    val_indices.extend(indices)
                else:
                    train_indices.extend(indices)

            print(f"[Dataset] Video-stratified split:", flush=True)
            print(f"[Dataset]   Train: {len(train_videos)} videos, {len(train_indices)} samples", flush=True)
            print(f"[Dataset]   Val: {len(val_videos)} videos, {len(val_indices)} samples", flush=True)

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        return train_subset, val_subset

except ImportError:
    # PyTorch not available
    PyTorchTrainingDataset = None
    create_video_stratified_split = None
