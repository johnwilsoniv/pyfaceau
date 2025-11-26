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
        video_name: str = '',
        frame_index: int = 0,
        quality_score: float = 1.0
    ):
        """
        Add a single sample to the dataset.

        Args:
            image: Face image (112, 112, 3) uint8
            hog_features: HOG features (4464,) float32
            landmarks: 2D landmarks (68, 2) float32
            global_params: Pose parameters (6,) float32
            local_params: PDM shape parameters (34,) float32
            au_intensities: AU intensities (17,) float32
            bbox: Face bounding box (4,) float32
            video_name: Source video name
            frame_index: Frame index in source video
            quality_score: Quality score (0-1)
        """
        if self.current_index >= self.expected_samples:
            # Resize datasets
            new_size = self.current_index + self.chunk_size
            for name in ['images', 'hog_features', 'landmarks', 'global_params',
                        'local_params', 'au_intensities', 'bboxes']:
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
                    'local_params', 'au_intensities', 'bboxes']:
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
    from torch.utils.data import Dataset as TorchDataset
    import torch

    class PyTorchTrainingDataset(TorchDataset):
        """PyTorch-compatible wrapper for TrainingDataset."""

        def __init__(
            self,
            h5_path: Union[str, Path],
            load_images: bool = True,
            load_hog: bool = False,
            augment: bool = False
        ):
            self.dataset = TrainingDataset(h5_path, load_images, load_hog)
            self.augment = augment

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]

            # Convert to tensors
            result = {
                'landmarks': torch.from_numpy(sample['landmarks']).float(),
                'global_params': torch.from_numpy(sample['global_params']).float(),
                'local_params': torch.from_numpy(sample['local_params']).float(),
                'au_intensities': torch.from_numpy(sample['au_intensities']).float(),
            }

            if 'image' in sample:
                # Normalize image to [0, 1] and convert to CHW
                image = sample['image'].astype(np.float32) / 255.0
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
                result['image'] = torch.from_numpy(image)

            if 'hog_features' in sample:
                result['hog_features'] = torch.from_numpy(sample['hog_features']).float()

            return result

except ImportError:
    # PyTorch not available
    PyTorchTrainingDataset = None
