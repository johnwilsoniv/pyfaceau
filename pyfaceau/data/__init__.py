"""
pyfaceau.data - Training data generation and storage for neural network training.

This package provides:
- HDF5 dataset storage for efficient training data access
- Training data generator to extract features from videos
- Quality filtering for training data
"""

from .hdf5_dataset import TrainingDataset, TrainingDataWriter
from .training_data_generator import TrainingDataGenerator
from .quality_filter import QualityFilter

__all__ = [
    'TrainingDataset',
    'TrainingDataWriter',
    'TrainingDataGenerator',
    'QualityFilter',
]
