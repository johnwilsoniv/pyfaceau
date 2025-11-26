"""
pyfaceau.nn - Neural Network Models for Face Analysis

This package provides neural network replacements for the traditional
pyCLNF landmark detection and AU prediction pipelines.

Models:
- UnifiedLandmarkPoseNet: Predicts 68 landmarks + 6 global params + 34 local params
- AUPredictionNet: Predicts 17 AU intensities from aligned face

Both models are optimized for real-time inference on ARM Mac (CoreML) and
Intel/CUDA (ONNX).

Inference:
- LandmarkPosePredictor: Unified interface for landmark/pose prediction
- AUPredictor: Unified interface for AU intensity prediction
- load_predictor: Load landmark/pose models
- load_au_predictor: Load AU prediction models

Training:
- LandmarkPoseLoss: Combined loss for landmark/pose training
- AUPredictionLoss: Combined loss for AU training
- train_landmark_pose: Training script (run as module)
- train_au_prediction: Training script (run as module)

Usage:
    # Landmark/Pose prediction
    from pyfaceau.nn import load_predictor
    predictor = load_predictor('models/landmark_pose/')
    result = predictor.predict(aligned_face)
    landmarks = result['landmarks']

    # AU prediction
    from pyfaceau.nn import load_au_predictor
    au_predictor = load_au_predictor('models/au_prediction/')
    result = au_predictor.predict(aligned_face)
    au_intensities = result['au_intensities']
"""

# Landmark/Pose models
from .landmark_pose_net import (
    UnifiedLandmarkPoseNet,
    LandmarkPoseLoss,
    WingLoss,
    export_to_onnx,
    export_to_coreml,
)
from .landmark_pose_inference import (
    LandmarkPosePredictor,
    load_predictor,
)

# AU prediction models
from .au_prediction_net import (
    AUPredictionNet,
    AUPredictionLoss,
    ConcordanceCorrelationLoss,
    AU_NAMES,
    NUM_AUS,
    export_au_to_onnx,
    export_au_to_coreml,
)
from .au_prediction_inference import (
    AUPredictor,
    load_au_predictor,
)

__all__ = [
    # Landmark/Pose Models
    'UnifiedLandmarkPoseNet',
    'LandmarkPoseLoss',
    'WingLoss',
    'export_to_onnx',
    'export_to_coreml',
    'LandmarkPosePredictor',
    'load_predictor',

    # AU Models
    'AUPredictionNet',
    'AUPredictionLoss',
    'ConcordanceCorrelationLoss',
    'AU_NAMES',
    'NUM_AUS',
    'export_au_to_onnx',
    'export_au_to_coreml',
    'AUPredictor',
    'load_au_predictor',
]
