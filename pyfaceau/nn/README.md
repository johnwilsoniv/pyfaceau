# pyfaceau.nn - Neural Network Models for Face Analysis

This package provides neural network replacements for the traditional pyCLNF landmark detection and AU prediction pipelines, targeting **20-30 FPS** on ARM Mac with high accuracy.

## Overview

| Component | Replaces | Input | Output | Parameters |
|-----------|----------|-------|--------|------------|
| `UnifiedLandmarkPoseNet` | pyCLNF iterative optimization | 112x112x3 RGB | 68 landmarks + pose params | 3.77M |
| `AUPredictionNet` | HOG+SVM AU classifiers | 112x112x3 RGB | 17 AU intensities | 7.39M |

## Architecture

### UnifiedLandmarkPoseNet

```
Input: 112x112x3 RGB image
    │
    ▼
┌─────────────────────────┐
│  MobileNetV2 Backbone   │  (efficient for ARM Mac)
│  - Inverted Residuals   │
│  - 1280 output features │
└───────────┬─────────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
┌───────┐┌───────┐┌───────┐
│Landmark││Global ││Local  │
│ Head   ││Params ││Params │
│ (68x2) ││ (6)   ││ (34)  │
└───────┘└───────┘└───────┘
```

**Outputs:**
- `landmarks`: (68, 2) - 2D facial landmarks in image coordinates [0, 112]
- `global_params`: (6,) - [scale, rx, ry, rz, tx, ty] pose parameters
- `local_params`: (34,) - PDM shape coefficients

### AUPredictionNet

```
Input: 112x112x3 RGB image
    │
    ▼
┌──────────────────────────┐
│ EfficientNet-Lite        │  (better for fine-grained tasks)
│ Backbone                 │
│ - MBConv blocks          │
│ - Squeeze-Excitation     │
│ - 1280 output features   │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│       AU Head            │
│  - FC layers + ReLU      │
│  - Output: 17 AUs        │
└──────────────────────────┘
```

**Outputs:**
- `au_intensities`: (17,) - AU intensity values in range [0, 5]

**AUs Predicted:**
```
AU01_r - Inner Brow Raiser     AU14_r - Dimpler
AU02_r - Outer Brow Raiser     AU15_r - Lip Corner Depressor
AU04_r - Brow Lowerer          AU17_r - Chin Raiser
AU05_r - Upper Lid Raiser      AU20_r - Lip Stretcher
AU06_r - Cheek Raiser          AU23_r - Lip Tightener
AU07_r - Lid Tightener         AU25_r - Lips Part
AU09_r - Nose Wrinkler         AU26_r - Jaw Drop
AU10_r - Upper Lip Raiser      AU45_r - Blink
AU12_r - Lip Corner Puller
```

## Installation

The nn module requires PyTorch. For optimal performance:

```bash
# CPU only
pip install torch torchvision

# Apple Silicon (MPS acceleration)
pip install torch torchvision  # MPS support is built-in

# CUDA (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For ONNX inference
pip install onnxruntime  # or onnxruntime-gpu for CUDA

# For CoreML export (macOS only)
pip install coremltools
```

## Training Data Generation

Before training, generate HDF5 training data using the existing pipeline:

```python
from pyfaceau.data import TrainingDataGenerator, TrainingDataWriter

# Create generator (uses pyCLNF for ground truth)
generator = TrainingDataGenerator()

# Process videos and save to HDF5
with TrainingDataWriter('training_data.h5', expected_samples=100000) as writer:
    for video_path in video_paths:
        for frame_data in generator.process_video(video_path):
            writer.add_sample(
                image=frame_data['image'],           # 112x112x3 RGB
                hog_features=frame_data['hog'],      # 4464-dim
                landmarks=frame_data['landmarks'],    # 68x2
                global_params=frame_data['global_params'],  # 6
                local_params=frame_data['local_params'],    # 34
                au_intensities=frame_data['au_intensities'], # 17
                bbox=frame_data['bbox'],
                video_name=video_path.name,
                frame_index=frame_data['frame_idx'],
            )
```

**HDF5 Structure:**
```
training_data.h5
├── images          (N, 112, 112, 3) uint8 RGB
├── hog_features    (N, 4464) float32
├── landmarks       (N, 68, 2) float32
├── global_params   (N, 6) float32
├── local_params    (N, 34) float32
├── au_intensities  (N, 17) float32
├── bboxes          (N, 4) float32
└── metadata/
    ├── video_names
    ├── frame_indices
    └── quality_scores
```

## Training

### Train Landmark/Pose Model

```bash
# Basic training
python -m pyfaceau.nn.train_landmark_pose \
    --data training_data.h5 \
    --output models/landmark_pose \
    --epochs 100 \
    --batch-size 32

# Full options
python -m pyfaceau.nn.train_landmark_pose \
    --data training_data.h5 \
    --output models/landmark_pose \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --lm-weight 1.0 \        # Landmark loss weight
    --gp-weight 0.1 \        # Global params loss weight
    --lp-weight 0.01 \       # Local params loss weight
    --width-mult 1.0 \       # Backbone width multiplier
    --val-split 0.1 \
    --patience 30 \          # Early stopping patience
    --save-every 10

# Resume training
python -m pyfaceau.nn.train_landmark_pose \
    --data training_data.h5 \
    --output models/landmark_pose \
    --resume models/landmark_pose/checkpoint_best.pt
```

**Output files:**
```
models/landmark_pose/
├── checkpoint_best.pt      # Best validation loss
├── checkpoint_final.pt     # Final epoch
├── checkpoint_epoch_*.pt   # Periodic checkpoints
├── training_history.json   # Loss/metric history
├── landmark_pose.onnx      # ONNX export
└── landmark_pose.mlpackage # CoreML export (if available)
```

### Train AU Prediction Model

```bash
# Basic training
python -m pyfaceau.nn.train_au_prediction \
    --data training_data.h5 \
    --output models/au_prediction \
    --epochs 100 \
    --batch-size 32

# Full options
python -m pyfaceau.nn.train_au_prediction \
    --data training_data.h5 \
    --output models/au_prediction \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --l1-weight 1.0 \        # Smooth L1 loss weight
    --ccc-weight 0.5 \       # Concordance correlation loss weight
    --dropout 0.3 \
    --width-mult 1.0 \
    --patience 30
```

**Output files:**
```
models/au_prediction/
├── checkpoint_best.pt
├── checkpoint_final.pt
├── training_history.json   # Includes per-AU metrics
├── au_prediction.onnx
└── au_prediction.mlpackage
```

## Inference

### Landmark/Pose Prediction

```python
from pyfaceau.nn import load_predictor
import cv2

# Load model (auto-selects best backend: CoreML > ONNX > PyTorch)
predictor = load_predictor('models/landmark_pose/')

# Or specify backend explicitly
from pyfaceau.nn import LandmarkPosePredictor
predictor = LandmarkPosePredictor.from_onnx('models/landmark_pose/landmark_pose.onnx')
predictor = LandmarkPosePredictor.from_coreml('models/landmark_pose/landmark_pose.mlpackage')
predictor = LandmarkPosePredictor.from_checkpoint('models/landmark_pose/checkpoint_best.pt')

# Predict (input: 112x112 BGR aligned face)
aligned_face = cv2.imread('aligned_face.png')  # 112x112x3 BGR
result = predictor.predict(aligned_face)

landmarks = result['landmarks']       # (68, 2) image coordinates
global_params = result['global_params']  # (6,) [scale, rx, ry, rz, tx, ty]
local_params = result['local_params']    # (34,) PDM coefficients

# Batch prediction
faces = np.stack([face1, face2, face3])  # (3, 112, 112, 3)
results = predictor.predict_batch(faces)
# results['landmarks'].shape == (3, 68, 2)
```

### AU Prediction

```python
from pyfaceau.nn import load_au_predictor

# Load model
au_predictor = load_au_predictor('models/au_prediction/')

# Predict
result = au_predictor.predict(aligned_face)

au_intensities = result['au_intensities']  # (17,) numpy array
au_dict = result['au_dict']  # {'AU01_r': 0.5, 'AU02_r': 1.2, ...}

# Print all AUs
for name, intensity in au_dict.items():
    print(f"{name}: {intensity:.2f}")

# Get AU names
au_names = au_predictor.get_au_names()
# ['AU01_r', 'AU02_r', ..., 'AU45_r']
```

### Combined Pipeline Usage

```python
from pyfaceau.nn import load_predictor, load_au_predictor
from pyfaceau import FaceAligner
import cv2

# Load models once
lm_predictor = load_predictor('models/landmark_pose/')
au_predictor = load_au_predictor('models/au_prediction/')
aligner = FaceAligner()

# Process video
cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face and get bbox (using MTCNN or other detector)
    bbox = detect_face(frame)

    # Align face to 112x112
    aligned = aligner.align(frame, bbox)

    # Predict landmarks and pose
    lm_result = lm_predictor.predict(aligned)

    # Predict AUs
    au_result = au_predictor.predict(aligned)

    # Use results
    landmarks_2d = lm_result['landmarks']
    au_intensities = au_result['au_intensities']
```

## Loss Functions

### LandmarkPoseLoss

Combined loss for landmark and pose prediction:

```python
Loss = w_lm * WingLoss(landmarks) + w_gp * MSE(global_params) + w_lp * L1(local_params)
```

- **WingLoss**: Better handles small/medium errors than L2
- **MSE**: For global pose parameters
- **L1**: For local params (encourages sparsity)

### AUPredictionLoss

Combined loss for AU prediction:

```python
Loss = w_l1 * SmoothL1(au) + w_ccc * (1 - CCC(au))
```

- **SmoothL1**: Robust to outliers
- **CCC (Concordance Correlation Coefficient)**: Measures agreement, better than simple correlation

## Export Formats

### ONNX Export

```python
from pyfaceau.nn import UnifiedLandmarkPoseNet, export_to_onnx

model = UnifiedLandmarkPoseNet()
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])

export_to_onnx(model, 'model.onnx', opset_version=12)
```

### CoreML Export (macOS)

```python
from pyfaceau.nn import UnifiedLandmarkPoseNet, export_to_coreml

model = UnifiedLandmarkPoseNet()
model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])

export_to_coreml(model, 'model.mlpackage')
```

## Backend Selection

The inference wrappers automatically select the best available backend:

| Platform | Preferred Backend | Fallback |
|----------|------------------|----------|
| Apple Silicon Mac | CoreML | ONNX → PyTorch |
| Intel Mac | ONNX | PyTorch |
| NVIDIA GPU | ONNX (CUDA) | PyTorch (CUDA) |
| CPU | ONNX | PyTorch |

Force a specific backend:

```python
# Force PyTorch
predictor = LandmarkPosePredictor.from_checkpoint('checkpoint.pt', device='cpu')

# Force ONNX with specific providers
predictor = LandmarkPosePredictor.from_onnx('model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Force CoreML
predictor = LandmarkPosePredictor.from_coreml('model.mlpackage')
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| FPS (ARM Mac) | 20-30 | Using CoreML backend |
| FPS (Intel + CUDA) | 50+ | Using ONNX with CUDA |
| Landmark Correlation | >0.95 | vs pyCLNF ground truth |
| AU Correlation | >0.90 | vs HOG+SVM ground truth |
| Landmark MAE | <2 pixels | On 112x112 image |
| AU MAE | <0.5 | On 0-5 scale |

## File Structure

```
pyfaceau/pyfaceau/nn/
├── __init__.py                    # Package exports
├── README.md                      # This document
├── landmark_pose_net.py           # UnifiedLandmarkPoseNet model
├── landmark_pose_inference.py     # Inference wrapper
├── train_landmark_pose.py         # Training script
├── au_prediction_net.py           # AUPredictionNet model
├── au_prediction_inference.py     # Inference wrapper
└── train_au_prediction.py         # Training script
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python -m pyfaceau.nn.train_landmark_pose --batch-size 16
```

### Slow Training

- Use GPU if available (CUDA or MPS)
- Reduce `num_workers` if I/O bound
- Use smaller `width_mult` for faster but less accurate model

### Poor Accuracy

- Ensure training data quality (check for bad alignments)
- Increase training epochs
- Try different loss weights
- Use data augmentation (not yet implemented)

### CoreML Export Fails

- Ensure macOS 13+ and coremltools installed
- Some operations may not be supported; fall back to ONNX

## API Reference

### Models

```python
# Landmark/Pose
UnifiedLandmarkPoseNet(width_mult=1.0, num_landmarks=68, num_local_params=34, image_size=112)

# AU Prediction
AUPredictionNet(width_mult=1.0, dropout=0.3)
```

### Loss Functions

```python
# Landmark/Pose
LandmarkPoseLoss(landmark_weight=1.0, global_params_weight=0.1, local_params_weight=0.01)
WingLoss(w=10.0, epsilon=2.0)

# AU Prediction
AUPredictionLoss(smooth_l1_weight=1.0, ccc_weight=0.5, au_weights=None)
ConcordanceCorrelationLoss(eps=1e-8)
```

### Inference

```python
# Landmark/Pose
load_predictor(model_path, backend=None) -> LandmarkPosePredictor
LandmarkPosePredictor.from_checkpoint(path, device=None, width_mult=1.0)
LandmarkPosePredictor.from_onnx(path, providers=None)
LandmarkPosePredictor.from_coreml(path)
LandmarkPosePredictor.from_auto(model_dir, prefer_coreml=True)

# AU Prediction
load_au_predictor(model_path, backend=None) -> AUPredictor
AUPredictor.from_checkpoint(path, device=None, width_mult=1.0, dropout=0.3)
AUPredictor.from_onnx(path, providers=None)
AUPredictor.from_coreml(path)
AUPredictor.from_auto(model_dir, prefer_coreml=True)
```

### Constants

```python
from pyfaceau.nn import AU_NAMES, NUM_AUS

AU_NAMES  # ['AU01_r', 'AU02_r', ..., 'AU45_r']
NUM_AUS   # 17
```
