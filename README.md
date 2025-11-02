# pyfaceau - Action Unit Generation based on Python and Openface 2.2


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Overview

pyfaceau is a Python reimplementation of the [OpenFace 2.2](https://github.com/TadasBaltrusaitis/OpenFace) Facial Action Unit extraction pipeline. It achieves **r =92 correlation** with the original C++ implementation while requiring **zero compilation** and running on any platform.

### Key Features

- ** 100% Python** - No C++ compilation required
- ** Easy Installation** - `pip install` and go
- ** High Accuracy** - r=0.92 overall
- ** High Performance** - 50-100 FPS with parallel processing (6-10x speedup!)
- ** Multi-Core Support** - Automatic parallelization across CPU cores
- ** Modular** - Use individual components independently
- ** 17 Action Units** - Full AU extraction (AU01, AU02, AU04, etc.)

---

## Quick Start

### Installation

#### Option 1: Install from PyPI (Recommended)

```bash
# Install pyfaceau
pip install pyfaceau

# Download model weights (14MB)
python -m pyfaceau.download_weights

# Or manually download from GitHub
# https://github.com/johnwilsoniv/face-analysis/tree/main/S0%20PyfaceAU/weights
```

#### Option 2: Install from Source

```bash
# Clone repository
git clone https://github.com/johnwilsoniv/face-analysis.git
cd "face-analysis/S0 PyfaceAU"

# Install in development mode
pip install -e .

# Model weights are included in the repository
```

### Basic Usage

#### High-Performance Mode (Recommended - 50-100 FPS)

```python
from pyfaceau import ParallelAUPipeline

# Initialize parallel pipeline 
pipeline = ParallelAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
    au_models_dir='path/to/AU_predictors',
    triangulation_file='weights/tris_68_full.txt',
    num_workers=6,  # Adjust based on CPU cores
    batch_size=30
)

# Process video
results = pipeline.process_video(
    video_path='input.mp4',
    output_csv='results.csv'
)

print(f"Processed {len(results)} frames")
# Typical output: ~28-50 FPS depending on CPU cores
```

#### Standard Mode (4.6 FPS)

```python
from pyfaceau import FullPythonAUPipeline

# Initialize standard pipeline
pipeline = FullPythonAUPipeline(
    retinaface_model='weights/retinaface_mobilenet025_coreml.onnx',
    pfld_model='weights/pfld_cunjian.onnx',
    pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
    au_models_dir='path/to/AU_predictors',
    triangulation_file='weights/tris_68_full.txt',
    use_calc_params=True,
    use_coreml=True,  # macOS only
    verbose=False
)

# Process video
results = pipeline.process_video(
    video_path='input.mp4',
    output_csv='results.csv'
)
```

### Example Output

```csv
frame,success,AU01_r,AU02_r,AU04_r,AU06_r,AU12_r,...
0,True,0.60,0.90,0.00,1.23,2.45,...
1,True,0.55,0.85,0.00,1.20,2.50,...
```

---

## Architecture

pyfaceau replicates the complete OpenFace 2.2 AU extraction pipeline:

```
Video Input
    ↓
Face Detection (RetinaFace ONNX)
    ↓
Landmark Detection (PFLD 68-point)
    ↓
3D Pose Estimation (Python Implementation of CalcParams with 98% fidelity)
    ↓
Face Alignment
    ↓
HOG Feature Extraction (PyFHOG)
    ↓
Geometric Features (PDM reconstruction)
    ↓
Running Median Tracking (Cython-optimized)
    ↓
AU Prediction (17 SVR models)
    ↓
Output: 17 AU intensities
```

---

## Custom Components & Innovations

pyfaceau includes several novel components that can be used independently in other projects:

### Python-based CalcParams - 3D Pose Estimation

A pure Python implementation of OpenFace's CalcParams algorithm for 3D head pose estimation. Achieves 99.45% correlation with the C++ reference implementation.

```python
from pyfaceau.alignment import CalcParams

# Initialize with PDM model
calc_params = CalcParams(pdm_file='weights/In-the-wild_aligned_PDM_68.txt')

# Estimate 3D pose from 2D landmarks
params_local, params_global, detected_landmarks = calc_params.estimate_pose(
    landmarks_2d,  # 68x2 array of detected landmarks
    img_width,
    img_height
)

# Extract pose parameters
tx, ty = params_global[4], params_global[5]  # Translation
rx, ry, rz = params_global[1:4]  # Rotation (radians)
scale = params_global[0]  # Scale factor
```

**Use cases:** Head pose tracking, gaze estimation, facial alignment

### CLNF Landmark Refinement

Constrained Local Neural Fields (CLNF) refinement using SVR patch experts for improved landmark accuracy. Particularly effective for challenging poses and expressions.

```python
from pyfaceau.detectors import CLNFRefiner

# Initialize refiner
refiner = CLNFRefiner(
    pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
    patch_expert_file='weights/svr_patches_0.25_general.txt'
)

# Refine landmarks
refined_landmarks = refiner.refine_landmarks(
    frame,
    initial_landmarks,
    face_bbox,
    num_iterations=5
)
```

**Use cases:** Landmark tracking, facial feature extraction, expression analysis

### Cython Histogram Median Tracker (260x speedup)

High-performance running median tracking for temporal smoothing of geometric features. Implements OpenFace's histogram-based median algorithm in optimized Cython.

```python
from pyfaceau.features import HistogramMedianTracker

# Initialize tracker
tracker = HistogramMedianTracker(
    num_features=136,  # 68 landmarks x 2 (x,y)
    history_length=120
)

# Update with new frame
smoothed_features = tracker.update(current_features)
```

**Use cases:** Temporal smoothing, noise reduction, video feature tracking

### Batched AU Predictor

Optimized AU prediction using batch processing for HOG features. Reduces overhead when processing multiple frames.

```python
from pyfaceau.prediction import BatchedAUPredictor

# Initialize predictor
predictor = BatchedAUPredictor(
    au_models_dir='weights/AU_predictors',
    batch_size=30
)

# Predict AUs for multiple frames
au_results = predictor.predict_batch(
    hog_features_list,  # List of HOG feature arrays
    geom_features_list  # List of geometric feature arrays
)
```

**Use cases:** Video processing, batch AU extraction, real-time analysis

### OpenFace22 Face Aligner

Pure Python implementation of OpenFace 2.2's face alignment algorithm. Produces pixel-perfect aligned faces matching the C++ implementation.

```python
from pyfaceau.alignment import OpenFace22FaceAligner

# Initialize aligner
aligner = OpenFace22FaceAligner(
    pdm_file='weights/In-the-wild_aligned_PDM_68.txt',
    triangulation_file='weights/tris_68_full.txt'
)

# Align face for AU extraction
aligned_face = aligner.align_face(
    frame,
    landmarks_2d,
    tx, ty, rz  # From CalcParams
)
```

**Output:** 112x112 RGB aligned face, ready for HOG extraction

**Use cases:** Face normalization, AU extraction preprocessing, facial feature analysis

---

## Supported Action Units

pyfaceau extracts 17 Facial Action Units:

**Dynamic AUs (11):**
- AU01 - Inner Brow Raiser
- AU02 - Outer Brow Raiser
- AU05 - Upper Lid Raiser
- AU09 - Nose Wrinkler
- AU15 - Lip Corner Depressor
- AU17 - Chin Raiser
- AU20 - Lip Stretcher
- AU23 - Lip Tightener
- AU25 - Lips Part
- AU26 - Jaw Drop
- AU45 - Blink

**Static AUs (6):**
- AU04 - Brow Lowerer
- AU06 - Cheek Raiser
- AU07 - Lid Tightener
- AU10 - Upper Lip Raiser
- AU12 - Lip Corner Puller
- AU14 - Dimpler

---

## Requirements

### Python Dependencies

```
python >= 3.10
numpy >= 1.20.0
opencv-python >= 4.5.0
pandas >= 1.3.0
scipy >= 1.7.0
onnxruntime >= 1.10.0
pyfhog >= 0.1.0
```

### Model Files

Download OpenFace 2.2 AU predictor models:
- Available from: [OpenFace repository](https://github.com/TadasBaltrusaitis/OpenFace)
- Place in: `AU_predictors/` directory
- Required: 17 `.dat` files (AU_1_dynamic_intensity_comb.dat, etc.)

---

## Project Structure

```
S0 pyfaceau/
├── pyfaceau/                  # Core library
│   ├── pipeline.py            # Full AU extraction pipeline
│   ├── detectors/             # Face and landmark detection
│   ├── alignment/             # Face alignment and pose estimation
│   ├── features/              # HOG and geometric features
│   ├── prediction/            # AU prediction and running median
│   └── utils/                 # Utilities and Cython extensions
├── weights/                   # Model weights
├── tests/                     # Test suite
├── examples/                  # Usage examples
└── docs/                      # Documentation
```

---

## Advanced Usage

### Process Single Frame

```python
from pyfaceau import FullPythonAUPipeline
import cv2

pipeline = FullPythonAUPipeline(...)

# Read frame
frame = cv2.imread('image.jpg')

# Process (requires landmarks and pose from CSV or detector)
aligned = pipeline.aligner.align_face(frame, landmarks, tx, ty, rz)
hog_features = pipeline.extract_hog(aligned)
aus = pipeline.predict_aus(hog_features, geom_features)
```

### Use Individual Components

```python
# Face detection only
from pyfaceau.detectors import ONNXRetinaFaceDetector
detector = ONNXRetinaFaceDetector('weights/retinaface_mobilenet025_coreml.onnx')
faces = detector.detect_faces(frame)

# Landmark detection only
from pyfaceau.detectors import CunjianPFLDDetector
landmarker = CunjianPFLDDetector('weights/pfld_cunjian.onnx')
landmarks, conf = landmarker.detect_landmarks(frame, bbox)

# Face alignment only
from pyfaceau.alignment import OpenFace22FaceAligner
aligner = OpenFace22FaceAligner('weights/In-the-wild_aligned_PDM_68.txt')
aligned = aligner.align_face(frame, landmarks, tx, ty, rz)
```

---

## Citation

If you use pyfaceau in your research, please cite:

```bibtex
@article{wilson2025splitface,
  title={A Split-Face Computer Vision/Machine Learning Assessment of Facial Paralysis Using Facial Action Units},
  author={Wilson IV, John and Rosenberg, Joshua and Gray, Mingyang L and Razavi, Christopher R},
  journal={Facial Plastic Surgery \& Aesthetic Medicine},
  year={2025},
  publisher={Mary Ann Liebert, Inc.}
}
```

Also cite the original OpenFace:

```bibtex
@inproceedings{baltrusaitis2018openface,
  title={OpenFace 2.0: Facial behavior analysis toolkit},
  author={Baltru{\v{s}}aitis, Tadas and Zadeh, Amir and Lim, Yao Chong and Morency, Louis-Philippe},
  booktitle={2018 13th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2018)},
  pages={59--66},
  year={2018},
  organization={IEEE}
}
```

---

## Acknowledgments

- **OpenFace** - Original C++ implementation by Tadas Baltrusaitis
- **PyFHOG** - HOG feature extraction library
- **RetinaFace** - Face detection model
- **PFLD** - Landmark detection by Cunjian Chen

---

## Support

- **Issues:** https://github.com/yourname/pyfaceau/issues
- **Documentation:** [docs/](docs/)
- **Examples:** [examples/](examples/)

---

**Built for the facial behavior research community**
