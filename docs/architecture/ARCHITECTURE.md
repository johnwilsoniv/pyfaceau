# pyAUface Architecture

**Detailed component descriptions and data flow**

---

## Pipeline Overview

pyAUface implements the complete OpenFace 2.2 AU extraction pipeline as 12 distinct components:

```
[1] Video Input → [2] Face Detection → [3] Landmark Detection →
[4] Pose Estimation → [5] Face Alignment → [6] Face Masking →
[7] HOG Extraction → [8] Geometric Features → [9] Running Median →
[10] Normalization → [11] AU Prediction → [12] Output
```

---

## Component 1: Video Input

**Module:** Standard OpenCV
**Function:** `cv2.VideoCapture()`

### Purpose
Read video frames sequentially for processing.

### Implementation
```python
import cv2
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame...
```

### Notes
- Standard OpenCV functionality
- No custom implementation needed

---

## Component 2: Face Detection

**Module:** `pyauface.detectors.retinaface`
**File:** `pyauface/detectors/retinaface.py`
**Class:** `ONNXRetinaFaceDetector`

### Purpose
Detect faces in video frames and return bounding boxes.

### Model
- **Architecture:** RetinaFace with MobileNet0.25 backbone
- **Format:** ONNX (CoreML-optimized for macOS)
- **Input:** Variable size RGB image
- **Output:** Bounding boxes + 5 facial landmarks + confidence
- **Size:** 1.7 MB

### Implementation
```python
from pyauface.detectors import ONNXRetinaFaceDetector

detector = ONNXRetinaFaceDetector(
    'weights/retinaface_mobilenet025_coreml.onnx',
    use_coreml=True,  # macOS Neural Engine acceleration
    confidence_threshold=0.5
)

faces = detector.detect_faces(frame)
# Returns: List of [x1, y1, x2, y2, confidence]
```

### Performance
- **CPU mode:** 469ms per frame
- **CoreML mode:** 150-230ms per frame
- **With tracking (99% skip):** ~2ms average

### Key Features
- CoreML Neural Engine acceleration on macOS
- Face tracking to skip redundant detections
- Confidence-based filtering

---

## Component 3: Landmark Detection

**Module:** `pyauface.detectors.pfld`
**File:** `pyauface/detectors/pfld.py`
**Class:** `CunjianPFLDDetector`

### Purpose
Detect 68 facial landmark points in 2D (x, y pixel coordinates).

### Model
- **Architecture:** PFLD (Practical Facial Landmark Detector)
- **Source:** cunjian/pytorch_face_landmark
- **Input:** 112×112 RGB image (face crop)
- **Output:** 68 (x, y) coordinates normalized to [0, 1]
- **Size:** 2.9 MB
- **Accuracy:** NME = 4.37% on 300W dataset

### Implementation
```python
from pyauface.detectors import CunjianPFLDDetector

detector = CunjianPFLDDetector('weights/pfld_cunjian.onnx')
landmarks, confidence = detector.detect_landmarks(frame, bbox)
# Returns: (68, 2) array of (x, y) coordinates
```

### Performance
- **Time:** 5ms per frame
- **Status:** Fully optimized (ONNX Runtime)

### Preprocessing
1. Create square bounding box with 10% padding
2. Crop and pad face if at image edge
3. Resize to 112×112
4. Convert BGR → RGB
5. Normalize to [0, 1]

### Post-processing
1. Reshape output to (68, 2)
2. Scale from [0, 1] to bbox coordinates
3. Translate to original image coordinates

---

## Component 4: 3D Pose Estimation (CalcParams)

**Module:** `pyauface.alignment.calc_params`
**File:** `pyauface/alignment/calc_params.py`
**Class:** `CalcParams`

### Purpose
Estimate 3D head pose and face shape from 2D landmarks using iterative optimization.

### Algorithm
Gauss-Newton optimization to find:
- **6 global parameters:** scale, rx, ry, rz, tx, ty (3D pose)
- **34 local parameters:** PCA coefficients (face shape variation)

### Implementation
```python
from pyauface.alignment import CalcParams

calc_params = CalcParams(pdm_parser)
params_global, params_local = calc_params.calc_params(landmarks_68.flatten())

# params_global: [scale, rx, ry, rz, tx, ty]
# params_local: [p_0, p_1, ..., p_33] (34 PCA coefficients)
```

### Performance
- **Time:** 80ms per frame (37% of total pipeline time)
- **Accuracy:** 99.45% correlation with C++ OpenFace
- **Status:** Warning: Bottleneck (optimization opportunity)

### Key Steps
1. Initialize from bounding box
2. Iterate (up to 1000 times):
   - Reconstruct 3D shape from current params
   - Project to 2D with current pose
   - Compute error vs detected landmarks
   - Compute Jacobian (derivatives)
   - Compute Hessian with regularization
   - Solve for parameter update
   - Update params (with rotation composition)
   - Check convergence
3. Return optimized params

### Validation Results
- Global params (pose): r = 0.9991 (99.91%)
- Local params (shape): r = 0.9899 (98.99%)
- Overall: r = 0.9945 (99.45%)

---

## Component 5: Face Alignment

**Module:** `pyauface.alignment.face_aligner`
**File:** `pyauface/alignment/face_aligner.py`
**Class:** `OpenFace22FaceAligner`

### Purpose
Align detected face to canonical frontal pose using landmarks and pose parameters.

### Algorithm
1. Extract reference 2D landmarks from PDM at canonical pose
2. Apply inverse p_rz rotation to correct for 2D head tilt
3. Compute similarity transform (rotation, translation, scale) using Kabsch algorithm
4. Transform tx, ty through scale_rot_matrix
5. Build 2×3 affine warp matrix
6. Apply `cv2.warpAffine()` to create aligned 112×112 image

### Implementation
```python
from pyauface.alignment import OpenFace22FaceAligner

aligner = OpenFace22FaceAligner('weights/In-the-wild_aligned_PDM_68.txt')
aligned = aligner.align_face(
    frame, landmarks, tx, ty, p_rz,
    apply_mask=True,
    triangulation=triangulation
)
# Returns: 112×112 BGR aligned face image
```

### Performance
- **Time:** 20ms per frame (9% of total)
- **Accuracy:** r = 0.94 for static AUs (proves correctness)

### Key Parameters
- **sim_scale:** 0.7 (must match OpenFace exactly)
- **output_size:** (112, 112) pixels
- **rigid_points:** 24 facial outline points for Kabsch

### Critical Details
1. **Inverse p_rz rotation:** Must use `-p_rz` (inverse) to rotate back to frontal
2. **tx, ty transformation:** Must transform through `scale_rot_matrix`
3. **24 rigid points:** Use only face outline (not flexible features)

---

## Component 6: Face Masking (Triangulation)

**Module:** `pyauface.features.triangulation`
**File:** `pyauface/features/triangulation.py`
**Class:** `TriangulationParser`

### Purpose
Apply binary mask to aligned face, keeping only pixels within facial region.

### Implementation
```python
from pyauface.features import TriangulationParser

triangulation = TriangulationParser('weights/tris_68_full.txt')
mask = triangulation.create_mask(aligned_landmarks, (112, 112))

# Apply mask
masked_face = cv2.bitwise_and(aligned_face, aligned_face, mask=mask)
```

### Triangulation File
- **Format:** 111 triangles defined by landmark indices
- **File:** `tris_68_full.txt`
- **Structure:** Each line contains 3 landmark indices forming a triangle

### Performance
- **Time:** Negligible (<1ms)
- **Status:** Fully optimized

---

## Component 7: HOG Feature Extraction (PyFHOG)

**Module:** External library `pyfhog`
**Package:** `pip install pyfhog`

### Purpose
Extract Felzenszwalb Histogram of Oriented Gradients (FHOG) features from 112×112 aligned face.

### Implementation
```python
import pyfhog
aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
hog_features = pyfhog.extract_fhog_features(aligned_rgb, cell_size=8)
# Returns: (4464,) numpy array
```

### Performance
- **Time:** 50ms per frame (23% of total pipeline time)
- **Accuracy:** r = 1.0 (PERFECT correlation with C++ OpenFace)
- **Status:** Warning: Bottleneck (optimization: increase cell_size)

### Feature Dimensions
- **Input:** 112×112 RGB image
- **Cell size:** 8×8 pixels
- **Grid:** 14×14 cells = 196 cells
- **Channels:** Multiple HOG channels per cell
- **Output:** 4464 features

### Requirements
- **Input format:** Must be RGB (not BGR)
- **Cell size:** 8 pixels (matches OpenFace exactly)

---

## Component 8: Geometric Feature Extraction (PDM)

**Module:** `pyauface.features.pdm`
**File:** `pyauface/features/pdm.py`
**Class:** `PDMParser`

### Purpose
Extract geometric features from 3D face shape for AU prediction.

### Algorithm
```python
# Reconstruct 3D shape from PDM
shape_3d = mean_shape + princ_comp @ params_local
shape_3d_flat = shape_3d.flatten()  # (204,)

# Geometric features: 204 (3D shape) + 34 (params_local) = 238
geom_features = np.concatenate([shape_3d_flat, params_local])
```

### Implementation
```python
from pyauface.features import PDMParser

pdm = PDMParser('weights/In-the-wild_aligned_PDM_68.txt')
geom_features = pdm.extract_geometric_features(params_local)
# Returns: (238,) numpy array
```

### Performance
- **Time:** 5ms per frame (2% of total)
- **Status:** Fully optimized

### PDM Components
- **Mean shape:** (204, 1) - Average 3D face shape
- **Principal components:** (204, 34) - PCA basis
- **Eigenvalues:** (34,) - Variance explained

---

## Component 9: Running Median Tracking

**Module:** `pyauface.prediction.running_median`
**File:** `pyauface/prediction/running_median.py`
**Class:** `DualHistogramMedianTracker`

### Purpose
Maintain histogram-based running median of HOG and geometric features for person-specific normalization (dynamic AU models).

### Algorithm
Two separate histogram trackers:
- **HOG median:** 4464 dimensions, 1000 bins each, range [-0.005, 1.0]
- **Geometric median:** 238 dimensions, 10000 bins each, range [-60.0, 60.0]

**Critical:** Updates histogram **every 2nd frame only** (matches OpenFace 2.2)

### Implementation
```python
from pyauface.prediction import DualHistogramMedianTracker

median_tracker = DualHistogramMedianTracker(
    hog_dim=4464, geom_dim=238,
    hog_bins=1000, hog_min=-0.005, hog_max=1.0,
    geom_bins=10000, geom_min=-60.0, geom_max=60.0
)

# Update every 2nd frame
update_histogram = (frame_idx % 2 == 1)
median_tracker.update(hog_features, geom_features, update_histogram)

# Get current medians
hog_median = median_tracker.get_hog_median()
geom_median = median_tracker.get_geom_median()
```

### Performance
- **Python version:** 47ms per frame
- **Cython version:** 0.2ms per frame (260x faster!)
- **Status:** Fully optimized (Cython)

### Cython Optimization
- **File:** `pyauface/utils/cython_extensions/cython_histogram_median.pyx`
- **Speedup:** 260x faster than pure Python
- **Features:** C-level histogram update, early termination, HOG clamping

---

## Component 10: Feature Normalization

**Purpose:** Normalize features by subtracting running median for dynamic AU models.

### Implementation
```python
for au_name, model in au_models.items():
    if model['model_type'] == 'dynamic':
        # Normalize by running median
        hog_norm = hog_features - hog_median
        geom_norm = geom_features - geom_median
        full_vector = np.concatenate([hog_norm, geom_norm])
    else:
        # Static models use original features
        full_vector = np.concatenate([hog_features, geom_features])
```

### Model Types
- **Dynamic (11 AUs):** AU01, AU02, AU05, AU09, AU15, AU17, AU20, AU23, AU25, AU26, AU45
- **Static (6 AUs):** AU04, AU06, AU07, AU10, AU12, AU14

---

## Component 11: AU Prediction (SVR Models)

**Module:** `pyauface.prediction.model_parser`
**File:** `pyauface/prediction/model_parser.py`
**Class:** `OF22ModelParser`

### Purpose
Predict AU intensities using Support Vector Regression (SVR) models.

### Algorithm
```python
# Center features
features_centered = features - model['means'].flatten()

# Compute score (SVR prediction)
score = np.dot(features_centered, model['support_vectors'].flatten()) + model['bias']

# Apply cutoff for dynamic models
if model['model_type'] == 'dynamic' and score < model['cutoff']:
    prediction = 0.0
else:
    prediction = score
```

### Implementation
```python
from pyauface.prediction import OF22ModelParser

parser = OF22ModelParser('path/to/AU_predictors/')
au_models = parser.load_all_models()

# Predict AU
prediction = parser.predict_au(features, au_models['AU12_r'])
```

### Performance
- **Time:** 30ms per frame for all 17 AUs (14% of total)
- **Status:** Warning: Optimization opportunity (vectorize predictions)

### Model Files
- **Format:** Binary `.dat` files
- **Location:** OpenFace `AU_predictors/` directory
- **Count:** 17 models (one per AU)

### Model Structure
Each model contains:
- **Support vectors:** (4702, 1) - SVR weights
- **Means:** (1, 4702) - Feature centering
- **Bias:** Scalar offset
- **Cutoff:** Threshold for dynamic models

---

## Component 12: Output

### Purpose
Output 17 AU intensities for each frame.

### Format
```python
{
    'frame': 0,
    'success': True,
    'AU01_r': 0.60,
    'AU02_r': 0.90,
    'AU04_r': 0.00,
    ...
    'AU45_r': 0.00
}
```

### CSV Output
```csv
frame,success,AU01_r,AU02_r,...,AU45_r
0,True,0.60,0.90,...,0.00
1,True,0.55,0.85,...,0.00
```

---

## Performance Summary

| Component | Time (ms) | % Total | Status |
|-----------|-----------|---------|--------|
| Video Input | ~0 | 0% | Native |
| Face Detection | 2 | 1% | Optimized (tracking) |
| Landmark Detection | 5 | 2% | Optimized (ONNX) |
| **Pose Estimation** | **80** | **37%** | Warning: Bottleneck |
| Face Alignment | 20 | 9% | Good |
| Face Masking | <1 | 0% | Optimized |
| **HOG Extraction** | **50** | **23%** | Warning: Bottleneck |
| Geometric Features | 5 | 2% | Optimized |
| Running Median | 0.2 | 0% | Optimized (Cython) |
| Normalization | <1 | 0% | Optimized |
| **AU Prediction** | **30** | **14%** | Warning: Optimization target |
| Output | <1 | 0% | Fast |
| **TOTAL** | **~217ms** | **100%** | **4.6 FPS** |

---

## Accuracy Validation

| Component | Validation Method | Result |
|-----------|-------------------|--------|
| HOG Extraction | Compare with C++ .hog files | r = 1.0 (PERFECT) |
| CalcParams | Compare params frame-by-frame | r = 0.9945 (99.45%) |
| Face Alignment | Validate via static AUs | r = 0.94 |
| Running Median | Two-pass validation | Working correctly |
| AU Prediction | Compare with C++ CSV | r = 0.83 overall |

---

## Dependencies

### External Libraries
- **pyfhog:** HOG feature extraction (C library)
- **ONNX Runtime:** Model inference (CPU/CoreML)
- **OpenCV:** Image processing
- **NumPy/SciPy:** Numerical operations

### Model Files
- **RetinaFace:** Face detection ONNX (1.7 MB)
- **PFLD:** Landmark detection ONNX (2.9 MB)
- **PDM:** 3D face model (67 KB)
- **Triangulation:** Face mask definition (1 KB)
- **AU Models:** 17 SVR models (external, ~50 MB total)

---

## Future Optimizations

### High Priority
1. **CalcParams:** Replace with OpenCV `solvePnP()` (50ms savings)
2. **PyFHOG:** Increase cell_size 8→12 (25ms savings)
3. **AU Prediction:** Vectorize all 17 models (15ms savings)

### Target
- **Current:** 217ms/frame (4.6 FPS)
- **Optimized:** 127ms/frame (7.9 FPS)
- **Improvement:** 2x faster

---

**For implementation details, see source code in `pyauface/` directory.**
