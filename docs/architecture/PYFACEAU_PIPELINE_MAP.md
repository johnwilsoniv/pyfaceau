# PyfaceAU Pipeline Component Map

**Purpose:** Systematic accuracy validation against C++ OpenFace 2.2
**Goal:** Achieve 99.9% fidelity (r > 0.83 correlation) with original implementation
**Current Status:** 87.4 FPS, 100% success rate, accuracy validation pending

---

## Pipeline Architecture

```
Input Video Frame (BGR, HxWx3)
    ↓
[1] Face Detection (RetinaFace)
    ↓
[2] Landmark Detection (PFLD 68-point)
    ↓
[3] Face Alignment (Similarity Transform)
    ↓
[4] Pose Estimation (CalcParams - PDM fitting)
    ↓
[5] HOG Feature Extraction
    ↓
[6] Geometric Feature Extraction
    ↓
[7] Running Median Tracking (for dynamic AUs)
    ↓
[8] AU Prediction (17 SVR models)
    ↓
Output: AU Intensities (17 values, 0-5 scale)
```

---

## Component 1: Face Detection

**File:** `pyfaceau/detectors/retinaface.py`
**Class:** `ONNXRetinaFaceDetector`
**Function:** Detect face bounding box in frame

### Input:
- BGR image: `(H, W, 3)` uint8 array

### Output:
- Bounding box: `[x_min, y_min, x_max, y_max]` in pixel coordinates
- 5-point landmarks: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4], [x5,y5]]`
- Confidence: float (0-1)

### C++ Equivalent:
- **OpenFace 2.2:** `Utilities::DetectFaces()` or similar
- **Note:** OpenFace 2.2 uses different face detector (may not be direct comparison)

### Validation Strategy:
```python
# Test on same frame
cpp_bbox = run_cpp_openface_face_detection(frame)
py_bbox = detector.detect_faces(frame)

# Compare IoU (Intersection over Union)
iou = compute_iou(cpp_bbox, py_bbox)
assert iou > 0.9  # Allow some variation in detector
```

### Optimization Applied:
- ONNX Runtime with attempted CoreML (falls back to CPU)
- Face tracking (skip detection on subsequent frames)

### Expected Accuracy:
- **Not directly comparable** - OpenFace 2.2 uses different detector
- **Skip this component** for accuracy validation

---

## Component 2: Landmark Detection

**File:** `pyfaceau/detectors/pfld.py`
**Class:** `CunjianPFLDDetector`
**Function:** Detect 68 facial landmarks

### Input:
- BGR image: `(H, W, 3)` uint8 array
- Face bbox: `[x_min, y_min, x_max, y_max]`

### Output:
- Landmarks: `(68, 2)` float32 array in image coordinates
- Format: `[[x0,y0], [x1,y1], ..., [x67,y67]]`

### C++ Equivalent:
- **OpenFace 2.2:** `LandmarkDetector::DetectLandmarksInVideo()`
- **Model:** CLNF (Constrained Local Neural Fields)
- **Output:** 68-point landmarks (same format)

### Validation Strategy:
```python
# Test on same frame with same bbox
cpp_landmarks = run_cpp_openface_landmarks(frame, bbox)
py_landmarks = pfld_detector.detect_landmarks(frame, bbox)

# Compute mean Euclidean distance (normalized by interocular distance)
iod = compute_interocular_distance(cpp_landmarks)
distances = np.linalg.norm(cpp_landmarks - py_landmarks, axis=1)
nme = np.mean(distances) / iod

# NME should be < 5% for good alignment
assert nme < 0.05
```

### Optimization Applied:
- ONNX Runtime with CoreML acceleration (91.5% ops on Neural Engine)
- 5.0x speedup (4.9ms → 0.98ms)

### Expected Accuracy:
- **Moderate correlation expected** - Different landmark detector
- PFLD vs CLNF: Different architectures, similar accuracy
- Target: NME < 5%

---

## Component 3: Face Alignment

**File:** `pyfaceau/alignment/face_aligner.py`
**Class:** `FaceAligner`
**Function:** Align face to canonical 112×112 pose using similarity transform

### Input:
- BGR image: `(H, W, 3)` uint8 array
- 68 landmarks: `(68, 2)` float32 array

### Output:
- Aligned face: `(112, 112, 3)` uint8 BGR image
- Similarity matrix: `(2, 3)` affine transform matrix

### C++ Equivalent:
- **OpenFace 2.2:** Implicit in face normalization before HOG extraction
- **Function:** `FaceAnalyser::AlignFace()` or similar

### Validation Strategy:
```python
# Both should produce similar aligned face
cpp_aligned = run_cpp_openface_align(frame, landmarks)
py_aligned = aligner.align(frame, landmarks)

# Compare pixel-wise similarity (allow some JPEG artifacts)
mse = np.mean((cpp_aligned - py_aligned) ** 2)
psnr = 20 * np.log10(255.0 / np.sqrt(mse))

# PSNR > 30 dB is good alignment
assert psnr > 30
```

### Optimization Applied:
- Vectorized NumPy operations
- Pre-computed reference points

### Expected Accuracy:
- **High correlation expected** - Standard similarity transform
- Target: PSNR > 30 dB

---

## Component 4: Pose Estimation (CalcParams)

**File:** `pyfaceau/alignment/calc_params.py`
**Class:** `CalcParams`
**Function:** Fit 3D PDM to 2D landmarks using Gauss-Newton optimization

### Input:
- 2D landmarks: `(68, 2)` or `(136,)` float32 array

### Output:
- Global params: `(6,)` array `[scale, rx, ry, rz, tx, ty]`
- Local params: `(34,)` array of PCA coefficients

### C++ Equivalent:
- **OpenFace 2.2:** `PDM::CalcParams()` in `PDM.cpp` lines 508-705
- **Exact same algorithm** - This is our Python replication!

### Validation Strategy:
```python
# CRITICAL: This should match C++ almost exactly
cpp_global, cpp_local = run_cpp_openface_calcparams(landmarks)
py_global, py_local = calc_params.calc_params(landmarks)

# Global params correlation
r_global = np.corrcoef(cpp_global, py_global)[0, 1]
assert r_global > 0.999  # Should be nearly identical

# Local params correlation
r_local = np.corrcoef(cpp_local, py_local)[0, 1]
assert r_local > 0.995  # Should be very high
```

### Optimization Applied:
- Numba JIT compilation (13.3x speedup: 42.5ms → 3.2ms)
- Cython rotation update (99.9% accuracy guaranteed)

### Expected Accuracy:
- **CRITICAL COMPONENT** - Must match C++ exactly
- Target: r > 0.999 for global params, r > 0.995 for local params
- **This is our gold standard validation!**

---

## Component 5: HOG Feature Extraction

**File:** `pyfaceau/features/hog_extractor.py`
**Class:** `HOGExtractor`
**Function:** Extract HOG features from aligned face

### Input:
- Aligned face: `(112, 112, 3)` BGR uint8 image

### Output:
- HOG features: `(4464,)` float32 array

### C++ Equivalent:
- **OpenFace 2.2:** `FaceAnalyser::ExtractHOGFeatures()`
- Uses dlib's HOG implementation

### Validation Strategy:
```python
# HOG should match exactly if alignment matches
cpp_hog = run_cpp_openface_hog(aligned_face)
py_hog = hog_extractor.extract(aligned_face)

# Element-wise correlation
r_hog = np.corrcoef(cpp_hog.flatten(), py_hog.flatten())[0, 1]
assert r_hog > 0.99

# Also check dimension
assert cpp_hog.shape == py_hog.shape == (4464,)
```

### Optimization Applied:
- NumPy/OpenCV optimized operations

### Expected Accuracy:
- **High correlation expected** - Deterministic algorithm
- Target: r > 0.99 (allows minor floating-point differences)

---

## Component 6: Geometric Feature Extraction

**File:** `pyfaceau/features/geometric_features.py`
**Class:** `GeometricFeatureExtractor`
**Function:** Extract 3D geometric features from PDM parameters

### Input:
- Global params: `(6,)` array
- Local params: `(34,)` array
- PDM model: Mean shape, principal components

### Output:
- Geometric features: `(238,)` float32 array

### C++ Equivalent:
- **OpenFace 2.2:** `FaceAnalyser::ExtractGeometricFeatures()`
- Computes inter-landmark distances, angles, etc.

### Validation Strategy:
```python
# Should match exactly if CalcParams matches
cpp_geom = run_cpp_openface_geometric(params_global, params_local)
py_geom = geom_extractor.extract(params_global, params_local)

# Element-wise correlation
r_geom = np.corrcoef(cpp_geom, py_geom)[0, 1]
assert r_geom > 0.999  # Should be nearly identical

# Check dimension
assert cpp_geom.shape == py_geom.shape == (238,)
```

### Optimization Applied:
- Vectorized NumPy operations
- Minimal overhead (0.01ms)

### Expected Accuracy:
- **Very high correlation expected** - Deterministic computation
- Target: r > 0.999

---

## Component 7: Running Median Tracking

**File:** `pyfaceau/features/histogram_median_tracker.py`
**Class:** `HistogramMedianTracker`
**Function:** Track temporal median of features for dynamic AU models

### Input:
- Combined features: `(4702,)` float32 array (4464 HOG + 238 geometric)

### Output:
- Running median: `(4702,)` float32 array

### C++ Equivalent:
- **OpenFace 2.2:** `FaceAnalyser::UpdateRunningMedian()`
- Uses histogram-based median tracker

### Validation Strategy:
```python
# Process same sequence of frames
for frame in test_frames:
    cpp_median = cpp_tracker.update(features[frame])
    py_median = py_tracker.update(features[frame])

# Check final median values
r_median = np.corrcoef(cpp_median, py_median)[0, 1]
assert r_median > 0.99
```

### Optimization Applied:
- Cython JIT compilation (179x speedup: 161ms → 0.9ms)

### Expected Accuracy:
- **High correlation expected** - Deterministic histogram algorithm
- Cython version: 99.9% match to C++ (verified in previous work)
- Target: r > 0.99

---

## Component 8: AU Prediction

**File:** `pyfaceau/prediction/batched_au_predictor.py`
**Class:** `BatchedAUPredictor`
**Function:** Predict 17 AU intensities using SVR models

### Input:
- HOG features: `(4464,)` float32 array
- Geometric features: `(238,)` float32 array
- Running median: `(4702,)` float32 array

### Output:
- AU predictions: Dictionary of 17 AU intensities (0-5 scale)
  ```python
  {
      'AU01_r': 0.234,
      'AU02_r': 0.567,
      ...
      'AU45_r': 1.234
  }
  ```

### C++ Equivalent:
- **OpenFace 2.2:** `FaceAnalyser::PredictAUs()`
- Uses LibSVM models loaded from `.dat` files

### Validation Strategy:
```python
# CRITICAL: This is the final output we care about!
cpp_aus = run_cpp_openface_aus(hog, geom, median)
py_aus = au_predictor.predict(hog, geom, median)

# Per-AU correlation
for au_name in ['AU01_r', 'AU02_r', ..., 'AU45_r']:
    cpp_vals = [cpp_aus[frame][au_name] for frame in test_frames]
    py_vals = [py_aus[frame][au_name] for frame in test_frames]

    r_au = np.corrcoef(cpp_vals, py_vals)[0, 1]
    print(f"{au_name}: r = {r_au:.4f}")

    # OpenFace 2.2 paper reports r > 0.83 as good
    assert r_au > 0.83

# Overall correlation
all_cpp = np.concatenate([cpp_aus[f].values() for f in test_frames])
all_py = np.concatenate([py_aus[f].values() for f in test_frames])
r_overall = np.corrcoef(all_cpp, all_py)[0, 1]

print(f"Overall AU correlation: r = {r_overall:.4f}")
assert r_overall > 0.83
```

### Optimization Applied:
- Batched vectorized SVR prediction (2-5x speedup)
- Single matrix operation for all 17 AUs

### Expected Accuracy:
- **CRITICAL COMPONENT** - This is our final output!
- Target: r > 0.83 per AU (OpenFace 2.2 benchmark)
- SVR models are identical (loaded from same `.dat` files)
- Only differences: floating-point precision, feature extraction chain

---

## Testing Strategy

### Phase 1: Component-Level Validation (Bottom-Up)

1. **Start with known-good inputs from C++**
   - Export intermediate outputs from C++ OpenFace 2.2
   - Use as ground truth for each component

2. **Test each component in isolation**
   ```
   CalcParams → Geometric Features → AU Prediction
   ```

3. **Identify accuracy bottlenecks**
   - Which component has lowest correlation?
   - Focus optimization there

### Phase 2: End-to-End Validation

1. **Process same video in both implementations**
   ```bash
   # C++ OpenFace 2.2
   FeatureExtraction -f test_video.mp4 -out_dir cpp_output/

   # Python PyfaceAU
   python3.10 benchmark_detailed.py --video test_video.mp4 --output py_output.csv
   ```

2. **Compare AU outputs frame-by-frame**
   ```python
   cpp_aus = pd.read_csv('cpp_output/test_video.csv')
   py_aus = pd.read_csv('py_output.csv')

   for au in AU_NAMES:
       r = np.corrcoef(cpp_aus[au], py_aus[au])[0, 1]
       print(f"{au}: r = {r:.4f}")
   ```

3. **Visualize discrepancies**
   ```python
   plt.plot(cpp_aus['AU12_r'], label='C++ OpenFace')
   plt.plot(py_aus['AU12_r'], label='Python PyfaceAU')
   plt.legend()
   plt.show()
   ```

### Phase 3: Accuracy Certification

**Success Criteria:**
- Overall AU correlation: **r > 0.83** (matches OpenFace 2.2 benchmark)
- Per-AU correlation: **r > 0.83** for at least 15/17 AUs
- CalcParams correlation: **r > 0.995** (critical component)
- No systematic bias (mean error ≈ 0)

---

## Test Data Requirements

### Recommended Test Videos

1. **DISFA Dataset** (if available)
   - OpenFace 2.2 was benchmarked on this
   - 27 videos, 130,000+ frames with AU annotations

2. **BP4D Dataset** (if available)
   - Another OpenFace benchmark dataset

3. **Custom Test Video**
   - `/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0434.MOV`
   - 972 frames, 1080x1920, 59.96 FPS

### C++ OpenFace 2.2 Reference Outputs

Need to generate:
```bash
cd /path/to/openface/build/bin
./FeatureExtraction -f test_video.mp4 -out_dir reference_output/ -2Dfp -3Dfp -pdmparams -pose -aus -hogalign
```

This produces:
- `test_video.csv` - Frame-by-frame AU intensities
- `test_video_of_details.txt` - Detailed parameters (pose, PDM, etc.)
- `test_video.hog` - HOG features (if needed)

---

## Known Differences (Expected)

### 1. Face Detection
- **C++:** Uses different detector (Haar cascades or MTCNN)
- **Python:** Uses RetinaFace (ONNX)
- **Impact:** Different bboxes, but shouldn't affect AU predictions if landmarks are good

### 2. Landmark Detection
- **C++:** Uses CLNF (Constrained Local Neural Fields)
- **Python:** Uses PFLD (deep learning, ONNX)
- **Impact:** Moderate - different detectors, similar accuracy

### 3. Floating-Point Precision
- **C++:** Uses `float` (32-bit) in some places, `double` (64-bit) in others
- **Python:** Uses `np.float32` consistently
- **Impact:** Minor differences (<0.001) acceptable

### 4. Numerical Solvers
- **C++:** Uses OpenCV's `cv::solve()`
- **Python:** Uses `cv2.solve()` or `scipy.linalg.solve()`
- **Impact:** Should be identical if same algorithm

---

## Debugging Tools

### Compare Intermediate Outputs
```python
def compare_outputs(cpp_val, py_val, component_name, threshold=0.99):
    """Compare C++ vs Python outputs"""
    if isinstance(cpp_val, dict):
        # Compare dictionaries (e.g., AU predictions)
        for key in cpp_val.keys():
            r = np.corrcoef([cpp_val[key]], [py_val[key]])[0, 1]
            status = "✅" if r > threshold else "❌"
            print(f"{status} {component_name}.{key}: r = {r:.4f}")
    else:
        # Compare arrays
        r = np.corrcoef(cpp_val.flatten(), py_val.flatten())[0, 1]
        mae = np.mean(np.abs(cpp_val - py_val))
        status = "✅" if r > threshold else "❌"
        print(f"{status} {component_name}: r = {r:.4f}, MAE = {mae:.6f}")
```

### Visualize Differences
```python
def visualize_au_comparison(cpp_aus, py_aus, au_name):
    """Plot C++ vs Python AU predictions"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(cpp_aus[au_name], label='C++ OpenFace', alpha=0.7)
    plt.plot(py_aus[au_name], label='Python PyfaceAU', alpha=0.7)
    plt.title(f'{au_name} Time Series')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(cpp_aus[au_name], py_aus[au_name], alpha=0.5)
    plt.plot([0, 5], [0, 5], 'r--')
    plt.xlabel('C++ OpenFace')
    plt.ylabel('Python PyfaceAU')
    plt.title('Correlation')

    plt.subplot(1, 3, 3)
    plt.hist(cpp_aus[au_name] - py_aus[au_name], bins=50)
    plt.xlabel('Error (C++ - Python)')
    plt.title('Error Distribution')

    plt.tight_layout()
    plt.show()
```

---

## Next Steps

1. **Generate C++ Reference Outputs**
   - Run OpenFace 2.2 FeatureExtraction on test video
   - Save all intermediate outputs

2. **Create Validation Script**
   - Load C++ and Python outputs
   - Compute correlations for each component
   - Generate accuracy report

3. **Fix Any Accuracy Issues**
   - Start with lowest-correlation component
   - Debug numerical differences
   - Ensure bit-for-bit accuracy where possible

4. **Document Final Accuracy**
   - Create accuracy certification report
   - Include correlation plots for all 17 AUs
   - Publish results

---

## Current Status

| Component | Optimization | Speed | Accuracy Validated |
|-----------|-------------|-------|-------------------|
| Face Detection | ONNX + Tracking | 1.5ms | ⏳ N/A (different detector) |
| Landmark Detection | ONNX + CoreML | 1.0ms | ⏳ Pending |
| Face Alignment | Vectorized | 0.4ms | ⏳ Pending |
| Pose Estimation | Numba JIT | 2.9ms | ⏳ Pending (CRITICAL) |
| HOG Extraction | NumPy | 0.15ms | ⏳ Pending |
| Geometric Features | Vectorized | 0.01ms | ⏳ Pending |
| Running Median | Cython | 0.9ms | 99.9% (verified) |
| AU Prediction | Batched | 0.22ms | ⏳ Pending (CRITICAL) |

**Overall Pipeline:** 87.4 FPS, 11.4ms per frame, 100% success rate

**Next Priority:** Validate CalcParams and AU Prediction accuracy!
