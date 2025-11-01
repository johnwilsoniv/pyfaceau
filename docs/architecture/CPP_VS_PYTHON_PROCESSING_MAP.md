# C++ OpenFace 2.2 vs Python PyFaceAU Processing Comparison

**Purpose:** Systematically map the exact C++ OpenFace 2.2 processing pipeline to identify remaining differences that prevent 99%+ correlation across all AUs.

**Current Status:**
- Using identical C++ CLNF landmarks (eliminates landmark detector as variable)
- With two-pass processing: 14/17 AUs passing (91.22% mean)
- Some AUs at 99%+ (AU45: 99.43%, AU12: 99.78%, AU06: 99.77%)
- Others below target (AU05: 72.68%, AU15: 60.22%, AU20: 74.73%)

---

## Frame Processing Pipeline

### Step 1: Landmark Detection
**C++ (FaceAnalyser.cpp:290-295)**
```cpp
// We are using CLNF landmarks from C++, so this is IDENTICAL
```

**Python (validate_accuracy.py:79-89)**
```python
# Extract landmarks from C++ reference CSV
cpp_row = cpp_df.iloc[frame_idx]
x_cols = [f'x_{i}' for i in range(68)]
y_cols = [f'y_{i}' for i in range(68)]
landmarks_68 = np.zeros((68, 2), dtype=np.float64)
landmarks_68[:, 0] = cpp_row[x_cols].values
landmarks_68[:, 1] = cpp_row[y_cols].values
```

**Status:** IDENTICAL (using same C++ landmarks)

---

### Step 2: CalcParams (Pose + Shape Estimation)

**C++ (FaceAnalyser.cpp:312-320)**
```cpp
bool success = pdm.CalcParams(global_params, local_params, local_params_landmarks,
                               landmarks_2D, validities);
params_global.at<double>(0) = global_params[0];  // scale
params_global.at<double>(1) = global_params[1];  // rx
params_global.at<double>(2) = global_params[2];  // ry
params_global.at<double>(3) = global_params[3];  // rz
params_global.at<double>(4) = global_params[4];  // tx
params_global.at<double>(5) = global_params[5];  // ty
params_local = local_params;  // PCA coefficients (34 dims)
```

**Python (validate_accuracy.py:100-103)**
```python
params_result = calc_params.process(landmarks_68)
params_global = params_result['params_global']  # (6,)
params_local = params_result['params_local']    # (34,)
```

**Validation Results:**
- Global params: p_rx correlation = 0.9923 (below 0.995 target)
- Local params: mean correlation = 0.9824 (below 0.995 target)

**Status:** Warning: MOSTLY CORRECT (98%+ correlation, but not 99.5%+)
- Could affect geometric features slightly

---

### Step 3: Face Alignment

**C++ (FaceAnalyser.cpp:330-345)**
```cpp
// Align face to canonical view
cv::Mat_<uchar> aligned_face;
cv::Mat sim_ref_frame = AlignedFace(aligned_face, frame, landmarks_2D,
                                    AlignFaceParams());
// AlignFaceParams(): sim_scale=0.7, out_width=112, out_height=112
```

**Python (validate_accuracy.py:108-109)**
```python
aligned_face = face_aligner.align_face(frame, landmarks_68)
# Uses sim_scale=0.7, output=(112, 112)
```

**Status:** LIKELY CORRECT
- Need to verify exact similarity transform parameters
- Need to verify interpolation method (C++ uses cv::warpAffine)

---

### Step 4: HOG Feature Extraction

**C++ (FaceAnalyser.cpp:356-401)**
```cpp
// Extract FHOG features (4x4 cells, 31 bins + 4 orientation bins = 35 total per cell)
cv::Mat_<double> hog_descriptor;
Extract_FHOG_descriptor(hog_descriptor, aligned_face, num_hog_rows, num_hog_cols);
// Result: (num_rows * num_cols * 31) dimensional vector

// Reshape and flatten
hog_descriptor = hog_descriptor.t();
hog_descriptor = hog_descriptor.reshape(1, 1);
```

**Python (validate_accuracy.py:114-115)**
```python
hog_features = extract_hog_features(aligned_face, visualize=False)
# Uses pyfhog (C binding to C++ FHOG implementation)
```

**Status:** ASSUMED CORRECT
- Uses same C++ FHOG library via pyfhog
- Should be bit-identical

**TODO:** Verify pyfhog produces identical output to OpenFace's Extract_FHOG_descriptor

---

### Step 5: Geometric Feature Construction

**C++ (FaceAnalyser.cpp:410-423)**
```cpp
// Convert params_local to CV_64F
params_local.convertTo(geom_descriptor_frame, CV_64F);

// Stack with actual feature point locations (without mean)
cv::Mat_<double> princ_comp_d;
pdm.princ_comp.convertTo(princ_comp_d, CV_64F);
cv::Mat_<double> locs = princ_comp_d * geom_descriptor_frame.t();

// CRITICAL: Concatenate [locs, params_local]
cv::hconcat(locs.t(), geom_descriptor_frame.clone(), geom_descriptor_frame);
// Result: (204 + 34 = 238) dimensional vector
```

**Python (validate_accuracy.py:120-121)**
```python
geom_features = extract_geom_features(params_global, params_local)
# Constructs [shape_3d, params_local] -> (204 + 34 = 238) dims
```

**Implementation (pyfaceau/features/geom_extractor.py):**
```python
def extract_geom_features(params_global, params_local, pdm):
    # Reconstruct 3D shape from PCA
    shape_3d = pdm.mean_shape + pdm.principal_components @ params_local
    # Concatenate [shape_3d (204,), params_local (34,)] -> (238,)
    return np.concatenate([shape_3d.flatten(), params_local.flatten()])
```

**Status:** Warning: VERIFY ORDER
- C++ concatenates `[locs.t(), params_local]`
- Need to confirm `locs = princ_comp * params_local.t()` produces shape in same order

---

### Step 6: Running Median Update

**C++ (FaceAnalyser.cpp:404-428)**
```cpp
// Update HOG median (every frame)
UpdateRunningMedian(this->hog_desc_hist[orientation_to_use],
                    this->hog_hist_sum[orientation_to_use],
                    this->hog_desc_median,
                    hog_descriptor,
                    update_median,  // true if frames_tracking % 2 == 0
                    this->num_bins_hog,    // 1000
                    this->min_val_hog,     // -0.005
                    this->max_val_hog);    // 1.0

// CRITICAL: Clamp HOG median to >= 0
this->hog_desc_median.setTo(0, this->hog_desc_median < 0);

// Update geometric median (every other frame)
if(frames_tracking % 2 == 1)
{
    UpdateRunningMedian(this->geom_desc_hist,
                        this->geom_hist_sum,
                        this->geom_descriptor_median,
                        geom_descriptor_frame,
                        update_median,
                        this->num_bins_geom,   // 10000
                        this->min_val_geom,    // -60
                        this->max_val_geom);   // 60
}
```

**Python (validate_accuracy.py:132-139)**
```python
# Update running median tracker
update_histogram = (frame_idx < 3000)
pipeline.running_median.update(hog_features, geom_features, update_histogram)

# Get current median
running_median = pipeline.running_median.get_combined_median()
```

**Python Implementation (histogram_median_tracker.py:194-211)**
```python
def update(self, hog_features, geom_features, update_histogram=True):
    self.hog_tracker.update(hog_features, update_histogram)

    # CRITICAL: Clamp HOG median to >= 0 (matches C++ line 405)
    self.hog_tracker.current_median[self.hog_tracker.current_median < 0] = 0.0

    self.geom_tracker.update(geom_features, update_histogram)
```

**Status:** Warning: VERIFY TIMING
- C++ updates HOG every frame, geometric every other frame
- Python updates both every frame
- **THIS COULD BE THE ISSUE!**

---

### Step 7: SVR Prediction

**C++ (SVR_dynamic_lin_regressors.cpp:114)**
```cpp
// For dynamic models (with running median)
cv::Mat_<double> input;
cv::hconcat(fhog_descriptor, geom_params, input);

cv::Mat_<double> run_med;
cv::hconcat(running_median, running_median_geom, run_med);

preds = (input - this->means - run_med) * this->support_vectors + this->biases;
```

**Python (batched_au_predictor.py:133-148)**
```python
# Concatenate features
full_vector = np.concatenate([hog_features, geom_features])  # (4702,)

# Center and subtract running median for dynamic models
centered = full_vector - self.all_means
centered[self.dynamic_mask] -= running_median

# SVR prediction
predictions = np.sum(centered * self.all_support_vectors, axis=1) + self.all_biases
predictions = np.clip(predictions, 0.0, 5.0)
```

**Status:** CORRECT (verified from C++ source)

---

### Step 8: Two-Pass Re-prediction

**C++ (FaceAnalyser.cpp:550-620)**
```cpp
// After all frames processed, re-predict early frames with final median
void FaceAnalyser::PostprocessOutputFile(...)
{
    // For frames 0-3000, re-predict using final stabilized median
    if(dynamic)
    {
        // ... re-prediction logic ...
    }
}
```

**Python (validate_accuracy.py:166-184)**
```python
# Two-pass processing
final_median = pipeline.running_median.get_combined_median()

for result in results:
    hog_features = result['hog_features']
    geom_features = result['geom_features']

    # Re-predict with final median
    au_results_repredicted = pipeline._predict_aus(hog_features, geom_features, final_median)
    result['aus'] = au_results_repredicted
```

**Status:** IMPLEMENTED (fixed major correlation issues)

---

## Key Findings & Suspected Issues

### 🔴 CRITICAL: Geometric Median Update Frequency

**C++:** Updates geometric median **every other frame** (line 424: `if(frames_tracking % 2 == 1)`)
**Python:** Updates geometric median **every frame**

This difference could cause geometric feature normalization to diverge!

### Warning: MEDIUM: CalcParams Accuracy

**Issue:** CalcParams correlations are slightly below target (98% vs 99.5%)
- p_rx: 0.9923 (should be > 0.995)
- Local params mean: 0.9824 (should be > 0.995)

**Impact:** Small errors in params_local could accumulate in geometric features

### Warning: LOW: Geometric Feature Construction Order

**Need to verify:** Does `shape_3d = mean_shape + princ_comp @ params_local` match C++ `locs = princ_comp * params_local.t()`?

---

## Action Items

1. **HIGHEST PRIORITY:** Fix geometric median update frequency to match C++ (every other frame)
2. **HIGH:** Investigate CalcParams accuracy gap (p_rx and local params)
3. **MEDIUM:** Verify geometric feature construction produces identical ordering to C++
4. **LOW:** Verify pyfhog produces bit-identical output to C++ Extract_FHOG_descriptor

---

## Current Performance Summary

**With C++ CLNF landmarks + two-pass processing:**

| Category | Performance |
|----------|-------------|
| Static AUs (mean) | 98.70% |
| Dynamic AUs (mean) | ~88% |
| **Overall (mean)** | **91.22%** |
| AUs passing (r>0.83) | 14/17 (82.4%) |

**Top performers (99%+):**
- AU45 (Blink): 99.43%
- AU12 (Lip Corner Puller): 99.78%
- AU06 (Cheek Raiser): 99.77%
- AU25 (Lips Part): 98.92%

**Underperformers (<80%):**
- AU05 (Upper Lid Raiser): 72.68%
- AU15 (Lip Corner Depressor): 60.22%
- AU20 (Lip Stretcher): 74.73%

---

## Next Steps

Implement geometric median update frequency fix and re-test accuracy.
