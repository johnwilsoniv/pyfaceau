# Performance Optimization Challenge - pyAUface AU Extraction Pipeline

**Date:** 2025-10-30
**Status:** WORKING but 7x SLOWER than native C++ OpenFace
**Goal:** Optimize from 4.6 FPS → target 15-20 FPS (2-3x improvement)

---

## Executive Summary

We have a **fully functional pure Python implementation** of OpenFace 2.2's AU extraction pipeline with CoreML acceleration and face tracking. It works correctly (100% success rate), but runs at **4.6 FPS** compared to native C++ OpenFace's **32.9 FPS**.

**Your mission:** Identify and fix the performance bottlenecks to get us closer to C++ performance.

---

## Current Performance Benchmarks

### Test Video
- **Source:** IMG_0942_left_mirrored.mp4
- **Resolution:** 1080x1920
- **Frames:** 1,110 frames
- **FPS:** 59 FPS (original)

### Performance Results

| Implementation | Total Time | FPS | Per Frame | Speedup vs pyAUface |
|----------------|------------|-----|-----------|-------------------|
| **Native C++ OpenFace 2.2** | **33.8s** | **32.9** | **30ms** | **7.1x faster** |
| S1 (OpenFace 3.0 Neural) | ~40s | 28 | 35ms | 6.1x faster |
| **pyAUface (Current)** | **240s** | **4.6** | **217ms** | **Baseline** |

### 100 Frame Test Results
```
pyAUface Performance (100 frames):
Success: 100/100 frames (100%)
⏱️ Time: 21.69s
 Per frame: 217ms
 FPS: 4.6

Face Tracking Statistics:
Tracking enabled: True
Frames since last detection: 99
Detection failures: 0
Total RetinaFace calls: 1/100 (99 skipped!)
```

**Face tracking IS working** - we successfully skip 99% of RetinaFace detections. The bottleneck is elsewhere.

---

## Component Performance Estimates

Based on profiling and analysis, here's where we think the 217ms/frame is going:

| Component | Estimated Time | % of Total | Status |
|-----------|----------------|------------|--------|
| **CalcParams (Pose Estimation)** | **~80ms** | **37%** | 🔴 BOTTLENECK |
| **PyFHOG (HOG Extraction)** | **~50ms** | **23%** | 🔴 BOTTLENECK |
| **17 SVR Models (AU Prediction)** | **~30ms** | **14%** | 🟡 OPTIMIZE |
| Landmark Detection (PFLD) | ~30ms | 14% | OK (ONNX) |
| Face Alignment | ~20ms | 9% | OK |
| Running Median | ~5ms | 2% | OK (Cython) |
| Face Detection (RetinaFace) | ~2ms avg | 1% | OK (skipped 99%) |
| **TOTAL** | **~217ms** | **100%** | |

### Key Bottlenecks Identified

**1. CalcParams (~80ms/frame - 37% of time)**
- **What:** Iterative Gauss-Newton optimization for 3D pose estimation
- **Current:** Pure Python/numpy implementation
- **C++ equivalent:** ~5-10ms (optimized, compiled)
- **Potential improvement:** 50-70ms savings

**2. PyFHOG (~50ms/frame - 23% of time)**
- **What:** HOG feature extraction on 112x112 aligned face
- **Current:** Python C extension, cell_size=8
- **C++ equivalent:** ~10-15ms (native FHOG library)
- **Potential improvement:** 25-35ms savings

**3. 17 SVR Predictions (~30ms/frame - 14% of time)**
- **What:** 17 separate SVR model predictions
- **Current:** Sequential numpy dot products
- **C++ equivalent:** ~0.5ms (optimized linear algebra)
- **Potential improvement:** 20-25ms savings

---

## Architecture Overview

### Pipeline Flow

```
Frame Input (1080x1920)
    ↓
[1] Face Detection (RetinaFace CoreML) - ~2ms avg (tracking skips 99%)
    ↓
[2] Landmark Detection (PFLD ONNX) - ~30ms
    ↓
[3] Pose Estimation (CalcParams) - ~80ms ← BOTTLENECK #1
    ↓
[4] Face Alignment - ~20ms
    ↓
[5] HOG Extraction (PyFHOG) - ~50ms ← BOTTLENECK #2
    ↓
[6] Geometric Features - ~5ms
    ↓
[7] Running Median Update - ~5ms
    ↓
[8] AU Prediction (17 SVRs) - ~30ms ← BOTTLENECK #3
    ↓
Output: 17 AU intensities
```

### Key Files

**Main Pipeline:**
- `full_python_au_pipeline.py` - Main pipeline orchestration
  - `_process_frame()` - Per-frame processing (lines 572-750)
  - Uses queue architecture for CoreML (main thread: VideoCapture, worker thread: processing)

**Bottleneck Components:**
1. **CalcParams:** `calc_params.py`
   - `calc_params()` method (lines 398-700)
   - Gauss-Newton optimization in pure Python/numpy

2. **PyFHOG:** `../pyfhog/src/pyfhog/`
   - C extension wrapper around FHOG
   - Called via `pyfhog.extract_fhog_features(aligned_face, cell_size=8)`

3. **AU Prediction:** `openface22_au_predictor.py`
   - 17 separate SVR models loaded from `.dat` files
   - `_predict_aus()` in `full_python_au_pipeline.py` (lines 710-750)

**Supporting Components:**
- `onnx_retinaface_detector.py` - Face detection (CoreML accelerated)
- `cunjian_pfld_detector.py` - Landmark detection (ONNX)
- `openface22_face_aligner.py` - Face alignment
- `pdm_parser.py` - PDM shape model
- `running_median_tracker.py` - Running median (Cython optimized)

---

## What We've Already Optimized

**Face Detection:** CoreML acceleration + face tracking → 99% reduction
**Running Median:** Cython implementation → 260x faster than Python
**Queue Architecture:** Proper threading for macOS CoreML
**Face Tracking:** "Detect on failure" strategy working perfectly

---

## Optimization Opportunities

### High ROI Optimizations

#### 1. CalcParams Optimization (~80ms → 30-40ms target)

**Current Problem:**
- Pure Python/numpy Gauss-Newton optimization
- Iterative least-squares solving
- Many matrix operations in Python loops

**Potential Solutions:**
- **Option A:** Use OpenCV's `solvePnP()` instead (single optimized C++ call)
- **Option B:** Reduce number of iterations (currently uses default)
- **Option C:** Numba JIT compilation of critical loops
- **Option D:** Cython implementation of Gauss-Newton solver
- **Option E:** Pre-compute/cache stable pose parameters

**Files to Modify:**
- `calc_params.py` - Main optimization target
- `full_python_au_pipeline.py:676-683` - Where CalcParams is called

**Expected Gain:** 40-50ms (20-25% total improvement)

---

#### 2. PyFHOG Optimization (~50ms → 20-30ms target)

**Current Problem:**
- HOG extraction on 112x112 image with cell_size=8
- Results in many cells → more computation

**Potential Solutions:**
- **Option A:** Increase cell_size (8 → 12 or 16) - fewer cells to compute
- **Option B:** Reduce aligned face size (112x112 → 96x96)
- **Option C:** Use skimage.feature.hog (potentially faster implementation)
- **Option D:** Parallel HOG extraction if not already parallelized
- **Option E:** Investigate if HOG parameters can be relaxed

**Files to Modify:**
- `full_python_au_pipeline.py:711-714` - HOG extraction call
- `openface22_face_aligner.py` - Aligned face output size

**Expected Gain:** 20-30ms (10-15% total improvement)

**Note:** Changing HOG parameters may affect AU accuracy - requires validation!

---

#### 3. AU Prediction Batching (~30ms → 15-20ms target)

**Current Problem:**
- 17 sequential SVR predictions
- Each prediction: `np.dot(centered_features, support_vectors) + bias`
- No batching/vectorization across models

**Potential Solutions:**
- **Option A:** Stack all SVs into single matrix, do one big matrix multiply
- **Option B:** Numba JIT compilation of prediction loop
- **Option C:** Parallelize predictions across AUs (multiprocessing/threading)
- **Option D:** Cache/pre-compute mean centering

**Files to Modify:**
- `full_python_au_pipeline.py:710-750` - `_predict_aus()` method
- `openface22_model_parser.py` - Model loading/structure

**Expected Gain:** 10-15ms (5-7% total improvement)

---

### Medium ROI Optimizations

#### 4. Landmark Detection (~30ms → 20-25ms target)

**Current:** ONNX Runtime CPU inference

**Potential Solutions:**
- Reduce PFLD input size if possible
- ONNX graph optimization
- Check if CoreML can accelerate PFLD (probably not worth complexity)

**Expected Gain:** 5-10ms (2-5% improvement)

---

#### 5. Face Alignment (~20ms → 15ms target)

**Current:** warpAffine + masking

**Potential Solutions:**
- Skip masking if not critical
- Optimize warpAffine parameters
- Reduce output size (ties into HOG optimization)

**Expected Gain:** 5ms (2% improvement)

---

## Profiling Data Needed

To optimize effectively, we need **actual profiling data**, not estimates. Here's what to collect:

### Recommended Profiling Approach

```python
import time
import cProfile
import pstats

# Add timing to each component in _process_frame()
timings = {}

# Example:
t0 = time.perf_counter()
params_global, params_local = self.calc_params.calc_params(landmarks_68.flatten())
timings['calc_params'] = time.perf_counter() - t0

# Repeat for all components, aggregate over 100 frames
```

### What We Need

1. **Per-component timing** (average over 100 frames)
2. **CalcParams breakdown:**
   - Time per iteration
   - Number of iterations to convergence
   - Matrix operation hotspots
3. **PyFHOG breakdown:**
   - Is it the C extension or Python wrapper overhead?
4. **SVR prediction breakdown:**
   - Time per model
   - Numpy operation hotspots

---

## Constraints & Requirements

### Must Maintain

**100% Python** - No C++ compilation required
**Cross-platform** - Works on Windows/Mac/Linux
**Accuracy** - Must match OpenFace 2.2 outputs
**All 17 AUs** - Complete AU extraction
**Face tracking** - Keep the optimization we have

### Can Relax (with validation)

- HOG parameters (cell_size, bins, etc.)
- Aligned face size
- CalcParams convergence criteria
- Number of Gauss-Newton iterations

### Cannot Change

- Video input format
- Output format (DataFrame with AU intensities)
- CoreML queue architecture (macOS requirement)
- Face tracking strategy

---

## Success Metrics

### Minimum Acceptable (50% improvement)
- **Target:** 7-8 FPS (130-150ms/frame)
- **Requires:** Fix 1-2 major bottlenecks

### Good Performance (2x improvement)
- **Target:** 10-12 FPS (80-100ms/frame)
- **Requires:** Fix all 3 major bottlenecks

### Excellent Performance (3x improvement)
- **Target:** 15-20 FPS (50-65ms/frame)
- **Requires:** All optimizations + creative solutions

### Stretch Goal (approach C++ parity)
- **Target:** 25-30 FPS (33-40ms/frame)
- **Requires:** Major architectural changes or compiled components

---

## Testing & Validation

### Performance Testing

**Test Script:** `coreml_only_test.py`
```bash
/usr/local/bin/python3.10 coreml_only_test.py
```

**Test Video:** `/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4`

**What to measure:**
- Per-frame time (target: <100ms)
- FPS (target: >10)
- Success rate (must stay 100%)

### Accuracy Validation

After optimizations, compare AU outputs against baseline:
```python
# Compare first 100 frames before/after optimization
baseline_df = pd.read_csv('baseline_results.csv')
optimized_df = pd.read_csv('optimized_results.csv')

for au_col in [f'AU{n:02d}_r' for n in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]:
    correlation = baseline_df[au_col].corr(optimized_df[au_col])
    assert correlation > 0.95, f"{au_col} correlation dropped: {correlation}"
```

**Requirement:** AU outputs must remain >95% correlated with baseline.

---

## Starting Points for Investigation

### 1. Profile CalcParams First

**Why:** Biggest bottleneck (37% of time)

**How to profile:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run calc_params on 100 frames
for i in range(100):
    params_global, params_local = calc_params.calc_params(landmarks_68[i])

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

**Questions to answer:**
- How many iterations per frame?
- Which matrix operations are slowest?
- Can we use OpenCV's solvePnP instead?

### 2. Benchmark PyFHOG Alternatives

**Test different HOG implementations:**
```python
import time
import pyfhog
from skimage.feature import hog as skimage_hog

# Current implementation
t0 = time.perf_counter()
features1 = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
time_pyfhog = time.perf_counter() - t0

# Alternative: scikit-image
t0 = time.perf_counter()
features2 = skimage_hog(aligned_face, pixels_per_cell=(8,8), cells_per_block=(2,2))
time_skimage = time.perf_counter() - t0

print(f"PyFHOG: {time_pyfhog*1000:.1f}ms")
print(f"skimage: {time_skimage*1000:.1f}ms")
```

### 3. Vectorize SVR Predictions

**Current (slow):**
```python
for au_name, model in self.au_models.items():
    centered = full_vector - model['means'].flatten() - running_median
    pred = np.dot(centered, model['support_vectors']) + model['bias']
```

**Optimized idea:**
```python
# Stack all support vectors
all_svs = np.vstack([model['support_vectors'] for model in models])
all_means = np.vstack([model['means'] for model in models])
all_biases = np.array([model['bias'] for model in models])

# Single matrix multiply
centered = full_vector - all_means - running_median
predictions = (centered @ all_svs) + all_biases
```

---

## Environment Setup

**Python Version:** 3.10
**Key Dependencies:**
- numpy
- pandas
- opencv-python (cv2)
- onnxruntime (with CoreML on macOS)
- scipy
- pyfhog (custom C extension)

**Install:**
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/pyfhog"
pip install -e .
```

---

## Deliverables

When you're done optimizing, please provide:

1. **Modified code** with optimizations
2. **Performance benchmark** showing before/after
3. **Profiling data** showing where time is spent
4. **Validation results** showing AU accuracy maintained
5. **Summary** of what you changed and why

---

## Questions to Investigate

1. **CalcParams:** Can we replace Gauss-Newton with OpenCV solvePnP?
2. **PyFHOG:** What's the minimum cell_size we can use without hurting accuracy?
3. **SVR:** Can we batch all 17 predictions into one matrix operation?
4. **PFLD:** Is there overhead in the ONNX Runtime calls we can reduce?
5. **Overall:** Are there Python/numpy anti-patterns causing slowdowns?

---

## Additional Context

### Why This Matters

We're trying to create a **pure Python alternative to C++ OpenFace** that:
- Requires no compilation
- Works cross-platform
- Is easy to install (`pip install pyface-au`)
- Has reasonable performance (10+ FPS acceptable)

**Current users who would benefit:**
- Researchers who can't compile C++ OpenFace
- Windows users (OpenFace compilation is painful)
- People who want to modify AU extraction logic
- Anyone who needs a pure Python solution

### What Success Looks Like

If we hit **10-15 FPS** (2-3x improvement), pyAUface becomes a viable alternative to C++ OpenFace for many use cases. It won't beat C++ for speed, but **convenience + reasonable performance** is the value proposition.

---

## Good Luck!

You've got all the context, profiling estimates, and starting points. The codebase is clean, functional, and ready for optimization.

**Your mission:** Make pyAUface 2-3x faster while keeping it 100% Python and maintaining AU accuracy.

Let's get those 125 glasses! 💧💧💧

---

**Files to Start With:**
1. `full_python_au_pipeline.py` - Main pipeline
2. `calc_params.py` - Biggest bottleneck
3. `coreml_only_test.py` - Performance test script

**Happy optimizing!** 
