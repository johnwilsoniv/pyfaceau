# Hardware Acceleration Opportunities - PyfaceAU

**Comprehensive guide to accelerating PyfaceAU pipeline components using GPUs, JIT compilation, and specialized hardware**

---

## Overview

Current performance: **4.6 FPS (217ms/frame)** sequential, **~30 FPS** with multiprocessing

**Target with hardware acceleration: 50-100+ FPS**

---

## Component Bottlenecks (Sequential Mode)

| Component | Time | % of Total | Acceleration Potential |
|-----------|------|------------|----------------------|
| **CalcParams (Pose)** | 80ms | 37% | 🔥 5-15x with GPU/JIT |
| **PyFHOG (HOG)** | 50ms | 23% | 🔥 2-5x with GPU |
| **AU Prediction** | 30ms | 14% |  2-10x with GPU/batching |
| PFLD (Landmarks) | 30ms | 14% |  2-5x with GPU ONNX |
| Face Alignment | 20ms | 9% |  2-3x with Numba |
| Running Median | 5ms | 2% | Already optimized (Cython) |
| Face Detection | 2ms | 1% | Already optimized (tracking) |

---

## 1. CalcParams (Pose Estimation) - HIGHEST PRIORITY

**Current:** 80ms/frame (37% of total time)
**Target:** 5-15ms/frame (5-15x speedup)

### Option A: Numba JIT Compilation (EASIEST)

**Expected Speedup:** 2-5x (80ms → 16-40ms)

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True, cache=True)
def gauss_newton_iteration(J, residuals, params):
    """JIT-compiled Gauss-Newton iteration"""
    JtJ = J.T @ J
    Jtr = J.T @ residuals
    delta = np.linalg.solve(JtJ, Jtr)
    return params - delta

# In calc_params.py:
class CalcParams:
    def calc_params(self, landmarks):
        # ... setup code ...
        for iteration in range(max_iters):
            # Use JIT-compiled function for hot loop
            params = gauss_newton_iteration(J, residuals, params)
            # ... convergence check ...
```

**Installation:**
```bash
pip install numba
```

**Pros:** Easy to implement, no GPU required, 2-5x speedup
**Cons:** Limited speedup compared to GPU

---

### Option B: CuPy GPU Acceleration (MEDIUM)

**Expected Speedup:** 5-10x (80ms → 8-16ms)

```python
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    import numpy as cp
    HAS_GPU = False

class CalcParams:
    def __init__(self, pdm_parser, use_gpu=True):
        self.use_gpu = use_gpu and HAS_GPU
        self.xp = cp if self.use_gpu else np

    def calc_params(self, landmarks):
        # Convert to GPU arrays
        if self.use_gpu:
            landmarks = cp.asarray(landmarks)

        # All matrix operations now run on GPU
        M3D = self.xp.array(self.pdm_parser.mean_shape)
        # ... Gauss-Newton on GPU ...

        # Convert back to CPU if needed
        if self.use_gpu:
            return cp.asnumpy(params_global), cp.asnumpy(params_local)
        return params_global, params_local
```

**Installation:**
```bash
# NVIDIA GPU (CUDA 11.x)
pip install cupy-cuda11x

# NVIDIA GPU (CUDA 12.x)
pip install cupy-cuda12x

# AMD GPU (ROCm)
pip install cupy-rocm-5-0
```

**Pros:** Significant speedup, reuses existing code structure
**Cons:** Requires NVIDIA/AMD GPU, memory transfer overhead

---

### Option C: JAX GPU Implementation (ADVANCED)

**Expected Speedup:** 10-15x (80ms → 5-8ms)

JAX provides the best GPU performance with JIT compilation + automatic differentiation:

```python
import jax
import jax.numpy as jnp
from jax import jit

@jit
def calc_params_jax(landmarks, mean_shape, basis_vectors):
    """JAX-optimized CalcParams"""

    def gauss_newton_step(params):
        # Compute residuals
        predicted = reconstruct_shape(params, mean_shape, basis_vectors)
        residuals = landmarks - predicted

        # Jacobian via automatic differentiation
        J = jax.jacfwd(lambda p: reconstruct_shape(p, mean_shape, basis_vectors))(params)

        # Solve normal equations
        JtJ = J.T @ J
        Jtr = J.T @ residuals
        delta = jnp.linalg.solve(JtJ, Jtr)
        return params - delta

    # Iterate Gauss-Newton
    params = init_params
    for i in range(max_iters):
        params = gauss_newton_step(params)

    return params
```

**Installation:**
```bash
# GPU version
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CPU version (for testing)
pip install jax
```

**Pros:** Best performance, automatic differentiation, XLA optimization
**Cons:** Complete rewrite of CalcParams, learning curve

---

## 2. HOG Extraction (PyFHOG) - HIGH PRIORITY

**Current:** 50ms/frame (23% of total time)
**Target:** 10-20ms/frame (2-5x speedup)

### Option A: GPU-Accelerated HOG (OpenCV CUDA)

**Expected Speedup:** 3-5x (50ms → 10-17ms)

```python
import cv2

class GPUHOGExtractor:
    def __init__(self, cell_size=8):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.hog = cv2.cuda_HOGDescriptor()
            self.use_gpu = True
        else:
            self.use_gpu = False

    def extract(self, aligned_face):
        if self.use_gpu:
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(aligned_face)

            # Compute HOG on GPU
            descriptors = self.hog.compute(gpu_img)

            # Download result
            return descriptors.download()
        else:
            # Fallback to PyFHOG
            return pyfhog.extract_fhog_features(aligned_face, cell_size=8)
```

**Installation:**
```bash
# Build OpenCV with CUDA support
pip install opencv-contrib-python-headless==4.8.0.76

# Or build from source with CUDA enabled
```

**Pros:** Significant speedup, maintains accuracy
**Cons:** Requires OpenCV CUDA build, GPU memory

---

### Option B: Reduce HOG Complexity (SIMPLE)

**Expected Speedup:** 1.5-2x (50ms → 25-33ms)

```python
# Current: 112x112 image, cell_size=8 → 14x14 cells = 196 cells
hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)

# Optimized: Increase cell_size to 12
hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=12)
# → 9x9 cells = 81 cells (2.4x fewer cells to compute)

# Or reduce aligned face size
aligned_face = cv2.resize(aligned_face, (96, 96))  # 112x112 → 96x96
hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)
# → 12x12 cells = 144 cells (1.36x fewer cells)
```

**Pros:** Zero code changes, immediate speedup
**Cons:** May affect AU accuracy (requires validation)

---

## 3. AU Prediction (17 SVRs) - MEDIUM PRIORITY

**Current:** 30ms/frame (14% of total time)
**Target:** 3-10ms/frame (3-10x speedup)

### Option A: Batch All SVR Predictions (SIMPLE)

**Expected Speedup:** 2-5x (30ms → 6-15ms)

```python
class BatchedAUPredictor:
    def __init__(self, au_models):
        # Stack all support vectors into single matrix
        self.all_support_vectors = np.vstack([
            model['support_vectors'] for model in au_models.values()
        ])  # Shape: (17, 4702)

        self.all_means = np.vstack([
            model['means'].flatten() for model in au_models.values()
        ])  # Shape: (17, 4702)

        self.all_biases = np.array([
            model['bias'] for model in au_models.values()
        ])  # Shape: (17,)

        self.dynamic_mask = np.array([
            model['model_type'] == 'dynamic' for model in au_models.values()
        ])  # Shape: (17,)

    def predict_all(self, full_vector, running_median):
        # Vectorized prediction for all 17 AUs at once
        full_vector = full_vector.reshape(1, -1)  # (1, 4702)

        # Center features (broadcast across all models)
        centered = full_vector - self.all_means  # (17, 4702)

        # Apply running median only to dynamic models
        centered[self.dynamic_mask] -= running_median

        # Single matrix multiply for all AUs
        predictions = np.sum(centered * self.all_support_vectors, axis=1) + self.all_biases

        # Clamp to [0, 5]
        predictions = np.clip(predictions, 0.0, 5.0)

        return predictions  # Shape: (17,)
```

**Pros:** Easy to implement, significant speedup, no dependencies
**Cons:** None - should be implemented!

---

### Option B: GPU Matrix Multiplication (CuPy)

**Expected Speedup:** 5-10x (30ms → 3-6ms)

Combine batching with GPU matrix operations:

```python
import cupy as cp

class GPUBatchedAUPredictor:
    def __init__(self, au_models):
        # Move all matrices to GPU
        self.all_support_vectors = cp.asarray(np.vstack([...]))
        self.all_means = cp.asarray(np.vstack([...]))
        self.all_biases = cp.asarray(np.array([...]))
        self.dynamic_mask = cp.asarray(np.array([...]))

    def predict_all(self, full_vector, running_median):
        # All operations on GPU
        full_vector = cp.asarray(full_vector).reshape(1, -1)
        running_median = cp.asarray(running_median)

        centered = full_vector - self.all_means
        centered[self.dynamic_mask] -= running_median

        predictions = cp.sum(centered * self.all_support_vectors, axis=1) + self.all_biases
        predictions = cp.clip(predictions, 0.0, 5.0)

        # Return to CPU
        return cp.asnumpy(predictions)
```

**Pros:** Maximum speedup for AU prediction
**Cons:** Requires GPU, memory transfer overhead

---

## 4. Landmark Detection (PFLD) - MEDIUM PRIORITY

**Current:** 30ms/frame (14% of total time)
**Target:** 6-15ms/frame (2-5x speedup)

### Option A: GPU ONNX Execution Provider

**Expected Speedup:** 2-3x (30ms → 10-15ms)

```python
import onnxruntime as ort

class CunjianPFLDDetector:
    def __init__(self, model_path, use_gpu=True):
        providers = []
        if use_gpu:
            # Try GPU providers in order of preference
            if 'TensorrtExecutionProvider' in ort.get_available_providers():
                providers.append('TensorrtExecutionProvider')
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')

        # Fallback to CPU
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        print(f"PFLD using: {self.session.get_providers()[0]}")
```

**Installation:**
```bash
# NVIDIA GPU (CUDA)
pip install onnxruntime-gpu

# TensorRT (NVIDIA, fastest)
pip install onnxruntime-gpu nvidia-tensorrt
```

**Pros:** Easy to enable, works with existing ONNX models
**Cons:** Requires GPU, some setup

---

### Option B: CoreML Acceleration (macOS only)

**Expected Speedup:** 2-4x (30ms → 7-15ms)

```python
# Already available via use_coreml=True in pipeline
pipeline = FullPythonAUPipeline(
    pfld_model='weights/pfld_cunjian.onnx',
    use_coreml=True,  # Enable Neural Engine acceleration
    ...
)
```

**Pros:** Automatic on macOS, uses Neural Engine
**Cons:** macOS only, requires ONNX→CoreML conversion

---

## 5. Face Alignment - LOW PRIORITY

**Current:** 20ms/frame (9% of total time)
**Target:** 7-10ms/frame (2-3x speedup)

### Option: Numba JIT for Kabsch Algorithm

**Expected Speedup:** 2-3x (20ms → 7-10ms)

```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def kabsch_transform(source, target):
    """JIT-compiled Kabsch algorithm"""
    # Center points
    centroid_source = source.mean(axis=0)
    centroid_target = target.mean(axis=0)

    source_centered = source - centroid_source
    target_centered = target - centroid_target

    # Compute covariance matrix
    H = source_centered.T @ target_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = centroid_target - R @ centroid_source

    return R, t
```

**Pros:** Moderate speedup, no GPU required
**Cons:** Low priority (only 9% of time)

---

## Implementation Priority

### 🔥 IMMEDIATE (High ROI)

1. **Batch SVR Predictions** - 2-5x speedup on AU prediction
   - Effort: 2-3 hours
   - Impact: 30ms → 6-15ms
   - No dependencies

2. **Numba JIT for CalcParams** - 2-5x speedup on pose estimation
   - Effort: 1 day
   - Impact: 80ms → 16-40ms
   - Dependencies: `pip install numba`

###  HIGH (GPU Required)

3. **CuPy GPU CalcParams** - 5-10x speedup on pose estimation
   - Effort: 2-3 days
   - Impact: 80ms → 8-16ms
   - Dependencies: NVIDIA GPU, `cupy-cuda11x`

4. **GPU ONNX for PFLD** - 2-3x speedup on landmarks
   - Effort: 1 hour
   - Impact: 30ms → 10-15ms
   - Dependencies: `onnxruntime-gpu`

###  ADVANCED (Maximum Performance)

5. **JAX GPU CalcParams** - 10-15x speedup on pose estimation
   - Effort: 1 week (rewrite)
   - Impact: 80ms → 5-8ms
   - Dependencies: `jax[cuda]`

6. **GPU HOG Extraction** - 3-5x speedup
   - Effort: 3-4 days
   - Impact: 50ms → 10-17ms
   - Dependencies: OpenCV CUDA build

---

## Expected Performance with Optimizations

| Configuration | FPS | Speedup | Optimizations |
|--------------|-----|---------|---------------|
| **Baseline** | 4.6 | 1x | Current |
| **Parallel (6 workers)** | ~28 | 6x | Multiprocessing |
| **+ Batched SVR** | ~32 | 7x | + numpy batching |
| **+ Numba CalcParams** | ~42 | 9x | + JIT compilation |
| **+ GPU CalcParams** | ~55 | 12x | + CuPy GPU |
| **+ GPU ONNX** | ~65 | 14x | + GPU PFLD |
| **+ JAX CalcParams** | **~80** | **17x** | + JAX GPU |
| **+ GPU HOG** | **~100+** | **21x** | + CUDA HOG |

---

## Installation Guide

### Basic Acceleration (Numba + Batching)

```bash
pip install numba
# Modify pipeline to use batched SVR and Numba CalcParams
```

**Expected: 9x speedup (4.6 → 42 FPS)**

### GPU Acceleration (NVIDIA CUDA)

```bash
# Install CUDA toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install GPU packages
pip install cupy-cuda11x          # CuPy for matrix operations
pip install onnxruntime-gpu        # GPU ONNX runtime
pip install numba                  # JIT compilation

# Optional: JAX for maximum performance
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Expected: 12-17x speedup (4.6 → 55-80 FPS)**

---

## Recommendations

**For maximum FPS with minimal effort:**
1. Use **ParallelAUPipeline** (6 workers) → 28 FPS
2. Add **batched SVR predictions** → 32 FPS
3. Add **Numba JIT CalcParams** → 42 FPS

**For GPU-accelerated performance:**
4. Add **CuPy GPU CalcParams** → 55 FPS
5. Add **GPU ONNX PFLD** → 65 FPS

**For maximum performance (100+ FPS):**
6. Rewrite **CalcParams in JAX** → 80 FPS
7. Add **GPU HOG extraction** → 100+ FPS

---

## Testing Hardware Acceleration

Run the detailed benchmark to identify bottlenecks:

```bash
python benchmark_detailed.py --video test.mp4 --max-frames 200
```

This will show:
- Per-component timing
- Bottleneck analysis
- Hardware acceleration recommendations
- Expected speedups

---

**Questions? Open an issue on GitHub!**
