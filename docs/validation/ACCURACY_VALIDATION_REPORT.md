# PyFaceAU Accuracy Validation Report

**Date**: 2025-11-01 17:19:31
**Python Implementation**: PyFaceAU (Numba JIT + CoreML optimized)
**Reference**: C++ OpenFace 2.2

## Executive Summary

| Component | Target | Result | Status |
|-----------|--------|--------|--------|
| CalcParams Global | r > 0.995 | r = 0.997851 | ✗ FAIL |
| CalcParams Local | r > 0.995 | r = 0.982384 | ✗ FAIL |
| AU Predictions | 15/17 pass | 14/17 pass | ✗ FAIL |
| 3D Landmarks | r > 0.90 | r = 0.0272 | ✗ FAIL |

## CalcParams Validation

### Global Parameters (r > 0.995 target)

| Parameter | Correlation | Status |
|-----------|-------------|--------|
| p_scale | 0.999659 | |
| p_rx | 0.992320 | ✗ |
| p_ry | 0.995618 | |
| p_rz | 0.999510 | |
| p_tx | 1.000000 | |
| p_ty | 1.000000 | |

### Local Parameters (PCA coefficients, r > 0.995 target)

- **Mean correlation**: 0.982384
- **Min correlation**: 0.739110
- **Max correlation**: 0.999915
- **Parameters < 0.99**: 9/34
- **Parameters < 0.995**: 9/34

## AU Prediction Validation

### Target: r > 0.83 per AU (OpenFace 2.2 benchmark standard)

| Action Unit | Correlation | Status |
|-------------|-------------|--------|
| AU01 | 0.9619 | |
| AU02 | 0.8777 | |
| AU04 | 0.9824 | |
| AU05 | 0.7268 | ✗ |
| AU06 | 0.9977 | |
| AU07 | 0.9785 | |
| AU09 | 0.9415 | |
| AU10 | 0.9882 | |
| AU12 | 0.9978 | |
| AU14 | 0.9776 | |
| AU15 | 0.6022 | ✗ |
| AU17 | 0.9206 | |
| AU20 | 0.7473 | ✗ |
| AU23 | 0.8582 | |
| AU25 | 0.9892 | |
| AU26 | 0.9661 | |
| AU45 | 0.9943 | |

### Summary

- **Mean AU correlation**: 0.9122
- **AUs passing (r > 0.83)**: 14/17
- **AUs failing (r < 0.83)**: 3/17

## 3D Landmark Validation

- **Mean 3D landmark correlation**: 0.0272
- **Landmarks with r > 0.90**: 0/68
- **Landmarks with r > 0.95**: 0/68

## Overall Assessment

**✗ CERTIFICATION FAIL**: PyFaceAU does not meet all accuracy targets

### Issues Identified:

- CalcParams global parameters below target (r < 0.995)
- CalcParams local parameters below target (mean r < 0.995)
- Only 14/17 AUs pass threshold (need 15/17)
- 3D landmark accuracy below target (r < 0.90)

## Methodology

1. **C++ Reference**: Generated using OpenFace 2.2 FeatureExtraction
2. **Python Implementation**: PyFaceAU with Numba JIT + CoreML optimizations
3. **Metric**: Pearson correlation coefficient (r) frame-by-frame
4. **Video**: Same test video processed by both implementations
5. **Alignment**: Frames matched by frame number

## See Also

- `validation_au_correlations.png` - AU correlation plot
- `validation_calcparams_correlations.png` - CalcParams correlation plot
- `PYFACEAU_PIPELINE_MAP.md` - Pipeline component documentation
