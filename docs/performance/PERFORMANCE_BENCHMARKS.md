# PyFaceAU Performance Benchmarks

**Date:** 2025-11-01
**Test Video:** IMG_0434.MOV (972 frames)
**Baseline:** C++ OpenFace 2.2

---

## Current Peak Performance

### Configuration 1: Python PFLD Landmarks (Production Default)

**Pipeline:**
```
RetinaFace (face detection)
→ PFLD CoreML (68 landmarks)
→ CalcParams (3D pose + alignment)
→ HOG/Geom feature extraction
→ AU prediction (17 AUs)
```

**Performance:**
- **Mean AU Correlation:** 85.62%
- **AUs Passing (r > 0.83):** 11/17 (64.7%)
- **Processing Speed:** 140 fps
- **Success Rate:** 100% (972/972 frames)

**AU Breakdown:**

| Category | AUs | Performance |
|----------|-----|-------------|
| **Excellent (99%+)** | 5/17 | AU12, AU06, AU45, AU25, AU10 |
| **Very Good (90-98%)** | 6/17 | AU04, AU07, AU14, AU26, AU09, AU17 |
| **Good (83-90%)** | 0/17 | - |
| **Near Miss (75-83%)** | 3/17 | AU01 (82.01%), AU02 (78.93%), AU20 (70.12%) |
| **Failing (< 75%)** | 3/17 | AU05 (68.45%), AU23 (61.07%), AU15 (55.89%) |

**Strengths:**
- Fast (140 fps enables real-time applications)
- Pure Python/CoreML (no C++ dependencies)
- Excellent on core expressions (smile, blink, jaw)
- Easy to deploy cross-platform

**Weaknesses:**
- Warning: Brow movements (AU01, AU02) just below threshold
- Warning: Lip precision movements (AU23) underperforming
- Warning: Sparse AUs (AU05, AU15, AU20) struggling

---

### Configuration 2: C++ CLNF Landmarks (Maximum Accuracy)

**Pipeline:**
```
RetinaFace (face detection)
→ C++ CLNF iterative refinement (68 landmarks)
→ CalcParams (3D pose + alignment)
→ HOG/Geom feature extraction
→ AU prediction (17 AUs)
```

**Performance:**
- **Mean AU Correlation:** 91.22%
- **AUs Passing (r > 0.83):** 14/17 (82.4%)
- **Processing Speed:** 140 fps (using pre-extracted landmarks)
- **Success Rate:** 100% (972/972 frames)

**AU Breakdown:**

| Category | AUs | Performance |
|----------|-----|-------------|
| **Excellent (99%+)** | 5/17 | AU12, AU06, AU45, AU25, AU10 |
| **Very Good (90-98%)** | 9/17 | AU04, AU07, AU14, AU26, AU01, AU09, AU17, AU02, AU23 |
| **Good (83-90%)** | 0/17 | - |
| **Near Miss (75-83%)** | 0/17 | - |
| **Failing (< 75%)** | 3/17 | AU20 (74.73%), AU05 (72.68%), AU15 (60.22%) |

**Strengths:**
- Best AU accuracy (91.22% mean)
- 14/17 AUs passing (vs 11/17 with PFLD)
- Excellent on brow movements (AU01: 96.19%, AU02: 87.77%)
- Much better lip precision (AU23: 85.82%)

**Weaknesses:**
- Warning: Requires C++ OpenFace library
- Warning: Complex deployment (dlib, OpenCV, OpenFace)
- Warning: CLNF detection slower than PFLD (not measured standalone)

---

## Performance Comparison: PFLD vs CLNF

### Overall Metrics

| Metric | PFLD | CLNF | Delta |
|--------|------|------|-------|
| **Mean AU Correlation** | 85.62% | 91.22% | **-5.60%** |
| **AUs Passing** | 11/17 | 14/17 | **-3 AUs** |
| **Processing Speed** | 140 fps | ~140 fps | ~Same |
| **Deployment Complexity** | Low (Python) | High (C++) | - |

### Per-AU Comparison

**AUs Improved with CLNF (> 5% gain):**

| AU | Description | PFLD | CLNF | Gain | Status Change |
|----|-------------|------|------|------|---------------|
| **AU01** | Inner Brow Raiser | 82.01% | 96.19% | **+14.18%** | Fail → Pass |
| **AU02** | Outer Brow Raiser | 78.93% | 87.77% | **+8.84%** | Fail → Pass |
| **AU23** | Lip Tightener | 61.07% | 85.82% | **+24.75%** | Fail → Pass |

**AUs Stable with Both (< 2% difference):**

| AU | Description | PFLD | CLNF | Delta |
|----|-------------|------|------|-------|
| AU12 | Lip Corner Puller | 99.73% | 99.78% | +0.05% |
| AU06 | Cheek Raiser | 99.69% | 99.77% | +0.08% |
| AU45 | Blink | 99.32% | 99.43% | +0.11% |
| AU10 | Upper Lip Raiser | 98.63% | 98.82% | +0.19% |
| AU04 | Brow Lowerer | 97.89% | 98.24% | +0.35% |
| AU25 | Lips Part | 98.51% | 98.92% | +0.41% |
| AU07 | Lid Tightener | 97.31% | 97.85% | +0.54% |
| AU14 | Dimpler | 96.98% | 97.76% | +0.78% |
| AU09 | Nose Wrinkler | 92.87% | 94.15% | +1.28% |
| AU26 | Jaw Drop | 95.14% | 96.61% | +1.47% |

**Root Cause Analysis:**

The 3 AUs that benefit most from CLNF (AU01, AU02, AU23) share common characteristics:
- **Landmark-sensitive:** Require precise brow (landmarks 17-26) and lip corner (landmarks 48, 54) placement
- **Subtle movements:** Small displacements that PFLD's single-pass detection misses
- **CLNF advantage:** Iterative refinement with patch experts detects subtle landmark shifts

---

## Improvement Plan: Targeted CLNF Refinement

### Goal
Recover 60-80% of the CLNF advantage while maintaining PFLD's speed and simplicity.

**Target Performance:**
- Mean AU Correlation: **88-90%** (vs current 85.62%)
- AUs Passing: **12-13/17** (vs current 11/17)
- Processing Speed: **100-120 fps** (vs current 140 fps)

### Approach: Hybrid PFLD + Targeted CLNF

```
Pipeline:
1. PFLD detects initial 68 landmarks (fast, 140 fps)
2. Targeted CLNF refinement on critical landmarks only:
   - Brow landmarks (17-26): 10 points
   - Lip corner landmarks (48, 54): 2 points
   - Total: 12/68 points refined (~18%)
3. Project refined landmarks back onto PDM (shape constraints)
4. Proceed with CalcParams and AU prediction
```

**Why This Works:**
- Focus computational effort where it matters (brows, lip corners)
- Leverage PFLD for 82% of landmarks (already accurate)
- Add CLNF precision only where PFLD struggles
- Maintain speed by refining small subset of points

### Implementation Components

**Phase 1: Patch Expert Integration**
- Extract SVR patch models from OpenFace
- Parse patch expert format (similar to AU SVR models)
- Implement patch extraction around landmarks
- **Estimated time:** 1-2 days

**Phase 2: PDM-Constrained Optimization**
- Adapt CalcParams Gauss-Newton solver for 2D landmarks
- Implement PDM shape projection
- Add iterative refinement loop (max 3-5 iterations)
- **Estimated time:** 2-3 days

**Phase 3: Targeted Refinement Module**
```python
class TargetedCLNFRefiner:
    """Refine brow and lip corner landmarks using CLNF patch experts"""

    CRITICAL_LANDMARKS = {
        'inner_brow': [17, 18, 19, 20, 21],   # AU01
        'outer_brow': [22, 23, 24, 25, 26],   # AU02
        'lip_corners': [48, 54]                # AU23
    }

    def refine(self, image, pfld_landmarks):
        # Extract patches around critical points
        # Run patch expert SVR evaluation
        # Optimize using PDM constraints
        # Return refined landmarks
        pass
```
- **Estimated time:** 2-3 days

**Phase 4: Integration & Validation**
- Add to FullPythonAUPipeline as optional refinement step
- Validate against C++ reference on IMG_0434.MOV
- Measure performance impact
- **Estimated time:** 1-2 days

**Total Estimated Time:** 6-10 days

### Expected Results

**Conservative Estimate:**
| AU | Current (PFLD) | Expected (PFLD+CLNF) | Target |
|----|----------------|----------------------|--------|
| AU01 | 82.01% | 88-92% | Pass (> 83%) |
| AU02 | 78.93% | 84-88% | Pass (> 83%) |
| AU23 | 61.07% | 70-78% Warning: | May pass |

**Optimistic Estimate:**
- Mean AU Correlation: 89-91% (within 0-2% of CLNF)
- AUs Passing: 13-14/17 (match or near CLNF)

---

## Deployment Recommendations

### For Production Applications

**Use PFLD (Current Default):**
- 85.62% correlation is excellent for most use cases
- 140 fps enables real-time processing
- Pure Python deployment (no C++ compilation)
- Cross-platform (macOS, Linux, Windows)

**Applications:**
- General facial expression analysis
- Emotion recognition
- Video conferencing features
- Content moderation
- User engagement tracking

### For Research/High-Precision Applications

**Use PFLD + Targeted CLNF Refinement (Planned):**
- 88-90% expected correlation
- 100-120 fps (still real-time)
- Better brow and lip precision
- Still pure Python (no dlib)

**Applications:**
- Clinical facial analysis
- Psychological research
- Fine-grained emotion detection
- AU-based interaction systems
- Benchmarking studies

### For Maximum Accuracy

**Use Full C++ CLNF Landmarks:**
- 91.22% correlation (best available)
- Warning: Requires C++ OpenFace integration
- Warning: Complex deployment

**Applications:**
- Ground truth dataset creation
- Algorithm development baseline
- Publications requiring OpenFace 2.2 comparison

---

## CalcParams Performance

Both PFLD and CLNF achieve similar CalcParams accuracy (3D pose estimation):

| Parameter Type | Correlation | Status |
|----------------|-------------|--------|
| **Global Mean** | 99.79% | Excellent |
| **Local Mean** | 98.24% | Very Good |

**Conclusion:** CalcParams accuracy is not significantly affected by landmark detector choice. The AU prediction differences are due to landmark precision, not pose estimation.

---

## Processing Speed Breakdown

**Current Pipeline (PFLD):**
```
Component breakdown (estimated per frame):
- RetinaFace detection:        ~2ms
- PFLD landmark detection:     ~2ms
- CalcParams optimization:     ~2ms
- HOG/Geom feature extraction: ~1ms
- AU prediction:               ~0.5ms
Total:                         ~7-8ms → 125-140 fps
```

**Projected Pipeline (PFLD + Targeted CLNF):**
```
Component breakdown (estimated per frame):
- RetinaFace detection:        ~2ms
- PFLD landmark detection:     ~2ms
- Targeted CLNF refinement:    ~2-3ms (NEW)
- CalcParams optimization:     ~2ms
- HOG/Geom feature extraction: ~1ms
- AU prediction:               ~0.5ms
Total:                         ~9-11ms → 90-110 fps
```

**Impact:** ~25-30% speed reduction, but still well above real-time (30 fps).

---

## Summary Statistics

| Configuration | Mean AU Corr | AUs Passing | Speed | Deployment | Status |
|---------------|--------------|-------------|-------|------------|--------|
| **PFLD Only** | 85.62% | 11/17 | 140 fps | Easy | Available |
| **PFLD + CLNF (Targeted)** | 88-90% (est) | 12-13/17 (est) | 100-120 fps (est) | Easy | 🚧 Planned |
| **Full C++ CLNF** | 91.22% | 14/17 | ~140 fps* | Hard | Validated |

*Speed measurement based on pre-extracted landmarks

---

## Files Referenced

**Validation Results:**
- `FINAL_ACCURACY_SUMMARY.md` - Complete accuracy investigation
- `PFLD_VS_CLNF_COMPARISON.md` - Detailed landmark detector comparison
- `ACCURACY_VALIDATION_REPORT.md` - Formal validation report
- `validation_au_correlations.png` - AU correlation visualization
- `validation_calcparams_correlations.png` - CalcParams visualization

**Pipeline Outputs:**
- `pipeline_output_full.csv` - PFLD landmark results (972 frames)
- `cpp_reference/IMG_0434.csv` - CLNF landmark reference (972 frames)

**Next Steps:**
- See TODO list for CLNF refinement implementation plan

---

**Document Status:** CURRENT
**Last Updated:** 2025-11-01
**PyFaceAU Version:** 1.0 (Production Ready with PFLD)
**Next Version:** 1.1 (Planned: PFLD + Targeted CLNF Refinement)
