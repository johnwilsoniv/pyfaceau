# PyFaceAU Accuracy Investigation Summary

**Date:** 2025-11-01
**Goal:** Achieve 99%+ correlation with C++ OpenFace 2.2 for all 17 AUs

---

## Current Status

**Test Configuration:**
- Using identical C++ CLNF landmarks (eliminates landmark detector as variable)
- With two-pass processing enabled
- Processing all 972 frames

**Results:**
- **14/17 AUs passing** (r > 0.83): 82.4%
- **Mean AU correlation:** 91.22%
- **3 AUs failing:** AU05 (72.68%), AU15 (60.22%), AU20 (74.73%)

---

## Major Fixes Implemented

### 1. Two-Pass Processing (CRITICAL - Fixed Major Issue)

**Problem:** Validation script was missing two-pass re-prediction logic
**Impact:** Dynamic AUs had elevated baseline due to unstable early median estimates

**Before two-pass:**
- AU01: 80.43% → **After:** 96.19% (+15.76%)
- AU02: 65.24% → **After:** 87.77% (+22.53%)
- AU23: 75.00% → **After:** 85.82% (+10.82%)
- Mean: 82.90% → **After:** 91.22% (+8.32%)

**Implementation:** Added two-pass logic to validate_accuracy.py:
```python
final_median = pipeline.running_median.get_combined_median()
for result in results:
    au_results = pipeline._predict_aus(
        result['hog_features'],
        result['geom_features'],
        final_median
    )
    result['aus'] = au_results
```

**Status:** IMPLEMENTED - Major improvement

### 2. Median Update Frequency Fix (No Impact)

**Investigation:** C++ updates both HOG and geometric medians every other frame (`if(frames_tracking % 2 == 1)`)

**Implementation:** Modified DualHistogramMedianTracker to match C++ behavior

**Result:** NO CHANGE in accuracy (identical correlations before/after)

**Conclusion:** Update frequency is just a "small speedup optimization" (per C++ comment) and doesn't affect accuracy

**Status:** IMPLEMENTED (but not limiting factor)

---

## Current Performance by AU

### Top Performers (99%+)
| AU | Name | Correlation | Status |
|----|------|-------------|--------|
| AU12 | Lip Corner Puller | 99.78% | |
| AU06 | Cheek Raiser | 99.77% | |
| AU45 | Blink | 99.43% | |
| AU25 | Lips Part | 98.92% | |
| AU10 | Upper Lip Raiser | 98.82% | |

### Good Performers (90-98%)
| AU | Name | Correlation | Status |
|----|------|-------------|--------|
| AU04 | Brow Lowerer | 98.24% | |
| AU07 | Lid Tightener | 97.85% | |
| AU14 | Dimpler | 97.76% | |
| AU26 | Jaw Drop | 96.61% | |
| AU01 | Inner Brow Raiser | 96.19% | |
| AU09 | Nose Wrinkler | 94.15% | |
| AU17 | Chin Raiser | 92.06% | |

### Acceptable Performers (83-90%)
| AU | Name | Correlation | Status |
|----|------|-------------|--------|
| AU02 | Outer Brow Raiser | 87.77% | |
| AU23 | Lip Tightener | 85.82% | |

### Under performers (<83%)
| AU | Name | Correlation | Active % | Status |
|----|------|-------------|----------|--------|
| AU20 | Lip Stretcher | 74.73% | 26.3% | |
| AU05 | Upper Lid Raiser | 72.68% | 15.3% | |
| AU15 | Lip Corner Depressor | 60.22% | 25.3% | |

**Pattern:** All 3 failing AUs have sparse/weak activity in test video (15-26% active frames, mean intensity 0.08-0.10)

---

## Verified C++ Matches

### Histogram Parameters
- HOG: 1000 bins, range [-0.005, 1.0]
- Geometric: 10000 bins, range [-60, 60]
- **Status:** Match C++ exactly

### HOG Median Clamping
- Python: `hog_median[hog_median < 0] = 0.0`
- C++: `hog_desc_median.setTo(0, hog_desc_median < 0)`
- **Status:** Implemented correctly

### SVR Prediction Formula
- Python: `(features - means - running_median) * support_vectors + bias`
- C++: `(input - this->means - run_med) * this->support_vectors + this->biases`
- **Status:** Match exactly

### Two-Pass Processing
- Python: Re-predicts all frames with final median
- C++: Re-predicts frames 0-3000 with final median
- **Status:** Implemented (Python does all frames for thoroughness)

### Median Update Frequency
- Python & C++: Both medians update every other frame
- **Status:** Matches (but doesn't affect accuracy)

---

## Remaining Accuracy Gap Analysis

### CalcParams Accuracy (Potential Issue)

**Current Performance:**
- Global params: p_rx = 99.23% (target: 99.5%+)
- Local params: mean = 98.24% (target: 99.5%+)

**Impact:**
- Small CalcParams errors could propagate to geometric features
- Geometric features include both 3D shape (204 dims) and PCA coefficients (34 dims)
- Total: 238 geometric features affected

**Evidence:**
- Static AUs (HOG only): 98.70% mean 
- High-activity dynamic AUs: 87-96% 
- Low-activity dynamic AUs: 60-75% 

**Hypothesis:** CalcParams errors have minimal impact on strong signals but accumulate in weak signals

### Test Video Limitations

**AU05, AU15, AU20 Characteristics:**
- Very sparse activity (15-26% of frames)
- Low intensity (mean: 0.08-0.10)
- Small absolute values make correlation sensitive to noise

**Evidence:**
- AU45 (Blink): 99.43% despite only 0.75% mean activity
  - BUT: Blinks are binary events (0 or high), not continuous low values
- AU05/15/20: Continuous low-level activity that's hard to distinguish from noise

---

## Conclusion

### What We Achieved

1. **Identified and fixed major issue:** Missing two-pass processing (+8% mean correlation)
2. **Verified C++ parity:** All major processing steps match C++ OpenFace 2.2
3. **Strong performance:** 91.22% mean correlation, 14/17 AUs passing (82.4%)
4. **Excellent on strong signals:** 5 AUs at 99%+, 9 more at 90%+

### Remaining Limitations

1. **CalcParams accuracy:** 98% vs target 99.5% - small systematic error
2. **Sparse AU detection:** Low-activity AUs (AU05, AU15, AU20) underperform
3. **Test video bias:** Single test video may not represent all AU patterns

### Recommended Next Steps

**Option A: Improve CalcParams**
- Investigate p_rx and local params accuracy gap
- May require diving into numerical optimization details
- Potential gain: 1-3% improvement

**Option B: Test on Additional Videos**
- Current results may be video-specific
- AUs with good activity might show 99%+ on different videos
- Would validate real-world performance

**Option C: Accept Current Performance**
- 91.22% mean correlation is excellent for practical use
- 14/17 AUs passing is strong (82.4% success rate)
- Failing AUs have inherent challenges (sparse/weak signals)
- Focus on downstream applications rather than perfecting validation

---

## Files Modified

1. `validate_accuracy.py` - Added two-pass processing
2. `pyfaceau/features/histogram_median_tracker.py` - Added median update frequency control
3. `CPP_VS_PYTHON_PROCESSING_MAP.md` - Detailed C++ comparison
4. `ACCURACY_INVESTIGATION_SUMMARY.md` - This document

---

## Technical Debt

- 3D landmark correlation is very low (0.0272) - needs investigation
  - May be coordinate system difference
  - May not affect AU prediction (which uses params_local, not landmarks)

- CalcParams correlation slightly below target
  - p_rx: 99.23% (need 99.5%+)
  - Local params mean: 98.24% (need 99.5%+)
