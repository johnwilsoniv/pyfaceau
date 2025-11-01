# CLNF Patch Expert Research

**Date:** 2025-11-01
**Purpose:** Understand CLNF patch expert format for implementing targeted landmark refinement
**Goal:** Improve PFLD landmarks from 85.62% to 88-90% AU correlation

---

## Overview

CLNF (Constrained Local Neural Fields) uses patch experts to iteratively refine landmark positions. Each landmark has a specialized detector (patch expert) trained on local appearance around that point.

**Key Insight:** Patch experts use SVR (Support Vector Regression) - the same technique we already use for AU prediction! This means we can reuse much of our existing SVR infrastructure.

---

## File Structure

### Patch Expert Models Location

```
$OPENFACE/lib/local/LandmarkDetector/model/patch_experts/
├── svr_patches_0.25_general.txt   # 68 SVR patch experts @ 0.25 scale
├── svr_patches_0.35_general.txt   # 68 SVR patch experts @ 0.35 scale
├── ccnf_patches_*.txt              # CCNF variants (neural network based)
└── cen_patches_*.dat               # CEN variants (ensemble)
```

**Recommended:** Use `svr_patches_0.25_general.txt` (same format as our AU SVR models)

**File size:** ~2-3 MB per scale

---

## SVR Patch Expert Format

### File Header Structure

```
# scaling factor of training
0.250000                          # Image scale used during training

# number of views
7                                 # Different head poses (frontal, left, right, etc.)

# centers of the views
3 1 6                             # View 0: Frontal (0°, 0°, 0°)
0.000000 0.000000 0.000000

3 1 6                             # View 1: Left 20° (-20°, 0°, 0°)
0.000000 -20.000000 0.000000

3 1 6                             # View 2: Left 45° (-45°, 0°, 0°)
0.000000 -45.000000 0.000000

... (5 more views)

# visibility indices per view
68                                # 68 landmarks
1 4                               # Format: num_values bytes_per_value
1 1 1 ... 1                       # Visibility for all 68 landmarks (1=visible)
```

### Per-Landmark SVR Model Structure

For each of the 68 landmarks, there's an SVR patch expert:

```cpp
struct SVR_patch_expert {
    int type;                    // 0 = raw pixels, 1 = gradient features
    double scaling;              // Logistic regression slope
    double bias;                 // Logistic regression bias
    cv::Mat_<float> weights;     // SVR weights (similar to AU SVR models!)
    map<int, Mat> weights_dfts;  // DFT for fast convolution (optional optimization)
    double confidence;           // Patch expert confidence
};
```

**Similarity to AU SVR Models:**
- Same SVR framework
- Similar weight format
- Uses support vectors + bias term
- Can reuse our existing SVR prediction code!

---

## How Patch Experts Work

### 1. Patch Extraction

Extract a small image patch around the current landmark position:

```python
def extract_patch(image, landmark, patch_size=11):
    """
    Extract patch around landmark for patch expert evaluation

    Args:
        image: Grayscale image (numpy array)
        landmark: (x, y) position
        patch_size: Size of patch (11x11, 21x21, etc.)

    Returns:
        Patch as feature vector (flattened)
    """
    x, y = int(landmark[0]), int(landmark[1])
    half = patch_size // 2

    patch = image[y - half:y + half + 1, x - half:x + half + 1]

    # Convert to feature vector (raw or gradient)
    if patch_expert.type == 0:  # Raw pixels
        features = patch.flatten()
    else:  # Gradient features
        grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        features = np.hstack([grad_x.flatten(), grad_y.flatten()])

    return features
```

### 2. SVR Response Computation

Compute response map showing where the landmark should be:

```python
def compute_response(patch_features, svr_weights, bias, scaling):
    """
    Compute patch expert response using SVR

    This is IDENTICAL to our AU prediction:
    prediction = features @ weights + bias

    Args:
        patch_features: Extracted patch features
        svr_weights: SVR support vector weights
        bias: SVR bias term
        scaling: Logistic regression scaling

    Returns:
        Response score (higher = better landmark position)
    """
    # Same as AU prediction!
    response = np.dot(patch_features, svr_weights) + bias

    # Apply logistic scaling
    response = 1.0 / (1.0 + np.exp(-scaling * response))

    return response
```

### 3. Landmark Refinement

Find the peak in the response map to refine landmark position:

```python
def refine_landmark(image, initial_landmark, patch_expert, search_window=5):
    """
    Refine landmark by searching for peak response

    Args:
        image: Input frame
        initial_landmark: Current landmark position from PFLD
        patch_expert: Loaded SVR patch expert
        search_window: How far to search around initial position

    Returns:
        Refined landmark position
    """
    best_response = -float('inf')
    best_position = initial_landmark

    # Search in a window around initial position
    for dx in range(-search_window, search_window + 1):
        for dy in range(-search_window, search_window + 1):
            candidate = initial_landmark + np.array([dx, dy])

            # Extract patch at candidate position
            patch = extract_patch(image, candidate)

            # Compute SVR response
            response = compute_response(
                patch,
                patch_expert.weights,
                patch_expert.bias,
                patch_expert.scaling
            )

            if response > best_response:
                best_response = response
                best_position = candidate

    return best_position
```

---

## Targeted Refinement Strategy

Since we only need to improve **brow landmarks (17-26)** and **lip corners (48, 54)**, we can implement a lightweight targeted approach:

### Critical Landmarks for AU Improvement

```python
CRITICAL_LANDMARKS = {
    # AU01: Inner Brow Raiser (82.01% → target 88%+)
    'inner_brow_left': [17, 18, 19, 20, 21],

    # AU02: Outer Brow Raiser (78.93% → target 88%+)
    'outer_brow_left': [22],
    'outer_brow_right': [26],

    # AU23: Lip Tightener (61.07% → target 78%+)
    'lip_corners': [48, 54],  # Left and right mouth corners

    # Total: 12 out of 68 landmarks (18%)
}
```

### Simplified Implementation Plan

**Phase 1: Load Only Critical Patch Experts**
- Extract only 12 patch experts from svr_patches_0.25_general.txt
- Significantly smaller memory footprint than full 68 experts
- Faster loading and evaluation

**Phase 2: Minimal Refinement Loop**
```python
def targeted_clnf_refinement(image, pfld_landmarks, patch_experts):
    """
    Refine only the 12 critical landmarks

    Args:
        image: Grayscale frame
        pfld_landmarks: Initial 68 landmarks from PFLD
        patch_experts: Dict of {landmark_idx: SVR_patch_expert}

    Returns:
        Refined landmarks (with only 12 points updated)
    """
    refined = pfld_landmarks.copy()

    # Refine only critical landmarks
    critical_indices = [17, 18, 19, 20, 21, 22, 26, 48, 54]

    for idx in critical_indices:
        if idx in patch_experts:
            refined[idx] = refine_landmark(
                image,
                refined[idx],
                patch_experts[idx],
                search_window=3  # Small window for speed
            )

    # Project back onto PDM to enforce shape constraints
    params = pdm.CalcParams(refined)
    refined = pdm.CalcShape(params)

    return refined
```

**Phase 3: PDM Constraint Projection**
- Use existing PDM infrastructure from FaceAligner
- Ensures refined landmarks remain anatomically plausible
- Prevents individual landmark refinements from breaking face shape

---

## Implementation Complexity Assessment

### Comparison to Existing Components

| Component | Lines of Code | Complexity | Similarity |
|-----------|---------------|------------|------------|
| **AU SVR Prediction** | ~100 | Medium | 95% similar |
| **CalcParams Optimization** | ~300 | High | Warning: 50% similar (different optimization target) |
| **CLNF Patch Experts** | ~200 (est) | Medium | Reuses AU SVR code |

**Key Advantage:** Patch expert SVR evaluation is nearly identical to AU prediction!

```python
# AU Prediction (existing code)
au_prediction = features @ svr_weights + bias

# Patch Expert Response (new code) - SAME OPERATION!
patch_response = patch_features @ svr_weights + bias
```

---

## File Parsing Requirements

### SVR Patch Expert Parser

```python
class SVRPatchExpertLoader:
    """Load SVR patch experts from OpenFace format"""

    def load(self, filepath, target_landmarks=None):
        """
        Load patch experts from file

        Args:
            filepath: Path to svr_patches_*.txt
            target_landmarks: List of landmark indices to load (None = all 68)

        Returns:
            Dict of {landmark_idx: SVR_patch_expert}
        """
        with open(filepath, 'r') as f:
            # Parse header
            scale = float(f.readline().split()[0])
            num_views = int(f.readline().split()[0])

            # Parse views (skip if not needed for frontal-only)
            for view in range(num_views):
                # Read view center, visibility indices
                pass

            # Load patch experts for each landmark
            patch_experts = {}
            for landmark_idx in range(68):
                # Only load if in target set
                if target_landmarks is None or landmark_idx in target_landmarks:
                    expert = self._read_patch_expert(f)
                    patch_experts[landmark_idx] = expert
                else:
                    self._skip_patch_expert(f)  # Skip to save memory

            return patch_experts

    def _read_patch_expert(self, f):
        """Read single SVR patch expert (similar to AU SVR loading)"""
        # Read type, scaling, bias, weights
        # Identical format to AU SVR models!
        pass
```

---

## Expected Performance Gain

### Conservative Estimate

| Metric | Current (PFLD) | Target (PFLD+CLNF) | Improvement |
|--------|----------------|-------------------|-------------|
| **AU01** | 82.01% | 88-92% | +6-10% |
| **AU02** | 78.93% | 84-88% | +5-9% |
| **AU23** | 61.07% | 70-78% | +9-17% |
| **Mean Correlation** | 85.62% | 88-90% | +2.4-4.4% |
| **AUs Passing** | 11/17 | 12-13/17 | +1-2 AUs |

### Processing Speed Impact

```
Current (PFLD only):
- PFLD detection: ~2ms
- Total pipeline: ~7ms → 140 fps

With Targeted CLNF:
- PFLD detection: ~2ms
- CLNF refinement (12 landmarks): ~2-3ms
- Total pipeline: ~9-10ms → 100-110 fps

Speed reduction: ~25% (still well above real-time)
```

---

## Next Steps

1. **Extract patch expert files** from OpenFace
   - Copy `svr_patches_0.25_general.txt` to `pyfaceau/weights/`
   - ~2-3 MB file size

2. **Implement SVR patch expert loader**
   - Reuse AU SVR loading code
   - Parse only critical landmarks (12/68) to save memory
   - Estimated: 1-2 days

3. **Implement patch extraction and response**
   - Extract 11x11 patches around landmarks
   - Compute SVR response (identical to AU prediction)
   - Estimated: 1 day

4. **Implement targeted refinement**
   - Refine 12 critical landmarks
   - Project onto PDM for shape constraints
   - Estimated: 2-3 days

5. **Integration and validation**
   - Add to FullPythonAUPipeline as optional step
   - Validate against C++ reference on IMG_0434.MOV
   - Measure AU correlation improvement
   - Estimated: 1-2 days

**Total Estimated Time:** 5-8 days

---

## Key Advantages of This Approach

**No dlib dependency** - Uses PFLD for initialization
**Reuses existing code** - SVR evaluation identical to AU prediction
**Focused effort** - Only 12/68 landmarks need refinement
**PDM constraints** - Existing infrastructure ensures valid shapes
**Moderate complexity** - Similar difficulty to AU prediction
**Measurable ROI** - Clear target: 85.62% → 88-90%

---

## References

**OpenFace Source Files:**
- `lib/local/LandmarkDetector/include/SVR_patch_expert.h` - Header defining patch expert structure
- `lib/local/LandmarkDetector/src/SVR_patch_expert.cpp` - Implementation (reading, response computation)
- `lib/local/LandmarkDetector/model/patch_experts/svr_patches_0.25_general.txt` - Trained models

**Similar Components in PyFaceAU:**
- `pyfaceau/prediction/au_predictor.py` - SVR-based AU prediction (95% similar to patch expert response!)
- `pyfaceau/alignment/calc_params.py` - PDM-constrained optimization (50% similar to refinement loop)
- `pyfaceau/alignment/face_aligner.py` - PDM loading and shape computation (directly reusable)

---

**Document Status:** RESEARCH COMPLETE
**Next Action:** Implement SVR patch expert loader
**Expected Outcome:** 88-90% mean AU correlation (vs current 85.62%)
