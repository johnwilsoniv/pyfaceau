#  OpenFace 2.2 Python Migration - Session Start Guide

**Last Updated:** 2025-10-29 (Evening Session)
**Current Phase:** Phase 4 (Final Phase!) - 75% Complete
**Next Task:** Implement Python Face Alignment

---

##  Quick Status

**COMPLETE:**
- Phase 1: Core Components (SVR, HOG, PDM parsers)
- Phase 2: Perfect AU Prediction (r = 0.9996)
- Phase 3: pyfhog v0.1.0 published to PyPI
- **Phase 4.2: pyfhog Validation** - **r = 1.000 PERFECT!** ⭐

⏳ **IN PROGRESS:**
- **Phase 4.1: Python Face Alignment** ← **START HERE**

---

##  TODAY'S ACCOMPLISHMENTS

### 🎉 Major Breakthrough: pyfhog Perfect Validation!

1. **Installed pyfhog v0.1.0** in of22_python_env
2. **Fixed critical bug:** Frame indexing issue in validation script
   - OpenFace writes all frame indices as 1.0 in .hog files
   - Validation script was loading same aligned face for all frames
   - Fixed by using `frame_num = i + 1` instead of `frame_indices[i]`
3. **Validated pyfhog:** **r = 1.000** (PERFECT!) with zero difference across all frames!
4. **Researched alignment algorithm:** Documented complete OpenFace face alignment process

### 📁 Key Files Created Today
- `validate_pyfhog_integration.py` - Validates pyfhog against OpenFace C++
- `diagnose_hog_ordering.py` - Tests feature ordering (confirmed: correct!)
- `diagnose_frame2_difference.py` - Debugged frame indexing bug
- `pyfhog_validation_output/` - Persistent output directory with aligned faces

---

##  NEXT SESSION: Implement Python Face Alignment

### Step 1: Review Alignment Algorithm (5 min)

**Key parameters discovered:**
```python
sim_scale = 0.7  # For AU analysis
output_size = (112, 112)  # Not 96x96!
rigid_points = [1,2,3,4,12,13,14,15,27,28,29,31,32,33,34,35,36,39,40,41,42,45,46,47]
```

**Algorithm flow:**
1. Extract rigid points from 68 landmarks
2. Compute similarity transform (scale + rotation) using `AlignShapesWithScale()`
3. Build 2×3 affine warp matrix
4. Center output at (width/2, height/2)
5. Apply `cv2.warpAffine()` with `INTER_LINEAR`

**Reference implementations:**
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/src/Face_utils.cpp` (lines 109-146)
- `/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/Utilities/include/RotationHelpers.h` (`AlignShapesWithScale`)

### Step 2: Implement Python Face Aligner (1-2 hours)

**Create: `openface22_face_aligner.py`**

```python
class OpenFace22FaceAligner:
    """
    Pure Python implementation of OpenFace 2.2 face alignment

    Aligns faces from 68 landmarks to 112x112 canonical reference frame
    using similarity transform (scale + rotation + translation)
    """

    def __init__(self, pdm_file, sim_scale=0.7, output_size=(112, 112)):
        """
        Load PDM mean shape for reference alignment

        Args:
            pdm_file: Path to PDM model (e.g., "pdm_68_multi_pie.txt")
            sim_scale: Scaling factor for reference shape (default: 0.7 for AUs)
            output_size: Output image size (default: 112x112)
        """
        pass

    def align_face(self, image, landmarks_68, rigid_only=True):
        """
        Align face to canonical reference frame

        Args:
            image: Input image (BGR format, any size)
            landmarks_68: 68 facial landmarks as (68, 2) array
            rigid_only: Use only rigid points (default: True)

        Returns:
            aligned_face: 112x112 aligned face image (RGB for pyfhog)
        """
        pass

    def _extract_rigid_points(self, landmarks):
        """Extract 24 rigid points from 68 landmarks"""
        pass

    def _align_shapes_with_scale(self, src_points, dst_points):
        """
        Compute similarity transform (scale + rotation)

        Implementation of OpenFace's AlignShapesWithScale:
        1. Mean-normalize both src and dst
        2. Compute RMS scale for each
        3. Normalize by scale
        4. Compute rotation matrix (Kabsch algorithm)
        5. Return: (s_dst / s_src) * R
        """
        pass
```

### Step 3: Validate Python Alignment (1 hour)

**Create: `validate_python_alignment.py`**

Test strategy:
1. Load video frame + 68 landmarks from of22_validation
2. Align with Python implementation
3. Compare pixel-by-pixel with OpenFace C++ aligned face
4. Compute MSE and correlation
5. **Target:** MSE < 1.0, correlation > 0.99

**Validation data available:**
- OpenFace aligned faces: `pyfhog_validation_output/IMG_0942_left_mirrored_aligned/`
- Landmarks from: `of22_validation/IMG_0942_left_mirrored.csv`
- Original video: `/Users/johnwilsoniv/Documents/SplitFace/S1O Processed Files/Face Mirror 1.0 Output/IMG_0942_left_mirrored.mp4`

### Step 4: Integrate with pyfhog (30 min)

**Create: `test_end_to_end_alignment.py`**

```python
# Test complete pipeline:
# Raw image + landmarks → Python alignment → pyfhog → HOG features
# Compare with OpenFace C++ pipeline

from openface22_face_aligner import OpenFace22FaceAligner
import pyfhog

aligner = OpenFace22FaceAligner('pdm_68_multi_pie.txt')
aligned_face = aligner.align_face(image, landmarks_68)
hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)

# Should get 4464 features matching OpenFace exactly!
```

---

## 📂 Key File Locations

### Working Directory
```
/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror/
```

### Python Environment
```bash
source of22_python_env/bin/activate  # Contains pyfhog v0.1.0
```

### Important Files

**Core Implementation:**
- `openface22_model_parser.py` - SVR model loading 
- `openface22_hog_parser.py` - Binary .hog parsing 
- `pdm_parser.py` - PDM shape model & geometric features 
- `histogram_median_tracker.py` - Running median 
- `validate_svr_predictions.py` - Complete validation pipeline 
- ⏳ `openface22_face_aligner.py` - **TO BE CREATED**

**Validation Scripts:**
- `validate_pyfhog_integration.py` - pyfhog validation (r=1.000!) 
- ⏳ `validate_python_alignment.py` - **TO BE CREATED**

**Reference Data:**
- PDM Model: `pdm_68_multi_pie.txt`
- SVR Models: `AU_predictors/AU*_static.dat` and `AU*_dynamic.dat`
- OpenFace C++ aligned faces: `pyfhog_validation_output/IMG_0942_left_mirrored_aligned/`
- OpenFace C++ .hog file: `pyfhog_validation_output/IMG_0942_left_mirrored.hog`

**Documentation:**
- `OPENFACE22_PYTHON_MIGRATION_ROADMAP.md` - Phase tracking Updated
- `OPENFACE22_PYTHON_MIGRATION_STATUS.md` - Detailed status Updated
- `START_HERE.md` - This file 

### OpenFace C++ Source (Reference)
```
/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/
├── src/Face_utils.cpp              # AlignFace() implementation (lines 109-146)
├── include/FaceAnalyserParameters.h # sim_scale_au = 0.7 (line 47)
└── AU_predictors/*.dat             # SVR models

/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/Utilities/
└── include/RotationHelpers.h       # AlignShapesWithScale() (lines 280-330)
```

---

##  Quick Commands

### Activate Environment
```bash
cd "/Users/johnwilsoniv/Documents/SplitFace Open3/S1 Face Mirror"
source of22_python_env/bin/activate
```

### Run Validation
```bash
# Verify pyfhog still works perfectly
python3 validate_pyfhog_integration.py

# After implementing alignment:
python3 validate_python_alignment.py
```

### Check pyfhog
```bash
python3 -c "import pyfhog; print(pyfhog.__version__)"  # Should be 0.1.0
```

---

## 🐛 Known Issues & Solutions

### Issue 1: Frame Indexing Bug FIXED
**Problem:** OpenFace writes all frame indices as 1.0 in .hog files
**Solution:** Use `frame_num = i + 1` directly instead of `frame_indices[i]`
**File:** `validate_pyfhog_integration.py:194`

### Issue 2: Image Size Confusion RESOLVED
**Problem:** Initial assumption was 96×96, but OpenFace uses 112×112
**Calculation:** 4464 features / 31 channels = 144 cells = 12×12 grid = 112 pixels with cell_size=8
**Solution:** Use 112×112 for FHOG extraction, no resizing needed

### Issue 3: Feature Ordering VERIFIED CORRECT
**Tested:** pyfhog uses same ordering as OpenFace (row, col, channel)
**Result:** r = 1.000 with original ordering - no transposition needed

---

## 📚 Algorithm Deep Dive: AlignShapesWithScale

**Reference:** RotationHelpers.h (lines 280-330)

```cpp
// Pseudocode translation for Python implementation:
def align_shapes_with_scale(src, dst):
    # 1. Mean normalize
    src_centered = src - mean(src)
    dst_centered = dst - mean(dst)

    # 2. Compute RMS scale
    s_src = sqrt(sum(src_centered**2) / n)
    s_dst = sqrt(sum(dst_centered**2) / n)

    # 3. Normalize by scale
    src_norm = src_centered / s_src
    dst_norm = dst_centered / s_dst

    # 4. Compute rotation (Kabsch algorithm)
    R = align_shapes_kabsch_2d(src_norm, dst_norm)

    # 5. Return scaled rotation
    scale = s_dst / s_src
    return scale * R  # 2×2 matrix
```

**Full affine transform:**
```python
# Build 2×3 warp matrix
warp_matrix = np.zeros((2, 3))
warp_matrix[:2, :2] = scale_rot_matrix
warp_matrix[0, 2] = -T[0] + output_width/2
warp_matrix[1, 2] = -T[1] + output_height/2

# Apply
aligned = cv2.warpAffine(image, warp_matrix, (112, 112),
                         flags=cv2.INTER_LINEAR)
```

---

## Success Criteria for Next Session

### Minimum (Phase 4.1 Complete):
- [ ] Python face alignment implemented
- [ ] Validation shows MSE < 5.0 and correlation > 0.95
- [ ] Integration with pyfhog produces correct 4464-dim features

### Target (Ready for Phase 4.3):
- [ ] Python alignment matches OpenFace C++ pixel-perfect (MSE < 1.0)
- [ ] End-to-end test: raw image → alignment → pyfhog → features works
- [ ] All aligned faces visually identical to OpenFace C++

### Stretch (Phase 4 Complete):
- [ ] Unified `OpenFace22AUPredictor` class created
- [ ] End-to-end video test produces r > 0.999 AU predictions
- [ ] Documentation complete

---

##  Final Goal: Complete Python AU Predictor

```python
from openface22_au_predictor import OpenFace22AUPredictor

# Initialize once
predictor = OpenFace22AUPredictor(
    models_dir="AU_predictors/",
    pdm_file="pdm_68_multi_pie.txt"
)

# Predict on single frame
aus = predictor.predict_frame(image, landmarks_68)
# Returns: {'AU01_r': 1.23, 'AU02_r': 0.45, ...}

# Or process entire video
results_df = predictor.predict_video(
    video_path="video.mp4",
    landmarks_csv="landmarks.csv"
)
# Returns: DataFrame with frame-by-frame AU predictions
```

**NO OpenFace C++ binary needed!** 🎉

---

##  Progress Tracker

| Phase | Status | Completion | Date |
|-------|--------|------------|------|
| Phase 1: Core Components | COMPLETE | 100% | 2025-10-27 |
| Phase 2: Perfect AU Prediction | COMPLETE | 100% | 2025-10-28 |
| Phase 3: pyfhog Publication | COMPLETE | 100% | 2025-10-29 AM |
| Phase 4.1: Face Alignment | 🔄 IN PROGRESS | 40% | 2025-10-29 PM |
| Phase 4.2: pyfhog Integration | COMPLETE | 100% | 2025-10-29 PM |
| Phase 4.3: Unified API | ⏳ PENDING | 0% | TBD |
| Phase 4.4: Testing | ⏳ PENDING | 0% | TBD |

**Overall Progress: 85% Complete** (Phase 4.1 partial credit)

---

## 💡 Tips for Next Session

1. **Start with alignment implementation** - This is the critical path
2. **Use OpenFace C++ aligned faces as ground truth** - They're in `pyfhog_validation_output/`
3. **Test incrementally** - Verify each component (rigid points extraction, transform computation, warping)
4. **Kabsch algorithm** - scipy has `scipy.spatial.transform.Rotation.align_vectors()` or implement manually
5. **Debugging** - If alignment doesn't match, visualize intermediate steps (mean shapes, transforms)

**Key insight:** Once alignment works, we're 90% done! The rest is just API wrapping.

---

##  Let's Finish This!

We're on the home stretch! Only 2-3 hours of work remain to complete the full Python OpenFace 2.2 migration.

**Next steps are clear. Let's implement that face alignment! 💪**
