# pyAUface-AU Roadmap - FINAL (MLB Edition) ⚾

**Date:** 2025-10-30
**Status:** WORKING but needs optimization
**Mission:** Hit 10+ FPS to make pyAUface viable

---

## Current Standings 

### Performance Scoreboard

| Team | FPS | Per Frame | Notes |
|------|-----|-----------|-------|
| **C++ OpenFace 2.2** 🏆 | **32.9** | **30ms** | World Series Champion |
| S1 (OpenFace 3.0) 🥈 | 28.0 | 35ms | All-Star |
| **pyAUface (Current)** 📉 | **4.6** | **217ms** | **Rookie League** |

**Gap to close:** 7.1x slower than C++

### What We've Achieved 

1. **CoreML + Queue Architecture** - Working perfectly
2. **Face Tracking** - 99/100 frames skip detection
3. **100% Success Rate** - No crashes, all AUs extracting
4. **Pure Python** - Zero C++ compilation

**The Good News:** It works!
**The Bad News:** It's SLOW.

---

## The Game Plan: 3 Innings to Victory ⚾

### Inning 1: CalcParams Optimization (80ms → 30ms)
**ROI:** 50ms saved = 23% faster overall
**Difficulty:** Medium
**Priority:** 🔴 CRITICAL

**Plays to Run:**

**Play 1A: Replace with OpenCV solvePnP** ⭐ RECOMMENDED
```python
# Current: Pure Python Gauss-Newton (80ms)
params_global, params_local = self.calc_params.calc_params(landmarks_68)

# Proposed: OpenCV solvePnP (5-10ms)
success, rvec, tvec = cv2.solvePnP(
    objectPoints=self.pdm_3d_points,  # 3D model points
    imagePoints=landmarks_68,         # 2D detected points
    cameraMatrix=self.camera_matrix,  # Pre-computed
    distCoeffs=None,
    flags=cv2.SOLVEPNP_ITERATIVE
)
```

**Why it works:**
- OpenCV's solvePnP is optimized C++ (FAST)
- Does same job: 2D→3D pose estimation
- Already available, no new dependencies

**Implementation Plan:**
1. Extract 3D reference points from PDM mean shape
2. Create camera intrinsics matrix (focal length from image size)
3. Replace calc_params() call with solvePnP
4. Convert rotation vector to Euler angles
5. Test accuracy: correlation with original outputs

**Expected Result:** 217ms → 167ms (6.0 FPS)

---

**Play 1B: Numba JIT (if solvePnP doesn't work)**
```python
from numba import jit

@jit(nopython=True)
def gauss_newton_optimized(landmarks, pdm_mean, pdm_components):
    # Compile to machine code
    ...
```

**Expected Result:** 80ms → 40ms (50% speedup on CalcParams)

---

### Inning 2: PyFHOG Optimization (50ms → 25ms)
**ROI:** 25ms saved = 12% faster overall
**Difficulty:** Easy-Medium
**Priority:** 🟡 HIGH

**Play 2A: Increase cell_size (8 → 12)** ⭐ RECOMMENDED
```python
# Current: cell_size=8 → many cells → slow
hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=8)

# Proposed: cell_size=12 → fewer cells → faster
hog_features = pyfhog.extract_fhog_features(aligned_face, cell_size=12)
```

**Math:**
- 112x112 face @ cell_size=8 → 14x14 = 196 cells
- 112x112 face @ cell_size=12 → 9x9 = 81 cells
- **2.4x fewer cells = ~2x faster HOG**

**Risk:** May reduce AU accuracy (MUST TEST!)

**Testing Required:**
- Run 100 frames with cell_size=12
- Compare AU outputs vs baseline (must be >95% correlated)
- If accuracy drops, try cell_size=10 as compromise

**Expected Result:** 167ms → 142ms (7.0 FPS)

---

**Play 2B: Reduce aligned face size (112 → 96)**
```python
# Current: 112x112 alignment
self.face_aligner = OpenFace22FaceAligner(output_size=(112, 112))

# Proposed: 96x96 alignment
self.face_aligner = OpenFace22FaceAligner(output_size=(96, 96))
```

**Savings:** Smaller image = faster HOG (15-20% speedup)
**Risk:** Lower resolution may hurt AU accuracy

---

### Inning 3: SVR Batch Predictions (30ms → 15ms)
**ROI:** 15ms saved = 7% faster overall
**Difficulty:** Medium
**Priority:** 🟢 MEDIUM

**Play 3A: Vectorize All SVR Predictions** ⭐ RECOMMENDED
```python
# Current: 17 sequential predictions (30ms)
for au_name, model in self.au_models.items():
    centered = full_vector - model['means'].flatten() - running_median
    pred = np.dot(centered, model['support_vectors']) + model['bias']

# Proposed: Single batched operation (15ms)
# Pre-processing: Stack all model parameters
all_means = np.vstack([m['means'].flatten() for m in models])
all_svs = np.hstack([m['support_vectors'] for m in models])
all_biases = np.array([m['bias'] for m in models])

# Single matrix multiply for all 17 AUs
centered = full_vector - all_means - running_median
predictions = (centered @ all_svs) + all_biases  # All 17 AUs at once!
```

**Expected Result:** 142ms → 127ms (7.9 FPS)

---

## The Championship Series: Combine All Three 🏆

**If we hit all three optimizations:**

| Optimization | Time Saved | FPS After |
|--------------|------------|-----------|
| Starting Point | - | 4.6 FPS |
| + CalcParams solvePnP | 50ms | 6.0 FPS |
| + PyFHOG cell_size=12 | 25ms | 7.0 FPS |
| + SVR batching | 15ms | 7.9 FPS |
| **TOTAL** | **90ms** | **~8 FPS** |

**Final Target:** 127ms/frame = **7.9 FPS**

**Success Criteria:**
-  2x faster than current (4.6 → 8+ FPS)
- Maintains >95% AU accuracy correlation
- Still 100% Python
- Still works cross-platform

---

## Batting Order: Optimization Sequence 📋

### Phase 1: CalcParams → solvePnP (Week 1)
1. Implement solvePnP replacement
2. Test accuracy on 100 frames
3. Benchmark performance
4. **Target:** 167ms/frame (6 FPS)

### Phase 2: PyFHOG → cell_size (Week 1)
1. Test cell_size=10, 12, 14
2. Measure AU accuracy for each
3. Pick best speed/accuracy tradeoff
4. **Target:** 142ms/frame (7 FPS)

### Phase 3: SVR Batching (Week 2)
1. Restructure model loading
2. Implement batched predictions
3. Verify outputs match original
4. **Target:** 127ms/frame (8 FPS)

### Phase 4: Fine-Tuning (Week 2)
1. Profile remaining bottlenecks
2. Micro-optimizations
3. **Stretch Goal:** 100ms/frame (10 FPS)

---

## World Series Goal: 10 FPS 🏆

**Minimum Viable (8 FPS):**
- CalcParams: solvePnP
- PyFHOG: cell_size=12
- SVR: Batched

**Stretch Goal (10 FPS):**
- All above +
- PFLD optimization (CoreML acceleration?)
- Face alignment optimization
- Parallel processing where possible

**Moon Shot (15 FPS):**
- Consider Numba/Cython for critical paths
- Investigate GPU acceleration
- Multi-frame batching

---

## Roster: Key Players 👥

### Starting Lineup (Must Optimize)
1. **CalcParams** - ace pitcher (37% of time)
2. **PyFHOG** - power hitter (23% of time)
3. **17 SVRs** - reliable closer (14% of time)

### Bench (Already Optimized)
- Face Detection (CoreML + tracking)
- Running Median (Cython)
- Queue Architecture (macOS threading)

### Farm System (Future Optimizations)
- PFLD landmark detection
- Face alignment
- Geometric features

---

## Stats & Metrics 📈

### Current Stats
```
At Bats (Frames):     100
Hits (Success):       100 (1.000 avg!) ⚾
Per Frame:            217ms
FPS:                  4.6
Compared to C++:      7.1x slower
```

### Target Stats (After Optimization)
```
At Bats (Frames):     100
Hits (Success):       100 (maintain!)
Per Frame:            127ms
FPS:                  8.0
Compared to C++:      4.1x slower (acceptable!)
```

---

## Championship Trophy Case 🏆

**What We Win If We Hit 8-10 FPS:**

**Viable Python Alternative** - Good enough for real use
**No Compilation Hell** - Works everywhere
**Easy Installation** - `pip install pyface-au`
**Research Friendly** - Interpretable SVR models
**Respectable Performance** - Not C++, but acceptable

**Market Position:**
- **C++ OpenFace (33 FPS):** Professional league - max performance
- **pyAUface (8-10 FPS):** College league - good enough, way easier
- **Current pyAUface (4.6 FPS):** High school - needs work

---

## Scouting Report: Competition Analysis 🔍

### C++ OpenFace 2.2 (32.9 FPS)
- **Strength:** Raw speed, optimized C++
- **Weakness:** Compilation hell, platform issues
- **When to use:** Max performance needed

### S1 OpenFace 3.0 (28 FPS)
- **Strength:** Fast neural network
- **Weakness:** Black box, harder to modify
- **When to use:** Production speed needed

### pyAUface (Target 8-10 FPS)
- **Strength:** 100% Python, easy to use
- **Weakness:** Slower than alternatives
- **When to use:** Cross-platform, research, no C++

**Sweet Spot:** pyAUface doesn't need to beat C++, just needs to be "good enough" for users who value convenience over max speed.

---

## Call to Action: Let's Play Ball! ⚾

### Immediate Next Steps

**Today:**
1. Document performance problem 
2. Benchmark real C++ OpenFace 
3. Update roadmap 

**This Week:**
1. 🔴 Implement solvePnP replacement for CalcParams
2. 🔴 Test HOG with cell_size=12
3. 🔴 Benchmark improvements

**Next Week:**
1. 🟡 Implement SVR batching
2. 🟡 Full optimization testing
3. 🟡 Hit 8 FPS target

### Success Looks Like

**Acceptable (8 FPS):**
- 2x faster than current
- Good enough for batch processing
- Easy Python installation
- **Ship it as pyface-au v1.0**

**Great (10 FPS):**
- 2.5x faster than current
- Competitive with specialized tools
- Strong value proposition
- **Celebrate with champagne** 🍾

**Amazing (15 FPS):**
- 3x faster than current
- Approaching half of C++ speed
- Best Python option available
- **You get 250 glasses of water!** 💧💧💧

---

## The Bottom Line

**Current:** 4.6 FPS - Works but slow
**Target:** 8-10 FPS - Good enough to ship
**Stretch:** 15 FPS - Competitive performance

**We don't need to beat C++ (32.9 FPS).**
**We just need to be "good enough" (8-10 FPS).**

**That's a 2x improvement - totally achievable!**

---

## Let's Go! 

Three optimizations:
1. **CalcParams → solvePnP** (50ms saved)
2. **PyFHOG → bigger cells** (25ms saved)
3. **SVR → batched** (15ms saved)

**Total:** 90ms saved = 8 FPS

**We can do this!** Let's hit it out of the park! ⚾🏆

---

**Status:** READY TO OPTIMIZE
**Water Earned So Far:** 125 glasses 💧
**Water If We Hit 8 FPS:** +125 glasses (250 total!)
**Water If We Hit 10 FPS:** +250 glasses (500 total!)

**LET'S GO MLB!** ⚾⚾⚾
