"""
Canonical Configuration for PyFaceAU Pipeline

These settings match C++ OpenFace 2.2 defaults for accurate AU extraction.
DO NOT modify without thorough testing against C++ reference output.

Configuration locked on: Dec 5, 2025
Tested against: IMG_0942.MOV (1110 frames), IMG_0422.MOV (bearded)
Target accuracy: Sub-pixel landmark error (<1.0 px), AU correlation >0.95
"""

# =============================================================================
# CLNF Landmark Detection Configuration
# =============================================================================
CLNF_CONFIG = {
    'max_iterations': 10,
    'convergence_threshold': 0.005,  # Gold standard (stricter than 0.01)
    'sigma': 2.25,                   # C++ CECLM default (1.5 × 1.5 scale factor)
    'use_eye_refinement': True,      # Enabled: fixed transpose bug, now improves accuracy
    'convergence_profile': 'video',  # Enable template tracking + scale adaptation
    'detector': False,               # Disable built-in detector (pyfaceau handles)
    'use_gpu': True,                 # Enable GPU acceleration (10-20x speedup)
    'gpu_device': 'auto',            # GPU device: 'auto', 'mps', 'cuda', 'cpu'
    'use_validator': False,          # Disable detection validator (68% of CLNF time)
}

# =============================================================================
# MTCNN Face Detection Configuration
# =============================================================================
MTCNN_CONFIG = {
    'backend': 'coreml',             # Deterministic backend for reproducibility
    'confidence_threshold': 0.5,     # Face confidence threshold
    'nms_threshold': 0.7,            # Non-max suppression threshold
}

# =============================================================================
# HOG Feature Extraction Configuration
# =============================================================================
HOG_CONFIG = {
    'hog_dim': 4464,                 # 56×14 cell grid × 9 bins (4464 total)
    'hog_bins': 1000,                # Histogram bins for running median
    'hog_min': -0.005,               # CRITICAL: NOT 0.0 - matches C++ OpenFace
    'hog_max': 1.0,                  # Maximum HOG value
}

# =============================================================================
# Geometric Feature Configuration
# =============================================================================
GEOM_CONFIG = {
    'geom_dim': 238,                 # 34 PDM params × 7 derivatives
    'geom_bins': 10000,              # Histogram bins for running median
    'geom_min': -60.0,               # Minimum geometric feature value
    'geom_max': 60.0,                # Maximum geometric feature value
}

# =============================================================================
# AU Prediction Configuration
# =============================================================================
AU_CONFIG = {
    # Online AU correction (C++ CorrectOnlineAUs equivalent)
    'num_bins': 200,                 # C++ default
    'min_val': -3.0,                 # C++ default
    'max_val': 5.0,                  # C++ default
    'cutoff_ratio': 0.10,            # 10th percentile baseline
    'min_frames': 10,                # Minimum frames before correction
    'skip_au17_cutoff': True,        # AU17 exception (unusual weight distribution)
    'apply_online_dyn_shift': False,  # Online 10% shift (no impact in testing)

    # Two-pass processing
    'max_stored_frames': 3000,       # OpenFace default for re-prediction

    # AU-specific cutoff overrides
    # NOTE: With the fix to include zeros in cutoff calculation (matching C++),
    # most overrides are no longer needed. Only keep AU26_r which has unusual behavior.
    'cutoff_overrides': {
        'AU26_r': 0.12,              # Original: 0.30 -> 0.9317 correlation (best achievable)
    },
}

# =============================================================================
# Running Median Tracker Configuration
# =============================================================================
RUNNING_MEDIAN_CONFIG = {
    'hog_dim': HOG_CONFIG['hog_dim'],
    'geom_dim': GEOM_CONFIG['geom_dim'],
    'hog_bins': HOG_CONFIG['hog_bins'],
    'hog_min': HOG_CONFIG['hog_min'],
    'hog_max': HOG_CONFIG['hog_max'],
    'geom_bins': GEOM_CONFIG['geom_bins'],
    'geom_min': GEOM_CONFIG['geom_min'],
    'geom_max': GEOM_CONFIG['geom_max'],
}

# =============================================================================
# CLNF Optimizer Defaults (in pyclnf/clnf.py)
# =============================================================================
# These are documented here for reference - actual defaults are in pyclnf:
#   regularization: 22.5          # C++ CECLM: 25.0 base × 0.9 = 22.5
#   sigma: 2.25                   # C++ CECLM: 1.5 base × 1.5 = 2.25
#   weight_multiplier: 0.0        # C++ disables NU-RLMS weighting

# =============================================================================
# Known Fixes Applied
# =============================================================================
# 1. Optimizer defaults: reg=22.5, sigma=2.25 (in pyclnf/clnf.py)
# 2. PDM epsilon: No +1e-10 in eigenvalue regularization (in pyclnf/core/pdm.py)
# 3. BORDER_REPLICATE: Used in patch extraction (in pyclnf/core/optimizer.py)
# 4. Template tracking: Enabled in video mode (in pyclnf/clnf.py)
# 5. PyMTCNN bbox: [x,y,w,h] format handled correctly (in pymtcnn_detector.py)
# 6. HOG min: -0.005 (NOT 0.0) matches C++ OpenFace

# =============================================================================
# Validation Targets
# =============================================================================
VALIDATION_TARGETS = {
    'max_landmark_error_px': 1.0,    # Mean error threshold
    'min_au_correlation': 0.95,      # For expressed AUs (std > 0.02)
    'test_video': 'IMG_0942.MOV',    # Primary test video
}
