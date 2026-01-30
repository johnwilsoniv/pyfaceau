# pyfaceau

A python-based implementation of OpenFace 2.2's Facial Action Unit extraction pipeline with an accurate dlib substitute (ptmtcnn, pyclnf).

**Accuracy: r = 0.97 correlation with C++ OpenFace 2.2**

## Installation

```bash
pip install pyfaceau
```

This automatically installs dependencies:
- [pyclnf](https://github.com/johnwilsoniv/pyclnf) - Facial landmark detection (68 points)
- [pymtcnn](https://github.com/johnwilsoniv/pymtcnn) - Face detection
- [pyfhog](https://github.com/johnwilsoniv/pyfhog) - FHOG feature extraction

Model weights are downloaded automatically on first use (~50MB).

## Quick Start

### Video Processing (Recommended)

```python
from pyfaceau import OpenFaceProcessor

# Initialize processor
processor = OpenFaceProcessor(verbose=True)

# Process video to CSV (same format as OpenFace)
processor.process_video("input.mp4", "output.csv")
```

### Batch Processing

```python
from pyfaceau import process_videos

# Process all videos in a directory
process_videos(
    directory_path="/path/to/videos",
    output_dir="/path/to/output"
)
```

### Frame-by-Frame Processing

```python
from pyfaceau import FullPythonAUPipeline
from pathlib import Path
import cv2

# Initialize pipeline with model paths
weights_dir = Path("weights")
pipeline = FullPythonAUPipeline(
    pdm_file=str(weights_dir / "In-the-wild_aligned_PDM_68.txt"),
    au_models_dir=str(weights_dir / "AU_predictors"),
    triangulation_file=str(weights_dir / "tris_68_full.txt"),
    patch_expert_file=str(weights_dir / "svr_patches_0.25_general.txt")
)

# Process single frame
image = cv2.imread("face.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = pipeline.process_frame(image_rgb, frame_num=0)

if result['success']:
    print("AU intensities:", result['au_intensities'])
    print("Landmarks shape:", result['landmarks_2d'].shape)  # (68, 2)
    print("Pose (pitch, yaw, roll):", result['pose'])
```

## Output Format

### CSV Output Columns

The output CSV matches OpenFace format:
- `frame` - Frame number
- `timestamp` - Time in seconds
- `confidence` - Detection confidence
- `success` - Whether face was detected
- `AU01_r` through `AU45_r` - AU intensities (0.0 - 5.0)
- `pose_Rx`, `pose_Ry`, `pose_Rz` - Head pose in radians
- `x_0` through `x_67`, `y_0` through `y_67` - 68 landmark coordinates

### Action Units

17 facial action units with intensity values (0.0 - 5.0):

| AU | Description |
|----|-------------|
| AU01 | Inner Brow Raiser |
| AU02 | Outer Brow Raiser |
| AU04 | Brow Lowerer |
| AU05 | Upper Lid Raiser |
| AU06 | Cheek Raiser |
| AU07 | Lid Tightener |
| AU09 | Nose Wrinkler |
| AU10 | Upper Lip Raiser |
| AU12 | Lip Corner Puller |
| AU14 | Dimpler |
| AU15 | Lip Corner Depressor |
| AU17 | Chin Raiser |
| AU20 | Lip Stretcher |
| AU23 | Lip Tightener |
| AU25 | Lips Part |
| AU26 | Jaw Drop |
| AU45 | Blink |

## Accuracy

Validated against C++ OpenFace 2.2

| Metric | Correlation |
|--------|-------------|
| **Overall Mean** | r = 0.97 |
| **Overall Median** | r = 0.996 |
| Static AUs | r = 0.98 |
| Dynamic AUs | r = 0.96 |

Per-AU correlations:
- AU01: 0.997, AU02: 0.999, AU04: 0.989, AU05: 0.999
- AU06: 0.999, AU07: 0.996, AU09: 0.997, AU10: 0.994
- AU12: 0.998, AU14: 0.974, AU15: 0.893, AU17: 0.948
- AU20: 0.817, AU23: 0.996, AU25: 0.984, AU26: 0.902, AU45: 0.998

## Requirements

- Python 3.8+
- numpy
- opencv-python
- torch
- scipy

## Acknowledgments

Based on OpenFace 2.0:

> Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit. IEEE International Conference on Automatic Face and Gesture Recognition.

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

CC BY-NC 4.0 - Free for non-commercial use with attribution.
