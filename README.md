# pyfaceau

Pure Python implementation of OpenFace 2.2's Facial Action Unit extraction pipeline. Drop-in replacement for OpenFace with no C++ compilation required.

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

```python
import cv2
from pyfaceau import FaceAnalyzer

# Initialize analyzer
analyzer = FaceAnalyzer()

# Load and analyze an image
image = cv2.imread("face.jpg")
result = analyzer.analyze(image)

# Access results
print(result.au_intensities)  # {'AU01': 0.5, 'AU02': 0.3, ...}
print(result.landmarks)        # (68, 2) array
print(result.pose)             # (pitch, yaw, roll)
```

## Video Processing

```python
import cv2
from pyfaceau import FaceAnalyzer

analyzer = FaceAnalyzer()
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = analyzer.analyze(frame)
    if result is not None:
        # AU intensities as dict
        for au, intensity in result.au_intensities.items():
            print(f"{au}: {intensity:.2f}")

cap.release()
```

## Batch Processing

```python
from pyfaceau import FaceAnalyzer
import cv2
import pandas as pd

analyzer = FaceAnalyzer()
cap = cv2.VideoCapture("video.mp4")

results = []
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = analyzer.analyze(frame)
    if result is not None:
        row = {'frame': frame_num, **result.au_intensities}
        results.append(row)

    frame_num += 1

# Save to CSV (same format as OpenFace)
df = pd.DataFrame(results)
df.to_csv("output.csv", index=False)
```

## Output Format

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

### Landmarks

68-point facial landmarks following the Multi-PIE format, same as OpenFace.

### Pose

Head pose as (pitch, yaw, roll) in radians.

## Comparison with OpenFace 2.2

| Feature | OpenFace 2.2 | pyfaceau |
|---------|--------------|----------|
| Language | C++ | Pure Python |
| Installation | Compile from source | `pip install` |
| Platform | Linux, macOS, Windows | Same |
| GPU Acceleration | OpenCV CUDA | CoreML (Mac), CUDA |
| Action Units | 17 AUs | 17 AUs (identical) |
| Landmarks | 68 points | 68 points (identical) |
| Accuracy | Reference | r=0.86 correlation |

## Advanced Options

```python
analyzer = FaceAnalyzer(
    use_gpu=True,           # Enable GPU acceleration
    detector="pymtcnn",     # Face detector: "pymtcnn" or "retinaface"
    verbose=False           # Disable initialization messages
)
```

## Requirements

- Python 3.8+
- numpy
- opencv-python
- torch (for AU prediction)

## Acknowledgments

Based on OpenFace 2.0:

> Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit. IEEE International Conference on Automatic Face and Gesture Recognition.

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

CC BY-NC 4.0 - Free for non-commercial use with attribution.
