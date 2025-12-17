# pyfaceau

Python implementation of OpenFace 2.2's Facial Action Unit extraction pipeline.

## Installation

```bash
pip install pyfaceau
```

## Usage

```python
from pyfaceau import FaceAnalyzer

analyzer = FaceAnalyzer()
result = analyzer.analyze(image)

print(result.au_intensities)  # 17 action unit intensities
print(result.landmarks)       # 68 facial landmarks
print(result.pose)            # head pose
```

## What it does

- Extracts 17 facial action units (AU01, AU02, AU04, AU05, AU06, AU07, AU09, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU25, AU26, AU45)
- Detects 68 facial landmarks via [pyclnf](https://github.com/johnwilsoniv/pyclnf)
- Estimates 3D head pose
- No C++ compilation required

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

CC BY-NC 4.0 — free for non-commercial use with attribution.
