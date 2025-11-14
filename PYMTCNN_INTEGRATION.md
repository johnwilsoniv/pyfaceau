# PyMTCNN + PyFaceAU Integration Guide

## Overview

This guide explains how to use PyMTCNN for cross-platform face detection with PyFaceAU's Action Unit extraction pipeline.

### Why PyMTCNN + PyFaceAU?

**Cross-Platform Support:**
- ✅ **Linux/Windows** with CUDA (NVIDIA GPUs): 50+ FPS
- ✅ **macOS** with CoreML (Apple Silicon): 34 FPS
- ✅ **CPU fallback** (all platforms): 5-10 FPS

**Performance Comparison:**
```
┌─────────────────────┬──────────────┬────────────────┐
│ Platform            │ RetinaFace   │ PyMTCNN        │
├─────────────────────┼──────────────┼────────────────┤
│ Apple Silicon (M3)  │ ~20 FPS      │ 34 FPS (1.7x)  │
│ NVIDIA GPU (CUDA)   │ ~20 FPS      │ 50+ FPS (2.5x) │
│ CPU                 │ ~5 FPS       │ 5-10 FPS       │
└─────────────────────┴──────────────┴────────────────┘
```

## Installation

### Step 1: Install PyFaceAU

```bash
cd /path/to/pyfaceau
pip install -e .
```

### Step 2: Install PyMTCNN

Choose installation based on your platform:

**For NVIDIA GPU (CUDA):**
```bash
pip install pymtcnn[onnx-gpu]
```

**For Apple Silicon (M1/M2/M3):**
```bash
pip install pymtcnn[coreml]
```

**For CPU-only:**
```bash
pip install pymtcnn[onnx]
```

### Step 3: Download PyFaceAU Weights

```bash
python -m pyfaceau.download_weights
```

## Quick Start

### Option 1: Use the Integration Example

```bash
python examples/pymtcnn_integration_example.py \
    --video input.mp4 \
    --output results.csv \
    --backend auto  # Or: cuda, coreml, cpu
```

### Option 2: Use PyMTCNN Detector Directly

```python
from pyfaceau.detectors import PyMTCNNDetector
from pyfaceau import FullPythonAUPipeline

# Create PyMTCNN detector
face_detector = PyMTCNNDetector(
    backend='auto',  # auto, cuda, coreml, or cpu
    min_face_size=60,
    verbose=True
)

# Use with PyFaceAU pipeline (replacing RetinaFace)
# Simply pass face_detector to the pipeline instead of ONNXRetinaFaceDetector
```

### Option 3: Manual Integration

```python
from pyfaceau.detectors import PyMTCNNDetector, CunjianPFLDDetector
from pyfaceau.alignment import CalcParams, OpenFace22FaceAligner
import cv2

# Initialize components
face_detector = PyMTCNNDetector(backend='cuda', verbose=True)
landmark_detector = CunjianPFLDDetector('weights/pfld_cunjian.onnx')
calc_params = CalcParams('weights/In-the-wild_aligned_PDM_68.txt')
face_aligner = OpenFace22FaceAligner(
    'weights/In-the-wild_aligned_PDM_68.txt',
    'weights/tris_68_full.txt'
)

# Process frame
frame = cv2.imread('image.jpg')

# 1. Detect face
dets, _ = face_detector.detect_faces(frame)
if len(dets) > 0:
    bbox = dets[0][:4].astype(int)

    # 2. Get 68-point landmarks
    landmarks, conf = landmark_detector.detect_landmarks(frame, bbox)

    # 3. Estimate 3D pose
    h, w = frame.shape[:2]
    params_local, params_global, _ = calc_params.estimate_pose(landmarks, w, h)

    # 4. Align face
    tx, ty, rz = params_global[4], params_global[5], params_global[3]
    aligned_face = face_aligner.align_face(frame, landmarks, tx, ty, rz)

    # 5. Continue with HOG extraction and AU prediction...
```

## API Reference

### PyMTCNNDetector

```python
from pyfaceau.detectors import PyMTCNNDetector

detector = PyMTCNNDetector(
    backend='auto',              # 'auto', 'cuda', 'coreml', 'cpu', 'onnx'
    min_face_size=60,            # Minimum face size in pixels
    thresholds=[0.6, 0.7, 0.7],  # Detection thresholds [PNet, RNet, ONet]
    factor=0.709,                # Image pyramid scale factor
    confidence_threshold=0.5,    # Minimum confidence for filtering
    nms_threshold=0.7,           # NMS threshold
    vis_threshold=0.5,           # Visibility threshold
    verbose=False                # Print initialization messages
)

# Detect faces (compatible with RetinaFace interface)
dets, img = detector.detect_faces(image)
# Returns: dets shape (N, 15) = [x1, y1, x2, y2, conf, lm1_x, lm1_y, ..., lm5_x, lm5_y]

# Get primary face
face_crop, dets = detector.get_face(image)

# Get backend info
backend_info = detector.get_backend_info()
# Returns: {'backend': 'ONNX + CUDA', 'provider': 'CUDAExecutionProvider'}
```

### Factory Function

```python
from pyfaceau.detectors import create_pymtcnn_detector

# Quick setup with recommended settings
detector = create_pymtcnn_detector(
    backend='cuda',
    min_face_size=60,
    verbose=True
)
```

## Backend Selection

### Automatic (Recommended)

```python
detector = PyMTCNNDetector(backend='auto', verbose=True)
```

Priority order: **CUDA → CoreML → CPU**

### Manual Selection

**Force CUDA (NVIDIA GPU):**
```python
detector = PyMTCNNDetector(backend='cuda')
```

**Force CoreML (Apple Silicon):**
```python
detector = PyMTCNNDetector(backend='coreml')
```

**Force CPU:**
```python
detector = PyMTCNNDetector(backend='cpu')
```

**Force ONNX (auto-selects provider):**
```python
detector = PyMTCNNDetector(backend='onnx')
```

## Performance Tips

### For Maximum Speed

1. **Use appropriate backend for your hardware:**
   - NVIDIA GPU: Use `backend='cuda'`
   - Apple Silicon: Use `backend='coreml'`
   - CPU: Use `backend='cpu'` or `'onnx'`

2. **Adjust min_face_size:**
   ```python
   # Smaller min_face_size = slower but detects smaller faces
   detector = PyMTCNNDetector(min_face_size=40)  # Slower
   detector = PyMTCNNDetector(min_face_size=80)  # Faster
   ```

3. **Tune thresholds:**
   ```python
   # Higher thresholds = faster but fewer detections
   detector = PyMTCNNDetector(thresholds=[0.7, 0.8, 0.8])  # Fast, strict
   detector = PyMTCNNDetector(thresholds=[0.5, 0.6, 0.6])  # Slow, permissive
   ```

### For Best Accuracy

```python
detector = PyMTCNNDetector(
    backend='auto',
    min_face_size=40,              # Detect smaller faces
    thresholds=[0.6, 0.7, 0.7],    # Balanced
    confidence_threshold=0.7,      # Filter weak detections
    verbose=True
)
```

## Troubleshooting

### PyMTCNN not available

```
Error: pymtcnn not installed
```

**Solution:**
```bash
pip install pymtcnn[onnx-gpu]  # For CUDA
pip install pymtcnn[coreml]    # For Apple Silicon
pip install pymtcnn[onnx]      # For CPU
```

### CUDA provider not available

```
Requested provider 'CUDAExecutionProvider' is not available
```

**Solution:**
1. Install CUDA toolkit (11.x or 12.x)
2. Install onnxruntime-gpu:
   ```bash
   pip install onnxruntime-gpu
   ```
3. Verify:
   ```python
   import onnxruntime as ort
   print(ort.get_available_providers())
   # Should include 'CUDAExecutionProvider'
   ```

### CoreML provider not available

```
CoreML not available for this model
```

**Solution:**
- Ensure you're on macOS
- Install coremltools:
  ```bash
  pip install coremltools>=7.0
  ```

## Integration with Existing PyFaceAU Code

### Replace RetinaFace in pipeline.py

**Before:**
```python
from pyfaceau.detectors import ONNXRetinaFaceDetector

detector = ONNXRetinaFaceDetector('weights/retinaface_mobilenet025_coreml.onnx')
```

**After:**
```python
from pyfaceau.detectors import PyMTCNNDetector

detector = PyMTCNNDetector(backend='auto', verbose=True)
```

The API is identical - just swap the detector class!

## Examples

See `examples/pymtcnn_integration_example.py` for a complete working example.

## Performance Benchmarks

Tested on various platforms:

```
Apple M3 Max (16-core):
  - RetinaFace + CoreML: ~20 FPS
  - PyMTCNN + CoreML:    34.26 FPS ✨ 71% faster

NVIDIA RTX 4090:
  - RetinaFace: ~20 FPS
  - PyMTCNN + CUDA: 50+ FPS ✨ 2.5x faster

Intel i9-13900K (CPU):
  - RetinaFace: ~5 FPS
  - PyMTCNN + ONNX: 5-10 FPS
```

## License

PyMTCNN is licensed under CC BY-NC 4.0
PyFaceAU is licensed under CC BY-NC 4.0

## Support

- PyMTCNN Issues: https://github.com/johnwilsoniv/pymtcnn/issues
- PyFaceAU Issues: https://github.com/johnwilsoniv/pyfaceau/issues
- Documentation: See README.md files in respective repositories

---

**Built for high-performance facial behavior analysis**
