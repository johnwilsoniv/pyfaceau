#!/usr/bin/env python3
"""
Visualize PyMTCNN + PFLD vs C++ OpenFace landmark comparison
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Test frame to visualize
FRAME_IDX = 0  # First frame
TEST_DIR = Path("pfld_vs_openface_test")
FRAME_PATH = TEST_DIR / "frames" / f"frame_{FRAME_IDX:03d}.jpg"
CSV_PATH = TEST_DIR / "cpp_output" / f"frame_{FRAME_IDX:03d}.csv"

# Add pymtcnn to path
pymtcnn_path = Path(__file__).parent.parent / "pymtcnn"
sys.path.insert(0, str(pymtcnn_path))
sys.path.insert(0, str(Path(__file__).parent))

from pymtcnn import MTCNN
import importlib.util

# Load PFLD detector
spec = importlib.util.spec_from_file_location(
    "pfld",
    Path(__file__).parent / "pyfaceau" / "detectors" / "pfld.py"
)
pfld_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfld_module)
CunjianPFLDDetector = pfld_module.CunjianPFLDDetector

print("="*80)
print("Landmark Visualization & Bbox Format Check")
print("="*80)

# Load frame
img = cv2.imread(str(FRAME_PATH))
print(f"\nFrame: {FRAME_PATH.name}")
print(f"  Shape: {img.shape}")

# Initialize detectors
print("\nInitializing PyMTCNN + PFLD...")
pymtcnn = MTCNN(backend='auto', verbose=False)
weights_dir = Path(__file__).parent / "weights"
pfld_model = weights_dir / "pfld_cunjian.onnx"
pfld = CunjianPFLDDetector(str(pfld_model), use_coreml=True)

# Run PyMTCNN
print("\n" + "="*80)
print("PyMTCNN Detection")
print("="*80)
bboxes, landmarks_mtcnn = pymtcnn.detect(img)
print(f"Number of faces detected: {len(bboxes)}")

if len(bboxes) > 0:
    bbox = bboxes[0]
    print(f"\nBbox (from PyMTCNN): {bbox}")

    # PyMTCNN returns [x, y, w, h] format according to docs
    h, w = img.shape[:2]
    print(f"  Image dimensions: width={w}, height={h}")
    print(f"  Bbox format: (x, y, w, h)")
    print(f"    x={bbox[0]:.2f}, y={bbox[1]:.2f}, w={bbox[2]:.2f}, h={bbox[3]:.2f}")
    print(f"    → x2={bbox[0]+bbox[2]:.2f}, y2={bbox[1]+bbox[3]:.2f}")

    bbox_format = "xywh"
    # Convert to x1,y1,x2,y2 for PFLD
    bbox_for_pfld = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    print(f"  Converted to (x1,y1,x2,y2) for PFLD: ({bbox_for_pfld[0]:.1f}, {bbox_for_pfld[1]:.1f}, {bbox_for_pfld[2]:.1f}, {bbox_for_pfld[3]:.1f})")

# Calculate the ACTUAL bbox that PFLD uses internally (square with 10% padding)
x_min, y_min, x_max, y_max = bbox_for_pfld
w = x_max - x_min
h = y_max - y_min
size = int(max([w, h]) * 1.1)
cx = int(x_min + w / 2)
cy = int(y_min + h / 2)
pfld_x1 = cx - size // 2
pfld_x2 = pfld_x1 + size
pfld_y1 = cy - size // 2
pfld_y2 = pfld_y1 + size

# Clip to image bounds (what PFLD actually sees)
img_h, img_w = img.shape[:2]
pfld_x1_clipped = max(0, pfld_x1)
pfld_y1_clipped = max(0, pfld_y1)
pfld_x2_clipped = min(img_w, pfld_x2)
pfld_y2_clipped = min(img_h, pfld_y2)

print(f"\nPFLD internal bbox calculation:")
print(f"  MTCNN bbox (x1,y1,x2,y2): ({x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f})")
print(f"  Face size: w={w:.1f}, h={h:.1f}")
print(f"  Square size (max * 1.1): {size}")
print(f"  Square bbox before clipping: ({pfld_x1}, {pfld_y1}, {pfld_x2}, {pfld_y2})")
print(f"  Square bbox after clipping:  ({pfld_x1_clipped}, {pfld_y1_clipped}, {pfld_x2_clipped}, {pfld_y2_clipped})")

# Run PFLD
print("\n" + "="*80)
print("PFLD Landmark Detection")
print("="*80)
python_landmarks, conf = pfld.detect_landmarks(img, bbox_for_pfld)
print(f"Landmarks detected: {python_landmarks.shape}")
print(f"Confidence: {conf}")
print(f"\nSample landmarks (Python PFLD):")
for i in [0, 16, 27, 33, 48]:  # Jaw, eyebrows, nose, eyes, mouth
    print(f"  Point {i:2d}: ({python_landmarks[i, 0]:7.2f}, {python_landmarks[i, 1]:7.2f})")

# Load C++ OpenFace landmarks
print("\n" + "="*80)
print("C++ OpenFace Landmarks")
print("="*80)
df = pd.read_csv(CSV_PATH)
cpp_landmarks = np.zeros((68, 2))
for i in range(68):
    cpp_landmarks[i, 0] = df[f'x_{i}'].iloc[0]
    cpp_landmarks[i, 1] = df[f'y_{i}'].iloc[0]

print(f"Landmarks loaded: {cpp_landmarks.shape}")
print(f"\nSample landmarks (C++ OpenFace):")
for i in [0, 16, 27, 33, 48]:
    print(f"  Point {i:2d}: ({cpp_landmarks[i, 0]:7.2f}, {cpp_landmarks[i, 1]:7.2f})")

# Compute errors
errors = np.linalg.norm(python_landmarks - cpp_landmarks, axis=1)
print("\n" + "="*80)
print("Error Analysis")
print("="*80)
print(f"Mean error:   {np.mean(errors):.2f} px")
print(f"Median error: {np.median(errors):.2f} px")
print(f"Max error:    {np.max(errors):.2f} px")
print(f"Min error:    {np.min(errors):.2f} px")

# Create visualization
print("\n" + "="*80)
print("Creating Visualization")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# Plot 1: Python PFLD landmarks
img1 = img.copy()
for i, (x, y) in enumerate(python_landmarks):
    cv2.circle(img1, (int(x), int(y)), 3, (0, 255, 0), -1)
    if i % 5 == 0:  # Label every 5th point
        cv2.putText(img1, str(i), (int(x)+5, int(y)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

# Draw MTCNN bbox in red (x, y, w, h format)
cv2.rectangle(img1,
             (int(bbox[0]), int(bbox[1])),
             (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])),
             (0, 0, 255), 2)

# Draw PFLD's actual square bbox in blue
cv2.rectangle(img1,
             (pfld_x1_clipped, pfld_y1_clipped),
             (pfld_x2_clipped, pfld_y2_clipped),
             (255, 0, 0), 2)

axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'Python PFLD\nRed=MTCNN bbox, Blue=PFLD crop\nMean Error: {np.mean(errors):.2f}px', fontsize=12)
axes[0].axis('off')

# Plot 2: C++ OpenFace landmarks
img2 = img.copy()
for i, (x, y) in enumerate(cpp_landmarks):
    cv2.circle(img2, (int(x), int(y)), 3, (255, 0, 0), -1)
    if i % 5 == 0:
        cv2.putText(img2, str(i), (int(x)+5, int(y)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title('C++ OpenFace (Gold Standard)', fontsize=14)
axes[1].axis('off')

# Plot 3: Overlay with error magnitude
img3 = img.copy()

# Draw C++ landmarks in blue
for i, (x, y) in enumerate(cpp_landmarks):
    cv2.circle(img3, (int(x), int(y)), 3, (255, 0, 0), -1)

# Draw Python landmarks in green with lines showing error
for i, (x_py, y_py) in enumerate(python_landmarks):
    x_cpp, y_cpp = cpp_landmarks[i]

    # Draw line from C++ to Python
    cv2.line(img3, (int(x_cpp), int(y_cpp)), (int(x_py), int(y_py)),
            (0, 165, 255), 1)  # Orange line

    # Draw Python landmark
    cv2.circle(img3, (int(x_py), int(y_py)), 3, (0, 255, 0), -1)

    # Label worst landmarks
    if errors[i] > np.percentile(errors, 90):  # Top 10% worst
        cv2.putText(img3, f"{i}:{errors[i]:.0f}px",
                   (int(x_py)+5, int(y_py)-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

axes[2].imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
axes[2].set_title('Overlay (Blue=C++, Green=Python, Orange=Error)\nWorst errors labeled',
                  fontsize=14)
axes[2].axis('off')

plt.tight_layout()
output_path = TEST_DIR / "landmark_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")

# Also create error heatmap
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
landmark_names = [
    'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw',
    'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw', 'Jaw',  # 0-16
    'R.Eyebrow', 'R.Eyebrow', 'R.Eyebrow', 'R.Eyebrow', 'R.Eyebrow',  # 17-21
    'L.Eyebrow', 'L.Eyebrow', 'L.Eyebrow', 'L.Eyebrow', 'L.Eyebrow',  # 22-26
    'Nose', 'Nose', 'Nose', 'Nose',  # 27-30
    'Nose', 'Nose', 'Nose', 'Nose', 'Nose',  # 31-35
    'R.Eye', 'R.Eye', 'R.Eye', 'R.Eye', 'R.Eye', 'R.Eye',  # 36-41
    'L.Eye', 'L.Eye', 'L.Eye', 'L.Eye', 'L.Eye', 'L.Eye',  # 42-47
    'Outer.Mouth', 'Outer.Mouth', 'Outer.Mouth', 'Outer.Mouth',
    'Outer.Mouth', 'Outer.Mouth', 'Outer.Mouth', 'Outer.Mouth',
    'Outer.Mouth', 'Outer.Mouth', 'Outer.Mouth', 'Outer.Mouth',  # 48-59
    'Inner.Mouth', 'Inner.Mouth', 'Inner.Mouth', 'Inner.Mouth',
    'Inner.Mouth', 'Inner.Mouth', 'Inner.Mouth', 'Inner.Mouth'  # 60-67
]

colors = plt.cm.RdYlGn_r(errors / errors.max())
bars = ax.barh(range(68), errors, color=colors)
ax.set_yticks(range(68))
ax.set_yticklabels([f"{i}: {landmark_names[i]}" for i in range(68)], fontsize=7)
ax.set_xlabel('Pixel Error', fontsize=12)
ax.set_title(f'Per-Landmark Error (Frame {FRAME_IDX})', fontsize=14)
ax.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}px')
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
error_path = TEST_DIR / "error_breakdown.png"
plt.savefig(error_path, dpi=150, bbox_inches='tight')
print(f"✓ Error breakdown saved to: {error_path}")

print("\n" + "="*80)
print("Analysis Complete")
print("="*80)
print(f"\nBbox format detected: {bbox_format}")
print(f"Bbox used for PFLD: {bbox_for_pfld}")
print(f"\nVisualization files:")
print(f"  - {output_path}")
print(f"  - {error_path}")
print("\n" + "="*80)
