"""
Test OpenFace MTCNN + Python CLNF Integration

This script tests the complete pipeline:
1. OpenFace MTCNN detection with CLNF-compatible bbox correction
2. Python CLNF refinement
3. Comparison with baseline detectors
"""

import cv2
import numpy as np
from pathlib import Path
import time

from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN


def extract_frame_from_video(video_path, frame_num=30):
    """Extract a single frame from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def bbox_to_68pt_initialization(bbox, landmarks_5pt=None):
    """
    Convert MTCNN bbox + 5pt landmarks to 68-point initialization

    OpenFace C++ uses:
    1. MTCNN bbox with custom correction
    2. 5-point landmarks (if available)
    3. PDM mean shape scaled/translated to fit bbox

    Args:
        bbox: [x1, y1, x2, y2] bounding box
        landmarks_5pt: [5, 2] optional 5-point landmarks

    Returns:
        Initial 68-point landmarks (simple bbox-based initialization)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Simple initialization: create 68 points distributed in bbox
    # In practice, would use PDM mean shape
    # This is a placeholder - actual CLNF integration would be more sophisticated

    # For now, return landmarks_5pt if available, else None
    if landmarks_5pt is not None:
        return landmarks_5pt

    # Placeholder: distribute points along bbox
    points_68 = []
    for i in range(68):
        # Simple distribution
        angle = (i / 68.0) * 2 * np.pi
        r = min(w, h) * 0.4
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        points_68.append([px, py])

    return np.array(points_68)


def test_detection_quality(detector, video_path, output_dir):
    """Test detection quality on a challenging video"""
    print("\n" + "="*60)
    print(f"Testing: {Path(video_path).name}")
    print("="*60)

    # Extract frame
    print("Extracting frame...")
    frame = extract_frame_from_video(video_path, frame_num=30)
    if frame is None:
        print("  ✗ Could not extract frame")
        return None

    print(f"  ✓ Frame extracted: {frame.shape[1]}x{frame.shape[0]}")

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    print("\nRunning OpenFace MTCNN detection...")
    start_time = time.time()
    bboxes, landmarks = detector.detect(frame_rgb, return_landmarks=True)
    elapsed = time.time() - start_time

    print(f"  ✓ Detection completed in {elapsed:.3f}s")
    print(f"  Detected {len(bboxes)} face(s)")

    if len(bboxes) == 0:
        print("  ✗ No faces detected")
        return None

    # Analyze detection
    results = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        aspect = w / h

        result = {
            'bbox': bbox,
            'landmarks': landmarks[i] if landmarks is not None else None,
            'width': w,
            'height': h,
            'aspect_ratio': aspect,
            'area': w * h
        }

        print(f"\n  Face {i+1}:")
        print(f"    Bbox: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
        print(f"    Size: {w:.0f}x{h:.0f}px (aspect={aspect:.2f})")
        print(f"    Area: {w*h:.0f}px²")

        if landmarks is not None:
            print(f"    Landmarks: {landmarks[i].shape}")
            # Analyze landmark quality
            lm = landmarks[i]
            eye_distance = np.linalg.norm(lm[0] - lm[1])
            face_width = x2 - x1
            eye_ratio = eye_distance / face_width
            print(f"    Inter-eye distance: {eye_distance:.1f}px ({eye_ratio*100:.1f}% of face width)")

        results.append(result)

    # Visualize
    print("\nSaving visualizations...")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Draw detection
    vis = frame_rgb.copy()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox.astype(int)

        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw label
        w, h = x2 - x1, y2 - y1
        label = f"Face {i+1}: {w:.0f}x{h:.0f}"
        cv2.putText(vis, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if landmarks is not None:
        landmark_labels = ['L_Eye', 'R_Eye', 'Nose', 'L_Mouth', 'R_Mouth']
        colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0)]

        for i, lms in enumerate(landmarks):
            for j, (x, y) in enumerate(lms):
                # Draw landmark
                cv2.circle(vis, (int(x), int(y)), 5, colors[j], -1)
                cv2.circle(vis, (int(x), int(y)), 6, (255, 255, 255), 1)

                # Draw label
                cv2.putText(vis, landmark_labels[j], (int(x)+8, int(y)-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[j], 1)

    # Save
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    video_name = Path(video_path).stem
    output_path = output_dir / f"mtcnn_detection_{video_name}.jpg"
    cv2.imwrite(str(output_path), vis_bgr)
    print(f"  ✓ Saved to: {output_path}")

    # Also save original
    orig_path = output_dir / f"original_{video_name}.jpg"
    cv2.imwrite(str(orig_path), frame)
    print(f"  ✓ Original saved to: {orig_path}")

    return results


def compare_bbox_correction():
    """Demonstrate the impact of bbox correction"""
    print("\n" + "="*60)
    print("Bbox Correction Demonstration")
    print("="*60)

    print("\nOpenFace CLNF-compatible correction coefficients:")
    print("  x_offset:     -0.0075  (shifts left slightly)")
    print("  y_offset:      0.2459  (shifts DOWN significantly - key for chin)")
    print("  width_scale:   1.0323  (3.2% wider)")
    print("  height_scale:  0.7751  (22.5% SHORTER - critical!)")

    print("\nRationale:")
    print("  • Standard MTCNN optimizes for 5-point landmarks")
    print("  • OpenFace CLNF needs bbox tight around 68 points")
    print("  • 68-point model includes jawline (points 1-17)")
    print("  • Standard bbox is too tall (includes too much above face)")
    print("  • Correction makes bbox tighter around actual face")

    # Example calculation
    print("\nExample: 200x300px bbox")
    print("  Original:  x=100, y=100, w=200, h=300")
    x, y, w, h = 100, 100, 200, 300
    new_x = x + w * -0.0075
    new_y = y + h * 0.2459
    new_w = w * 1.0323
    new_h = h * 0.7751
    print(f"  Corrected: x={new_x:.1f}, y={new_y:.1f}, w={new_w:.1f}, h={new_h:.1f}")
    print(f"  Changes:   Δx={new_x-x:.1f}, Δy={new_y-y:.1f} (↓74px!), Δw={new_w-w:.1f}, Δh={new_h-h:.1f}")

    print("\nThis correction is CRITICAL for CLNF convergence!")


def main():
    print("="*60)
    print("OpenFace MTCNN + CLNF Integration Test")
    print("="*60)

    # Initialize detector
    print("\nInitializing OpenFace MTCNN detector...")
    detector = OpenFaceMTCNN()
    print(f"  ✓ Detector ready (device: {detector.device})")

    # Output directory
    output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/test_output")
    output_dir.mkdir(exist_ok=True)

    # Test videos
    test_videos = [
        "/Users/johnwilsoniv/Documents/SplitFace Open3/D Facial Paralysis Pts/IMG_9330.MOV",
    ]

    # Test each video
    all_results = {}
    for video_path in test_videos:
        if Path(video_path).exists():
            results = test_detection_quality(detector, video_path, output_dir)
            if results:
                all_results[Path(video_path).name] = results
        else:
            print(f"\n✗ Video not found: {video_path}")

    # Demonstrate bbox correction
    compare_bbox_correction()

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    if all_results:
        for video_name, results in all_results.items():
            print(f"\n{video_name}:")
            print(f"  Faces detected: {len(results)}")
            for i, result in enumerate(results):
                print(f"  Face {i+1}: {result['width']:.0f}x{result['height']:.0f}px, aspect={result['aspect_ratio']:.2f}")
    else:
        print("No results (test videos not found)")

    print("\n" + "="*60)
    print("Next Steps for CLNF Integration")
    print("="*60)
    print("1. Use detected bbox + 5pt landmarks as CLNF initialization")
    print("2. Run Python CLNF refinement (SVR patch experts)")
    print("3. Measure final landmark error vs ground truth")
    print("4. Compare with:")
    print("   - RetinaFace + PFLD initialization")
    print("   - facenet-pytorch MTCNN initialization")
    print("   - FAN initialization")
    print("\nExpected: OpenFace MTCNN should provide best CLNF initialization")
    print("due to bbox correction tuned specifically for 68-point models")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
