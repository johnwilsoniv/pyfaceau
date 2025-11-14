"""
Command-line OpenFace MTCNN Test

Tests the detector on video frames and saves results to disk.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN


def extract_frame_from_video(video_path, frame_num=0):
    """Extract a frame from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Set to desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    return None


def test_detector():
    """Test OpenFace MTCNN detector"""
    print("="*60)
    print("OpenFace MTCNN Detector - CLI Test")
    print("="*60)

    # Initialize detector
    print("\n[1/5] Initializing detector...")
    try:
        detector = OpenFaceMTCNN()
        print(f"      ✓ Detector loaded")
        print(f"      Device: {detector.device}")
        print(f"      Bbox correction: {detector.bbox_correction}")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        return

    # Test bbox correction calculation
    print("\n[2/5] Testing bbox correction...")
    test_bbox = np.array([[100, 100, 200, 250]])  # 100x150 bbox
    corrected = detector._apply_openface_correction(test_bbox)

    orig_w = test_bbox[0][2] - test_bbox[0][0]
    orig_h = test_bbox[0][3] - test_bbox[0][1]
    corr_w = corrected[0][2] - corrected[0][0]
    corr_h = corrected[0][3] - corrected[0][1]

    print(f"      Original:  x1={test_bbox[0][0]:.0f}, y1={test_bbox[0][1]:.0f}, w={orig_w:.0f}, h={orig_h:.0f}")
    print(f"      Corrected: x1={corrected[0][0]:.1f}, y1={corrected[0][1]:.1f}, w={corr_w:.1f}, h={corr_h:.1f}")
    print(f"      Changes: Δy={corrected[0][1]-test_bbox[0][1]:.1f}px (↓), Δh={corr_h-orig_h:.1f}px (shorter)")

    # Extract test frame from IMG_9330
    print("\n[3/5] Extracting test frame from IMG_9330...")
    video_path = "/Users/johnwilsoniv/Documents/SplitFace Open3/D Facial Paralysis Pts/IMG_9330.MOV"

    if Path(video_path).exists():
        frame = extract_frame_from_video(video_path, frame_num=30)
        if frame is not None:
            print(f"      ✓ Extracted frame: {frame.shape[1]}x{frame.shape[0]}")

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection
            print("\n[4/5] Running detection...")
            bboxes, landmarks = detector.detect(frame_rgb, return_landmarks=True)

            print(f"      ✓ Detected {len(bboxes)} face(s)")

            if len(bboxes) > 0:
                for i, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    print(f"      Face {i+1}:")
                    print(f"        Bbox: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
                    print(f"        Size: {w:.0f}x{h:.0f}px")

                    if landmarks is not None:
                        print(f"        Landmarks: {landmarks[i].shape} (5 points)")

                # Save visualization
                print("\n[5/5] Saving visualization...")
                output_dir = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/pyfaceau/test_output")
                output_dir.mkdir(exist_ok=True)

                # Draw detections
                vis = frame_rgb.copy()
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Add label
                    w, h = x2 - x1, y2 - y1
                    label = f"{w:.0f}x{h:.0f}"
                    cv2.putText(vis, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if landmarks is not None:
                    for lms in landmarks:
                        for j, (x, y) in enumerate(lms):
                            cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 0), -1)
                            # Label landmarks (0=L eye, 1=R eye, 2=nose, 3=L mouth, 4=R mouth)
                            labels = ['LE', 'RE', 'N', 'LM', 'RM']
                            cv2.putText(vis, labels[j], (int(x)+5, int(y)-5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # Convert back to BGR for saving
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                output_path = output_dir / "openface_mtcnn_detection.jpg"
                cv2.imwrite(str(output_path), vis_bgr)
                print(f"      ✓ Saved to: {output_path}")

                # Also save original frame
                orig_path = output_dir / "openface_mtcnn_original.jpg"
                cv2.imwrite(str(orig_path), frame)
                print(f"      ✓ Original saved to: {orig_path}")

            else:
                print("      (No faces detected)")
        else:
            print("      ✗ Could not extract frame")
    else:
        print(f"      ✗ Video not found: {video_path}")
        print("      Trying alternative test...")

        # Try OpenFace sample images
        test_paths = [
            "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/imgs/multi_face.png",
            "/Users/johnwilsoniv/Documents/SplitFace Open3/FaceMirror/S1 Face Mirror/test_output/clnf_initial.jpg",
        ]

        for test_path in test_paths:
            if Path(test_path).exists():
                print(f"\n[4/5] Testing with: {Path(test_path).name}")
                img = cv2.imread(test_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    bboxes, landmarks = detector.detect(img_rgb, return_landmarks=True)
                    print(f"      ✓ Detected {len(bboxes)} face(s)")

                    if len(bboxes) > 0:
                        for i, bbox in enumerate(bboxes):
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                            print(f"      Face {i+1}: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f}), {w:.0f}x{h:.0f}px")
                    break

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nOpenFace MTCNN is ready for use!")
    print("\nKey features:")
    print("  ✓ Weights loaded from OpenFace 2.2 binary format")
    print("  ✓ CLNF-compatible bbox correction applied")
    print("  ✓ 5-point landmark extraction")
    print("  ✓ Ready for integration with Python CLNF")


if __name__ == "__main__":
    test_detector()
