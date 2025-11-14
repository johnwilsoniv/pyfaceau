"""
Test OpenFace MTCNN Detector

Tests the PyTorch implementation of OpenFace MTCNN on challenging images
and compares with existing detectors.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN


def draw_detections(image, bboxes, landmarks=None, title="Detection Result"):
    """Draw bounding boxes and landmarks on image"""
    img_vis = image.copy()

    # Draw bounding boxes
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmarks
    if landmarks is not None:
        for lms in landmarks:
            for (x, y) in lms:
                cv2.circle(img_vis, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    return img_vis


def test_single_image(detector, image_path, title=""):
    """Test detector on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return None, None

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect faces
    print("Running detection...")
    start_time = time.time()
    bboxes, landmarks = detector.detect(image_rgb, return_landmarks=True)
    elapsed = time.time() - start_time

    print(f"Detection time: {elapsed:.3f}s")
    print(f"Detected {len(bboxes)} face(s)")

    if len(bboxes) > 0:
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            print(f"  Face {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) size={w:.1f}x{h:.1f}")

            if landmarks is not None:
                print(f"    Landmarks: {landmarks[i]}")

    # Visualize
    if len(bboxes) > 0:
        img_vis = draw_detections(image_rgb, bboxes, landmarks, title=title)
    else:
        print("No faces detected - skipping visualization")
        img_vis = None

    return bboxes, landmarks


def test_bbox_correction_impact(detector):
    """Test the impact of OpenFace bbox correction"""
    print("\n" + "="*60)
    print("Testing Impact of OpenFace Bbox Correction")
    print("="*60)

    # Load test image
    test_image = "/Users/johnwilsoniv/Documents/SplitFace Open3/test_images/IMG_8401.jpg"
    image = cv2.imread(test_image)
    if image is None:
        print(f"Test image not found: {test_image}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with correction
    print("\n1. With OpenFace bbox correction:")
    bboxes_corrected, _ = detector.detect(image_rgb, return_landmarks=False)
    if len(bboxes_corrected) > 0:
        x1, y1, x2, y2 = bboxes_corrected[0]
        print(f"   Corrected bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"   Size: {x2-x1:.1f} x {y2-y1:.1f}")

    # Temporarily disable correction to see raw bbox
    print("\n2. Without OpenFace correction (raw MTCNN output):")
    correction_backup = detector.bbox_correction.copy()
    detector.bbox_correction = {'x_offset': 0, 'y_offset': 0, 'width_scale': 1.0, 'height_scale': 1.0}

    bboxes_raw, _ = detector.detect(image_rgb, return_landmarks=False)
    if len(bboxes_raw) > 0:
        x1, y1, x2, y2 = bboxes_raw[0]
        print(f"   Raw bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        print(f"   Size: {x2-x1:.1f} x {y2-y1:.1f}")

    # Restore correction
    detector.bbox_correction = correction_backup

    # Compare
    if len(bboxes_corrected) > 0 and len(bboxes_raw) > 0:
        print("\n3. Difference:")
        diff_x1 = bboxes_corrected[0][0] - bboxes_raw[0][0]
        diff_y1 = bboxes_corrected[0][1] - bboxes_raw[0][1]
        diff_x2 = bboxes_corrected[0][2] - bboxes_raw[0][2]
        diff_y2 = bboxes_corrected[0][3] - bboxes_raw[0][3]
        print(f"   Delta x1: {diff_x1:.1f}px")
        print(f"   Delta y1: {diff_y1:.1f}px (should be positive - shifts down)")
        print(f"   Delta x2: {diff_x2:.1f}px")
        print(f"   Delta y2: {diff_y2:.1f}px (should be negative - reduces height)")

        # Visualize both
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Raw bbox
        img_raw = image_rgb.copy()
        x1, y1, x2, y2 = bboxes_raw[0].astype(int)
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (255, 0, 0), 3)
        axes[0].imshow(img_raw)
        axes[0].set_title('Raw MTCNN Bbox (no correction)', fontsize=14)
        axes[0].axis('off')

        # Corrected bbox
        img_corr = image_rgb.copy()
        x1, y1, x2, y2 = bboxes_corrected[0].astype(int)
        cv2.rectangle(img_corr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        axes[1].imshow(img_corr)
        axes[1].set_title('OpenFace Corrected Bbox (CLNF-ready)', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()


def main():
    print("="*60)
    print("OpenFace MTCNN Detector - Comprehensive Test")
    print("="*60)

    # Initialize detector
    print("\nInitializing OpenFace MTCNN detector...")
    detector = OpenFaceMTCNN(
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        device=None  # Auto-detect
    )
    print(f"Detector initialized on device: {detector.device}")

    # Test images
    test_images = [
        ("/Users/johnwilsoniv/Documents/SplitFace Open3/test_images/IMG_8401.jpg", "IMG_8401 (Surgical Markings)"),
        ("/Users/johnwilsoniv/Documents/SplitFace Open3/test_images/IMG_9330.jpg", "IMG_9330 (Extreme Pose)"),
    ]

    results = {}

    # Test each image
    for image_path, title in test_images:
        if Path(image_path).exists():
            bboxes, landmarks = test_single_image(detector, image_path, title)
            results[title] = {'bboxes': bboxes, 'landmarks': landmarks}
        else:
            print(f"\nWarning: Test image not found: {image_path}")

    # Test bbox correction impact
    test_bbox_correction_impact(detector)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for title, result in results.items():
        if result['bboxes'] is not None:
            print(f"{title}: {len(result['bboxes'])} face(s) detected")
        else:
            print(f"{title}: Image not found")

    # Performance notes
    print("\n" + "="*60)
    print("OpenFace MTCNN Key Features")
    print("="*60)
    print("1. Custom bbox correction for CLNF compatibility")
    print(f"   - x_offset: {detector.bbox_correction['x_offset']}")
    print(f"   - y_offset: {detector.bbox_correction['y_offset']} (shifts bbox down)")
    print(f"   - width_scale: {detector.bbox_correction['width_scale']} (slightly wider)")
    print(f"   - height_scale: {detector.bbox_correction['height_scale']} (significantly shorter)")
    print("\n2. Returns 5-point facial landmarks")
    print("3. Weights from OpenFace 2.2 MTCNN models")
    print("4. Optimized for 68-point CLNF initialization")

    # Show all plots
    plt.show()

    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
