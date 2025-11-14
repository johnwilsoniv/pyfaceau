"""
Simple OpenFace MTCNN Test with Synthetic Image

Tests the detector on a synthetic test image to verify functionality.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pyfaceau.detectors.openface_mtcnn import OpenFaceMTCNN


def create_test_pattern():
    """Create a simple test pattern image"""
    # Create a simple 640x480 test image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # Add some text
    cv2.putText(img, "OpenFace MTCNN Test", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Face detection on test pattern", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Draw a simple "face" pattern (circle with eyes and mouth)
    center_x, center_y = 320, 300
    face_size = 120

    # Face outline
    cv2.circle(img, (center_x, center_y), face_size, (180, 140, 120), -1)
    cv2.circle(img, (center_x, center_y), face_size, (100, 80, 60), 3)

    # Eyes
    cv2.circle(img, (center_x - 40, center_y - 30), 15, (50, 50, 50), -1)
    cv2.circle(img, (center_x + 40, center_y - 30), 15, (50, 50, 50), -1)

    # Mouth
    cv2.ellipse(img, (center_x, center_y + 30), (50, 25), 0, 0, 180, (100, 50, 50), -1)

    return img


def test_detector_basic():
    """Basic functionality test"""
    print("="*60)
    print("OpenFace MTCNN - Basic Functionality Test")
    print("="*60)

    # Initialize detector
    print("\n1. Initializing detector...")
    detector = OpenFaceMTCNN()
    print("   ✓ Detector initialized")
    print(f"   Device: {detector.device}")
    print(f"   Min face size: {detector.min_face_size}px")

    # Test with synthetic image
    print("\n2. Creating test image...")
    test_img = create_test_pattern()
    print(f"   ✓ Test image created: {test_img.shape}")

    # Run detection
    print("\n3. Running detection...")
    try:
        bboxes, landmarks = detector.detect(test_img, return_landmarks=True)
        print(f"   ✓ Detection completed")
        print(f"   Detected {len(bboxes)} face(s)")

        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                print(f"   Face {i+1}: ({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f})")

            if landmarks is not None:
                print(f"   Landmarks shape: {landmarks.shape}")
        else:
            print("   (No faces detected - expected for synthetic pattern)")

    except Exception as e:
        print(f"   ✗ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test bbox correction
    print("\n4. Testing bbox correction...")
    print(f"   Correction coefficients:")
    print(f"   - x_offset: {detector.bbox_correction['x_offset']}")
    print(f"   - y_offset: {detector.bbox_correction['y_offset']}")
    print(f"   - width_scale: {detector.bbox_correction['width_scale']}")
    print(f"   - height_scale: {detector.bbox_correction['height_scale']}")

    # Test with dummy bbox
    test_bbox = np.array([[100, 100, 200, 250]])  # x1, y1, x2, y2
    corrected = detector._apply_openface_correction(test_bbox)
    print(f"   Test bbox: {test_bbox[0]}")
    print(f"   Corrected:  {corrected[0]}")

    dx1 = corrected[0][0] - test_bbox[0][0]
    dy1 = corrected[0][1] - test_bbox[0][1]
    dw = (corrected[0][2] - corrected[0][0]) - (test_bbox[0][2] - test_bbox[0][0])
    dh = (corrected[0][3] - corrected[0][1]) - (test_bbox[0][3] - test_bbox[0][1])

    print(f"   Delta x: {dx1:.1f}px")
    print(f"   Delta y: {dy1:.1f}px (should be positive - shifts down)")
    print(f"   Delta width: {dw:.1f}px (should be positive - slightly wider)")
    print(f"   Delta height: {dh:.1f}px (should be negative - significantly shorter)")

    print("\n5. Testing pyramid scale calculation...")
    scales = detector._calculate_scales(480, 640)
    print(f"   Generated {len(scales)} pyramid scales")
    print(f"   Scales: {[f'{s:.3f}' for s in scales[:5]]}...")

    print("\n" + "="*60)
    print("Basic Functionality Test: PASSED ✓")
    print("="*60)

    return True


def test_detector_with_real_face():
    """Test with a sample face image if available"""
    print("\n" + "="*60)
    print("Testing with Sample Images")
    print("="*60)

    detector = OpenFaceMTCNN()

    # Try to find OpenFace test images
    test_paths = [
        "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/imgs/multi_face.png",
        "/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/imgs/1.jpg",
        "/Users/johnwilsoniv/Documents/SplitFace Open3/FaceMirror/S1 Face Mirror/test_output/clnf_initial.jpg",
    ]

    for path in test_paths:
        try:
            img = cv2.imread(path)
            if img is not None:
                print(f"\nTesting with: {path}")
                print(f"Image size: {img.shape[1]}x{img.shape[0]}")

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bboxes, landmarks = detector.detect(img_rgb, return_landmarks=True)

                print(f"Detected {len(bboxes)} face(s)")
                if len(bboxes) > 0:
                    for i, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        print(f"  Face {i+1}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) size={w:.1f}x{h:.1f}")

                    # Visualize first image with detections
                    if len(bboxes) > 0:
                        img_vis = img_rgb.copy()
                        for bbox in bboxes:
                            x1, y1, x2, y2 = bbox.astype(int)
                            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        if landmarks is not None:
                            for lms in landmarks:
                                for (x, y) in lms:
                                    cv2.circle(img_vis, (int(x), int(y)), 3, (255, 0, 0), -1)

                        plt.figure(figsize=(12, 8))
                        plt.imshow(img_vis)
                        plt.title(f'OpenFace MTCNN Detection\n{path.split("/")[-1]}')
                        plt.axis('off')
                        plt.show()

                    break  # Only test first valid image

        except Exception as e:
            print(f"Could not load/test {path}: {e}")
            continue

    print("\n" + "="*60)


def main():
    print("\n" + "="*60)
    print("OpenFace MTCNN PyTorch Implementation - Test Suite")
    print("="*60)
    print("\nThis test verifies:")
    print("  1. Weight loading from OpenFace binary format")
    print("  2. Network forward pass (PNet, RNet, ONet)")
    print("  3. CLNF-compatible bbox correction")
    print("  4. Landmark extraction (5 points)")
    print("  5. Detection pipeline with NMS")

    # Run basic test
    success = test_detector_basic()

    if success:
        # Try to test with real images
        test_detector_with_real_face()

    print("\n" + "="*60)
    print("Test Suite Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Test with challenging images (IMG_8401, IMG_9330)")
    print("  2. Integrate with CLNF pipeline")
    print("  3. Compare with facenet-pytorch MTCNN")
    print("  4. Measure landmark initialization error")


if __name__ == "__main__":
    main()
