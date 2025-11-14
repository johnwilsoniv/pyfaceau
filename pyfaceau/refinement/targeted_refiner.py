"""
Targeted CLNF refinement for critical landmarks

This module implements focused CLNF refinement for brow landmarks (17-26) and
lip corners (48, 54) to improve AU01, AU02, and AU23 detection accuracy.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any

# Handle both relative and absolute imports
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from pyfaceau.refinement.svr_patch_expert import SVRPatchExpert, SVRPatchExpertLoader
else:
    from .svr_patch_expert import SVRPatchExpert, SVRPatchExpertLoader


class TargetedCLNFRefiner:
    """
    Targeted CLNF landmark refinement for critical AU landmarks.

    Refines only brow landmarks (17-26) and lip corners (48, 54) using
    OpenFace SVR patch experts. This focused approach maintains speed
    while improving accuracy for AU01, AU02, and AU23.

    Optionally enforces PDM (Point Distribution Model) constraints to ensure
    refined landmarks remain anatomically plausible.
    """

    # Critical landmarks for AU improvement
    CRITICAL_LANDMARKS = [17, 18, 19, 20, 21, 22, 26, 48, 54]

    def __init__(self, patch_expert_file: str, search_window: int = 3,
                 pdm: Optional[Any] = None, enforce_pdm: bool = False):
        """
        Initialize targeted CLNF refiner.

        Args:
            patch_expert_file: Path to svr_patches_*.txt file
            search_window: Search radius around initial landmark (pixels)
            pdm: Optional PDM for shape constraint enforcement
            enforce_pdm: Whether to project refined landmarks onto PDM
        """
        self.search_window = search_window
        self.pdm = pdm
        self.enforce_pdm = enforce_pdm

        # Load only critical patch experts
        loader = SVRPatchExpertLoader(patch_expert_file)
        self.patch_experts = loader.load(target_landmarks=self.CRITICAL_LANDMARKS)

        if self.enforce_pdm and self.pdm is None:
            print("⚠️ Warning: enforce_pdm=True but no PDM provided. PDM constraints disabled.")
            self.enforce_pdm = False

        pdm_status = " + PDM constraints" if self.enforce_pdm else ""
        print(f"Loaded {len(self.patch_experts)} patch experts for refinement{pdm_status}")

    def refine_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Refine critical landmarks using CLNF patch experts.

        Args:
            image: Grayscale image (H, W) as uint8
            landmarks: Initial 68 landmarks as (68, 2) array

        Returns:
            Refined landmarks (68, 2) array
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure float image for processing
        if image.dtype == np.uint8:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)

        # Copy landmarks for refinement
        refined = landmarks.copy()

        # Refine each critical landmark
        for landmark_idx in self.CRITICAL_LANDMARKS:
            if landmark_idx not in self.patch_experts:
                continue

            expert = self.patch_experts[landmark_idx]
            initial_pos = refined[landmark_idx]

            # Search for best position
            refined_pos = self._search_for_best_position(
                image_float,
                initial_pos,
                expert
            )

            refined[landmark_idx] = refined_pos

        # Optional: Enforce PDM constraints to ensure anatomically plausible shape
        if self.enforce_pdm and self.pdm is not None:
            refined = self._project_to_pdm(refined)

        return refined

    def _project_to_pdm(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Project refined landmarks onto PDM to enforce shape constraints.

        This ensures refined landmarks remain anatomically plausible by projecting
        them onto the learned PCA shape space and constraining shape parameters
        to stay within 3 standard deviations (similar to OpenFace 2.2 C++ CLNF).

        Args:
            landmarks: Refined landmarks (68, 2)

        Returns:
            PDM-constrained landmarks (68, 2)
        """
        # Project landmarks onto PDM shape space with 3 std dev constraint
        # This regularizes landmarks to stay within plausible facial shapes
        constrained = self.pdm.project_to_shape_space(landmarks, n_std_devs=3.0)

        return constrained

    def _search_for_best_position(
        self,
        image: np.ndarray,
        initial_pos: np.ndarray,
        expert: SVRPatchExpert
    ) -> np.ndarray:
        """
        Search for optimal landmark position using patch expert.

        Args:
            image: Grayscale image (normalized to 0-1)
            initial_pos: Initial landmark position (x, y)
            expert: SVR patch expert for this landmark

        Returns:
            Refined position (x, y)
        """
        best_response = -float('inf')
        best_pos = initial_pos.copy()

        # Search in window around initial position
        for dx in range(-self.search_window, self.search_window + 1):
            for dy in range(-self.search_window, self.search_window + 1):
                candidate = initial_pos + np.array([dx, dy], dtype=np.float32)

                # Extract patch at candidate position
                patch = self._extract_patch(image, candidate, expert)

                if patch is None:
                    continue

                # Compute patch expert response
                response = self._compute_response(patch, expert)

                if response > best_response:
                    best_response = response
                    best_pos = candidate

        return best_pos

    def _extract_patch(
        self,
        image: np.ndarray,
        position: np.ndarray,
        expert: SVRPatchExpert
    ) -> Optional[np.ndarray]:
        """
        Extract patch around landmark position.

        Args:
            image: Grayscale image (normalized to 0-1)
            position: Landmark position (x, y)
            expert: Patch expert (determines patch type)

        Returns:
            Feature vector for patch, or None if out of bounds
        """
        # Patch experts use 11x11 patches
        patch_size = expert.weights.shape[0]  # Should be 11
        half_size = patch_size // 2

        # Get integer position
        x, y = int(round(position[0])), int(round(position[1]))

        # Check bounds
        if (x - half_size < 0 or x + half_size >= image.shape[1] or
            y - half_size < 0 or y + half_size >= image.shape[0]):
            return None

        # Extract patch
        patch = image[y - half_size:y + half_size + 1,
                     x - half_size:x + half_size + 1]

        # Convert to features based on patch expert type
        if expert.type == 0:
            # Raw pixel features
            features = patch.flatten()
        else:
            # Gradient features (type == 1)
            # Compute image gradients
            grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

            # Concatenate gradient features
            features = np.concatenate([grad_x.flatten(), grad_y.flatten()])

        return features

    def _compute_response(
        self,
        patch_features: np.ndarray,
        expert: SVRPatchExpert
    ) -> float:
        """
        Compute patch expert response using SVR.

        This is identical to AU prediction:
        response = features @ weights + bias

        Args:
            patch_features: Extracted patch features
            expert: SVR patch expert with weights

        Returns:
            Response score (higher = better landmark position)
        """
        # Flatten weights for dot product
        weights_flat = expert.weights.flatten()

        # Ensure feature dimensions match
        if len(patch_features) != len(weights_flat):
            # If patch expert expects raw pixels but we have gradients (or vice versa)
            # This shouldn't happen if patch extraction is correct
            return -float('inf')

        # SVR prediction: features @ weights + bias
        response = np.dot(patch_features, weights_flat) + expert.bias

        # Apply logistic scaling (optional, makes response 0-1)
        # OpenFace uses: 1.0 / (1.0 + exp(-scaling * response))
        response = 1.0 / (1.0 + np.exp(-expert.scaling * response))

        return response


def test_refiner():
    """Test the targeted CLNF refiner on a sample image"""
    import os
    import sys

    # Add parent directory to path to enable absolute imports
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from pyfaceau.detectors.pfld import CunjianPFLDDetector
    from pyfaceau.detectors.retinaface import OptimizedFaceDetector

    print("="*80)
    print("TESTING TARGETED CLNF REFINER")
    print("="*80)

    # Initialize detectors
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(current_dir, '../../weights')
    retinaface_model = os.path.join(weights_dir, 'retinaface_mobilenet025_coreml.onnx')
    pfld_model = os.path.join(weights_dir, 'pfld_cunjian.onnx')
    patch_expert_file = os.path.join(weights_dir, 'svr_patches_0.25_general.txt')

    print("\n1. Loading models...")
    face_detector = OptimizedFaceDetector(retinaface_model)
    landmark_detector = CunjianPFLDDetector(pfld_model)
    refiner = TargetedCLNFRefiner(patch_expert_file, search_window=3)

    # Load test image
    test_video = '/Users/johnwilsoniv/Documents/SplitFace Open3/D Normal Pts/IMG_0434.MOV'
    cap = cv2.VideoCapture(test_video)

    if not cap.isOpened():
        print(f"Could not open video: {test_video}")
        return

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read frame from video")
        return

    print(f"   Frame shape: {frame.shape}")

    # Detect face
    print("\n2. Detecting face...")
    detections, img_raw = face_detector.detect_faces(frame)

    if len(detections) == 0:
        print("   No faces detected")
        return

    # Extract bbox coordinates (first 4 values: x_min, y_min, x_max, y_max)
    first_face = detections[0]
    bbox = first_face[:4]  # x_min, y_min, x_max, y_max
    print(f"   Face bbox: {bbox}")

    # Detect landmarks with PFLD
    print("\n3. Detecting initial landmarks with PFLD...")
    pfld_landmarks, confidence = landmark_detector.detect_landmarks(frame, bbox)
    print(f"   PFLD landmarks shape: {pfld_landmarks.shape}")
    print(f"   Sample landmarks (17-22):")
    for i in range(17, 23):
        print(f"     {i}: ({pfld_landmarks[i, 0]:.1f}, {pfld_landmarks[i, 1]:.1f})")

    # Refine landmarks with CLNF
    print("\n4. Refining landmarks with CLNF...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    refined_landmarks = refiner.refine_landmarks(gray, pfld_landmarks)

    print(f"   Refined landmarks shape: {refined_landmarks.shape}")
    print(f"   Sample refined landmarks (17-22):")
    for i in range(17, 23):
        delta_x = refined_landmarks[i, 0] - pfld_landmarks[i, 0]
        delta_y = refined_landmarks[i, 1] - pfld_landmarks[i, 1]
        print(f"     {i}: ({refined_landmarks[i, 0]:.1f}, {refined_landmarks[i, 1]:.1f}) "
              f"[Δx={delta_x:+.1f}, Δy={delta_y:+.1f}]")

    # Compute displacement statistics
    print("\n5. Refinement statistics:")
    displacements = []
    for idx in refiner.CRITICAL_LANDMARKS:
        delta = refined_landmarks[idx] - pfld_landmarks[idx]
        displacement = np.linalg.norm(delta)
        displacements.append(displacement)

    print(f"   Mean displacement: {np.mean(displacements):.2f} pixels")
    print(f"   Max displacement: {np.max(displacements):.2f} pixels")
    print(f"   Min displacement: {np.min(displacements):.2f} pixels")

    print("\nTargeted CLNF refiner test complete!")


if __name__ == '__main__':
    test_refiner()
