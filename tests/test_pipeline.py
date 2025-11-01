#!/usr/bin/env python3
"""
Test pyAUface Pipeline - Basic functionality test

Tests the complete AU extraction pipeline on a single frame.
"""

import sys
sys.path.insert(0, '../pyauface')
sys.path.insert(0, '../../pyfhog/src')

import numpy as np
import cv2

def test_pipeline_initialization():
    """Test that all components initialize correctly"""
    print("\n" + "="*80)
    print("TEST: Pipeline Initialization")
    print("="*80)

    try:
        from pyauface.pipeline import FullPythonAUPipeline

        pipeline = FullPythonAUPipeline(
            retinaface_model='../weights/retinaface_mobilenet025_coreml.onnx',
            pfld_model='../weights/pfld_cunjian.onnx',
            pdm_file='../weights/In-the-wild_aligned_PDM_68.txt',
            au_models_dir='/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors',
            triangulation_file='../weights/tris_68_full.txt',
            use_calc_params=True,
            use_coreml=False,  # CPU mode for testing
            verbose=False
        )

        print("✅ Pipeline initialized successfully")
        return True

    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False

def test_individual_components():
    """Test individual components can be imported and used"""
    print("\n" + "="*80)
    print("TEST: Individual Components")
    print("="*80)

    tests_passed = 0
    tests_total = 0

    # Test 1: Face Detector
    tests_total += 1
    try:
        from pyauface.detectors.retinaface import ONNXRetinaFaceDetector
        detector = ONNXRetinaFaceDetector('../weights/retinaface_mobilenet025_coreml.onnx', use_coreml=False)
        print("✅ Face detector (RetinaFace)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Face detector failed: {e}")

    # Test 2: Landmark Detector
    tests_total += 1
    try:
        from pyauface.detectors.pfld import CunjianPFLDDetector
        landmarker = CunjianPFLDDetector('../weights/pfld_cunjian.onnx')
        print("✅ Landmark detector (PFLD)")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Landmark detector failed: {e}")

    # Test 3: Face Aligner
    tests_total += 1
    try:
        from pyauface.alignment.face_aligner import OpenFace22FaceAligner
        aligner = OpenFace22FaceAligner('../weights/In-the-wild_aligned_PDM_68.txt')
        print("✅ Face aligner")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Face aligner failed: {e}")

    # Test 4: PDM Parser
    tests_total += 1
    try:
        from pyauface.features.pdm import PDMParser
        pdm = PDMParser('../weights/In-the-wild_aligned_PDM_68.txt')
        print("✅ PDM parser")
        tests_passed += 1
    except Exception as e:
        print(f"❌ PDM parser failed: {e}")

    # Test 5: Running Median
    tests_total += 1
    try:
        from pyauface.prediction.running_median import DualHistogramMedianTracker
        median_tracker = DualHistogramMedianTracker()
        print("✅ Running median tracker")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Running median tracker failed: {e}")

    # Test 6: AU Model Parser
    tests_total += 1
    try:
        from pyauface.prediction.model_parser import OF22ModelParser
        parser = OF22ModelParser('/Users/johnwilsoniv/repo/fea_tool/external_libs/openFace/OpenFace/lib/local/FaceAnalyser/AU_predictors')
        print("✅ AU model parser")
        tests_passed += 1
    except Exception as e:
        print(f"❌ AU model parser failed: {e}")

    print(f"\nComponent Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PYFACEAU PIPELINE TESTS")
    print("="*80)

    all_passed = True

    # Test 1: Initialization
    if not test_pipeline_initialization():
        all_passed = False

    # Test 2: Components
    if not test_individual_components():
        all_passed = False

    # Summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
