# PyMTCNN detector (required for face detection)
try:
    from .pymtcnn_detector import PyMTCNNDetector, create_pymtcnn_detector, PYMTCNN_AVAILABLE
except ImportError:
    PYMTCNN_AVAILABLE = False
    PyMTCNNDetector = None
    create_pymtcnn_detector = None

# PFLD landmark detector
try:
    from .pfld import CunjianPFLDDetector
except ImportError:
    CunjianPFLDDetector = None

__all__ = [
    'PyMTCNNDetector',
    'create_pymtcnn_detector',
    'PYMTCNN_AVAILABLE',
    'CunjianPFLDDetector'
]
