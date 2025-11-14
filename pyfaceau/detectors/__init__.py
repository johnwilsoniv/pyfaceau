from .pfld import CunjianPFLDDetector

# PyMTCNN detector (required for face detection)
try:
    from .pymtcnn_detector import PyMTCNNDetector, create_pymtcnn_detector, PYMTCNN_AVAILABLE
except ImportError:
    PYMTCNN_AVAILABLE = False
    PyMTCNNDetector = None
    create_pymtcnn_detector = None

__all__ = [
    'CunjianPFLDDetector',
    'PyMTCNNDetector',
    'create_pymtcnn_detector',
    'PYMTCNN_AVAILABLE'
]
