"""
pyfaceau - Pure Python OpenFace 2.2 AU Extraction

A complete Python implementation of OpenFace 2.2's AU extraction pipeline
with high-performance parallel processing support and CLNF landmark refinement.
"""

__version__ = "1.3.9"

from .pipeline import FullPythonAUPipeline
from .parallel_pipeline import ParallelAUPipeline
from .processor import OpenFaceProcessor, process_videos

__all__ = [
    'FullPythonAUPipeline',
    'ParallelAUPipeline',
    'OpenFaceProcessor',
    'process_videos'
]
