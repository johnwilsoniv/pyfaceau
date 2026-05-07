"""
pyfaceau - Pure Python OpenFace 2.2 AU Extraction

A complete Python implementation of OpenFace 2.2's AU extraction pipeline
with high-performance parallel processing support and CLNF landmark refinement.
"""

__version__ = "1.3.13"

# Weight management functions can be imported without heavy dependencies
from .download_weights import (
    download_weights,
    ensure_weights,
    get_weights_dir,
    weights_exist
)


def __getattr__(name):
    """Lazy import of heavy modules to speed up package loading."""
    if name == 'FullPythonAUPipeline':
        from .pipeline import FullPythonAUPipeline
        return FullPythonAUPipeline
    elif name == 'ParallelAUPipeline':
        from .parallel_pipeline import ParallelAUPipeline
        return ParallelAUPipeline
    elif name == 'OpenFaceProcessor':
        from .processor import OpenFaceProcessor
        return OpenFaceProcessor
    elif name == 'process_videos':
        from .processor import process_videos
        return process_videos
    raise AttributeError(f"module 'pyfaceau' has no attribute '{name}'")


__all__ = [
    'FullPythonAUPipeline',
    'ParallelAUPipeline',
    'OpenFaceProcessor',
    'process_videos',
    # Weight management
    'download_weights',
    'ensure_weights',
    'get_weights_dir',
    'weights_exist',
]
