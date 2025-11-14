"""
Python implementation of Constrained Local Neural Fields (CLNF).

This package provides a pure Python implementation of the CLNF landmark
detection algorithm for handling challenging cases like surgical markings
and severe facial paralysis.

Components:
- cen_patch_experts: CEN patch expert loader and inference
- pdm: Point Distribution Model for shape-constrained fitting
- nu_rlms: Non-Uniform Regularized Landmark Mean Shift optimization
- clnf_detector: Main CLNF detector integrating all components
"""

from .cen_patch_experts import CENPatchExperts
from .pdm import PointDistributionModel
from .nu_rlms import NURLMSOptimizer
from .clnf_detector import CLNFDetector

__all__ = ['CENPatchExperts', 'PointDistributionModel', 'NURLMSOptimizer', 'CLNFDetector']
