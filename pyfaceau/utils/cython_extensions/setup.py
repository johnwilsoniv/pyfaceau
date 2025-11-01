#!/usr/bin/env python3
"""
Setup script for Cython extensions

Compiles optimized Cython modules for:
1. cython_histogram_median - Fast histogram median tracker (260x speedup)
2. cython_rotation_update - Accurate rotation update for CalcParams (99.9% accuracy)

Usage:
    python3.10 setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cython_histogram_median",
        ["cython_histogram_median.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
        extra_link_args=[]
    ),
    Extension(
        "cython_rotation_update",
        ["cython_rotation_update.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3'],
        extra_link_args=[]
    ),
]

setup(
    name="pyfaceau_cython_extensions",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'initializedcheck': False,
        }
    ),
    include_dirs=[np.get_include()],
)
