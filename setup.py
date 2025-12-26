"""
setup.py for pyfaceau - Pure Python OpenFace 2.2 AU Extraction
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pyfaceau",
    version="1.3.6",
    author="John Wilson",
    author_email="",  # Add email if desired
    description="Pure Python OpenFace 2.2 AU extraction with PyMTCNN face detection and CLNF refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnwilsoniv/face-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/johnwilsoniv/face-analysis/issues",
        "Documentation": "https://github.com/johnwilsoniv/face-analysis/tree/main/S0%20PyfaceAU",
        "Source Code": "https://github.com/johnwilsoniv/face-analysis",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: Other/Proprietary License",  # CC BY-NC 4.0 (dual licensing)
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "onnxruntime>=1.10.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pyfhog>=0.1.0",
        "pyclnf>=0.2.0",
        "pymtcnn>=1.1.0",  # Cross-platform face detection
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        # Face detection backends
        "cuda": [
            "pymtcnn[onnx-gpu]>=1.1.0",  # NVIDIA GPU acceleration
        ],
        "coreml": [
            "pymtcnn[coreml]>=1.1.0",  # Apple Silicon acceleration
            "coremltools>=7.0",
        ],
        "cpu": [
            "pymtcnn[onnx]>=1.1.0",  # CPU-only face detection
        ],
        # All acceleration options
        "all": [
            "pymtcnn[all]>=1.1.0",
            "coremltools>=7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pyfaceau=pyfaceau.processor:main",
            "pyfaceau-gui=pyfaceau_gui:main",
        ],
    },
    scripts=['pyfaceau_gui.py'],
    include_package_data=True,
    package_data={
        "pyfaceau": ["*.txt", "*.json"],
    },
    zip_safe=False,
)
