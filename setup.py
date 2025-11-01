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
    version="1.0.0",
    author="John Wilson",
    author_email="",  # Add email if desired
    description="Pure Python OpenFace 2.2 AU extraction with CLNF landmark refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourname/pyfaceau",  # Update with actual repository
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "onnxruntime>=1.10.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "accel": [
            "onnxruntime-coreml>=1.10.0",  # macOS CoreML acceleration
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
