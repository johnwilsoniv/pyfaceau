#!/usr/bin/env python3
"""
Download model weights for PyFaceAU

Usage:
    python -m pyfaceau.download_weights
"""

import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


WEIGHTS_BASE_URL = "https://github.com/johnwilsoniv/face-analysis/raw/main/S0%20PyfaceAU/weights/"

REQUIRED_WEIGHTS = {
    # Note: Face detection uses PyMTCNN (installed separately)
    # Note: Landmark detection uses CLNF (Constrained Local Neural Fields)
    "In-the-wild_aligned_PDM_68.txt": "67KB - PDM shape model",
    "svr_patches_0.25_general.txt": "1.1MB - CLNF patch experts",
    "tris_68_full.txt": "1KB - Triangulation data",
}

AU_PREDICTOR_FILES = [
    "AU_1_dynamic_intensity_comb.dat",
    "AU_2_dynamic_intensity_comb.dat",
    "AU_4_static_intensity_comb.dat",
    "AU_5_dynamic_intensity_comb.dat",
    "AU_6_static_intensity_comb.dat",
    "AU_7_static_intensity_comb.dat",
    "AU_9_dynamic_intensity_comb.dat",
    "AU_10_static_intensity_comb.dat",
    "AU_12_static_intensity_comb.dat",
    "AU_14_static_intensity_comb.dat",
    "AU_15_dynamic_intensity_comb.dat",
    "AU_17_dynamic_intensity_comb.dat",
    "AU_20_dynamic_intensity_comb.dat",
    "AU_23_dynamic_intensity_comb.dat",
    "AU_25_dynamic_intensity_comb.dat",
    "AU_26_dynamic_intensity_comb.dat",
    "AU_45_dynamic_intensity_comb.dat",
]


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_weights_dir():
    """Get or create weights directory"""
    # First try package installation location
    try:
        import pyfaceau
        pkg_dir = Path(pyfaceau.__file__).parent
        weights_dir = pkg_dir / "weights"
    except:
        # Fall back to current directory
        weights_dir = Path.cwd() / "weights"

    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def main():
    """Download all required weights"""
    print("PyFaceAU Weight Downloader")
    print("=" * 60)

    weights_dir = get_weights_dir()
    print(f"Downloading weights to: {weights_dir}")
    print()

    # Download main weights
    print("Downloading main model weights...")
    for filename, description in REQUIRED_WEIGHTS.items():
        output_path = weights_dir / filename

        if output_path.exists():
            print(f"✓ {filename} (already exists)")
            continue

        url = WEIGHTS_BASE_URL + filename
        try:
            download_file(url, output_path, desc=f"{filename} ({description})")
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return 1

    # Download AU predictors
    print("\nDownloading AU predictor models...")
    au_dir = weights_dir / "AU_predictors"
    au_dir.mkdir(exist_ok=True)

    for filename in AU_PREDICTOR_FILES:
        output_path = au_dir / filename

        if output_path.exists():
            print(f"✓ {filename} (already exists)")
            continue

        url = WEIGHTS_BASE_URL + "AU_predictors/" + filename
        try:
            download_file(url, output_path, desc=filename)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            return 1

    print("\n" + "=" * 60)
    print("✓ All weights downloaded successfully!")
    print(f"Weights location: {weights_dir}")
    print("\nYou can now use PyFaceAU:")
    print("  from pyfaceau import FullPythonAUPipeline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
