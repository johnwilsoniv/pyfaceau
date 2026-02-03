#!/usr/bin/env python3
"""
Download model weights for PyFaceAU

Usage:
    python -m pyfaceau.download_weights

Or programmatically:
    from pyfaceau.download_weights import ensure_weights
    weights_dir = ensure_weights()
"""

import os
import sys
import urllib.request
from pathlib import Path

# Try to import tqdm, fall back to simple progress if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# GitHub release URL for weights (use releases for large files, not raw repo)
# Create a release at: https://github.com/johnwilsoniv/pyfaceau/releases with weights.zip
WEIGHTS_BASE_URL = "https://github.com/johnwilsoniv/pyfaceau/releases/download/weights-v1.0/"

# Fallback: Raw GitHub URL (for smaller text files)
# Note: Binary .dat files may need LFS or release hosting
WEIGHTS_FALLBACK_URL = "https://raw.githubusercontent.com/johnwilsoniv/pyfaceau/main/weights/"

REQUIRED_WEIGHTS = {
    # Note: Face detection uses PyMTCNN (installed separately)
    # Note: Landmark detection uses CLNF (Constrained Local Neural Fields)
    "In-the-wild_aligned_PDM_68.txt": "67KB - PDM shape model",
    "svr_patches_0.25_general.txt": "1.1MB - CLNF patch experts",
    "tris_68_full.txt": "1KB - Triangulation data",
}

# AU predictor files stored in AU_predictors/svr_combined/ subdirectory
AU_PREDICTOR_FILES = [
    "svr_combined/AU_1_dynamic_intensity_comb.dat",
    "svr_combined/AU_2_dynamic_intensity_comb.dat",
    "svr_combined/AU_4_static_intensity_comb.dat",
    "svr_combined/AU_5_dynamic_intensity_comb.dat",
    "svr_combined/AU_6_static_intensity_comb.dat",
    "svr_combined/AU_7_static_intensity_comb.dat",
    "svr_combined/AU_9_dynamic_intensity_comb.dat",
    "svr_combined/AU_10_static_intensity_comb.dat",
    "svr_combined/AU_12_static_intensity_comb.dat",
    "svr_combined/AU_14_static_intensity_comb.dat",
    "svr_combined/AU_15_dynamic_intensity_comb.dat",
    "svr_combined/AU_17_dynamic_intensity_comb.dat",
    "svr_combined/AU_20_dynamic_intensity_comb.dat",
    "svr_combined/AU_23_dynamic_intensity_comb.dat",
    "svr_combined/AU_25_dynamic_intensity_comb.dat",
    "svr_combined/AU_26_dynamic_intensity_comb.dat",
    "svr_combined/AU_45_dynamic_intensity_comb.dat",
]

# AU predictor config files
AU_CONFIG_FILES = [
    "AU_all_best.txt",
    "main_dynamic_svms.txt",
]


if TQDM_AVAILABLE:
    class DownloadProgressBar(tqdm):
        """Progress bar for downloads"""
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)


def download_file(url, output_path, desc=None):
    """Download a file with progress bar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if TQDM_AVAILABLE:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        # Simple fallback without progress bar
        print(f"  Downloading {desc or url}...")
        urllib.request.urlretrieve(url, filename=output_path)


def get_user_weights_dir():
    """
    Get user-writable weights directory.

    Priority:
    1. PYFACEAU_WEIGHTS_DIR environment variable
    2. ~/.pyfaceau/weights/
    """
    # Check environment variable first
    env_dir = os.environ.get('PYFACEAU_WEIGHTS_DIR')
    if env_dir:
        weights_dir = Path(env_dir)
    else:
        # Default to user home directory
        weights_dir = Path.home() / ".pyfaceau" / "weights"

    weights_dir.mkdir(parents=True, exist_ok=True)
    return weights_dir


def get_weights_dir():
    """
    Get weights directory, checking multiple locations.

    Priority:
    1. PYFACEAU_WEIGHTS_DIR environment variable
    2. Sibling 'weights' directory (for development installs)
    3. User home directory ~/.pyfaceau/weights/

    Returns:
        Path to weights directory (may not exist if weights not downloaded)
    """
    # Check environment variable first
    env_dir = os.environ.get('PYFACEAU_WEIGHTS_DIR')
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    # Check sibling weights directory (for development/editable installs)
    try:
        pkg_dir = Path(__file__).parent.parent  # pyfaceau package parent
        dev_weights = pkg_dir / "weights"
        if dev_weights.exists() and (dev_weights / "In-the-wild_aligned_PDM_68.txt").exists():
            return dev_weights
    except Exception:
        pass

    # Check user home directory
    user_weights = Path.home() / ".pyfaceau" / "weights"
    if user_weights.exists() and (user_weights / "In-the-wild_aligned_PDM_68.txt").exists():
        return user_weights

    # Return user weights dir (will need download)
    return user_weights


def weights_exist(weights_dir=None):
    """Check if required weights exist"""
    if weights_dir is None:
        weights_dir = get_weights_dir()
    weights_dir = Path(weights_dir)

    # Check main PDM file
    pdm_file = weights_dir / "In-the-wild_aligned_PDM_68.txt"
    if not pdm_file.exists():
        return False

    # Check at least one AU model
    au_dir = weights_dir / "AU_predictors" / "svr_combined"
    if not au_dir.exists():
        return False

    return True


def ensure_weights(auto_download=True, verbose=True):
    """
    Ensure weights are available, downloading if necessary.

    Args:
        auto_download: If True, automatically download missing weights
        verbose: If True, print progress messages

    Returns:
        Path to weights directory

    Raises:
        FileNotFoundError: If weights are missing and auto_download is False
    """
    weights_dir = get_weights_dir()

    if weights_exist(weights_dir):
        return weights_dir

    if not auto_download:
        raise FileNotFoundError(
            f"PyFaceAU weights not found.\n"
            f"Expected location: {weights_dir}\n"
            f"\n"
            f"To download weights, run:\n"
            f"  python -m pyfaceau.download_weights\n"
            f"\n"
            f"Or set PYFACEAU_WEIGHTS_DIR environment variable to your weights directory."
        )

    if verbose:
        print("PyFaceAU weights not found. Downloading...")

    # Download to user weights directory
    download_dir = get_user_weights_dir()
    result = download_weights(download_dir, verbose=verbose)

    if result != 0:
        raise RuntimeError("Failed to download PyFaceAU weights")

    return download_dir


def download_weights(weights_dir=None, verbose=True):
    """
    Download all required weights.

    Args:
        weights_dir: Directory to download weights to (default: ~/.pyfaceau/weights/)
        verbose: If True, print progress messages

    Returns:
        0 on success, 1 on failure
    """
    if weights_dir is None:
        weights_dir = get_user_weights_dir()
    weights_dir = Path(weights_dir)

    if verbose:
        print("PyFaceAU Weight Downloader")
        print("=" * 60)
        print(f"Downloading weights to: {weights_dir}")
        print()

    # Create directory structure
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "AU_predictors" / "svr_combined").mkdir(parents=True, exist_ok=True)

    # Try primary URL first, fall back to secondary
    base_urls = [WEIGHTS_BASE_URL, WEIGHTS_FALLBACK_URL]

    # Download main weights
    if verbose:
        print("Downloading main model weights...")

    for filename, description in REQUIRED_WEIGHTS.items():
        output_path = weights_dir / filename

        if output_path.exists():
            if verbose:
                print(f"[OK] {filename} (already exists)")
            continue

        downloaded = False
        for base_url in base_urls:
            url = base_url + filename
            try:
                download_file(url, str(output_path), desc=f"{filename} ({description})")
                if verbose:
                    print(f"[OK] Downloaded {filename}")
                downloaded = True
                break
            except Exception as e:
                if verbose:
                    print(f"  (trying fallback URL...)")
                continue

        if not downloaded:
            print(f"[FAILED] Failed to download {filename}")
            print(f"  Please download manually from GitHub and place in: {output_path}")
            return 1

    # Download AU predictor config files
    if verbose:
        print("\nDownloading AU predictor config files...")

    au_dir = weights_dir / "AU_predictors"
    for filename in AU_CONFIG_FILES:
        output_path = au_dir / filename

        if output_path.exists():
            if verbose:
                print(f"[OK] {filename} (already exists)")
            continue

        downloaded = False
        for base_url in base_urls:
            url = base_url + "AU_predictors/" + filename
            try:
                download_file(url, str(output_path), desc=filename)
                if verbose:
                    print(f"[OK] Downloaded {filename}")
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print(f"[FAILED] Failed to download {filename}")
            return 1

    # Download AU predictor models
    if verbose:
        print("\nDownloading AU predictor models (SVR weights)...")

    for filename in AU_PREDICTOR_FILES:
        output_path = au_dir / filename

        if output_path.exists():
            if verbose:
                print(f"[OK] {filename} (already exists)")
            continue

        downloaded = False
        for base_url in base_urls:
            url = base_url + "AU_predictors/" + filename
            try:
                download_file(url, str(output_path), desc=filename.split('/')[-1])
                if verbose:
                    print(f"[OK] Downloaded {filename}")
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print(f"[FAILED] Failed to download {filename}")
            print(f"  Please download manually from GitHub and place in: {output_path}")
            return 1

    if verbose:
        print("\n" + "=" * 60)
        print("[OK] All weights downloaded successfully!")
        print(f"Weights location: {weights_dir}")
        print("\nYou can now use PyFaceAU:")
        print("  from pyfaceau import OpenFaceProcessor")
        print("  processor = OpenFaceProcessor()")

    return 0


def main():
    """Command-line entry point for weight download"""
    return download_weights(verbose=True)


if __name__ == "__main__":
    sys.exit(main())
