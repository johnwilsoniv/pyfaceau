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
import zipfile
import tempfile
from pathlib import Path

# Try to import tqdm, fall back to simple progress if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# GitHub release URL for weights zip (~1.6MB compressed)
WEIGHTS_ZIP_URL = "https://github.com/johnwilsoniv/pyfaceau/releases/download/weights-v1.0/pyfaceau-weights-v1.0.zip"

# Required files to verify successful download
REQUIRED_FILES = [
    "In-the-wild_aligned_PDM_68.txt",
    "svr_patches_0.25_general.txt",
    "tris_68_full.txt",
    "AU_predictors/AU_all_best.txt",
    "AU_predictors/svr_combined/AU_1_dynamic_intensity_comb.dat",
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
    if TQDM_AVAILABLE:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print(f"  Downloading {desc or url}...")
        urllib.request.urlretrieve(url, filename=output_path)


def get_user_weights_dir():
    """
    Get user-writable weights directory.

    Priority:
    1. PYFACEAU_WEIGHTS_DIR environment variable
    2. ~/.pyfaceau/weights/
    """
    env_dir = os.environ.get('PYFACEAU_WEIGHTS_DIR')
    if env_dir:
        weights_dir = Path(env_dir)
    else:
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
        if weights_exist(Path(env_dir)):
            return Path(env_dir)

    # Check sibling weights directory (for development/editable installs)
    try:
        pkg_dir = Path(__file__).parent.parent  # pyfaceau package parent
        dev_weights = pkg_dir / "weights"
        if dev_weights.exists() and weights_exist(dev_weights):
            return dev_weights
    except Exception:
        pass

    # Check user home directory
    user_weights = Path.home() / ".pyfaceau" / "weights"
    if user_weights.exists() and weights_exist(user_weights):
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
    au_model = weights_dir / "AU_predictors" / "svr_combined" / "AU_1_dynamic_intensity_comb.dat"
    if not au_model.exists():
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
        print("PyFaceAU weights not found. Downloading (~1.6MB)...")

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

    # Create directory
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Download zip file to temp location
    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        if verbose:
            print(f"Downloading weights from GitHub release...")
            print(f"  URL: {WEIGHTS_ZIP_URL}")

        download_file(WEIGHTS_ZIP_URL, tmp_path, desc="pyfaceau-weights.zip")

        if verbose:
            print(f"\nExtracting weights...")

        # Extract zip
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)

        if verbose:
            print(f"  Extracted to: {weights_dir}")

        # Verify extraction
        missing = []
        for req_file in REQUIRED_FILES:
            if not (weights_dir / req_file).exists():
                missing.append(req_file)

        if missing:
            print(f"\n[WARNING] Some files missing after extraction:")
            for f in missing:
                print(f"  - {f}")
            return 1

        if verbose:
            print("\n" + "=" * 60)
            print("[OK] All weights downloaded successfully!")
            print(f"Weights location: {weights_dir}")
            print("\nYou can now use PyFaceAU:")
            print("  from pyfaceau import OpenFaceProcessor")
            print("  processor = OpenFaceProcessor()")

        return 0

    except urllib.error.URLError as e:
        print(f"\n[ERROR] Failed to download weights: {e}")
        print("\nPlease check your internet connection and try again.")
        print("Or download manually from:")
        print(f"  {WEIGHTS_ZIP_URL}")
        print(f"And extract to: {weights_dir}")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return 1

    finally:
        # Clean up temp file
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception:
            pass


def main():
    """Command-line entry point for weight download"""
    return download_weights(verbose=True)


if __name__ == "__main__":
    sys.exit(main())
