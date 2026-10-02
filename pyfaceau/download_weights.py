#!/usr/bin/env python3
"""
Compatibility layer for code written for pyfaceau 1.3 and earlier.

pyfaceau no longer downloads a weights zip from its own GitHub release. The
OpenFace model files are downloaded from OpenFace's official repository after
you accept the OpenFace license; see ``pyfaceau.models`` and the
``pyfaceau-download-models`` command. The functions below keep their old
names and return values but use the new shared model folder.
"""

import os
import shutil
import sys
from pathlib import Path

from . import models
from .models import ModelsNotInstalledError, ensure_models

__all__ = ["get_weights_dir", "weights_exist", "ensure_weights", "download_weights",
           "ModelsNotInstalledError", "ensure_models"]

# Files that must exist in a weights folder (old and new layout are the same).
REQUIRED_FILES = [
    models.PDM_FILE,
    models.TRIANGULATION_FILE,
    models.AU_MODELS_DIR + "/svr_combined/AU_1_dynamic_intensity_comb.dat",
]


def _legacy_env_dir():
    """PYFACEAU_WEIGHTS_DIR, if set and complete (kept from pyfaceau 1.3)."""
    env_dir = os.environ.get("PYFACEAU_WEIGHTS_DIR", "").strip()
    if env_dir and weights_exist(env_dir):
        return Path(env_dir)
    return None


def get_weights_dir():
    """The weights folder pyfaceau will use (it may not exist yet).

    PYFACEAU_WEIGHTS_DIR wins if it points to a complete folder; otherwise the
    shared OpenFace model folder (``pyfaceau.models.models_dir()``).
    """
    return _legacy_env_dir() or models.models_dir()


def weights_exist(weights_dir=None):
    """True if the required model files are present in ``weights_dir``."""
    if weights_dir is None:
        legacy = _legacy_env_dir()
        if legacy is not None:
            return True
        return models.models_ready()
    weights_dir = Path(weights_dir)
    return all((weights_dir / f).exists() for f in REQUIRED_FILES)


def ensure_weights(auto_download=True, verbose=True):
    """Return the weights folder, preparing it from already-downloaded files if needed.

    ``auto_download`` is kept for compatibility but no longer downloads on its
    own: downloading requires accepting the OpenFace license once, with the
    ``pyfaceau-download-models`` command or OPENFACE_MODELS_ACCEPT_LICENSE=1.

    Raises:
        ModelsNotInstalledError (a FileNotFoundError): the files are not installed.
    """
    legacy = _legacy_env_dir()
    if legacy is not None:
        return legacy
    return ensure_models()


def download_weights(weights_dir=None, verbose=True):
    """Run the interactive download (asks you to accept the OpenFace license).

    Returns 0 on success and 1 on failure, as before. If ``weights_dir`` is
    given, the ready files are also copied there.
    """
    from .download_models import main as download_main
    result = download_main([])
    if result == 0 and weights_dir is not None:
        shutil.copytree(models.models_dir(), Path(weights_dir), dirs_exist_ok=True)
        if verbose:
            print(f"Copied the model files to: {weights_dir}")
    return result


def main():
    """The old pyfaceau-download-weights command; same as pyfaceau-download-models."""
    from .download_models import main as download_main
    return download_main()


if __name__ == "__main__":
    sys.exit(main())
