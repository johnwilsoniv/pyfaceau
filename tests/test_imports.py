"""Importing pyfaceau must not put its own package folder on sys.path.

If it does, pyfaceau's modules (config, models, ...) shadow top-level imports
of other packages; pyclnf, for example, imports "models.openface_loader".
"""

import sys
from pathlib import Path

import pytest


def test_calc_params_import_leaves_sys_path_alone():
    pytest.importorskip("scipy")
    pytest.importorskip("cv2")
    import pyfaceau
    package_dir = str(Path(pyfaceau.__file__).parent)
    import pyfaceau.alignment.calc_params  # noqa: F401
    assert package_dir not in sys.path
