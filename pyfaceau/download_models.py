"""
pyfaceau-download-models: download the OpenFace model files pyfaceau needs.

Usage:
    pyfaceau-download-models                   # shows the license summary, asks you to type YES
    pyfaceau-download-models --accept-license  # accepts the license without asking
    python -m pyfaceau.download_models         # same, if the command is not found

The files come from OpenFace's official GitHub repository, are checked against
SHA-256 checksums, and are stored in a cache shared with pyclnf and pymtcnn
(see ``pyfaceau.models``). When pyclnf or pymtcnn provide the same
``ensure_models`` function, their files are prepared too.
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from importlib import metadata
from typing import List, Optional, Tuple

from . import models

# Releases that bundle their own model files and have no ensure_models() yet.
# Newer versions are asked for ensure_models() and used when they provide it.
_BUNDLED_UP_TO = {"pyclnf": (0, 3, 4), "pymtcnn": (1, 1, 5)}

LICENSE_SUMMARY = f"""\
pyfaceau uses model files from OpenFace 2.2.0. They are not part of pyfaceau:
they belong to Carnegie Mellon University and are downloaded from OpenFace's
official GitHub page, then checked.

OpenFace license, in short (the full text is what counts):
  - Use only for academic or non-profit, non-commercial research.
  - Do not share, sell or give others access to these files.
  - Commercial use needs a separate license from Carnegie Mellon University.

Full license: {models.LICENSE_URL}
"""


def _say(text: str = "", end: str = "\n") -> None:
    try:
        print(text, end=end, flush=True)
    except (BrokenPipeError, OSError):
        pass


class _ProgressBar:
    """Single-line progress for a terminal, one line per file otherwise."""

    def __init__(self, label: str):
        self.label = label
        self.tty = sys.stdout.isatty()
        self.last_name = None
        self.last_pct = -1

    def __call__(self, *args, **kwargs) -> None:
        # Tolerant of other packages' callbacks: (done, total, name) is expected.
        try:
            done, total = int(args[0]), int(args[1])
            name = str(args[2]) if len(args) > 2 else ""
        except (IndexError, TypeError, ValueError):
            return
        if total <= 0:
            return
        pct = min(100, int(done * 100 / total))
        if self.tty:
            if pct != self.last_pct:
                bar = "#" * (pct // 4)
                _say(f"\r  {self.label}: [{bar:<25}] {pct:3d}%  "
                     f"{models._human_size(done)} of {models._human_size(total)}", end="")
                if done >= total:
                    _say()
        else:
            if name and name != self.last_name:
                _say(f"  downloading {name.rsplit('/', 1)[-1]}")
            elif done >= total and self.last_pct < 100:
                _say(f"  {self.label}: done ({models._human_size(total)})")
        self.last_name = name or self.last_name
        self.last_pct = pct


def _version_tuple(version: str) -> Tuple[int, ...]:
    m = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(x) for x in m.group(0).split(".")) if m else ()


def _prepare_dependencies(cache_dir: Optional[str]) -> List[str]:
    """Run pyclnf/pymtcnn ensure_models() when the installed versions provide it."""
    lines = []
    for dep, bundled_up_to in _BUNDLED_UP_TO.items():
        try:
            version = metadata.version(dep)
        except metadata.PackageNotFoundError:
            lines.append(f"{dep}: not installed (skipped)")
            continue
        if _version_tuple(version) <= bundled_up_to:
            lines.append(f"{dep} {version}: includes its own model files, nothing to download")
            continue
        try:
            ensure = getattr(importlib.import_module(f"{dep}.models"), "ensure_models")
        except (ImportError, AttributeError):
            lines.append(f"{dep} {version}: has no separate model download (skipped)")
            continue
        _say(f"\nPreparing the model files for {dep} {version}...")
        path = ensure(accept_license=True, cache_dir=cache_dir, progress=_ProgressBar(dep))
        lines.append(f"{dep} {version}: ready in {path}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=models.DOWNLOAD_COMMAND,
        description="Download the OpenFace 2.2.0 model files used by pyfaceau "
                    "(and by pyclnf and pymtcnn, when they support it).")
    parser.add_argument("--accept-license", action="store_true",
                        help="accept the OpenFace license without being asked")
    parser.add_argument("--cache-dir", metavar="FOLDER",
                        help=f"use this folder instead of the default (same as {models.ENV_MODELS_DIR})")
    args = parser.parse_args(argv)

    root = models.models_root(args.cache_dir)
    _say("OpenFace model files for pyfaceau")
    _say("=================================")
    _say(LICENSE_SUMMARY)

    accepted = models.license_accepted(args.accept_license)
    if accepted:
        _say("License accepted (--accept-license or OPENFACE_MODELS_ACCEPT_LICENSE=1).\n")
    else:
        try:
            answer = input("Type YES to accept the OpenFace license and download the files: ")
        except EOFError:
            answer = ""
        except KeyboardInterrupt:
            _say("\nCancelled. Nothing was downloaded.")
            return 130
        if answer.strip().upper() != "YES":
            _say("\nNothing was downloaded, because the license was not accepted.")
            _say(f"To accept it without being asked, run:  {models.DOWNLOAD_COMMAND} --accept-license")
            return 1
        _say()

    _say(f"Saving to: {root}")
    try:
        path = models.ensure_models(accept_license=True, cache_dir=args.cache_dir,
                                    progress=_ProgressBar("pyfaceau"))
        dependency_lines = _prepare_dependencies(args.cache_dir)
    except KeyboardInterrupt:
        _say("\nCancelled. Run the command again to finish; files already checked are kept.")
        return 130
    except models.ModelDownloadError as e:
        _say(f"\nThe download did not finish.\n{e}")
        return 1
    except Exception as e:  # a dependency's ensure_models failed in its own way
        _say(f"\nSomething went wrong: {e}")
        return 1

    _say("\nDone. The OpenFace model files are installed and checked.")
    _say(f"  pyfaceau: ready in {path}")
    for line in dependency_lines:
        _say(f"  {line}")
    _say("\nYou can now use pyfaceau.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
