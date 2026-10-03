"""
OpenFace model files for pyfaceau.

pyfaceau does not include any OpenFace model files. OpenFace's files are
licensed by Carnegie Mellon University for academic or non-profit,
non-commercial research only, and may not be redistributed. pyfaceau therefore
downloads them, only after you accept the OpenFace license, from OpenFace's
official GitHub repository (tag OpenFace_2.2.0). Every file is checked against
the SHA-256 checksum listed in ``openface_models.json``.

The files are kept in a cache shared with pyclnf and pymtcnn::

    macOS    ~/Library/Application Support/OpenFaceModels/2.2.0/
    Windows  %LOCALAPPDATA%\\OpenFaceModels\\2.2.0\\
    Linux    $XDG_DATA_HOME/OpenFaceModels/2.2.0/   (default ~/.local/share)

Set the environment variable ``OPENFACE_MODELS_DIR`` to use another folder in
place of ``<user data dir>/OpenFaceModels`` (the ``2.2.0`` sub-folder is always
added). Inside the version folder:

    originals/<path in OpenFace's repository>   the downloaded files, unchanged
    derived/pyfaceau/<converter version>/       the folder pyfaceau loads from
    .lock                                       held while files are written

Typical use::

    from pyfaceau.models import ensure_models
    weights_dir = ensure_models()          # raises ModelsNotInstalledError if missing
    weights_dir = ensure_models(accept_license=True)   # downloads what is missing

Setting ``OPENFACE_MODELS_ACCEPT_LICENSE=1`` counts as accepting the license.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import shutil
import ssl
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

__all__ = [
    "ensure_models",
    "ModelsNotInstalledError",
    "ModelDownloadError",
    "models_root",
    "models_dir",
    "models_ready",
    "model_paths",
    "license_accepted",
    "load_manifest",
    "user_data_dir",
    "OPENFACE_VERSION",
    "LICENSE_URL",
]

OPENFACE_VERSION = "2.2.0"
LICENSE_URL = "https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt"
ENV_MODELS_DIR = "OPENFACE_MODELS_DIR"
ENV_ACCEPT_LICENSE = "OPENFACE_MODELS_ACCEPT_LICENSE"
PACKAGE_NAME = "pyfaceau"
# Bump when the layout of derived/pyfaceau/<version>/ changes.
CONVERTER_VERSION = "1"
MANIFEST_FILE = "openface_models.json"
STAMP_FILE = "pyfaceau-models.json"

DOWNLOAD_COMMAND = "pyfaceau-download-models"
DOWNLOAD_MODULE = "pyfaceau.download_models"

# Files inside the ready folder that the AU pipeline opens.
PDM_FILE = "In-the-wild_aligned_PDM_68.txt"
TRIANGULATION_FILE = "tris_68_full.txt"
PATCH_EXPERT_FILE = "svr_patches_0.25_general.txt"
AU_MODELS_DIR = "AU_predictors"

# Downloads are only accepted from OpenFace's official repository at this tag.
_OFFICIAL_URL_PREFIX = "https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/OpenFace_2.2.0/"
_RETRIES = 4            # attempts per URL
_TIMEOUT_S = 30         # socket timeout per request
_LOCK_TIMEOUT_S = 15 * 60
_CHUNK = 1 << 16

ProgressCallback = Callable[[int, int, str], None]
"""progress(done_bytes, total_bytes, name): bytes downloaded so far in this
call, total bytes this call will download, and the OpenFace path of the file
being downloaded (empty string at the start and the end)."""

_thread_lock = threading.Lock()


class ModelsNotInstalledError(FileNotFoundError):
    """The OpenFace model files are not installed and the license was not accepted.

    Subclass of FileNotFoundError so code written for older pyfaceau versions,
    which raised FileNotFoundError for missing weights, keeps working.
    """


class ModelDownloadError(RuntimeError):
    """The model files could not be downloaded, checked or saved."""


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

def user_data_dir() -> Path:
    """The per-user data folder of this operating system."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) if base else Path.home() / "AppData" / "Local"
    xdg = os.environ.get("XDG_DATA_HOME", "")
    if xdg and os.path.isabs(xdg):
        return Path(xdg)
    return Path.home() / ".local" / "share"


def models_root(cache_dir: Optional[os.PathLike] = None) -> Path:
    """The shared cache folder for this OpenFace version.

    ``cache_dir`` (or else ``OPENFACE_MODELS_DIR``) replaces
    ``<user data dir>/OpenFaceModels``; ``2.2.0`` is always appended.
    """
    if cache_dir is not None:
        base = Path(cache_dir).expanduser()
    elif os.environ.get(ENV_MODELS_DIR, "").strip():
        base = Path(os.environ[ENV_MODELS_DIR].strip()).expanduser()
    else:
        base = user_data_dir() / "OpenFaceModels"
    return base / OPENFACE_VERSION


def models_dir(cache_dir: Optional[os.PathLike] = None) -> Path:
    """The folder pyfaceau loads its model files from (it may not exist yet)."""
    return models_root(cache_dir) / "derived" / PACKAGE_NAME / CONVERTER_VERSION


def model_paths(weights_dir: os.PathLike) -> Dict[str, str]:
    """The model file paths the AU pipeline expects inside a weights folder."""
    weights_dir = Path(weights_dir)
    return {
        "pdm_file": str(weights_dir / PDM_FILE),
        "au_models_dir": str(weights_dir / AU_MODELS_DIR),
        "triangulation_file": str(weights_dir / TRIANGULATION_FILE),
        "patch_expert_file": str(weights_dir / PATCH_EXPERT_FILE),
    }


# --------------------------------------------------------------------------
# Manifest and license
# --------------------------------------------------------------------------

def load_manifest() -> dict:
    """The list of OpenFace files pyfaceau needs (path, URLs, SHA-256, size)."""
    with open(Path(__file__).with_name(MANIFEST_FILE), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    _check_manifest(manifest)
    return manifest


def _check_manifest(manifest: dict) -> None:
    """Reject unsafe paths and any URL outside OpenFace's official repository."""
    for entry in manifest["files"]:
        for key in ("path", "layout_path"):
            parts = entry[key].split("/")
            if entry[key].startswith("/") or ".." in parts or "" in parts:
                raise ValueError(f"unsafe path in {MANIFEST_FILE}: {entry[key]!r}")
        for url in entry["urls"]:
            if url != _OFFICIAL_URL_PREFIX + entry["path"]:
                raise ValueError(f"unexpected URL in {MANIFEST_FILE}: {url!r}")


def license_accepted(accept_license: bool = False) -> bool:
    """True if the caller accepted the license or OPENFACE_MODELS_ACCEPT_LICENSE=1."""
    return bool(accept_license) or os.environ.get(ENV_ACCEPT_LICENSE, "").strip() == "1"


def _manifest_key(manifest: dict) -> str:
    rows = sorted((e["path"], e["sha256"], e["size"], e["layout_path"]) for e in manifest["files"])
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def _original_path(root: Path, entry: dict) -> Path:
    return root.joinpath("originals", *entry["path"].split("/"))


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _original_ok(root: Path, entry: dict) -> bool:
    path = _original_path(root, entry)
    try:
        if path.stat().st_size != entry["size"]:
            return False
        return _sha256_of(path) == entry["sha256"]
    except OSError:
        return False


def _derived_ok(ready: Path, manifest: dict) -> bool:
    try:
        stamp = json.loads((ready / STAMP_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if stamp.get("manifest_key") != _manifest_key(manifest):
        return False
    for entry in manifest["files"]:
        try:
            if ready.joinpath(*entry["layout_path"].split("/")).stat().st_size != entry["size"]:
                return False
        except OSError:
            return False
    return True


def models_ready(cache_dir: Optional[os.PathLike] = None) -> bool:
    """True if pyfaceau's model folder is complete (nothing to download or copy)."""
    return _derived_ok(models_dir(cache_dir), load_manifest())


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def ensure_models(accept_license: bool = False, *, cache_dir=None,
                  progress: Optional[ProgressCallback] = None) -> Path:
    """Return the folder with pyfaceau's OpenFace model files, preparing it if needed.

    Missing files are downloaded from OpenFace's official repository only when
    the OpenFace license is accepted (``accept_license=True`` or the
    environment variable ``OPENFACE_MODELS_ACCEPT_LICENSE=1``). Every file is
    checked against its SHA-256 checksum. Files already downloaded for another
    package are reused.

    Args:
        accept_license: True to accept the OpenFace license and allow downloads.
        cache_dir: Folder to use in place of ``<user data dir>/OpenFaceModels``
            (default: ``OPENFACE_MODELS_DIR`` or the per-user data folder).
        progress: Optional ``progress(done_bytes, total_bytes, name)`` callback.

    Returns:
        The ready folder (``.../OpenFaceModels/2.2.0/derived/pyfaceau/1``). It
        has the same layout as the old ``weights`` folder, so it can be passed
        as ``weights_dir=`` to ``OpenFaceProcessor``.

    Raises:
        ModelsNotInstalledError: files are missing and the license was not accepted.
        ModelDownloadError: a download failed, a checksum did not match, or the
            folder could not be written.
    """
    manifest = load_manifest()
    root = models_root(cache_dir)
    ready = models_dir(cache_dir)
    if _derived_ok(ready, manifest):
        return ready

    missing = [e for e in manifest["files"] if not _original_ok(root, e)]
    if missing and not license_accepted(accept_license):
        raise ModelsNotInstalledError(_not_installed_message(root, missing))

    with _thread_lock, _FileLock(root / ".lock"):
        # Another process may have finished while we waited for the lock.
        if _derived_ok(ready, manifest):
            return ready
        missing = [e for e in manifest["files"] if not _original_ok(root, e)]
        if missing:
            # Files can disappear while we wait for the lock: ask again before downloading.
            if not license_accepted(accept_license):
                raise ModelsNotInstalledError(_not_installed_message(root, missing))
            _download_all(root, missing, progress)
        _derive(root, ready, manifest)
    return ready


# --------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------

def _human_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    return f"{max(1, round(n / 1000))} KB"


def _not_installed_message(root: Path, missing: List[dict]) -> str:
    size = _human_size(sum(e["size"] for e in missing))
    python = sys.executable or "python"
    return (
        "The OpenFace model files that pyfaceau needs are not installed yet.\n"
        "\n"
        "pyfaceau does not include them: they belong to OpenFace (Carnegie Mellon\n"
        "University) and may only be used for academic or non-profit, non-commercial\n"
        f"research. License: {LICENSE_URL}\n"
        "\n"
        f"To install them ({size} for pyfaceau, one time), open a terminal and run:\n"
        "\n"
        f"    {DOWNLOAD_COMMAND}\n"
        "\n"
        "The same command also prepares the files pyclnf and pymtcnn need (up to\n"
        "about 440 MB in total).\n"
        "\n"
        "If that command is not found, run this instead:\n"
        "\n"
        f"    \"{python}\" -m {DOWNLOAD_MODULE}\n"
        "\n"
        "The files will be saved in:\n"
        f"    {root}\n"
    )


def _ssl_hint(exc: BaseException) -> str:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
        if sys.platform == "darwin":
            return ("\nYour Python cannot check the website's security certificate. If you\n"
                    "installed Python from python.org, open the Python folder in\n"
                    "Applications, double-click 'Install Certificates.command', then try again.")
        return ("\nYour Python cannot check the website's security certificate. Update your\n"
                "system's certificates (or set SSL_CERT_FILE), then try again.")
    return ""


# --------------------------------------------------------------------------
# Download, verify, derive
# --------------------------------------------------------------------------

class _ChecksumError(Exception):
    pass


def _user_agent() -> str:
    try:
        from . import __version__ as version
    except Exception:  # pragma: no cover
        version = "unknown"
    return f"{PACKAGE_NAME}/{version} (OpenFace model downloader)"


def _download_all(root: Path, entries: List[dict], progress: Optional[ProgressCallback]) -> None:
    total = sum(e["size"] for e in entries)
    done = 0
    if progress:
        progress(0, total, "")
    for entry in entries:
        dest = _original_path(root, entry)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ModelDownloadError(_write_error(dest.parent, e)) from e
        base = done

        def file_progress(n: int, _base=base, _name=entry["path"]) -> None:
            if progress:
                progress(_base + n, total, _name)

        _download_one(entry, dest, file_progress)
        done += entry["size"]
    if progress:
        progress(total, total, "")


def _download_one(entry: dict, dest: Path, file_progress: Callable[[int], None]) -> None:
    last_error: Optional[BaseException] = None
    for url in entry["urls"]:
        for attempt in range(_RETRIES):
            if attempt:
                time.sleep(min(30.0, 2 ** attempt) + random.uniform(0, 1))
            tmp = dest.with_name(f"{dest.name}.part-{os.getpid()}-{random.randrange(1 << 30):x}")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
                h = hashlib.sha256()
                n = 0
                with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response, open(tmp, "wb") as out:
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        n += len(chunk)
                        if n > entry["size"]:
                            raise _ChecksumError(f"{entry['path']}: larger than expected")
                        out.write(chunk)
                        h.update(chunk)
                        file_progress(n)
                    out.flush()
                    os.fsync(out.fileno())
                if n != entry["size"] or h.hexdigest() != entry["sha256"]:
                    raise _ChecksumError(
                        f"{entry['path']}: checksum mismatch (got {n} bytes, sha256 {h.hexdigest()})")
                os.replace(tmp, dest)
                return
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code in (400, 401, 403, 404, 410):
                    break  # permanent for this URL; try the next one
            except PermissionError as e:
                raise ModelDownloadError(_write_error(dest.parent, e)) from e
            except OSError as e:
                if e.errno in (errno.ENOSPC, errno.EROFS, errno.EACCES):
                    raise ModelDownloadError(_write_error(dest.parent, e)) from e
                last_error = e  # URLError, timeouts, resets, SSL errors
                if _ssl_hint(e):
                    break  # certificate problems do not go away by retrying
            except (_ChecksumError, ValueError, EOFError) as e:
                last_error = e
            except Exception as e:  # http.client.IncompleteRead and similar
                last_error = e
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
    raise ModelDownloadError(
        f"Could not download {entry['path']} from OpenFace's repository.\n"
        f"Last error: {last_error}\n"
        "Check your internet connection and try again. If you are behind a firewall\n"
        "or proxy, make sure https://raw.githubusercontent.com can be reached."
        + _ssl_hint(last_error if last_error is not None else Exception())
    )


def _write_error(folder: Path, exc: BaseException) -> str:
    return (f"Could not write to {folder}: {exc}\n"
            f"Free some disk space, or set {ENV_MODELS_DIR} to a folder you can write to.")


def _derive(root: Path, ready: Path, manifest: dict) -> None:
    """Copy the originals into pyfaceau's layout and move the folder into place atomically."""
    parent = ready.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=parent))
    except OSError as e:
        raise ModelDownloadError(_write_error(parent, e)) from e
    try:
        for entry in manifest["files"]:
            src = _original_path(root, entry)
            dst = tmp.joinpath(*entry["layout_path"].split("/"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if _sha256_of(dst) != entry["sha256"]:
                raise ModelDownloadError(f"{entry['path']}: copy does not match its checksum")
        stamp = {
            "package": PACKAGE_NAME,
            "converter_version": CONVERTER_VERSION,
            "openface_version": OPENFACE_VERSION,
            "manifest_key": _manifest_key(manifest),
            "files": {e["layout_path"]: e["sha256"] for e in manifest["files"]},
        }
        (tmp / STAMP_FILE).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        os.chmod(tmp, 0o755)  # mkdtemp makes it private; other users may share the cache
        if ready.exists():
            shutil.rmtree(ready)  # incomplete or outdated; we hold the lock
        os.replace(tmp, ready)
    except OSError as e:
        raise ModelDownloadError(_write_error(parent, e)) from e
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


class _FileLock:
    """Inter-process lock on a file (released by the OS if the process dies)."""

    def __init__(self, path: Path, timeout: float = _LOCK_TIMEOUT_S):
        self.path = Path(path)
        self.timeout = timeout
        self._fh = None

    def __enter__(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+b")
        except OSError as e:
            raise ModelDownloadError(_write_error(self.path.parent, e)) from e
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() > deadline:
                    self._fh.close()
                    raise ModelDownloadError(
                        f"Another program has been preparing the model files in {self.path.parent}\n"
                        "for a long time. Close other programs using pyfaceau, pyclnf or pymtcnn and try again.")
                time.sleep(0.5)

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
        return False
