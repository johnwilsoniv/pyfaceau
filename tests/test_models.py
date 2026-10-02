"""Offline tests for pyfaceau.models and the pyfaceau-download-models command.

A small fake manifest with file:// URLs stands in for OpenFace's files, so no
network access and no OpenFace files are needed.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from pyfaceau import models
from pyfaceau import download_models


def _fake_manifest(tmp_path, corrupt=None):
    src = tmp_path / "server"
    files = {
        "lib/local/FaceAnalyser/AU_predictors/In-the-wild_aligned_PDM_68.txt": ("In-the-wild_aligned_PDM_68.txt", b"pdm " * 50),
        "lib/local/FaceAnalyser/AU_predictors/tris_68_full.txt": ("tris_68_full.txt", b"tris " * 20),
        "lib/local/LandmarkDetector/model/patch_experts/svr_patches_0.25_general.txt": ("svr_patches_0.25_general.txt", b"svr " * 30),
        "lib/local/FaceAnalyser/AU_predictors/svr_combined/AU_1_dynamic_intensity_comb.dat": ("AU_predictors/svr_combined/AU_1_dynamic_intensity_comb.dat", bytes(range(256)) * 40),
    }
    entries = []
    for path, (layout, data) in files.items():
        f = src / path
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data if path != corrupt else data + b"tampered")
        entries.append({"path": path, "urls": [f.as_uri()], "sha256": hashlib.sha256(data).hexdigest(),
                        "size": len(data), "layout_path": layout})
    return {"schema": 1, "openface_version": "2.2.0", "files": entries}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv(models.ENV_ACCEPT_LICENSE, raising=False)
    monkeypatch.delenv(models.ENV_MODELS_DIR, raising=False)
    monkeypatch.delenv("PYFACEAU_WEIGHTS_DIR", raising=False)
    manifest = _fake_manifest(tmp_path)
    monkeypatch.setattr(models, "load_manifest", lambda: json.loads(json.dumps(manifest)))
    monkeypatch.setattr(models, "_RETRIES", 1)
    return tmp_path, manifest


def test_real_manifest_is_complete_and_points_to_openface():
    manifest = models.load_manifest()
    files = manifest["files"]
    assert manifest["openface_version"] == "2.2.0"
    assert len(files) == 32
    layouts = [e["layout_path"] for e in files]
    assert len(set(layouts)) == len(layouts)
    assert {"In-the-wild_aligned_PDM_68.txt", "tris_68_full.txt", "svr_patches_0.25_general.txt"} <= set(layouts)
    assert sum(l.startswith("AU_predictors/svr_combined/") for l in layouts) == 29
    for e in files:
        assert e["urls"] == ["https://raw.githubusercontent.com/TadasBaltrusaitis/OpenFace/OpenFace_2.2.0/" + e["path"]]
        assert len(e["sha256"]) == 64 and int(e["sha256"], 16) >= 0
        assert e["size"] > 0


def test_package_contains_no_model_files():
    pkg = Path(models.__file__).parent
    bad = [p for p in pkg.rglob("*") if p.suffix.lower() in
           {".dat", ".onnx", ".pth", ".pt", ".mlmodel", ".h5", ".npz", ".npy", ".pkl"}
           or p.name in {"In-the-wild_aligned_PDM_68.txt", "tris_68_full.txt"}]
    assert bad == []


def test_cache_location(monkeypatch, tmp_path):
    monkeypatch.delenv(models.ENV_MODELS_DIR, raising=False)
    monkeypatch.setattr(models.sys, "platform", "darwin")
    assert models.models_root() == Path.home() / "Library" / "Application Support" / "OpenFaceModels" / "2.2.0"
    monkeypatch.setattr(models.sys, "platform", "linux")
    monkeypatch.setattr(models.os, "name", "posix")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert models.models_root() == tmp_path / "xdg" / "OpenFaceModels" / "2.2.0"
    monkeypatch.setenv("XDG_DATA_HOME", "relative/ignored")
    assert models.models_root() == Path.home() / ".local" / "share" / "OpenFaceModels" / "2.2.0"
    monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp_path / "custom"))
    assert models.models_root() == tmp_path / "custom" / "2.2.0"
    assert models.models_root(tmp_path / "arg") == tmp_path / "arg" / "2.2.0"
    assert models.models_dir(tmp_path / "arg") == tmp_path / "arg" / "2.2.0" / "derived" / "pyfaceau" / "1"


def test_missing_without_license_raises_plain_message(env):
    tmp, _ = env
    with pytest.raises(models.ModelsNotInstalledError) as info:
        models.ensure_models(cache_dir=tmp / "cache")
    assert isinstance(info.value, FileNotFoundError)
    text = str(info.value)
    assert "pyfaceau-download-models" in text and "-m pyfaceau.download_models" in text
    assert str(tmp / "cache" / "2.2.0") in text
    assert not (tmp / "cache" / "2.2.0" / "originals").exists()


def test_download_verify_and_derive(env):
    tmp, manifest = env
    events = []
    ready = models.ensure_models(True, cache_dir=tmp / "cache", progress=lambda d, t, n: events.append((d, t, n)))
    assert ready == models.models_dir(tmp / "cache")
    for e in manifest["files"]:
        data = (tmp / "cache" / "2.2.0" / "originals" / e["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == e["sha256"]
        assert (ready / e["layout_path"]).read_bytes() == data
    paths = models.model_paths(ready)
    assert Path(paths["pdm_file"]).exists() and Path(paths["au_models_dir"], "svr_combined").is_dir()
    total = sum(e["size"] for e in manifest["files"])
    assert events[0] == (0, total, "") and events[-1] == (total, total, "")
    leftovers = [p for p in (tmp / "cache").rglob("*") if ".part-" in p.name or p.name.startswith(".tmp-")]
    assert leftovers == []
    # Ready now: works without license and without network.
    for e in manifest["files"]:
        e["urls"] = ["file:///nonexistent"]
    assert models.ensure_models(cache_dir=tmp / "cache") == ready


def test_env_var_counts_as_acceptance(env, monkeypatch):
    tmp, _ = env
    monkeypatch.setenv(models.ENV_ACCEPT_LICENSE, "1")
    assert models.ensure_models(cache_dir=tmp / "cache").is_dir()


def test_env_var_cache_dir(env, monkeypatch):
    tmp, _ = env
    monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp / "envcache"))
    ready = models.ensure_models(True)
    assert ready == tmp / "envcache" / "2.2.0" / "derived" / "pyfaceau" / "1"


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    manifest = _fake_manifest(tmp_path, corrupt="lib/local/FaceAnalyser/AU_predictors/tris_68_full.txt")
    monkeypatch.setattr(models, "load_manifest", lambda: manifest)
    monkeypatch.setattr(models, "_RETRIES", 1)
    with pytest.raises(models.ModelDownloadError):
        models.ensure_models(True, cache_dir=tmp_path / "cache")
    root = tmp_path / "cache" / "2.2.0"
    assert not (root / "originals" / "lib/local/FaceAnalyser/AU_predictors/tris_68_full.txt").exists()
    assert not models.models_dir(tmp_path / "cache").exists()
    assert [p for p in root.rglob("*") if ".part-" in p.name] == []


def test_damaged_files_are_repaired(env):
    tmp, manifest = env
    ready = models.ensure_models(True, cache_dir=tmp / "cache")
    original = tmp / "cache" / "2.2.0" / "originals" / manifest["files"][0]["path"]
    original.write_bytes(b"damaged")
    (ready / manifest["files"][1]["layout_path"]).unlink()
    with pytest.raises(models.ModelsNotInstalledError):
        models.ensure_models(cache_dir=tmp / "cache")  # a download is needed: license required
    ready = models.ensure_models(True, cache_dir=tmp / "cache")
    for e in manifest["files"]:
        assert hashlib.sha256((ready / e["layout_path"]).read_bytes()).hexdigest() == e["sha256"]


def test_derive_from_existing_originals_needs_no_license(env):
    tmp, _ = env
    ready = models.ensure_models(True, cache_dir=tmp / "cache")
    import shutil
    shutil.rmtree(ready)
    assert models.ensure_models(cache_dir=tmp / "cache") == ready  # local copy only


def test_concurrent_calls(env):
    tmp, _ = env
    results, errors = [], []

    def run():
        try:
            results.append(models.ensure_models(True, cache_dir=tmp / "cache"))
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(set(results)) == 1


def test_cli_accept_license(env, capsys):
    tmp, _ = env
    assert download_models.main(["--accept-license", "--cache-dir", str(tmp / "cache")]) == 0
    out = capsys.readouterr().out
    assert models.LICENSE_URL in out and "Done." in out
    assert models.models_ready(tmp / "cache")


def test_cli_declined(env, monkeypatch, capsys):
    tmp, _ = env
    monkeypatch.setattr("builtins.input", lambda prompt="": "no")
    assert download_models.main(["--cache-dir", str(tmp / "cache")]) == 1
    assert "not accepted" in capsys.readouterr().out
    assert not (tmp / "cache" / "2.2.0" / "originals").exists()


def test_cli_typed_yes(env, monkeypatch):
    tmp, _ = env
    monkeypatch.setattr("builtins.input", lambda prompt="": "yes")
    assert download_models.main(["--cache-dir", str(tmp / "cache")]) == 0


def test_cli_no_terminal(env, monkeypatch):
    tmp, _ = env

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert download_models.main(["--cache-dir", str(tmp / "cache")]) == 1


def test_dependency_detection(monkeypatch):
    versions = {"pyclnf": "0.3.4", "pymtcnn": "1.1.5"}
    monkeypatch.setattr(download_models.metadata, "version", lambda name: versions[name])
    lines = download_models._prepare_dependencies(None)
    assert all("includes its own model files" in l for l in lines)

    def missing(name):
        raise download_models.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(download_models.metadata, "version", missing)
    assert all("not installed" in l for l in download_models._prepare_dependencies(None))


def test_legacy_helpers(env, monkeypatch):
    tmp, _ = env
    import importlib
    download_weights = importlib.import_module("pyfaceau.download_weights")
    monkeypatch.setenv(models.ENV_MODELS_DIR, str(tmp / "cache"))
    assert not download_weights.weights_exist()
    with pytest.raises(FileNotFoundError):
        download_weights.ensure_weights()
    ready = models.ensure_models(True)
    assert download_weights.weights_exist() and download_weights.ensure_weights() == ready
    assert download_weights.get_weights_dir() == ready
    monkeypatch.setenv("PYFACEAU_WEIGHTS_DIR", str(ready))
    assert download_weights.ensure_weights() == ready
