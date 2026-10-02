# Changelog

## 1.4.0 (2026-10-02)

### What changes for you

- **pyfaceau no longer contains OpenFace's model files, and no longer downloads
  them by itself.** These files belong to OpenFace (Carnegie Mellon University)
  and may only be used for academic or non-profit, non-commercial research.
- **One-time setup:** after installing or upgrading, open a terminal and run
  `pyfaceau-download-models`. It shows a short summary of the
  [OpenFace license](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt),
  asks you to type `YES`, downloads about 3.4 MB from OpenFace's official
  GitHub page, and checks every file.
- If you skip this step, pyfaceau stops with a message that tells you exactly
  what to run. It never downloads anything without your agreement.
- Your results do not change: on the same video, pyfaceau 1.4.0 gives exactly
  the same AU values as 1.3.16 with a complete set of model files.
- If you used the automatic download of pyfaceau 1.3.x, you may now also see
  values for AU05, AU09, AU14 and AU20. That download was missing the four
  model files these AUs need; the new download includes them.
- If you pass your own model folder (`OpenFaceProcessor(weights_dir=...)`),
  nothing changes: that folder is still used.

### Details

- New `pyfaceau-download-models` command (also `python -m pyfaceau.download_models`;
  `--accept-license` skips the question). The old `pyfaceau-download-weights`
  command now does the same thing.
- New `pyfaceau.models.ensure_models(accept_license=False, *, cache_dir=None, progress=None)`
  returns the ready model folder, downloading and checking missing files only
  when the license is accepted (`accept_license=True` or
  `OPENFACE_MODELS_ACCEPT_LICENSE=1`). Otherwise it raises
  `ModelsNotInstalledError` (a `FileNotFoundError`) with installation steps.
- Model files live in a folder shared with pyclnf and pymtcnn:
  `~/Library/Application Support/OpenFaceModels/2.2.0/` (macOS),
  `%LOCALAPPDATA%\OpenFaceModels\2.2.0\` (Windows),
  `~/.local/share/OpenFaceModels/2.2.0/` (Linux). Set `OPENFACE_MODELS_DIR` to
  use another folder.
- `OpenFaceProcessor()` and `FullPythonAUPipeline()` use that folder when no
  model paths are given. `PYFACEAU_WEIGHTS_DIR` is still honored if it points
  to a complete folder. The `auto_download_weights` argument is ignored.
- The old `~/.pyfaceau/weights` folder and the `weights-v1.0` download from
  pyfaceau's GitHub page are no longer used.
- The package and the GitHub repository no longer contain any model files;
  the repository history was cleaned as well.
- The GUI (`pyfaceau_gui.py`) now uses the shared model folder and shows the
  error message when the pipeline cannot start.
