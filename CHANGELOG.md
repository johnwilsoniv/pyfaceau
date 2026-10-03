# Changelog

## 1.4.1 (2026-10-03)

### What changes for you

- **The `pyfaceau` command works.** In a terminal, type `pyfaceau video.mp4` to
  measure the 17 action units in every frame of a video. The results are saved
  as `video.csv` in the folder you are in; to choose the file, type
  `pyfaceau video.mp4 -o results.csv`. Before, the command stopped with an
  error.
- **`pyfaceau-gui` is now a proper part of pyfaceau.** It opens the same
  window as before, but no longer depends on a loose copy of the GUI script
  in the folder of the command, which Python does not find in every
  installation. If the command is not found, `python -m pyfaceau.gui` opens
  the window too.
- **pyfaceau stays on OpenCV 4.** Since OpenCV 5 came out, a new installation of
  pyfaceau got OpenCV 5, which resamples images slightly differently. On
  OpenFace's sample videos that moves AU intensities by up to 0.54 and
  landmarks by up to 2.45 pixels, and flips 0.4–0.6% of AU presence flags, so
  results would depend on the day pyfaceau was installed. pyfaceau now asks
  for OpenCV 4; if you have OpenCV 5, `pip install --upgrade pyfaceau` puts
  OpenCV 4 back.
- pyfaceau now needs pyclnf 0.4.1 (installed automatically), which has the
  same OpenCV 4 requirement and also works without PyTorch.
- Your results do not change: with the same OpenCV 4 version, pyfaceau 1.4.1
  gives exactly the same AU values as 1.4.0.
- The README now lists the columns the CSV file really has: `frame`,
  `timestamp`, `success` and the 17 AU intensities (`AU01_r` to `AU45_r`).

### Details

- `pyfaceau VIDEO [-o OUTPUT.csv]` (also `python -m pyfaceau`) runs the default
  pipeline, `OpenFaceProcessor`, on one video. It shows progress in a
  terminal, prints the installation steps when the OpenFace model files are
  missing, and exits with 1 when the video cannot be processed or no frame has
  a face. The entry point `pyfaceau.processor:main` did not exist before.
- The GUI moved into the package as `pyfaceau.gui` (`pyfaceau-gui`, also
  `python -m pyfaceau.gui`). The entry point used to import a top-level module
  `pyfaceau_gui` that was not part of the package; it was only found because
  setup.py copied `pyfaceau_gui.py` into the folder of the commands, which
  Python puts first on its path when a command starts. That loose script (and
  with it the `pyfaceau_gui.py` command) is no longer installed. The GUI uses
  the shared model folder (or `PYFACEAU_WEIGHTS_DIR`).
- `opencv-python>=4.5.0,<5` and `pyclnf>=0.4.1` in pyproject.toml, setup.py and
  requirements.txt. OpenCV 5's `warpAffine` and `remap` no longer snap sample
  positions to 1/32 pixel, and no `cv2` flag restores the 4.x behaviour.

## 1.4.0 (2026-10-02)

### What changes for you

- **pyfaceau no longer contains OpenFace's model files, and no longer downloads
  them by itself.** These files belong to OpenFace (Carnegie Mellon University)
  and may only be used for academic or non-profit, non-commercial research.
- **One-time setup:** after installing or upgrading, open a terminal and run
  `pyfaceau-download-models`. It shows a short summary of the
  [OpenFace license](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt),
  asks you to type `YES`, and prepares the OpenFace files for pyfaceau and for
  the two packages it uses, pyclnf and pymtcnn: about 440 MB in total, from
  OpenFace's official sources, with every file checked.
- pyfaceau now needs pyclnf 0.4.0 and pymtcnn 1.2.0 (installed automatically).
  They no longer include OpenFace's files either; the same command prepares
  theirs.
- On recent macOS versions, pyfaceau 1.3.x could stop at the first video with
  "Failed to initialize any backend" when coremltools was not installed.
  pymtcnn 1.2.0 now falls back to a face detector that works, so this no
  longer happens.
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
  `--accept-license` skips the question). It prepares the files of pyfaceau,
  pyclnf and pymtcnn in one shared folder; a file needed by more than one
  package is downloaded once. The old `pyfaceau-download-weights` command now
  does the same thing.
- Requires `pyclnf>=0.4.0` and `pymtcnn>=1.2.0` (pyproject.toml, setup.py and
  requirements.txt). New `coreml` extra (`pip install "pyfaceau[coreml]"`) adds
  coremltools for pymtcnn's Core ML face detector on Apple Silicon.
- Downloads are accepted only from OpenFace's official repository, the license
  agreement is checked again right before downloading, and the prepared folder
  is readable by other users of a shared model folder.
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
