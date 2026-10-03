# pyfaceau

A python-based implementation of OpenFace 2.2's Facial Action Unit extraction pipeline with an accurate dlib substitute (pymtcnn, pyclnf).

**Accuracy: r = 0.97 correlation with C++ OpenFace 2.2**

## Installation

Installing pyfaceau takes two steps: install the package, then download the
OpenFace model files once.

### Step 1: Install pyfaceau

```bash
pip install pyfaceau
```

This also installs the packages pyfaceau needs:
- [pyclnf](https://github.com/johnwilsoniv/pyclnf) - Facial landmark detection (68 points)
- [pymtcnn](https://github.com/johnwilsoniv/pymtcnn) - Face detection
- [pyfhog](https://github.com/johnwilsoniv/pyfhog) - FHOG feature extraction

### Step 2: Download the OpenFace model files (one time)

pyfaceau uses model files from [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace).
They are **not included** in pyfaceau: they belong to Carnegie Mellon University
and may only be used for academic or non-profit, non-commercial research.
To download them, open a terminal and run:

```bash
pyfaceau-download-models
```

The command shows a short summary of the OpenFace license and a link to the
full text, then asks you to type `YES`. It then prepares the OpenFace files
for pyfaceau **and** for the two packages pyfaceau uses, pyclnf (facial
landmarks) and pymtcnn (face detection): about 440 MB in total, almost all of
it pyclnf's landmark models. The files come from OpenFace's official GitHub
page and, for pyclnf's four large files, from the Dropbox or OneDrive links
OpenFace itself uses. Every file is checked, and the command tells you when it
is done. You only need to do this once per computer, also after upgrading.

The files are saved in one folder shared by pyfaceau, pyclnf and pymtcnn:

| System | Folder |
|--------|--------|
| macOS | `~/Library/Application Support/OpenFaceModels/2.2.0/` |
| Windows | `%LOCALAPPDATA%\OpenFaceModels\2.2.0\` |
| Linux | `~/.local/share/OpenFaceModels/2.2.0/` (or `$XDG_DATA_HOME`) |

To use another folder, set the environment variable `OPENFACE_MODELS_DIR`
before running the command and your program (the `2.2.0` sub-folder is added
for you). On a computer without a terminal prompt (scripts, servers), use
`pyfaceau-download-models --accept-license`, or set
`OPENFACE_MODELS_ACCEPT_LICENSE=1` to accept the license; pyfaceau then
downloads missing files when it starts.

If you already have a folder with the model files, you can keep using it:
`OpenFaceProcessor(weights_dir="/path/to/weights")` always uses the folder
you give it.

### Troubleshooting

- **"The OpenFace model files that pyfaceau needs are not installed yet"**
  (or the same message for pyclnf or pymtcnn): run `pyfaceau-download-models`
  (Step 2). It prepares the files of all three packages.
- **`pyfaceau-download-models: command not found`**: run
  `python -m pyfaceau.download_models` instead, with the same Python you
  installed pyfaceau into.
- **Certificate error on macOS** (`CERTIFICATE_VERIFY_FAILED`): open the Python
  folder in Applications, double-click `Install Certificates.command`, and try
  again.
- **Download stopped or failed**: run the command again. Files that were
  already downloaded and checked are kept. Behind a firewall or proxy, make
  sure `raw.githubusercontent.com` and `www.dropbox.com` (or
  `onedrive.live.com`) can be reached.

### Upgrading from pyfaceau 1.3 or earlier

Run `pyfaceau-download-models` once. The old `~/.pyfaceau/weights` folder is
no longer used and can be deleted. If you used the automatic download of
earlier versions, you may now see values for AU05, AU09, AU14 and AU20, which
that download did not include.

### Install from GitHub (for development)

```bash
git clone https://github.com/johnwilsoniv/pyfaceau.git
cd pyfaceau
conda create -n pyfaceau python=3.11
conda activate pyfaceau
pip install -e .
pyfaceau-download-models
```

The repository does not contain model files either.

## Quick Start

### From a terminal

```bash
pyfaceau input.mp4
```

This measures the 17 action units in every frame of `input.mp4` and saves them
in `input.csv`, in the folder you are in (see [Output Format](#output-format)).
To choose the file name, add `-o`:

```bash
pyfaceau input.mp4 -o results.csv
```

If you see "command not found", type `python -m pyfaceau input.mp4` instead.

### With a window

```bash
pyfaceau-gui
```

opens a small window: add your videos, choose a folder for the results and
click **Process Videos**. You get one CSV file per video. (If you see "command
not found", type `python -m pyfaceau.gui` instead.)

### Video Processing in Python

```python
from pyfaceau import OpenFaceProcessor

# Initialize processor
processor = OpenFaceProcessor(verbose=True)

# Process the video and save the AU values as a CSV file (see Output Format)
processor.process_video("input.mp4", "output.csv")
```

### Batch Processing

```python
from pyfaceau import process_videos

# Process all videos in a directory
process_videos(
    directory_path="/path/to/videos",
    output_dir="/path/to/output"
)
```

### Frame-by-Frame Processing

```python
from pyfaceau import FullPythonAUPipeline
import cv2

# Initialize pipeline (model files come from the folder installed by
# `pyfaceau-download-models`; you can also pass pdm_file=, au_models_dir=,
# triangulation_file= and patch_expert_file= yourself)
pipeline = FullPythonAUPipeline()

# Process single frame
image = cv2.imread("face.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = pipeline.process_frame(image_rgb, frame_num=0)

if result['success']:
    print("AU intensities:", result['au_intensities'])
    print("Landmarks shape:", result['landmarks_2d'].shape)  # (68, 2)
    print("Pose (pitch, yaw, roll):", result['pose'])
```

## Output Format

### CSV Output Columns

The CSV file has one row per video frame and these columns:
- `frame` - Frame number, starting at 0
- `timestamp` - Time of the frame in seconds
- `success` - `True` if a face was found and measured in this frame, `False` if not
- `AU01_r`, `AU02_r`, ... `AU45_r` - the intensities of the 17 action units listed
  below, from 0 (absent) to 5 (maximum). Use them only in rows where `success` is
  `True`; in the other rows they are empty or not meaningful.

These are the AU intensity columns of OpenFace's output (OpenFace's own CSV has
more columns, such as landmarks and head pose, which pyfaceau does not write).

### Action Units

17 facial action units with intensity values (0.0 - 5.0):

| AU | Description |
|----|-------------|
| AU01 | Inner Brow Raiser |
| AU02 | Outer Brow Raiser |
| AU04 | Brow Lowerer |
| AU05 | Upper Lid Raiser |
| AU06 | Cheek Raiser |
| AU07 | Lid Tightener |
| AU09 | Nose Wrinkler |
| AU10 | Upper Lip Raiser |
| AU12 | Lip Corner Puller |
| AU14 | Dimpler |
| AU15 | Lip Corner Depressor |
| AU17 | Chin Raiser |
| AU20 | Lip Stretcher |
| AU23 | Lip Tightener |
| AU25 | Lips Part |
| AU26 | Jaw Drop |
| AU45 | Blink |

## Accuracy

Validated against C++ OpenFace 2.2

| Metric | Correlation |
|--------|-------------|
| **Overall Mean** | r = 0.97 |
| **Overall Median** | r = 0.996 |
| Static AUs | r = 0.98 |
| Dynamic AUs | r = 0.96 |

Per-AU correlations:
- AU01: 0.997, AU02: 0.999, AU04: 0.989, AU05: 0.999
- AU06: 0.999, AU07: 0.996, AU09: 0.997, AU10: 0.994
- AU12: 0.998, AU14: 0.974, AU15: 0.893, AU17: 0.948
- AU20: 0.817, AU23: 0.996, AU25: 0.984, AU26: 0.902, AU45: 0.998

## Requirements

- Python 3.10-3.12 (3.10 recommended; newer versions lack prebuilt wheels)
- numpy
- opencv-python 4 (installed automatically). pyfaceau stays on OpenCV 4 on
  purpose: OpenCV 5 resamples images slightly differently, which changes the AU
  values (see the [changelog](https://github.com/johnwilsoniv/pyfaceau/blob/main/CHANGELOG.md)).
- torch
- scipy

## Acknowledgments

Based on OpenFace 2.2:

> Baltrusaitis, T., Zadeh, A., Lim, Y. C., & Morency, L. P. (2018). OpenFace 2.0: Facial Behavior Analysis Toolkit. IEEE International Conference on Automatic Face and Gesture Recognition.

## Citation

If you use this in research, please cite:

> Wilson IV, J., Rosenberg, J., Gray, M. L., & Razavi, C. R. (2025). A split-face computer vision/machine learning assessment of facial paralysis using facial action units. *Facial Plastic Surgery & Aesthetic Medicine*. https://doi.org/10.1177/26893614251394382

## License

pyfaceau's code: CC BY-NC 4.0 - free for non-commercial use with attribution
(see [LICENSE](LICENSE)).

OpenFace model files: not included in pyfaceau. `pyfaceau-download-models`
downloads them from OpenFace after you accept the
[OpenFace license](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt)
(academic or non-profit, non-commercial research only; do not share the files).
For commercial use, contact Carnegie Mellon University.
