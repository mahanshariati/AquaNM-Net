# AquaNM-Net
This is the repository for AquaNM-Net Paper.

## Overview

AquaNMNet processes underwater images using three complementary color representations:

- RGB
- Lab
- HSI

The network combines multi-branch feature extraction, selective feature fusion, decoder reconstruction, and NAS-based operation selection.


## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install AquaNMNet as an editable package:

```bash
pip install -e .
```

The implementation depends on PyTorch and related libraries including:

- torchvision
- numpy
- OpenCV
- Pillow
- einops
- mamba-ssm
- pytorch-msssim
- torchinfo
- scikit-image
- lpips
- scipy

The installation requirements for `mamba-ssm` may depend on the local CUDA and PyTorch versions.

## Dataset Preparation

AquaNMNet uses paired underwater image data consisting of:

- underwater input images
- corresponding reference images
- train/validation/test split files

The expected structure is:

```text
data/
├── LSUI/
│   ├── input/
│   └── GT/
│
└── splits/
    └── lsui/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

LSUI is example. you can use other datasets and put it instead on LSUI.
The dataset itself should be downloaded separately and placed according to the configured paths. The split files are filled for a LSUI dataset. Feel free to remove them and use your own split files.

## Generating Dataset Splits

To generate LSUI split files:

```bash
python scripts/split_lsui.py
```

To create the search subset:

```bash
python scripts/make_lsui_search_subset.py
```

## Training

Training uses a sampled 13-gene architecture from the NAS search space.

Example:

```bash
python scripts/train.py
```

Training checkpoints contain:

```python
{
    "iter": ...,
    "best_psnr": ...,
    "bad_count": ...,
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
}
```

The implementation also supports loading weights-only state dictionaries.

## Neural Architecture Search

Evolutionary architecture search is implemented in:

```text
scripts/search.py
```

The search procedure uses:

- population-based evolutionary optimization
- mutation
- crossover
- top candidate retention

Candidate architectures are represented by 13 integers, where each integer selects one operation from the nine available choices.

Example architecture format:

```text
[0, 3, 5, 2, 8, 1, 4, 6, 7, 0, 5, 2, 8]
```

## Inference and Evaluation

Run evaluation for a selected 13-gene architecture using a trained checkpoint:

```bash
python scripts/infer.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --arch A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12
```

The architecture argument must contain exactly 13 integers, each from 0 to 8.
The checkpoint must match the model's `--base-channels` (default: 48).
By default, paired images are loaded from `data/LSUI/input` and `data/LSUI/GT`,
using filenames from `data/splits/lsui/test.txt`. Edit `aqua_nmnet/config.py`
if your dataset is located elsewhere. Use `--all-images` to evaluate every
paired filename in the input and target directories.

### Extended evaluation example

```bash
python scripts/infer.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --arch 7 4 8 3 1 8 2 3 5 2 2 8 1 \
    --image-size 256 \
    --benchmark-size 256 \
    --benchmark-warmup 10 \
    --benchmark-runs 50
```

The architecture above is an example of a 13-choice configuration; use the
architecture associated with your trained checkpoint or search experiment.
Inference resizes both input and reference images to 256 × 256 by default.

The pipeline loads the checkpoint, counts the supernet parameters,
optionally materializes the selected subnet, measures forward-pass latency,
computes image-quality metrics, and optionally saves enhanced PNG images.

### Evaluation outputs

Under `--output-dir` (or the default `outputs/checkpoints/LSUI_infer_<checkpoint>`):

- `per_image_metrics.csv`: filename, resolution, PSNR, SSIM, LPIPS, UIQM, UCIQE.
- `summary_metrics.json`: per-metric means and population standard deviations,
  model parameter counts, device information, benchmark settings, latency, and FPS.
- Enhanced `.png` images (unless `--no-save-outputs` is specified).

Additional options:

- `--all-images`: use all paired LSUI images instead of the test split.
- `--image-size N`: resize input and ground-truth images to `N × N`.
- `--base-channels N`: set model width (must match the checkpoint).
- `--benchmark-size N`: benchmark model forward passes at `N × N`.
- `--benchmark-warmup N`: run `N` unmeasured warmup iterations.
- `--benchmark-runs N`: run `N` timed iterations.
- `--no-materialize`: retain all supernet operations during evaluation.
- `--no-save-outputs`: write metrics but skip enhanced PNG files.

### Metrics

Full-reference metrics (require aligned ground-truth images):

- **PSNR (↑)**: peak signal-to-noise ratio, reported in dB.
- **SSIM (↑)**: structural similarity index.
- **LPIPS (↓)**: perceptual distance using pretrained AlexNet features.

No-reference metrics (computed only on enhanced images):

- **UIQM (↑)**: underwater image quality measure, combining colorfulness,
  sharpness, and contrast.

PSNR and SSIM are implemented in `aqua_nmnet/utils/image_metrics.py`.
UIQM, parameter counting, and latency benchmarking are implemented in
`aqua_nmnet/utils/eval_metrics.py`. Different published UIQM implementations
may use different processing conventions; comparisons should use the same code.
LPIPS may require downloading pretrained feature-network weights on first use.

### Model efficiency reporting

The summary records total parameters before and after optional subnet
materialization. It also reports latency mean and population standard
deviation in milliseconds, and batch-size-1 FPS, measured at the chosen
benchmark input size. Timing excludes image loading, host-to-device transfer,
metric calculation, and disk output. FPS is calculated as `1000 / mean_latency_ms`
and is not end-to-end pipeline throughput. Benchmark values depend on hardware
and the software environment. FLOPs/MACs are **not** computed.

## Checkpoint Compatibility

The model uses PyTorch `state_dict()` checkpoints.

When modifying the code, checkpoint-sensitive module names should be preserved because existing trained checkpoints depend on the registered module hierarchy.

## Third-Party Code and Attribution

Some components may originate from or be adapted from external research implementations.

## License

This project is released under the MIT License.
See the [LICENSE](LICENSE) file for details.

## Citation

Citation information will be added when the associated publication details are finalized.


## Credits

This project builds upon previous underwater image enhancement research.

Some components and implementation ideas are adapted from:

- AutoEnhancer:
  https://github.com/piggy2009/autoEnhancer

- UDnet:
  https://github.com/alzayats/UDnet

We thank the authors for publicly releasing their implementations.