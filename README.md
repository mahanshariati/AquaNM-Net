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

Inference is performed using:

```bash
python scripts/infer.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --arch A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12
```

The architecture argument must contain exactly 13 values, each between 0 and 8.

The inference pipeline:

- loads the trained checkpoint
- restores the selected architecture
- evaluates enhanced images
- computes image-quality metrics
- saves enhanced outputs

## Metrics

The repository includes utilities for:

- PSNR
- SSIM

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