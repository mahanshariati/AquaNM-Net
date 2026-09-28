import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional, Set

def _find_paired_filenames(
    input_dir: Path,
    target_dir: Path,
    include_names: Optional[Set[str]] = None,
    exclude_names: Optional[Set[str]] = None,
) -> list[str]:
    input_names = {path.name for path in input_dir.iterdir() if path.is_file()}
    target_names = {path.name for path in target_dir.iterdir() if path.is_file()}

    paired_names = input_names & target_names
    if include_names is not None:
        paired_names &= include_names

    if exclude_names:
        paired_names -= exclude_names

    return sorted(paired_names)


def _array_to_unit_tensor(array: np.ndarray) -> torch.Tensor:
    array = array.astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    
    return torch.from_numpy(array).float()


def _make_lab_target(rgb_uint8: np.ndarray) -> torch.Tensor:
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)
    return _array_to_unit_tensor(lab)


def _make_hsi_target(
    rgb_uint8: np.ndarray,
    eps: float = 1e-6
) -> torch.Tensor:
    rgb = rgb_uint8.astype(np.float32) / 255.0

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Intensity
    rgb_sum = r + g + b
    intensity = rgb_sum / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(r, g), b)
    safe_sum = np.maximum(rgb_sum, eps)

    saturation = 1.0 - (3.0 * min_rgb / safe_sum)

    # Black pixels have zero saturation.
    saturation = np.where(
        rgb_sum > eps,
        saturation,
        0.0
    )
    saturation = np.clip(saturation, 0.0, 1.0)

    hue_y = np.sqrt(3.0) * (g - b)
    hue_x = 2.0 * r - g - b

    valid_hue = (
        hue_x ** 2 + hue_y ** 2
    ) > eps ** 2

    safe_hue_x = np.where(valid_hue, hue_x, 1.0)
    safe_hue_y = np.where(valid_hue, hue_y, 0.0)

    angle = np.arctan2(safe_hue_y, safe_hue_x)

    hue = np.mod(angle, 2.0 * np.pi)
    hue = hue / (2.0 * np.pi)

    # Hue is undefined for grayscale pixels.
    hue = np.where(valid_hue, hue, 0.0)

    hsi = np.stack(
        [hue, saturation, intensity],
        axis=0
    ).astype(np.float32)

    return torch.from_numpy(hsi)


class AquaPairedDataset(Dataset):
    def __init__(
        self,
        input_dir,
        target_dir,
        joint_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        exclude_names: Optional[Set[str]] = None,
        include_names: Optional[Set[str]] = None,
    ):
        super().__init__()

        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)

        self.joint_transform = joint_transform
        self.transform = transform

        self.filenames = _find_paired_filenames(
            input_dir=self.input_dir,
            target_dir=self.target_dir,
            include_names=include_names,
            exclude_names=exclude_names,
        )

        if not self.filenames:
            raise RuntimeError(
                "No paired images were found.\n"
                f"Input directory: {self.input_dir}\n"
                f"Target directory: {self.target_dir}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]

        input_path = (self.input_dir / filename)
        target_path = (self.target_dir / filename)

        input_image = Image.open(input_path).convert("RGB")

        target_image = Image.open(target_path).convert("RGB")

        if self.joint_transform is not None:
            input_image, target_image = (self.joint_transform(input_image, target_image))

        input_array = np.asarray(input_image, dtype=np.uint8)

        target_array = np.asarray(target_image, dtype=np.uint8)

        if self.transform is not None:
            input_tensor = self.transform(input_image)
            target_tensor = self.transform(target_image)
        else:
            input_tensor = (_array_to_unit_tensor(input_array))
            target_tensor = (_array_to_unit_tensor(target_array))

        lab_target = _make_lab_target(target_array)
        hsi_target = _make_hsi_target(target_array)

        return input_tensor, target_tensor, lab_target, hsi_target
