import cv2
import numpy as np


_DEFAULT_DATA_RANGE = 255.0

_SSIM_KERNEL_SIZE = 11
_SSIM_SIGMA = 1.5
_SSIM_BORDER = _SSIM_KERNEL_SIZE // 2

_gaussian_1d = cv2.getGaussianKernel(_SSIM_KERNEL_SIZE, _SSIM_SIGMA)
_SSIM_WINDOW = _gaussian_1d @ _gaussian_1d.T


def compute_psnr(image_a, image_b, data_range=_DEFAULT_DATA_RANGE):
    image_a = np.asarray(image_a, dtype=np.float64)
    image_b = np.asarray(image_b, dtype=np.float64)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"PSNR inputs must have matching shapes, "
            f"got {image_a.shape} and {image_b.shape}."
        )

    mse = np.mean(np.square(image_a - image_b))
    if mse == 0:
        return float("inf")

    return 10.0 * np.log10((data_range ** 2) / mse)


def _compute_single_channel_ssim(image_a, image_b, data_range=_DEFAULT_DATA_RANGE):
    image_a = np.asarray(image_a, dtype=np.float64)
    image_b = np.asarray(image_b, dtype=np.float64)
    
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    valid = (
        slice(_SSIM_BORDER, -_SSIM_BORDER),
        slice(_SSIM_BORDER, -_SSIM_BORDER),
    )

    mean_a = cv2.filter2D(image_a, -1, _SSIM_WINDOW)[valid]
    mean_b = cv2.filter2D(image_b, -1, _SSIM_WINDOW)[valid]

    mean_a_sq = mean_a ** 2
    mean_b_sq = mean_b ** 2
    mean_ab = mean_a * mean_b

    variance_a = (
        cv2.filter2D(image_a ** 2, -1, _SSIM_WINDOW)[valid] - mean_a_sq
    )
    variance_b = (
        cv2.filter2D(image_b ** 2, -1, _SSIM_WINDOW)[valid] - mean_b_sq
    )
    covariance = (
        cv2.filter2D(image_a * image_b, -1, _SSIM_WINDOW)[valid] - mean_ab
    )

    luminance_contrast = (2.0 * mean_ab + c1) * (2.0 * covariance + c2)
    normalization = (mean_a_sq + mean_b_sq + c1) * (variance_a + variance_b + c2)

    ssim_map = luminance_contrast / normalization

    return float(np.mean(ssim_map))


def compute_ssim(image_a, image_b, data_range=_DEFAULT_DATA_RANGE):
    image_a = np.asarray(image_a)
    image_b = np.asarray(image_b)
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"SSIM inputs must have matching shapes, "
            f"got {image_a.shape} and {image_b.shape}."
        )

    if image_a.ndim == 2:
        return _compute_single_channel_ssim(image_a, image_b, data_range)

    if image_a.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D image, got shape {image_a.shape}."
        )

    num_channels = image_a.shape[-1]

    if num_channels == 1:
        return _compute_single_channel_ssim(image_a[..., 0], image_b[..., 0], data_range)

    if num_channels == 3:
        channel_scores = [
            _compute_single_channel_ssim(image_a[..., channel], image_b[..., channel], data_range)
            for channel in range(num_channels)
        ]
        return float(np.mean(channel_scores))

    raise ValueError(
        f"SSIM supports 1- or 3-channel images; "
        f"received {num_channels} channels."
    )
