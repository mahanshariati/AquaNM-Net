import math
import time
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from scipy import ndimage


def calculate_uciqe(rgb_uint8: np.ndarray) -> float:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("UCIQE expects an HxWx3 RGB image.")

    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
    l = lab[..., 0] / 255.0
    a = (lab[..., 1] - 128.0) / 255.0
    b = (lab[..., 2] - 128.0) / 255.0

    chroma = np.sqrt(a * a + b * b)
    sigma_c = float(np.std(chroma))

    l_sorted = np.sort(l.reshape(-1))
    if l_sorted.size == 0:
        return float("nan")
    k = max(1, int(round(0.01 * l_sorted.size)))
    contrast_l = float(np.mean(l_sorted[-k:]) - np.mean(l_sorted[:k]))

    denom = np.sqrt(chroma * chroma + l * l)
    saturation = np.divide(
        chroma,
        denom,
        out=np.zeros_like(chroma, dtype=np.float64),
        where=denom > 1e-12,
    )
    mean_s = float(np.mean(saturation))

    return float(0.4680 * sigma_c + 0.2745 * contrast_l + 0.2576 * mean_s)


def _alpha_trimmed_mean(x: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    k = x.size
    if k == 0:
        return 0.0
    t_l = int(math.ceil(alpha_l * k))
    t_r = int(math.floor(alpha_r * k))
    start = min(t_l + 1, k)
    end = max(start, k - t_r)
    trimmed = x[start:end]
    return float(trimmed.mean()) if trimmed.size else float(x.mean())


def _variance_about(x: np.ndarray, mu: float) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.mean((x - mu) ** 2))


def _uicm(rgb: np.ndarray) -> float:
    r = rgb[..., 0].reshape(-1)
    g = rgb[..., 1].reshape(-1)
    b = rgb[..., 2].reshape(-1)

    rg = r - g
    yb = (r + g) / 2.0 - b

    mu_rg = _alpha_trimmed_mean(rg)
    mu_yb = _alpha_trimmed_mean(yb)
    var_rg = _variance_about(rg, mu_rg)
    var_yb = _variance_about(yb, mu_yb)

    l_term = math.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)
    r_term = math.sqrt(max(var_rg + var_yb, 0.0))
    return float(-0.0268 * l_term + 0.1586 * r_term)


def _sobel_magnitude(channel: np.ndarray) -> np.ndarray:
    dx = ndimage.sobel(channel, axis=0)
    dy = ndimage.sobel(channel, axis=1)
    mag = np.hypot(dx, dy)
    max_mag = float(np.max(mag)) if mag.size else 0.0
    if max_mag > 0.0:
        mag = mag * (255.0 / max_mag)
    return mag


def _eme(x: np.ndarray, window_size: int = 10) -> float:
    h, w = x.shape[:2]
    k1 = w / float(window_size)
    k2 = h / float(window_size)
    n1 = int(k1)
    n2 = int(k2)
    if n1 == 0 or n2 == 0:
        return 0.0

    weight = 2.0 / (k1 * k2)
    val = 0.0
    for col in range(n1):
        for row in range(n2):
            block = x[
                row * window_size : (row + 1) * window_size,
                col * window_size : (col + 1) * window_size,
            ]
            block_max = float(np.max(block))
            block_min = float(np.min(block))
            if block_min > 0.0 and block_max > 0.0:
                val += math.log(block_max / block_min)
    return float(weight * val)


def _uism(rgb: np.ndarray) -> float:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    r_edge = _sobel_magnitude(r) * r
    g_edge = _sobel_magnitude(g) * g
    b_edge = _sobel_magnitude(b) * b

    r_eme = _eme(r_edge, 10)
    g_eme = _eme(g_edge, 10)
    b_eme = _eme(b_edge, 10)

    return float(0.299 * r_eme + 0.587 * g_eme + 0.114 * b_eme)


def _uiconm(rgb: np.ndarray, window_size: int = 10) -> float:
    h, w = rgb.shape[:2]
    k1 = w / float(window_size)
    k2 = h / float(window_size)
    n1 = int(k1)
    n2 = int(k2)
    if n1 == 0 or n2 == 0:
        return 0.0

    weight = -1.0 / (k1 * k2)
    val = 0.0
    for col in range(n1):
        for row in range(n2):
            block = rgb[
                row * window_size : (row + 1) * window_size,
                col * window_size : (col + 1) * window_size,
                :,
            ]
            block_max = float(np.max(block))
            block_min = float(np.min(block))
            top = block_max - block_min
            bottom = block_max + block_min
            if bottom > 0.0 and top > 0.0:
                ratio = top / bottom
                val += ratio * math.log(ratio)
    return float(weight * val)


def calculate_uiqm(rgb_uint8: np.ndarray) -> float:
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError("UIQM expects an HxWx3 RGB image.")
    rgb = rgb_uint8.astype(np.float64)
    uicm = _uicm(rgb)
    uism = _uism(rgb)
    uiconm = _uiconm(rgb, 10)
    return float(0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _normalize_hw(input_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(input_size, int):
        return int(input_size), int(input_size)
    if len(input_size) != 2:
        raise ValueError("input_size must be int or (height, width).")
    return int(input_size[0]), int(input_size[1])


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    best_arch: Sequence[int],
    device: torch.device,
    input_size: Union[int, Tuple[int, int]] = 256,
    warmup: int = 10,
    runs: int = 50,
) -> dict:
    h, w = _normalize_hw(input_size)
    x = torch.rand(1, 3, h, w, device=device)

    for _ in range(max(0, warmup)):
        model(x, best_arch)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        times_ms = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x, best_arch)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(float(start.elapsed_time(end)))
    else:
        times_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x, best_arch)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std(ddof=0))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else float("inf")

    return {
        "benchmark_height": h,
        "benchmark_width": w,
        "benchmark_batch_size": 1,
        "latency_ms_mean": mean_ms,
        "latency_ms_std": std_ms,
        "fps": fps,
        "warmup_runs": int(warmup),
        "timed_runs": int(runs),
    }
