import csv
import json
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import lpips
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from aqua_nmnet.utils.eval_metrics import (
    benchmark_forward,
    calculate_uciqe,
    calculate_uiqm,
    count_parameters,
)
from aqua_nmnet.utils.image_metrics import compute_psnr, compute_ssim
from aqua_nmnet.models.model import AquaNMNet
from aqua_nmnet.config import (
    LSUI_INPUT_DIR,
    LSUI_TARGET_DIR,
    TEST_SPLIT_FILE,
    CHECKPOINT_DIR,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TO_TENSOR = transforms.ToTensor()


CHOICE_TO_ATTR = {
    0: "resblock_skipconnect",
    1: "resblock_dil2Conv",
    2: "resblock_conv",
    3: "resblock_se",
    4: "resblock_trans_normal",
    5: "resblock_trans_dual",
    6: "resblock_trans_eca",
    7: "resblock_trans_sa",
    8: "resblock_trans_sge",
}


def validate_arch(best_arch: Sequence[int]) -> list[int]:
    best_arch = [int(x) for x in best_arch]
    if len(best_arch) != 13:
        raise ValueError(
            f"Architecture must contain exactly 13 choices, got {len(best_arch)}."
        )
    invalid = [x for x in best_arch if not 0 <= x <= 8]
    if invalid:
        raise ValueError(
            f"Architecture choices must be in [0, 8]. Invalid values: {invalid}"
        )
    return best_arch


def materialize_selected_subnet(model: AquaNMNet, best_arch: Sequence[int]) -> AquaNMNet:
    best_arch = validate_arch(best_arch)
    blocks_and_choices = [
        (model.base.block1, {best_arch[0], best_arch[4], best_arch[8]}),
        (model.base.block2, {best_arch[1], best_arch[5], best_arch[9]}),
        (model.base.block3, {best_arch[2], best_arch[6], best_arch[10]}),
        (model.base.block4, {best_arch[3], best_arch[7], best_arch[11]}),
        (model.refine, {best_arch[12]}),
    ]
    for block, keep_choices in blocks_and_choices:
        for choice_idx, attr_name in CHOICE_TO_ATTR.items():
            if choice_idx not in keep_choices and hasattr(block, attr_name):
                delattr(block, attr_name)
    return model


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a valid model state_dict.")
    if checkpoint and all(str(k).startswith("module.") for k in checkpoint.keys()):
        checkpoint = {str(k)[7:]: v for k, v in checkpoint.items()}
    return checkpoint


def _read_test_list(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"LSUI test-list file not found: {path}")
    names = []
    with open(path, "r") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    if not names:
        raise RuntimeError(f"LSUI test list is empty: {path}")
    if len(names) != len(set(names)):
        raise RuntimeError("LSUI test list contains duplicate filenames.")
    return names


def _selected_files(use_test_list: bool) -> list[str]:
    if use_test_list:
        files = _read_test_list(TEST_SPLIT_FILE)
        missing = [
            f for f in files
            if not os.path.isfile(os.path.join(LSUI_INPUT_DIR, f))
            or not os.path.isfile(os.path.join(LSUI_TARGET_DIR, f))
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} LSUI test pairs are missing. Examples: {missing[:5]}"
            )
        return files

    inp = {
        p.name for p in Path(LSUI_INPUT_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }
    gt = {
        p.name for p in Path(LSUI_TARGET_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }
    if inp != gt:
        raise RuntimeError(
            f"LSUI input/GT folders are not perfectly paired: "
            f"input-only={len(inp-gt)}, GT-only={len(gt-inp)}"
        )
    files = sorted(inp)
    if not files:
        raise RuntimeError("No paired LSUI images found.")
    return files


def _resize_pair(
    inp: Image.Image,
    gt: Image.Image,
    resize_to: Optional[Union[int, Tuple[int, int]]],
) -> Tuple[Image.Image, Image.Image]:
    if resize_to is None:
        if inp.size != gt.size:
            raise ValueError(
                f"Input/GT size mismatch: input={inp.size}, GT={gt.size}."
            )
        return inp, gt

    if isinstance(resize_to, int):
        size = (int(resize_to), int(resize_to))
    else:
        if len(resize_to) != 2:
            raise ValueError("resize_to tuple must be (width, height).")
        size = (int(resize_to[0]), int(resize_to[1]))

    return (
        inp.resize(size, Image.Resampling.BILINEAR),
        gt.resize(size, Image.Resampling.BILINEAR),
    )


def _stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def run_inference(
    ckpt_path: str,
    best_arch: Sequence[int],
    out_dir: Optional[str] = None,
    use_test_list: bool = True,
    resize_to: Optional[Union[int, Tuple[int, int]]] = 256,
    materialize: bool = True,
    benchmark_size: Union[int, Tuple[int, int]] = 256,
    benchmark_warmup: int = 10,
    benchmark_runs: int = 50,
    save_outputs: bool = True,
    base_channels: int = 48,
):
    if base_channels <= 0:
        raise ValueError("base_channels must be positive.")
    if benchmark_runs <= 0 or benchmark_warmup < 0:
        raise ValueError("benchmark_runs must be positive and benchmark_warmup nonnegative.")
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    best_arch = validate_arch(best_arch)
    file_list = _selected_files(use_test_list)

    if out_dir is None:
        ckpt_name = Path(ckpt_path).stem
        out_dir = os.path.join(CHECKPOINT_DIR, f"LSUI_infer_{ckpt_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Architecture: {best_arch}")
    print(f"LSUI images: {len(file_list)}")
    print(f"Metric resolution: {'native' if resize_to is None else resize_to}")

    model = AquaNMNet(base_channels=base_channels)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    supernet_params = count_parameters(model)

    if materialize:
        model = materialize_selected_subnet(model, best_arch)
    subnet_params = count_parameters(model)

    model = model.to(DEVICE).eval()

    benchmark = benchmark_forward(
        model,
        best_arch,
        DEVICE,
        input_size=benchmark_size,
        warmup=benchmark_warmup,
        runs=benchmark_runs,
    )

    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()

    rows = []
    psnr_values, ssim_values = [], []
    lpips_values, uiqm_values = [], []

    with torch.inference_mode():
        for idx, fname in enumerate(file_list, start=1):
            inp = Image.open(os.path.join(LSUI_INPUT_DIR, fname)).convert("RGB")
            gt = Image.open(os.path.join(LSUI_TARGET_DIR, fname)).convert("RGB")
            inp, gt = _resize_pair(inp, gt, resize_to)

            inp_var = TO_TENSOR(inp).unsqueeze(0).to(DEVICE, non_blocking=True)
            gt_var = TO_TENSOR(gt).unsqueeze(0).to(DEVICE, non_blocking=True)

            final, _, _, _ = model(inp_var, best_arch)
            final = final.clamp(0.0, 1.0)

            if final.shape != gt_var.shape:
                raise RuntimeError(
                    f"Prediction/GT tensor mismatch for {fname}: "
                    f"pred={tuple(final.shape)}, gt={tuple(gt_var.shape)}"
                )

            pred_np = final[0].permute(1, 2, 0).float().cpu().numpy()
            gt_np = np.asarray(gt, dtype=np.uint8)
            psnr = float(compute_psnr(pred_np * 255.0, gt_np))
            ssim = float(compute_ssim(pred_np * 255.0, gt_np))
            lpips_value = float(
                lpips_metric(final * 2.0 - 1.0, gt_var * 2.0 - 1.0)
                .mean()
                .item()
            )

            pred_u8 = np.clip(np.rint(pred_np * 255.0), 0, 255).astype(np.uint8)
            uiqm = float(calculate_uiqm(pred_u8))

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            lpips_values.append(lpips_value)
            uiqm_values.append(uiqm)

            rows.append({
                "file": fname,
                "width": int(pred_u8.shape[1]),
                "height": int(pred_u8.shape[0]),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_value,
                "uiqm": uiqm,
            })

            if save_outputs:
                out_name = Path(fname).stem + ".png"
                Image.fromarray(pred_u8, mode="RGB").save(os.path.join(out_dir, out_name))

            if idx % 50 == 0 or idx == len(file_list):
                print(
                    f"[{idx}/{len(file_list)}] "
                    f"PSNR={np.mean(psnr_values):.4f}, "
                    f"SSIM={np.mean(ssim_values):.5f}, "
                    f"LPIPS={np.mean(lpips_values):.5f}, "
                    f"UIQM={np.mean(uiqm_values):.4f}"
                )

    psnr_mean, psnr_std = _stats(psnr_values)
    ssim_mean, ssim_std = _stats(ssim_values)
    lpips_mean, lpips_std = _stats(lpips_values)
    uiqm_mean, uiqm_std = _stats(uiqm_values)

    summary = {
        "dataset": "LSUI",
        "split": "test_list" if use_test_list else "all_paired",
        "checkpoint": ckpt_path,
        "best_arch": best_arch,
        "num_images": len(rows),
        "resize_to": resize_to,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "uiqm_mean": uiqm_mean,
        "uiqm_std": uiqm_std,
        "supernet_params": supernet_params,
        "supernet_params_m": supernet_params / 1e6,
        "subnet_params": subnet_params,
        "subnet_params_m": subnet_params / 1e6,
        "materialized": bool(materialize),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None,
        **benchmark,
        "output_dir": out_dir,
    }

    csv_path = os.path.join(out_dir, "per_image_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(out_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"PSNR : {psnr_mean:.4f} ± {psnr_std:.4f} dB")
    print(f"SSIM : {ssim_mean:.5f} ± {ssim_std:.5f}")
    print(f"LPIPS: {lpips_mean:.5f} ± {lpips_std:.5f}  (lower is better)")
    print(f"UIQM : {uiqm_mean:.4f} ± {uiqm_std:.4f}")
    print(f"Supernet parameters : {supernet_params / 1e6:.3f} M")
    print(f"Fixed-subnet params  : {subnet_params / 1e6:.3f} M")
    print(
        f"Latency @{benchmark['benchmark_height']}x{benchmark['benchmark_width']}: "
        f"{benchmark['latency_ms_mean']:.3f} ± {benchmark['latency_ms_std']:.3f} ms"
    )
    print(f"FPS (batch=1): {benchmark['fps']:.2f}")
    print(f"Per-image CSV: {csv_path}")
    print(f"Summary JSON : {summary_path}")
    print(f"Outputs      : {out_dir}")

    return summary


run_inference = run_inference