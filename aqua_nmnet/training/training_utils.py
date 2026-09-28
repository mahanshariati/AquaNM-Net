import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def build_validation_architectures(choice, layers, seed, architecture_type):
    rng = np.random.RandomState(seed)

    if architecture_type == "all_conv":
        return [
            np.full(layers, 2, dtype=np.int64)
        ]

    if architecture_type != "balanced_panel":
        raise ValueError(
            "Unknown validation architecture type: "
            f"{architecture_type}"
        )

    panel = np.empty((choice, layers), dtype=np.int64)
    for layer_index in range(layers):
        panel[:, layer_index] = rng.permutation(choice)

    return [
        panel[index].copy()
        for index in range(choice)
    ]


@torch.no_grad()
def validate(net, val_loader, validation_architectures, device_id):
    net.eval()

    psnr_values = []
    l1_values = []

    for data in val_loader:
        images, targets, _, _ = data
        images = images.cuda(device_id, non_blocking=True)
        targets = targets.cuda(device_id, non_blocking=True)
        
        targets = torch.clamp(targets, 0.0, 1.0)

        for architecture in validation_architectures:
            prediction, _, _, _ = net(
                images,
                architecture,
            )

            prediction = torch.clamp(prediction, 0.0, 1.0)

            l1 = F.l1_loss(prediction, targets).item()
            mse = F.mse_loss(prediction, targets).item()
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))

            l1_values.append(l1)
            psnr_values.append(psnr)

    net.train()

    mean_psnr = (
        float(np.mean(psnr_values))
        if psnr_values
        else 0.0
    )

    mean_l1 = (
        float(np.mean(l1_values))
        if l1_values
        else 0.0
    )

    return mean_psnr, mean_l1


def save_full_checkpoint(
    path,
    net,
    optimizer,
    curr_iter,
    best_psnr,
    bad_count
):
    checkpoint = {
        "iter": curr_iter,
        "best_psnr": best_psnr,
        "bad_count": bad_count,
        "model": net.state_dict(),
        "optim": optimizer.state_dict(),
    }

    torch.save(checkpoint, path)


def save_best(
    net,
    optimizer,
    curr_iter,
    best_psnr,
    bad_count,
    save_dir
):
    save_full_checkpoint(
        path=os.path.join(save_dir, "best.pth"),
        net=net,
        optimizer=optimizer,
        curr_iter=curr_iter,
        best_psnr=best_psnr,
        bad_count=bad_count
    )


def load_training_checkpoint(
    path,
    net,
    optimizer,
    device_id,
    fallback_iteration=0,
):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Requested checkpoint was not found: {path}"
        )

    print("Resuming from:", path)
    checkpoint = torch.load(path, map_location=f"cuda:{device_id}")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        net.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optimizer_state = checkpoint.get(
            "optim",
            checkpoint.get("optimizer"),
        )

        if optimizer_state is not None:
            optimizer.load_state_dict(
                optimizer_state
            )

        current_iteration = int(
            checkpoint.get("iter", 0)
        )

        best_psnr = float(
            checkpoint.get(
                "best_psnr",
                float("-inf"),
            )
        )

        bad_count = int(
            checkpoint.get("bad_count", 0)
        )

        print(
            f"Resumed at iteration {current_iteration}, "
            f"best PSNR {best_psnr:.4f}"
        )

        return (
            current_iteration,
            best_psnr,
            bad_count,
        )

    # Previous weights-only checkpoints remain supported.
    net.load_state_dict(
        checkpoint,
        strict=True,
    )

    print(
        "Loaded weights-only checkpoint at iteration "
        f"{fallback_iteration}. Adam state could not be restored."
    )

    return (
        fallback_iteration,
        float("-inf"),
        0,
    )
