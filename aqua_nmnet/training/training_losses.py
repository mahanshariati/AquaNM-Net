import torch
import torch.nn.functional as F


def resize_hsi_target(target_hsi, size, eps=1e-6):
    hue = target_hsi[:, 0:1]
    saturation_intensity = target_hsi[:, 1:3]

    angle = 2.0 * torch.pi * hue

    hue_sin = F.interpolate(
        torch.sin(angle),
        size=size,
        mode="bilinear",
        align_corners=False
    )

    hue_cos = F.interpolate(
        torch.cos(angle),
        size=size,
        mode="bilinear",
        align_corners=False
    )

    magnitude = torch.sqrt(
        hue_sin.square() + hue_cos.square()
    ).clamp_min(eps)

    hue_sin = hue_sin / magnitude
    hue_cos = hue_cos / magnitude

    resized_hue = torch.atan2(hue_sin, hue_cos)
    resized_hue = torch.remainder(
        resized_hue,
        2.0 * torch.pi
    ) / (2.0 * torch.pi)

    resized_si = F.interpolate(
        saturation_intensity,
        size=size,
        mode="bilinear",
        align_corners=False
    )

    return torch.cat(
        [resized_hue, resized_si],
        dim=1
    )


def compute_hsi_auxiliary_losses(pred_hsi, target_hsi, eps=1e-6):
    pred_angle = 2.0 * torch.pi * pred_hsi[:, 0:1]
    target_angle = 2.0 * torch.pi * target_hsi[:, 0:1]

    pred_hue_vector = torch.cat(
        [
            torch.sin(pred_angle),
            torch.cos(pred_angle)
        ],
        dim=1
    )

    target_hue_vector = torch.cat(
        [
            torch.sin(target_angle),
            torch.cos(target_angle)
        ],
        dim=1
    )

    hue_mse_map = (
        pred_hue_vector - target_hue_vector
    ).square().mean(dim=1, keepdim=True)

    hue_l1_map = (
        pred_hue_vector - target_hue_vector
    ).abs().mean(dim=1, keepdim=True)

    # Hue matters less when target saturation is near zero.
    hue_weight = target_hsi[:, 1:2].detach()

    hue_mse = (
        hue_mse_map * hue_weight
    ).sum() / (hue_weight.sum() + eps)

    hue_l1 = (
        hue_l1_map * hue_weight
    ).sum() / (hue_weight.sum() + eps)

    si_mse = F.mse_loss(
        pred_hsi[:, 1:3],
        target_hsi[:, 1:3]
    )

    si_l1 = F.l1_loss(
        pred_hsi[:, 1:3],
        target_hsi[:, 1:3]
    )

    loss_mse = (hue_mse + 2.0 * si_mse) / 3.0
    loss_l1 = (hue_l1 + 2.0 * si_l1) / 3.0

    return loss_mse, loss_l1


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()


def gradient_loss(pred, target):
    c = pred.shape[1]
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=pred.dtype, device=pred.device)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=pred.dtype, device=pred.device)
    kx = kx.view(1,1,3,3).repeat(c,1,1,1)
    ky = ky.view(1,1,3,3).repeat(c,1,1,1)
    pgx = F.conv2d(pred, kx, padding=1, groups=c)
    pgy = F.conv2d(pred, ky, padding=1, groups=c)
    tgx = F.conv2d(target, kx, padding=1, groups=c)
    tgy = F.conv2d(target, ky, padding=1, groups=c)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def compute_training_losses(
    final_prediction,
    aux_rgb,
    aux_lab,
    aux_hsi,
    rgb_target,
    lab_target,
    hsi_target,
    mse_criterion,
    l1_criterion,
    perceptual_criterion,
    ssim_module,
    current_iteration,
):
    loss_char = charbonnier_loss(
        final_prediction,
        rgb_target,
    )

    loss_l1 = l1_criterion(
        final_prediction,
        rgb_target,
    )

    loss_perceptual = perceptual_criterion(
        final_prediction,
        rgb_target,
    )

    loss_ssim = 1.0 - ssim_module(
        final_prediction,
        rgb_target,
    )

    loss_edge = gradient_loss(
        final_prediction,
        rgb_target,
    )

    perceptual_warmup = min(1.0, current_iteration / 20000)

    aux_rgb_target = F.interpolate(
        rgb_target,
        size=aux_rgb.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    aux_lab_target = F.interpolate(
        lab_target,
        size=aux_lab.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    aux_hsi_target = resize_hsi_target(
        hsi_target,
        size=aux_hsi.shape[2:],
    )

    rgb_mse = mse_criterion(
        aux_rgb,
        aux_rgb_target,
    )

    rgb_l1 = l1_criterion(
        aux_rgb,
        aux_rgb_target,
    )

    lab_mse = mse_criterion(
        aux_lab,
        aux_lab_target,
    )

    lab_l1 = l1_criterion(
        aux_lab,
        aux_lab_target,
    )

    hsi_mse, hsi_l1 = (
        compute_hsi_auxiliary_losses(
            aux_hsi,
            aux_hsi_target,
        )
    )

    final_loss = (
        0.8 * loss_char
        + 0.2 * loss_l1
        + 0.4 * perceptual_warmup * loss_perceptual
        + 0.2 * loss_ssim
        + 0.1 * loss_edge
    )

    rgb_aux_loss = (
        rgb_mse + 0.25 * rgb_l1
    )

    lab_aux_loss = (
        lab_mse + 0.25 * lab_l1
    )

    hsi_aux_loss = (
        hsi_mse + 0.25 * hsi_l1
    )

    total_loss = (
        final_loss
        + 0.4 * rgb_aux_loss
        + 0.5 * lab_aux_loss
        + 0.8 * hsi_aux_loss
    )

    components = {
        "charbonnier": loss_char,
        "l1": loss_l1,
        "perceptual": loss_perceptual,
        "ssim": loss_ssim,
        "edge": loss_edge,
        "rgb_aux": rgb_aux_loss,
        "lab_aux": lab_aux_loss,
        "hsi_aux": hsi_aux_loss,
    }

    return total_loss, components
