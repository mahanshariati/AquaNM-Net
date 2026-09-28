import os
import random
import time

import numpy as np
import torch
from pytorch_msssim import SSIM
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from aqua_nmnet.data import transform_inputs
from aqua_nmnet.config import (
    CHECKPOINT_DIR,
    LSUI_INPUT_DIR,
    LSUI_TARGET_DIR,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
)
from aqua_nmnet.data.dataset_factory import create_aqua_dataset
from aqua_nmnet.utils.general_utils import AverageMeter, VGGPerceptualLoss, ensure_directory
from aqua_nmnet.models.model import AquaNMNet
from aqua_nmnet.training.training_losses import compute_training_losses
from aqua_nmnet.training.training_utils import (
    build_validation_architectures,
    initialize_weights,
    load_training_checkpoint,
    save_best,
    save_full_checkpoint,
    validate,
)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device_id = 0

model_to_continue_path = ""

args = {
    'choice': 9,  # select values will be in [0..8]
    'layers': 13, # 12 encoder + 1 refine
    'base_channels': 48,
    'iter_num': 500000,
    'iter_save': 5000,
    'last_iter': 0,
    'train_batch_size': 2,
    'val_batch_size': 1,
    'lr': 1e-4,
    'lr_decay': 0.9,
    'snapshot': '', # set to e.g. '56000' or 'best' or 'last' to resume; keep '' for fresh
    'image_size': 256,
    'crop_size': 256,
    'val_interval': 5000,
    'early_stop_patience': 30,
    'early_stop_min_delta': 0.01,
    'val_arch_seed': 0,
    'val_arch_type': 'balanced_panel', # 'balanced_panel' or 'all_conv'
    "seed": 2021,
}    

def _write_log(log_path, message, print_message=False):
    if print_message:
        print(message)

    with open(
        log_path,
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(message + "\n")
        
    
def train_step(
    net,
    images,
    targets,
    lab_target,
    hsi_target,
    optimizer,
    current_iteration,
    mse_criterion,
    l1_criterion,
    perceptual_criterion,
    ssim_module,
    loss_meters,
    log_path,
    device_id,
):
    images = images.cuda(device_id, non_blocking=True)
    targets = targets.cuda(device_id, non_blocking=True)
    lab_target = lab_target.cuda(device_id, non_blocking=True)
    hsi_target = hsi_target.cuda(device_id, non_blocking=True)

    architecture = np.random.randint(
        0,
        args["choice"],
        size=(args["layers"],),
        dtype=np.int64,
    )

    optimizer.zero_grad()

    final_prediction, aux_rgb, aux_lab, aux_hsi = net(images, architecture)

    total_loss, losses = compute_training_losses(
        final_prediction=final_prediction,
        aux_rgb=aux_rgb,
        aux_lab=aux_lab,
        aux_hsi=aux_hsi,
        rgb_target=targets,
        lab_target=lab_target,
        hsi_target=hsi_target,
        mse_criterion=mse_criterion,
        l1_criterion=l1_criterion,
        perceptual_criterion=perceptual_criterion,
        ssim_module=ssim_module,
        current_iteration=current_iteration,
    )

    total_loss.backward()
    optimizer.step()

    loss_meters["total"].update(
        total_loss.item(),
        images.size(0),
    )
    
    loss_meters["charbonnier"].update(
        losses["charbonnier"].item(),
        images.size(0),
    )

    loss_meters["l1"].update(
        losses["l1"].item(),
        images.size(0),
    )

    loss_meters["perceptual"].update(
        losses["perceptual"].item(),
        images.size(0),
    )

    if current_iteration % 20 == 0:
        message = (
            f"[iter {current_iteration}] "
            f"total {loss_meters['total'].avg:.6f} "
            f"(final_char {loss_meters['charbonnier'].avg:.6f}, "
            f"final_l1 {loss_meters['l1'].avg:.6f}, "
            f"final_perc {loss_meters['perceptual'].avg:.6f})"
        )
    
        _write_log(
            log_path,
            message,
            print_message=True,
        )


def main():
    seed = args["seed"]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    ckpt_path = CHECKPOINT_DIR
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    exp_name = 'AquaNMNet' + '_' + time_str
    save_dir = os.path.join(ckpt_path, exp_name)
    ensure_directory(ckpt_path)
    ensure_directory(save_dir)

    # ---------------------- DATA ----------------------
    train_joint_transform = transform_inputs.Compose([
        transform_inputs.RandomCrop(args["crop_size"]),
        transform_inputs.RandomHorizontalFlip(),
    ])
    
    val_joint_transform = transform_inputs.Compose([
        transform_inputs.Resize(args["image_size"]),
    ])
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_set = create_aqua_dataset(
        input_dir=LSUI_INPUT_DIR,
        target_dir=LSUI_TARGET_DIR,
        joint_transform=train_joint_transform,
        image_transform=img_transform,
        include_list_file=TRAIN_SPLIT_FILE,
    )
    
    val_set = create_aqua_dataset(
        input_dir=LSUI_INPUT_DIR,
        target_dir=LSUI_TARGET_DIR,
        joint_transform=val_joint_transform,
        image_transform=img_transform,
        include_list_file=VAL_SPLIT_FILE,
    )
    
    print("Train size:", len(train_set), "Val size:", len(val_set))
    
    # ---------------------- Losses ----------------------
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss().cuda(device_id)
    
    ssim_module = SSIM(
        data_range=1.0,
        size_average=True,
        channel=3,
    ).cuda(device_id)
    
    # ---------------------- Logging ----------------------
    log_path = os.path.join(
        save_dir,
        "training.log",
    )
    
    loss_meters = {
        "total": AverageMeter(),
        "charbonnier": AverageMeter(),
        "l1": AverageMeter(),
        "perceptual": AverageMeter(),
    }
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"{args}\n\n")
        
    _write_log(log_path, f"Train size: {len(train_set)}  Val size: {len(val_set)}")
    
    net = AquaNMNet(args["base_channels"]).cuda(device_id)
    net.apply(initialize_weights)

    optimizer = optim.Adam(
        net.parameters(),
        lr=args["lr"]
    )

    curr_iter = 0
    best_psnr = float("-inf")
    bad_count = 0

    if args["snapshot"] not in ("", "0") and model_to_continue_path:
        checkpoint_path = os.path.join(
            model_to_continue_path,
            f'{args["snapshot"]}.pth',
        )
    
        curr_iter, best_psnr, bad_count = (
            load_training_checkpoint(
                path=checkpoint_path,
                net=net,
                optimizer=optimizer,
                device_id=device_id,
                fallback_iteration=args["last_iter"],
            )
        )

        if curr_iter >= args["iter_num"]:
            print(
                f"Training already reached iteration {curr_iter}; "
                "nothing to resume."
            )
            return

    train_loader = DataLoader(
        train_set,
        batch_size=args['train_batch_size'],
        num_workers=4,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args["val_batch_size"],
        num_workers=2,
        shuffle=False,
    )

    fixed_val_archs = build_validation_architectures(
        choice=args["choice"],
        layers=args["layers"],
        seed=args["val_arch_seed"],
        architecture_type=args["val_arch_type"],
    )

    val_arch_log = [
        arch.tolist()
        for arch in fixed_val_archs
    ]
    
    _write_log(
        log_path,
        (
            f"Fixed validation architecture panel "
            f"({len(fixed_val_archs)} architectures): "
            f"{val_arch_log}"
        ),
    )


    while True:
        for data in train_loader:

            optimizer.param_groups[0]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']

            images, targets, lab_target, hsi_target = data
            train_step(
                net=net,
                images=images,
                targets=targets,
                lab_target=lab_target,
                hsi_target=hsi_target,
                optimizer=optimizer,
                current_iteration=curr_iter,
                mse_criterion=criterion_mse,
                l1_criterion=criterion_l1,
                perceptual_criterion=criterion_perceptual,
                ssim_module=ssim_module,
                loss_meters=loss_meters,
                log_path=log_path,
                device_id=device_id,
            )
            curr_iter += 1

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...')
                torch.save(net.state_dict(), os.path.join(save_dir, f'{curr_iter}.pth'))


            if len(val_set) > 0 and (curr_iter % args['val_interval'] == 0):
                val_psnr, val_l1 = validate(net, val_loader, fixed_val_archs, device_id)
                validation_message = (
                    f"[iter {curr_iter}][val], "
                    f"PSNR {val_psnr:.4f}, "
                    f"L1 {val_l1:.6f}, "
                    f"best {best_psnr:.4f}"
                )
                _write_log(log_path, validation_message, print_message=True)

                if val_psnr > (best_psnr + args['early_stop_min_delta']):
                    best_psnr = val_psnr
                    bad_count = 0
                    save_best(net, optimizer, curr_iter, best_psnr, bad_count, save_dir)

                    best_message = (
                        f"[iter {curr_iter}] New BEST PSNR: "
                        f"{best_psnr:.4f} -> saved best.pth"
                    )
                    _write_log(log_path, best_message, print_message=True)
                else:
                    bad_count += 1
                    no_improvement_message = (
                        f"[iter {curr_iter}] No improvement "
                        f"({bad_count}/{args['early_stop_patience']})"
                    )
                    _write_log(log_path, no_improvement_message, print_message=True)

                    if bad_count >= args['early_stop_patience']:
                        early_stop_message = (
                            f"EARLY STOP at iter {curr_iter}. "
                            f"Best PSNR = {best_psnr:.4f}"
                        )
                        _write_log(log_path, early_stop_message, print_message=True)
                        
                        save_full_checkpoint(
                            path=os.path.join(save_dir, "last.pth"),
                            net=net,
                            optimizer=optimizer,
                            curr_iter=curr_iter,
                            best_psnr=best_psnr,
                            bad_count=bad_count
                        )
                        return

            if curr_iter % args["iter_save"] == 0:
                save_full_checkpoint(
                    path=os.path.join(save_dir, "last.pth"),
                    net=net,
                    optimizer=optimizer,
                    curr_iter=curr_iter,
                    best_psnr=best_psnr,
                    bad_count=bad_count
                )

                print(
                    f"last.pth is the full checkpoint corresponding "
                    f"to {curr_iter}.pth"
                )            

            if curr_iter >= args['iter_num']:
                torch.save(net.state_dict(), os.path.join(save_dir, f'{curr_iter}.pth'))
                save_full_checkpoint(
                    path=os.path.join(save_dir, "last.pth"),
                    net=net,
                    optimizer=optimizer,
                    curr_iter=curr_iter,
                    best_psnr=best_psnr,
                    bad_count=bad_count
                )

                return

if __name__ == "__main__":
    main()