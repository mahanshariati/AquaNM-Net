import cv2
import numpy as np
import os
import torch
from PIL import Image
from skimage import img_as_ubyte
from torchvision import transforms

from aqua_nmnet.utils.general_utils import AverageMeter
from aqua_nmnet.models.model import AquaNMNet
from aqua_nmnet.utils.image_metrics import compute_psnr, compute_ssim
from aqua_nmnet.config import (
    CHECKPOINT_DIR,
    LSUI_INPUT_DIR,
    LSUI_TARGET_DIR,
    TEST_SPLIT_FILE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_to_tensor = transforms.ToTensor()

def read_list_txt(txt_path):
    names = []
    with open(txt_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                names.append(ln)
    return names


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


def validate_arch(best_arch):
    best_arch = [int(x) for x in best_arch]

    if len(best_arch) != 13:
        raise ValueError(
            f"Architecture must contain exactly 13 choices, "
            f"got {len(best_arch)}."
        )

    invalid = [
        x for x in best_arch
        if x < 0 or x > 8
    ]

    if invalid:
        raise ValueError(
            f"Architecture choices must be in [0, 8]. "
            f"Invalid values: {invalid}"
        )

    return best_arch


def materialize_selected_subnet(model, best_arch):
    best_arch = validate_arch(best_arch)

    blocks_and_choices = [
        (
            model.base.block1,
            {best_arch[0], best_arch[4], best_arch[8]}
        ),
        (
            model.base.block2,
            {best_arch[1], best_arch[5], best_arch[9]}
        ),
        (
            model.base.block3,
            {best_arch[2], best_arch[6], best_arch[10]}
        ),
        (
            model.base.block4,
            {best_arch[3], best_arch[7], best_arch[11]}
        ),
        (
            model.refine,
            {best_arch[12]}
        ),
    ]

    for block, keep_choices in blocks_and_choices:
        for choice_idx, attr_name in CHOICE_TO_ATTR.items():
            if choice_idx not in keep_choices:
                if hasattr(block, attr_name):
                    delattr(block, attr_name)

    return model


def run_inference(
    ckpt_path,
    best_arch=None,
    out_dir=None,
    use_test_list=True,
    image_size=256,
    base_channels=48,
):

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}"
        )
        
    if best_arch is None:
        raise ValueError(
            "best_arch must be provided explicitly for fixed-subnet inference."
        )
    
    best_arch = validate_arch(best_arch)
    
    if out_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        out_dir = os.path.join(CHECKPOINT_DIR, f"LSUI_infer_{ckpt_name}")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading checkpoint:", ckpt_path)

    model = AquaNMNet(base_channels=base_channels)
    
    state = torch.load(
        ckpt_path,
        map_location="cpu"
    )
    
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    
    model.load_state_dict(
        state,
        strict=True
    )
    
    model = materialize_selected_subnet(
        model,
        best_arch
    )
    
    model = model.to(device)
    model.eval()

    # which filenames to evaluate
    if use_test_list:
        file_list = read_list_txt(TEST_SPLIT_FILE)
    else:
        file_list = sorted(os.listdir(LSUI_INPUT_DIR))

    psnr_record = AverageMeter()
    ssim_record = AverageMeter()

    with torch.no_grad():
        for fname in file_list:
            inp_path = os.path.join(LSUI_INPUT_DIR, fname)
            gt_path  = os.path.join(LSUI_TARGET_DIR,  fname)

            if (not os.path.isfile(inp_path)) or (not os.path.isfile(gt_path)):
                # skip if missing (shouldn’t happen if LSUI is clean)
                continue

            inp = Image.open(inp_path).convert("RGB")

            inp = inp.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )

            inp = (
                image_to_tensor(inp)
                .unsqueeze(0)
                .to(device)
            )
            
            pred, _, _, _ = model(inp, best_arch)
            pred = torch.clamp(pred, 0, 1)
            
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()

            gt = Image.open(gt_path).convert("RGB")

            gt = gt.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )
            
            gt = np.array(gt)

            psnr = compute_psnr(pred_np * 255.0, gt)
            ssim = compute_ssim(pred_np * 255.0, gt)
            psnr_record.update(psnr)
            ssim_record.update(ssim)

            # save png result
            out_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(
                os.path.join(out_dir, out_name),
                cv2.cvtColor(img_as_ubyte(pred_np), cv2.COLOR_RGB2BGR)
            )

    if psnr_record.count == 0:
        raise RuntimeError(
            "No image pairs were evaluated."
        )
    
    print("Done.")
    print("Avg PSNR:", psnr_record.avg)
    print("Avg SSIM:", ssim_record.avg)
    print("Saved outputs to:", out_dir)
    return psnr_record.avg, ssim_record.avg, out_dir
