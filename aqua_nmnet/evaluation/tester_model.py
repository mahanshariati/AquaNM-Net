import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms

from aqua_nmnet.utils.general_utils import AverageMeter
from aqua_nmnet.utils.image_metrics import compute_psnr, compute_ssim

device_id = 0

@torch.inference_mode()
def get_cand_err(model, cand, args):
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('starting test for architecture:', cand)
    model.cuda(device_id).eval()

    all_files = sorted([
        f for f in os.listdir(args['image_path'])
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])
    image_files = [f for f in all_files]

    if not image_files:
        raise RuntimeError(
            "No images were found in the architecture-search "
            f"directory: {args['image_path']}"
        )

    psnr_record = AverageMeter()
    ssim_record = AverageMeter()
    
    search_size = int(args.get("image_size", 256))

    for fname in image_files:
        img_path = os.path.join(args['image_path'], fname)
        gt_path  = os.path.join(args['gt_path'],  fname)
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(
                f"Ground-truth image not found: {gt_path}"
            )

        img = Image.open(img_path).convert("RGB")
        gt  = Image.open(gt_path).convert("RGB")
    
        img = img.resize(
            (search_size, search_size),
            Image.Resampling.BILINEAR
        )
        gt = gt.resize(
            (search_size, search_size),
            Image.Resampling.BILINEAR
        )
    
        img_var = img_transform(img).unsqueeze(0).cuda(device_id, non_blocking=True)  
        
        final, _, _, _ = model(img_var, cand)

        prediction = torch.clamp(final, 0, 1)
        
        prediction = (
            prediction[0]
            .permute(1, 2, 0)
            .float()
            .cpu()
            .numpy()
        )
    
        gt_np = np.asarray(gt)
    
        psnr = compute_psnr(
            prediction * 255.0,
            gt_np
        )
        ssim = compute_ssim(
            prediction * 255.0,
            gt_np
        )
        
        psnr_record.update(psnr)
        ssim_record.update(ssim)

    print('Architecture:', cand)
    print('  PSNR: {:.5f}  SSIM: {:.5f}'.format(psnr_record.avg, ssim_record.avg))

    return psnr_record.avg, ssim_record.avg
