import torch
import torch.nn as nn
import torch.nn.functional as F

from aqua_nmnet.models.psmcse import PSMCSE
from aqua_nmnet.models.decoder import MultiScaleDecoder
from aqua_nmnet.models.encoder import MultiBranchEncoder
from aqua_nmnet.models.fusion import ResidualConvSE, SelectiveFusion
from aqua_nmnet.models.choice_block import ChoiceBlock

class AquaNMNet(nn.Module):
    def __init__(self, base_channels):
        super().__init__()

        self.colornet = PSMCSE()
        
        self.base = MultiBranchEncoder(base_channels)

        self.fusion1 = SelectiveFusion(base_channels)
        self.fusion2 = SelectiveFusion(base_channels * 2)
        self.fusion3 = SelectiveFusion(base_channels * 4)
        self.fusion4 = SelectiveFusion(base_channels * 8)

        self.msi1 = ResidualConvSE(base_channels)
        self.msi2 = ResidualConvSE(base_channels * 2)
        self.msi3 = ResidualConvSE(base_channels * 4)
        self.msi4 = ResidualConvSE(base_channels * 8)

        self.search = MultiScaleDecoder(base_channels)

        self.refine = ChoiceBlock(base_channels)

        self.de_predict = nn.Sequential(
            nn.Conv2d(base_channels, 3, kernel_size=1, stride=1)
        )

        self.de_predict_rgb = nn.Sequential(
            nn.Conv2d(base_channels * 8, 3, kernel_size=1, stride=1)
        )
        self.de_predict_lab = nn.Sequential(
            nn.Conv2d(base_channels * 8, 3, kernel_size=1, stride=1)
        )
        self.de_predict_hsi = nn.Sequential(
            nn.Conv2d(base_channels * 8, 3, kernel_size=1, stride=1)
        )

    def forward(self, img, architecture):
        if len(architecture) != 13:
            raise ValueError(
                "AquaNMNet expects exactly 13 architecture "
                f"choices, got {len(architecture)}."
            )
            
        _, _, original_h, original_w = img.shape

        pad_h = (8 - original_h % 8) % 8
        pad_w = (8 - original_w % 8) % 8

        if pad_h > 0 or pad_w > 0:
            mode = "reflect"
            if pad_h >= original_h or pad_w >= original_w:
                mode = "replicate"

            img = F.pad(
                img,
                (0, pad_w, 0, pad_h),
                mode=mode
            )

        rgb, lab, hsi = self.colornet(img)
        
        level1_rgb, level2_rgb, level3_rgb, level4_rgb, \
        level1_lab, level2_lab, level3_lab, level4_lab, \
        level1_hsi, level2_hsi, level3_hsi, level4_hsi = \
            self.base(rgb, lab, hsi, architecture[:12])

        fused_level1 = self.msi1(self.fusion1(level1_rgb, level1_lab, level1_hsi))
        fused_level2 = self.msi2(self.fusion2(level2_rgb, level2_lab, level2_hsi))
        fused_level3 = self.msi3(self.fusion3(level3_rgb, level3_lab, level3_hsi)) 
        fused_level4 = self.msi4(self.fusion4(level4_rgb, level4_lab, level4_hsi))

        decoded_features = self.search(fused_level4, fused_level3, fused_level2, fused_level1)
        refined_features = self.refine(decoded_features, architecture[12])


        mid_rgb = self.de_predict_rgb(level4_rgb)
        mid_lab = self.de_predict_lab(level4_lab)
        mid_hsi = self.de_predict_hsi(level4_hsi)

        final = torch.sigmoid(self.de_predict(refined_features))
        final = final[:, :, :original_h, :original_w]

        return final, mid_rgb, mid_lab, mid_hsi
