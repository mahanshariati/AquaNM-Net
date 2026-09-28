import torch
import torch.nn as nn

from aqua_nmnet.models.vmamba_block import MambaBlock


class MultiScaleDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(dim*8, dim*16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(dim*4, dim*8, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.PixelShuffle(2)
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.PixelShuffle(2)
        )

        self.conv_cat3 = nn.Conv2d(dim*8, dim*4, kernel_size=1, bias=False)
        self.conv_cat2 = nn.Conv2d(dim*4, dim*2, kernel_size=1, bias=False)
        self.conv_cat1 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=False)

        self.decoder_bottleneck = MambaBlock(dim * 8)
        self.decoder_block3 = MambaBlock(dim * 4)
        self.decoder_block2 = MambaBlock(dim * 2)
        self.decoder_block1 = MambaBlock(dim)

    def forward(self, level4, level3, level2, level1):
        level4    = self.decoder_bottleneck(level4)
        de_level3 = self.conv_up3(level4)
        de_level3 = torch.cat([de_level3, level3], 1)
        de_level3 = self.conv_cat3(de_level3)
        de_level3 = self.decoder_block3(de_level3)

        de_level2 = self.conv_up2(de_level3)
        de_level2 = torch.cat([de_level2, level2], 1)
        de_level2 = self.conv_cat2(de_level2)
        de_level2 = self.decoder_block2(de_level2)

        de_level1 = self.conv_up1(de_level2)
        de_level1 = torch.cat([de_level1, level1], 1)
        de_level1 = self.conv_cat1(de_level1)
        de_level1 = self.decoder_block1(de_level1)

        return de_level1
