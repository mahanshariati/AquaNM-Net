import torch.nn as nn
from aqua_nmnet.models.choice_block import ChoiceBlock


class MultiBranchEncoder(nn.Module):
    def __init__(self, base_channels):
        super().__init__()

        self.conv1_lab = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv1_hsi = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.conv2_lab = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )
        self.conv2_rgb = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )
        self.conv2_hsi = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )

        self.conv3_lab = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )
        self.conv3_rgb = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )
        self.conv3_hsi = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )

        self.conv4_lab = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )
        self.conv4_rgb = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )
        self.conv4_hsi = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.PixelUnshuffle(2),
        )

        self.block1 = ChoiceBlock(base_channels)
        self.block2 = ChoiceBlock(base_channels * 2)
        self.block3 = ChoiceBlock(base_channels * 4)
        self.block4 = ChoiceBlock(base_channels * 8)

    def _encode_branch(
        self,
        x,
        conv1,
        conv2,
        conv3,
        conv4,
        architecture,
    ):
        level1 = self.block1(conv1(x), architecture[0])
        level2 = self.block2(conv2(level1), architecture[1])
        level3 = self.block3(conv3(level2), architecture[2])
        level4 = self.block4(conv4(level3), architecture[3])

        return level1, level2, level3, level4

    def forward(self, rgb, lab, hsi, architecture):
        rgb_levels = self._encode_branch(
            rgb,
            self.conv1_rgb,
            self.conv2_rgb,
            self.conv3_rgb,
            self.conv4_rgb,
            architecture[0:4],
        )

        lab_levels = self._encode_branch(
            lab,
            self.conv1_lab,
            self.conv2_lab,
            self.conv3_lab,
            self.conv4_lab,
            architecture[4:8],
        )

        hsi_levels = self._encode_branch(
            hsi,
            self.conv1_hsi,
            self.conv2_hsi,
            self.conv3_hsi,
            self.conv4_hsi,
            architecture[8:12],
        )

        return (*rgb_levels, *lab_levels, *hsi_levels)
