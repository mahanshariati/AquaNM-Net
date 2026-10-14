import torch.nn as nn

from aqua_nmnet.models.trans_block_normal import TransformerBlock
from aqua_nmnet.models.trans_block_dual import TransformerBlock_dual
from aqua_nmnet.models.trans_block_eca import TransformerBlock_eca
from aqua_nmnet.models.trans_block_sa import TransformerBlock_sa
from aqua_nmnet.models.trans_block_sge import TransformerBlock_sge


class _ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)
        

class _DilatedConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class _ResidualSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                channels // reduction,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                channels // reduction,
                channels,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape

        attention = self.avg_pool(x).view(
            batch_size,
            channels,
        )
        attention = self.fc(attention).view(
            batch_size,
            channels,
            1,
            1,
        )

        return x * attention + x
        

class ChoiceBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.resblock_skipconnect = nn.Identity()
        self.resblock_dil2Conv = _DilatedConvBlock(channels)
        self.resblock_conv = _ConvBlock(channels)
        self.resblock_se = _ResidualSEBlock(channels)

        transformer_kwargs = {
            "dim": channels,
            "num_heads": 2,
            "ffn_expansion_factor": 2.66,
            "bias": False,
            "LayerNorm_type": "WithBias",
        }

        self.resblock_trans_normal = nn.Sequential(TransformerBlock(**transformer_kwargs))
        self.resblock_trans_dual = nn.Sequential(TransformerBlock_dual(**transformer_kwargs))
        self.resblock_trans_eca = nn.Sequential(TransformerBlock_eca(**transformer_kwargs))
        self.resblock_trans_sa = nn.Sequential(TransformerBlock_sa(**transformer_kwargs))
        self.resblock_trans_sge = nn.Sequential(TransformerBlock_sge(**transformer_kwargs))

    def forward(self, x, select):
        if select == 0:
            return self.resblock_skipconnect(x)
        if select == 1:
            return x + self.resblock_dil2Conv(x)
        if select == 2:
            return x + self.resblock_conv(x)
        if select == 3:
            return self.resblock_se(x)
        if select == 4:
            return self.resblock_trans_normal(x)
        if select == 5:
            return self.resblock_trans_dual(x)
        if select == 6:
            return self.resblock_trans_eca(x)
        if select == 7:
            return self.resblock_trans_sa(x)
        if select == 8:
            return self.resblock_trans_sge(x)

        raise ValueError(
            f"Invalid operation selection {select}. "
            "Expected an integer from 0 to 8."
        )
