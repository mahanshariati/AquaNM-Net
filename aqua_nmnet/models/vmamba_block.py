import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba2


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class CrossScan(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        xs = []
        xs.append(x.reshape(B, C, -1))
        xs.append(x.flip(-1).reshape(B, C, -1))
        xs.append(x.transpose(-2, -1).reshape(B, C, -1))
        xs.append(x.transpose(-2, -1).flip(-1).reshape(B, C, -1))
        return torch.stack(xs, dim=1)


class CrossMerge(nn.Module):
    def forward(self, xs, H, W):
        B, _, C, L = xs.shape
        x0 = xs[:, 0].view(B, C, H, W)
        x1 = xs[:, 1].view(B, C, H, W).flip(-1)
        x2 = xs[:, 2].view(B, C, W, H).transpose(-2, -1)
        x3 = xs[:, 3].view(B, C, W, H).flip(-1).transpose(-2, -1)
        return (x0 + x1 + x2 + x3) / 4


class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        d_inner = dim * expand
        headdim = 64
        
        if d_inner % headdim != 0:
            d_ssm = (d_inner // headdim) * headdim
        else:
            d_ssm = None

        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            d_ssm=d_ssm,
            use_mem_eff_path=False,
        )
        self.scan = CrossScan()
        self.merge = CrossMerge()

        self.norm2 = LayerNorm2d(dim)
        hidden = int(dim * 2.66)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        B, C, H, W = x.shape
        xs = self.scan(x)
        xs = rearrange(xs, 'b n c l -> (b n) l c')
        xs = self.mamba(xs)
        xs = rearrange(xs, '(b n) l c -> b n c l', b=B, n=4)
        x = self.merge(xs, H, W)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x + residual
        return x
