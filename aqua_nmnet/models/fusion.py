import torch
import torch.nn as nn

class SelectiveFusion(nn.Module):
    def __init__(self, channels, reduction=8, min_channels=32):
        super().__init__()

        hidden_channels = max(channels // reduction, min_channels)
        
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.attn1 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.attn2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.attn3 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(self, x1, x2, x3):
        if x1.shape != x2.shape or x1.shape != x3.shape:
            raise ValueError(
                f"All inputs must have the same shape, got "
                f"{x1.shape}, {x2.shape}, {x3.shape}"
            )

        x = x1 + x2 + x3 # [B, C, H, W]

        x = self.depthwise(x) # [B, C, H, W]

        # Global Average Pooling
        z = x.mean(dim=(2, 3), keepdim=True) # [B, C, 1, 1]
        
        z = self.reduce(z) # [B, hidden_channels, 1, 1]
        z = self.act(z)
        
        a1 = self.attn1(z)
        a2 = self.attn2(z)
        a3 = self.attn3(z)
        # each: [B, C, 1, 1]
 
        attention = torch.stack([a1, a2, a3], dim=1)
        attention = torch.softmax(attention, dim=1)
        # for every channel: s1 + s2 + s3 = 1

        s1 = attention[:, 0]
        s2 = attention[:, 1]
        s3 = attention[:, 2]
        # each: [B, C, 1, 1]

        out = (x1 * s1 + x2 * s2 + x3 * s3) # [B, C, H, W]

        return out


class ResidualConvSE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.act = nn.GELU()

        self.se_reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=True) 
        self.se_expand = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        # Residual convolution
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity

        # SE attention
        w = out.mean(dim=(2,3), keepdim=True)
        w = self.se_reduce(w)
        w = self.act(w)
        w = self.se_expand(w)
        w = self.sigmoid(w)
        
        out = out * w

        return out
