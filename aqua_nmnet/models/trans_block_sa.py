import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pdb import set_trace as stx


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


def choose_sa_groups(head_channels, max_groups=8):
    """
    Choose the largest group count that allows every group
    to be split evenly into channel and spatial branches.
    """
    for groups in range(
        min(max_groups, head_channels // 2),
        0,
        -1
    ):
        if head_channels % (2 * groups) == 0:
            return groups

    raise ValueError(
        f"Cannot create symmetric SA groups for "
        f"{head_channels} channels."
    )


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention_sa(nn.Module):
    def __init__(self, num_heads, groups, channel):
        super(Attention_sa, self).__init__()

        if channel % groups != 0:
            raise ValueError(
                f"Head channels ({channel}) must be divisible "
                f"by groups ({groups})."
            )

        channels_per_group = channel // groups

        if channels_per_group % 2 != 0:
            raise ValueError(
                f"Channels per group ({channels_per_group}) "
                f"must be even."
            )

        
        self.num_heads = num_heads

        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        heads = x.chunk(self.num_heads, dim=1)
        outputs = []
        for head in heads:
            b, c, h, w = head.shape

            head = head.reshape(b * self.groups, -1, h, w)
            x_0, x_1 = head.chunk(2, dim=1)

            # channel attention
            xn = self.avg_pool(x_0)
            xn = self.cweight * xn + self.cbias
            xn = x_0 * self.sigmoid(xn)

            # spatial attention
            xs = self.gn(x_1)
            xs = self.sweight * xs + self.sbias
            xs = x_1 * self.sigmoid(xs)

            # concatenate along channel axis
            out = torch.cat([xn, xs], dim=1)
            out = out.reshape(b, -1, h, w)

            out = self.channel_shuffle(out, 2)

            outputs.append(out)

        # Two different branches of ECA module
        output = torch.cat(outputs, dim=1)

        return output


class TransformerBlock_sa(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_sa, self).__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        head_channels = dim // num_heads
        groups = choose_sa_groups(
            head_channels,
            max_groups=8
        )

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_sa(num_heads, groups, head_channels)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
