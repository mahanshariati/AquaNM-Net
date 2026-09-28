import torch

def _t(data):
    return torch.tensor(data, requires_grad=False, dtype=torch.float32)

def _mul(coeffs, image):
    coeffs = coeffs.to(image.device).view(3, 3, 1, 1)
    return torch.nn.functional.conv2d(image, coeffs)


_RGB_TO_XYZ = {
    "srgb": _t([[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]]),

    "prophoto": _t([[0.7976749, 0.1351917, 0.0313534],
                    [0.2880402, 0.7118741, 0.0000857],
                    [0.0000000, 0.0000000, 0.8252100]])

    }


_XYZ_TO_RGB = {
    "srgb": _t([[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]]),

    "prophoto": _t([[ 1.3459433, -0.2556075, -0.0511118],
                    [-0.5445989,  1.5081673,  0.0205351],
                    [0.0000000,  0.0000000,  1.2118128]])
    }


WHITE_POINTS = {item[0]: _t(item[1:]).view(1, 3, 1, 1) for item in [
    ("a", 1.0985, 1.0000, 0.3558),
    ("b", 0.9807, 1.0000, 1.1822),
    ("e", 1.0000, 1.0000, 1.0000),
    ("d50", 0.9642, 1.0000, 0.8251),
    ("d55", 0.9568, 1.0000, 0.9214),
    ("d65", 0.9504, 1.0000, 1.0888),
    ("icc", 0.9642, 1.0000, 0.8249)
]}


_EPSILON = 0.008856
_KAPPA = 903.3
_XYZ_TO_LAB = _t([[0.0, 116.0, 0.], [500.0, -500.0, 0.], [0.0, 200.0, -200.0]])
_LAB_TO_XYZ = _t([[1.0 / 116.0, 1.0 / 500.0, 0], [1.0 / 116.0, 0, 0], [1.0 / 116.0, 0, -1.0 / 200.0]])
_LAB_OFF = _t([16.0, 0.0, 0.0]).view(1, 3, 1, 1)


# -------------------------------------------------------------------------
# HSI conversion
# -------------------------------------------------------------------------
def rgb2hsi(img, eps=1e-6):
    img = torch.clamp(img, 0.0, 1.0)

    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    # Intensity
    rgb_sum = r + g + b
    intensity = rgb_sum / 3.0

    # Saturation
    min_rgb = img.min(dim=1).values
    safe_sum = rgb_sum.clamp_min(eps)
    saturation = 1.0 - (3.0 * min_rgb / safe_sum)

    # Black pixels have no saturation.
    saturation = torch.where(
        rgb_sum > eps,
        saturation,
        torch.zeros_like(saturation)
    )
    saturation = saturation.clamp(0.0, 1.0)

    # Stable circular hue calculation.
    hue_y = torch.sqrt(
        torch.tensor(3.0, device=img.device, dtype=img.dtype)
    ) * (g - b)

    hue_x = 2.0 * r - g - b

    # Hue is undefined for grayscale pixels.
    valid_hue = (
        hue_x.square() + hue_y.square()
    ) > eps ** 2

    # Avoid atan2(0, 0), whose gradient is undefined.
    safe_hue_x = torch.where(
        valid_hue,
        hue_x,
        torch.ones_like(hue_x)
    )
    safe_hue_y = torch.where(
        valid_hue,
        hue_y,
        torch.zeros_like(hue_y)
    )

    angle = torch.atan2(safe_hue_y, safe_hue_x)

    # Convert angle from [-pi, pi] to hue in [0, 1).
    hue = torch.remainder(angle, 2.0 * torch.pi)
    hue = hue / (2.0 * torch.pi)

    # Give grayscale pixels a fixed hue of zero.
    hue = torch.where(
        valid_hue,
        hue,
        torch.zeros_like(hue)
    )

    return torch.stack(
        (hue, saturation, intensity),
        dim=1
    )


def hsi2rgb(img):
    img = torch.clamp(img, 0, 1)
    h = img[:, 0, :, :]
    s = img[:, 1, :, :]
    i = img[:, 2, :, :]
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    h1 = torch.zeros_like(h)
    hi0 = (h < 1 / 3)
    hi2 = (h >= 2 / 3)
    hi1 = 1 - hi0.int() - hi2.int()
    hi1 = (hi1 == 1)
    h1[hi0] = 2 * torch.pi * h[hi0]
    h1[hi1] = 2 * torch.pi * (h[hi1] - 1 / 3)
    h1[hi2] = 2 * torch.pi * (h[hi2] - 2 / 3)
    p = i * (1 - s)
    q = i * (1 + s * torch.cos(h1) / (torch.cos(torch.pi / 3 - h1) + 1e-5))

    r[hi0] = q[hi0]
    b[hi0] = p[hi0]
    g[hi0] = 3 * i[hi0] - r[hi0] - b[hi0]

    g[hi1] = q[hi1]
    r[hi1] = p[hi1]
    b[hi1] = 3 * i[hi1] - r[hi1] - g[hi1]

    b[hi2] = q[hi2]
    g[hi2] = p[hi2]
    r[hi2] = 3 * i[hi2] - g[hi2] - b[hi2]

    r = r.unsqueeze(1)
    g = g.unsqueeze(1)
    b = b.unsqueeze(1)
    out = torch.cat((r, g, b), dim=1)
    return out

def apply_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.0031308
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, 12.92 * rgb, (1.055 * torch.pow(torch.abs(rgb1), 1 / 2.4) - 0.055))
    elif gamma is None:
        return rgb
    else:
        return torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), 1.0 / gamma)



def remove_gamma(rgb, gamma="srgb"):
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(rgb < T, rgb / 12.92, torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4))
    elif gamma is None:
        return rgb
    else:
        res = torch.pow(torch.max(rgb, rgb.new_tensor(0.0)), gamma) + \
              torch.min(rgb, rgb.new_tensor(0.0)) # very important to avoid vanishing gradients
        return res


def rgb2xyz(rgb, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = remove_gamma(rgb, gamma_correction)
    return _mul(_RGB_TO_XYZ[space], rgb)


def xyz2rgb(xyz, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    rgb = _mul(_XYZ_TO_RGB[space], xyz)
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = apply_gamma(rgb, gamma_correction)
    return rgb


def _lab_f(x):
    x1 = torch.max(x, x.new_tensor(_EPSILON))
    return torch.where(x > _EPSILON, torch.pow(x1, 1.0 / 3), (_KAPPA * x + 16.0) / 116.0)


def xyz2lab(xyz, white_point="d65"):
    xyz = xyz / WHITE_POINTS[white_point].to(xyz.device)
    f_xyz = _lab_f(xyz)
    return _mul(_XYZ_TO_LAB, f_xyz) - _LAB_OFF.to(xyz.device)


def _inv_lab_f(x):
    x3 = torch.max(x, x.new_tensor(_EPSILON)) ** 3
    return torch.where(x3 > _EPSILON, x3, (116.0 * x - 16.0) / _KAPPA)


def lab2xyz(lab, white_point="d65"):
    f_xyz = _mul(_LAB_TO_XYZ, lab + _LAB_OFF.to(lab.device))
    xyz = _inv_lab_f(f_xyz)
    return xyz * WHITE_POINTS[white_point].to(lab.device)


def rgb2lab(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    lab = xyz2lab(rgb2xyz(rgb, gamma_correction, clip_rgb, space), white_point)
    return lab


def lab2rgb(rgb, white_point="d65", gamma_correction="srgb", clip_rgb=False, space="srgb"):
    return xyz2rgb(lab2xyz(rgb, white_point), gamma_correction, clip_rgb, space)
    