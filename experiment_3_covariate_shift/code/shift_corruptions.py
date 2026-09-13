# -*- coding: utf-8 -*-
"""
Covariate-shift corruptions for Experiment 3 (matches the paper's shift table).

Photometric corruptions alter pixel statistics only (image); geometric corruptions are applied to
BOTH image and mask (nearest-neighbour on the mask). Geometric ops that reveal empty regions fill
the image with 0 and the mask with IGNORE (255) so undefined pixels are not scored.

Two grids:
  GROWLI_SHIFTS  (Exp 3a, SegFormer-B2 / GrowliFlower-L): 20 continuous levels per corruption.
                 5 corruption functions are provided; the paper uses 4 (=80 conditions) — set
                 GROWLI_ACTIVE to the 4 you want (default lists all five; drop rotation OR zoom).
  BUP20_SHIFTS   (Exp 3b, Mask2Former / BUP20): 3 discrete levels per corruption
                 (hflip is a single binary level) -> 7*3 - 2 = 19 conditions.

All functions take (pil_img, seg_np or None, param) and return (pil_img, seg_np). seg_np is a uint8
HxW label map (or None for image-only use). IGNORE=255.
"""
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms.v2.functional as TF

IGNORE = 255


# ---------------------------------------------------------------- photometric (image only)
def brightness(img, seg, factor):
    return ImageEnhance.Brightness(img).enhance(factor), seg


def contrast(img, seg, factor):
    return ImageEnhance.Contrast(img).enhance(factor), seg


def gaussian_noise(img, seg, sigma):
    t = TF.to_image(img)
    t = TF.to_dtype(t, torch.float32, scale=True)
    t = (t + torch.randn_like(t) * sigma).clamp(0, 1)
    return TF.to_pil_image(t), seg


def gaussian_blur(img, seg, sigma):
    if sigma <= 0:
        return img, seg
    k = int(2 * round(2 * sigma) + 1)
    k = k if k % 2 == 1 else k + 1
    return TF.gaussian_blur(img, kernel_size=k, sigma=float(sigma)), seg


# ---------------------------------------------------------------- geometric (image + mask, NN)
def _mask_pil(seg):
    return Image.fromarray(seg.astype(np.uint8))


def rotation(img, seg, degrees):
    img_r = img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(0, 0, 0), expand=False)
    if seg is None:
        return img_r, None
    seg_r = _mask_pil(seg).rotate(degrees, resample=Image.NEAREST, fillcolor=IGNORE, expand=False)
    return img_r, np.array(seg_r, dtype=np.uint8)


def zoom(img, seg, scale):
    """Zoom IN by `scale` (>1): magnify then centre-crop back to original size."""
    if scale <= 1.0:
        return img, seg
    W, H = img.size
    nw, nh = int(round(W * scale)), int(round(H * scale))
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    img_z = img.resize((nw, nh), Image.BILINEAR).crop((x0, y0, x0 + W, y0 + H))
    if seg is None:
        return img_z, None
    seg_z = (_mask_pil(seg).resize((nw, nh), Image.NEAREST).crop((x0, y0, x0 + W, y0 + H)))
    return img_z, np.array(seg_z, dtype=np.uint8)


def translation(img, seg, frac):
    """Horizontal shift by frac * width; revealed strip -> image 0, mask IGNORE."""
    W, H = img.size
    dx = int(round(frac * W))
    affine = (1, 0, -dx, 0, 1, 0)
    img_t = img.transform((W, H), Image.AFFINE, affine, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if seg is None:
        return img_t, None
    seg_t = _mask_pil(seg).transform((W, H), Image.AFFINE, affine, resample=Image.NEAREST, fillcolor=IGNORE)
    return img_t, np.array(seg_t, dtype=np.uint8)


def horizontal_flip(img, seg, _param=None):
    img_f = TF.horizontal_flip(img)
    if seg is None:
        return img_f, None
    return img_f, np.ascontiguousarray(np.fliplr(seg))


PHOTOMETRIC = {"brightness", "contrast", "gaussian_noise", "gaussian_blur"}
FUNCS = {"brightness": brightness, "contrast": contrast, "gaussian_noise": gaussian_noise,
         "gaussian_blur": gaussian_blur, "rotation": rotation, "zoom": zoom,
         "translation": translation, "hflip": horizontal_flip}


def _lin(lo, hi, n):
    return [round(lo + i * (hi - lo) / (n - 1), 4) for i in range(n)]


# Exp 3a — GrowliFlower-L: 20 continuous levels each. Provide all five; paper uses four (=80).
GROWLI_SHIFTS = {
    "brightness":     _lin(1.0, 2.0, 20),      # Δ=0.05 (factor)
    "contrast":       _lin(1.0, 2.0, 20),      # Δ=0.05
    "gaussian_noise": _lin(0.0, 0.285, 20),    # Δ=0.015 (σ)
    "rotation":       _lin(0.0, 85.5, 20),     # Δ=4.5°
    "zoom":           _lin(1.0, 1.475, 20),    # Δ=0.025 (scale)
}
# The paper reports 80 = 4×20. Training aug = rotation + hflip ONLY, so we evaluate on the OTHER
# shift types (rotation is excluded from the test grid because it's a training augmentation).
GROWLI_ACTIVE = ["brightness", "contrast", "gaussian_noise", "zoom"]

# Exp 3b — BUP20: 3 discrete levels (hflip single) -> 7*3 - 2 = 19.
BUP20_SHIFTS = {
    "brightness":     [0.50, 0.65, 0.80],
    "contrast":       [0.50, 0.65, 0.80],
    "gaussian_noise": [0.05, 0.10, 0.20],
    "gaussian_blur":  [1, 2, 3],               # σ in px
    "zoom":           [1.10, 1.25, 1.50],
    "translation":    [0.05, 0.10, 0.15],      # fraction of width
    "hflip":          [None],                  # single binary level
}


def iter_conditions(grid, active=None):
    """Yield (name, level_index, param) for a grid dict (optionally restricted to `active`)."""
    names = active if active is not None else list(grid)
    for name in names:
        for i, p in enumerate(grid[name]):
            yield name, i, p


def apply(name, img, seg, param):
    return FUNCS[name](img, seg, param)
