"""Mask resizing for visual sheets. MAX-POOL when shrinking, NEAREST only when enlarging - never NEAREST down.

NEAREST-downsampling a mask (or a 1-px outline) drops pixels, which drew real, solid defect masks as broken specks
(2026-09-25). Every sheet script resizes masks through here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage as nd


def mask_resize(m: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Binary mask -> (h, w) bool. Max-pool if shrinking in either axis, NEAREST if only enlarging."""
    h, w = (size, size) if isinstance(size, int) else size
    b = np.asarray(m) > (127 if np.asarray(m).dtype == np.uint8 else 0.5)
    if h < b.shape[0] or w < b.shape[1]:
        t = torch.from_numpy(b.astype(np.float32))[None, None]
        return F.adaptive_max_pool2d(t, (h, w))[0, 0].numpy() > 0.5
    return np.array(Image.fromarray(b.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127


def outline_on(a: np.ndarray, sub: np.ndarray, colour, width: int = 1) -> None:
    """Draw a thin outline just OUTSIDE mask `sub` onto display array `a` (edge computed at DISPLAY resolution)."""
    disp = mask_resize(sub, a.shape[:2])
    edge = nd.binary_dilation(disp, iterations=width) & ~disp
    a[edge] = (0.6 * np.array(colour) + 0.4 * a[edge]).astype(np.uint8)
