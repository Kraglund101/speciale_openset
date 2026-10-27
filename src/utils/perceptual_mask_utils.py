"""Perceptual mask refinement utilities.

Pipeline: VGG perceptual distance → roundtripped dilated mask → per-component
thresholding → birefnet foreground intersection.

Used to compute refined binary masks that highlight where the generative model
actually changed the image, removing artifacts from the placed mask's dilation.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision import models

from src.utils.mask_utils import downsample_mask_maxpool

# VGG16 hook layers: relu1_2, relu2_2, relu3_3, relu4_3
HOOK_LAYERS = [3, 8, 15, 22]

# ImageNet normalization for VGG
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

# Default Gaussian blur sigma for perceptual distance smoothing
SIGMA = 4.0

# Default per-component threshold factor
THRESHOLD_FACTOR = 0.65


def build_vgg_features(device: str = "cpu") -> nn.Module:
    """Load VGG16 and register hooks on relu1_2, relu2_2, relu3_3, relu4_3.

    Returns a wrapper module that, given a [B,3,H,W] tensor (ImageNet-normed),
    returns a list of 4 feature maps.
    """
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False

    class VGGFeatures(nn.Module):
        def __init__(self, vgg_features: nn.Module):
            super().__init__()
            self.slices = nn.ModuleList()
            prev = 0
            for idx in HOOK_LAYERS:
                self.slices.append(nn.Sequential(*list(vgg_features.children())[prev:idx + 1]))
                prev = idx + 1

        @torch.no_grad()
        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            feats = []
            for s in self.slices:
                x = s(x)
                feats.append(x)
            return feats

    model = VGGFeatures(vgg).to(device)
    model.eval()
    return model


def _normalize_for_vgg(img: torch.Tensor) -> torch.Tensor:
    """Normalize [0,1] RGB tensor to ImageNet stats for VGG."""
    mean = torch.tensor(VGG_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(VGG_STD, device=img.device).view(1, 3, 1, 1)
    return (img - mean) / std


def compute_perceptual_distance(
    vgg: nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    device: str = "cpu",
    size: int = 512,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Compute VGG perceptual distance map between two images.

    Distance = L2 / C at each layer, upsampled to `size`, averaged, Gaussian-blurred.

    Args:
        vgg: VGG feature extractor from build_vgg_features().
        img_a: [B,3,H,W] in [0,1] (generated image).
        img_b: [B,3,H,W] in [0,1] (canvas/normal image).
        device: torch device.
        size: output spatial resolution.
        sigma: Gaussian blur sigma for smoothing.

    Returns:
        np.ndarray [size, size] perceptual distance map.
    """
    # Resize to target size if needed
    if img_a.shape[-1] != size or img_a.shape[-2] != size:
        img_a = F.interpolate(img_a, size=(size, size), mode="bilinear", align_corners=False)
    if img_b.shape[-1] != size or img_b.shape[-2] != size:
        img_b = F.interpolate(img_b, size=(size, size), mode="bilinear", align_corners=False)

    a_norm = _normalize_for_vgg(img_a.to(device))
    b_norm = _normalize_for_vgg(img_b.to(device))

    feats_a = vgg(a_norm)
    feats_b = vgg(b_norm)

    dist_sum = torch.zeros(1, 1, size, size, device=device)
    for fa, fb in zip(feats_a, feats_b):
        # Euclidean distance divided by channel count (L2/C)
        diff = fa - fb  # [B, C, H, W]
        C = diff.shape[1]
        l2 = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-8)  # [B, 1, H, W]
        l2_per_ch = l2 / C
        # Upsample to target size
        l2_up = F.interpolate(l2_per_ch, size=(size, size), mode="bilinear", align_corners=False)
        dist_sum += l2_up

    dist = dist_sum[0, 0].cpu().numpy()

    # Gaussian blur
    if sigma > 0:
        dist = ndimage.gaussian_filter(dist, sigma=sigma)

    return dist


def compute_dilated_mask_512(
    mask_np: np.ndarray,
    band_mode: int = 2,
    target_size: int = 512,
) -> np.ndarray:
    """Roundtrip mask through latent resolution with band dilation.

    Pipeline: mask → maxpool to 512 → maxpool to 64 → Chebyshev dilation → nearest to 512.

    Args:
        mask_np: Binary mask [H, W] as float32 (0/1).
        band_mode: Number of Chebyshev dilation pixels at 64×64 (1 or 2).
        target_size: Output resolution (default 512).

    Returns:
        np.ndarray [target_size, target_size] binary dilated mask (0/1 float).
    """
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]

    # Down to 512
    mask_512 = downsample_mask_maxpool(mask_t, target_size)  # [1, 512, 512]

    # Down to 64
    core_64 = downsample_mask_maxpool(mask_512.unsqueeze(0), 64)  # [1, 1, 64, 64]

    # Chebyshev dilation
    dilated_64 = core_64
    for _ in range(band_mode):
        dilated_64 = F.max_pool2d(dilated_64, kernel_size=3, stride=1, padding=1)
    dilated_64 = (dilated_64 > 0.5).float()

    # Nearest upsample back to target_size
    dilated_up = F.interpolate(dilated_64, size=(target_size, target_size), mode="nearest")
    return dilated_up[0, 0].numpy()


def drop_small_blobs(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove 8-connected blobs smaller than min_px from a binary mask (min_px <= 0: unchanged)."""
    m = mask > 0.5
    if min_px <= 0 or not m.any():
        return m.astype(np.float32)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, [j + 1 for j, v in enumerate(sizes) if v >= min_px]).astype(np.float32)


def build_refined_mask(
    dist: np.ndarray,
    dilated: np.ndarray,
    factor: float = THRESHOLD_FACTOR,
    fg_mask: Optional[np.ndarray] = None,
    abs_threshold: Optional[float] = None,
    level_mask: Optional[np.ndarray] = None,
    percentile: float = 90.0,
) -> Tuple[np.ndarray, dict]:
    """Build refined binary mask using per-component p90 thresholding.

    Pipeline:
    1. Intersect dist with dilated mask to get relevant distance values
    2. Find connected components via scipy.ndimage.label
    3. Per-component: threshold at factor * p90 of component distances
    4. Union all thresholded components
    5. Intersect with birefnet foreground mask (if provided)

    Args:
        dist: [S, S] perceptual distance map.
        dilated: [S, S] binary dilated mask (0/1 float).
        factor: Threshold factor applied to p90 (default 0.65).
        fg_mask: Optional [S, S] binary foreground mask (0/1 float).
        level_mask: Optional [S, S] binary mask (e.g. the footprint). If given, p90 is taken over
            component & level_mask only (falls back to the whole component if that is empty); the threshold is
            still applied over the whole component.
        percentile: which percentile the relative level uses (default 90 = p90).

    Returns:
        (refined_mask, stats) where refined_mask is [S, S] binary float,
        stats contains per-component info.
    """
    dilated_bin = (dilated > 0.5).astype(np.float32)
    masked_dist = dist * dilated_bin

    # Connected components
    labels, n_components = ndimage.label(dilated_bin)

    refined = np.zeros_like(dilated_bin)
    component_stats = []

    for comp_id in range(1, n_components + 1):
        comp_mask = (labels == comp_id)
        comp_dists = dist[comp_mask]

        if len(comp_dists) == 0:
            continue

        level_px = comp_mask & (level_mask > 0.5) if level_mask is not None else None
        p90 = np.percentile(dist[level_px] if level_px is not None and level_px.any() else comp_dists, percentile)
        if abs_threshold is None:
            # RELATIVE (default): the most-changed part of this region - always non-empty, also when nothing
            # was painted
            thresh = factor * p90
        else:
            # ABSOLUTE: keep pixels whose distance exceeds the surroundings' typical distance by more than
            # abs_threshold (calibrated on blank repaints). Empty when nothing was painted.
            ring = ndimage.binary_dilation(comp_mask, iterations=24) & ~ndimage.binary_dilation(dilated_bin > 0.5, iterations=4)
            if fg_mask is not None and fg_mask.shape == ring.shape:
                ring &= fg_mask > 0.5
            ring_med = float(np.median(dist[ring])) if ring.sum() >= 20 else float(np.median(dist[dilated_bin < 0.5]))
            thresh = ring_med + abs_threshold

        comp_refined = (dist >= thresh) & comp_mask
        refined[comp_refined] = 1.0

        component_stats.append({
            "component_id": int(comp_id),
            "n_pixels": int(comp_mask.sum()),
            "p90": float(p90),
            "threshold": float(thresh),
            "n_refined": int(comp_refined.sum()),
        })

    # Intersect with foreground mask
    if fg_mask is not None:
        fg_bin = (fg_mask > 0.5).astype(np.float32)
        # Resize fg_mask if needed
        if fg_bin.shape != refined.shape:
            import torch
            import torch.nn.functional as F
            # MAX-POOL when shrinking (never drop FG pixels); NEAREST only when enlarging
            if fg_bin.shape[0] > refined.shape[0] or fg_bin.shape[1] > refined.shape[1]:
                fg_bin = (F.adaptive_max_pool2d(torch.from_numpy(fg_bin)[None, None], refined.shape)[0, 0].numpy() > 0.5).astype(np.float32)
            else:
                from PIL import Image
                fg_pil = Image.fromarray((fg_bin * 255).astype(np.uint8)).resize((refined.shape[1], refined.shape[0]), Image.NEAREST)
                fg_bin = (np.array(fg_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        refined = refined * fg_bin

    stats = {
        "n_components": n_components,
        "components": component_stats,
        "n_refined_total": int(refined.sum()),
        "factor": factor,
        "abs_threshold": abs_threshold,
    }

    return refined, stats
