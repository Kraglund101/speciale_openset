#!/usr/bin/env python3
"""Compute perceptually-refined binary masks for all 100 generated cashew anomalies.

Current default (selected 2026-04-10 after human visual inspection):
  - Backbone: DINOv3-CNX-Base distilled (timm "convnext_base.dinov3_lvd1689m"),
    same weights as the UniNet downstream model
  - Canvas calibration: α-blend roundtrip. The canvas is passed through
    VAE encode→decode, then composited with the raw canvas using the same
    alpha map the generator uses at inference (see generate.py:272). This
    gives a reference image that matches what the generator would produce
    in unchanged regions, so the perceptual distance reflects only the
    diffusion delta — not VAE reconstruction noise.
  - Distance blur: σ=3 Gaussian (balances contour smoothness vs peripheral bleed)
  - Threshold: factor × p90 per connected component, factor=0.6

Per sample:
  - Easy (000-093): DINOv3-CNX cosine distance (α-blended canvas) →
    roundtripped dilated mask → 0.6×p90 per-component thresholding →
    intersect birefnet FG mask.
  - Hard (094-099): placed_mask ∩ birefnet FG mask (no distance-based refinement;
    hard cases are full-object deformations where the placed mask is already the
    correct answer).

Rationale for the current defaults (see also thesis discussion):
  - DINOv3-CNX gave ~20% tighter masks than the original VGG16 path at the
    same threshold, with smoother contours and no patch blockiness.
  - α-blend roundtrip canvas cancels VAE bias inside the dilated band
    (~6% tighter masks) — a second-order correction but principled.
  - factor=0.6 chosen over 0.65 after visual inspection: slightly more permissive
    captures faint anomaly edges without visible noise increase.
  - σ=3 chosen as compromise: σ=4 was blurring real anomaly peripheries
    outward into a halo (loose contours); σ=0/1 produced ragged per-pixel
    thresholding artifacts. σ=3 gives smooth contiguous regions without
    the over-smoothing of σ=4.

Previously (before 2026-04-10): VGG16 perceptual distance, raw canvas (no
roundtrip), factor=0.65. Old flag semantics preserved via opt-outs
(--no-vae-roundtrip, --no-dinov3-cnx) for reproducibility of older results.

Usage:
    python scripts/compute_refined_masks.py \
        --gen-dir <run>/cashew_100_rp/generated/cfg7_vis \
        --prep-dir results/cashew_100_rp/prep \
        --output-dir <run>/cashew_100_rp/refined_masks \
        --fg-dir anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/birefnet_masks/Normal

Non-SD1.5 generators (e.g. FLUX.1 Kontext): the SD1.5-VAE α-blend reference is
wrong for them. Use ``--canvas-ref raw`` to compare against the raw canvas,
resized the way the generator resized it. ``--canvas-resize-via N`` inserts the
generator's intermediate LANCZOS resize (canvas -> NxN -> 512), e.g. 1024 for a
FLUX backend that works on a 1024x1024 canvas and saves a LANCZOS 512 copy:
    python scripts/compute_refined_masks.py --canvas-ref raw --canvas-resize-via 1024 \
        --gen-dir <run>/generated/<flux_variant> --prep-dir results/cashew_100_rp/prep \
        --output-dir <run>/refined_masks_<flux_variant> --fg-dir <birefnet Normal dir>
In raw mode each sample also logs ``bg_mae_255``: mean |gen - reference| (0-255)
far outside the dilated mask. It is ~0 when the generator pastes back raw canvas
pixels and the resize chain matches; it is clearly > 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch.nn.functional as F

from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool
from src.utils.perceptual_mask_utils import (
    drop_small_blobs,
    build_refined_mask,
    build_vgg_features,
    compute_dilated_mask_512,
    compute_perceptual_distance,
)

BAND_MODE = 2
TARGET_SIZE = 512
VAE_MODEL_ID = "runwayml/stable-diffusion-inpainting"


def _load_vae(device: str):
    """Load SD 1.5 inpainting VAE for canvas roundtrip."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_ID, subfolder="vae").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def _load_dinov3_cnx(device: str, stages=(0, 1, 2)):
    """Load DINOv3 ConvNeXt-Base (distilled) via timm and return a callable
    returning a spatial perceptual distance map.

    Uses timm's "convnext_base.dinov3_lvd1689m" — same weights UniNet uses.
    ConvNeXt gives true spatial feature maps (128x128/64x64/32x32/16x16) instead
    of a ViT patch grid, so no blockiness.

    Per stage we do cosine distance per-pixel (direction only), upsample to size,
    and average across stages. Default stages (0,1,2) = 128x128, 64x64, 32x32 —
    skipping stage 3 (16x16) which is too coarse to resolve small anomalies.
    """
    import timm as _timm
    model = _timm.create_model(
        "convnext_base.dinov3_lvd1689m", pretrained=True, features_only=True
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def _compute(img_a: torch.Tensor, img_b: torch.Tensor, size: int = 512, sigma: float = 4.0) -> np.ndarray:
        a = img_a.to(device)
        b = img_b.to(device)
        if a.shape[-1] != size or a.shape[-2] != size:
            a = F.interpolate(a, size=(size, size), mode="bilinear", align_corners=False)
        if b.shape[-1] != size or b.shape[-2] != size:
            b = F.interpolate(b, size=(size, size), mode="bilinear", align_corners=False)
        a = (a - mean) / std
        b = (b - mean) / std
        feats_a = model(a)
        feats_b = model(b)

        dist_sum = torch.zeros(1, 1, size, size, device=device)
        for i in stages:
            fa = feats_a[i]  # [1, C, H, W]
            fb = feats_b[i]
            fa_n = F.normalize(fa, dim=1)
            fb_n = F.normalize(fb, dim=1)
            cos = (fa_n * fb_n).sum(dim=1, keepdim=True)  # [1, 1, H, W]
            dist = 1.0 - cos
            dist_up = F.interpolate(dist, size=(size, size), mode="bilinear", align_corners=False)
            dist_sum = dist_sum + dist_up
        dist_sum = dist_sum / len(stages)

        dist_np = dist_sum[0, 0].cpu().numpy()
        if sigma > 0:
            from scipy import ndimage as _nd
            dist_np = _nd.gaussian_filter(dist_np, sigma=sigma)
        return dist_np

    return _compute


def _load_dinov3_vit(device: str):
    """Load DINOv3 ViT-B/16 via timm and return a spatial cosine distance callable.

    Uses "vit_base_patch16_dinov3.lvd1689m" — same DINOv3 weights family as the
    ConvNeXt variant but transformer-based. At 512×512 input the patch grid is
    32×32 (patch size 16). Coarser spatial resolution than DINOv3-CNX (which
    has stage features down to 128×128) but useful as a non-convolutional
    reference point.
    """
    import timm as _timm
    model = _timm.create_model(
        "vit_base_patch16_dinov3.lvd1689m",
        pretrained=True, num_classes=0,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    GRID = 32  # 512 / 16
    # DINOv3 has 1 CLS + 4 register tokens = 5 prefix tokens before patches
    N_PREFIX = 5

    @torch.no_grad()
    def _compute(img_a: torch.Tensor, img_b: torch.Tensor, size: int = 512, sigma: float = 4.0) -> np.ndarray:
        a = img_a.to(device)
        b = img_b.to(device)
        if a.shape[-1] != size or a.shape[-2] != size:
            a = F.interpolate(a, size=(size, size), mode="bilinear", align_corners=False)
        if b.shape[-1] != size or b.shape[-2] != size:
            b = F.interpolate(b, size=(size, size), mode="bilinear", align_corners=False)
        a = (a - mean) / std
        b = (b - mean) / std
        fa = model.forward_features(a)  # [1, N_prefix+H*W, C]
        fb = model.forward_features(b)
        # Drop prefix tokens, keep patches
        fa = fa[:, N_PREFIX:]
        fb = fb[:, N_PREFIX:]
        # Unit-normalize per patch
        fa = F.normalize(fa, dim=-1)
        fb = F.normalize(fb, dim=-1)
        cos = (fa * fb).sum(dim=-1)  # [1, H*W]
        dist = (1.0 - cos).view(1, 1, GRID, GRID).float()
        dist_up = F.interpolate(dist, size=(size, size), mode="bilinear", align_corners=False)
        dist_np = dist_up[0, 0].cpu().numpy()
        if sigma > 0:
            from scipy import ndimage as _nd
            dist_np = _nd.gaussian_filter(dist_np, sigma=sigma)
        return dist_np

    return _compute


def _load_dinov2(device: str):
    """Load DINOv2 ViT-B/14 and return a callable computing a spatial patch-token
    L2 distance map. Input: two [B,3,H,W] tensors in [0,1]. Output: np.ndarray [size,size].

    DINOv2 expects 518x518 inputs (37x37 patches at 14px patch size). We use cosine
    distance in feature space (1 - cosine_similarity) on normalized patch features —
    this mirrors the "direction-only" comparison that LPIPS uses and matches the way
    DINO features are typically compared in the literature.
    """
    model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", source="github", trust_repo=True
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # ImageNet normalization (DINOv2 uses ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    DINO_SIZE = 518  # 37 * 14
    GRID = 37

    @torch.no_grad()
    def _compute(img_a: torch.Tensor, img_b: torch.Tensor, size: int = 512, sigma: float = 4.0) -> np.ndarray:
        a = F.interpolate(img_a.to(device), size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False)
        b = F.interpolate(img_b.to(device), size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False)
        a = (a - mean) / std
        b = (b - mean) / std
        fa = model.forward_features(a)["x_norm_patchtokens"]  # [1, 1369, 768]
        fb = model.forward_features(b)["x_norm_patchtokens"]
        # Cosine distance per patch
        fa = F.normalize(fa, dim=-1)
        fb = F.normalize(fb, dim=-1)
        cos = (fa * fb).sum(dim=-1)  # [1, 1369]
        dist = (1.0 - cos).view(1, 1, GRID, GRID)
        # Upsample to target size
        dist_up = F.interpolate(dist, size=(size, size), mode="bilinear", align_corners=False)
        dist_np = dist_up[0, 0].cpu().numpy()
        if sigma > 0:
            from scipy import ndimage as _nd
            dist_np = _nd.gaussian_filter(dist_np, sigma=sigma)
        return dist_np

    return _compute


def _load_lpips(device: str):
    """Load real LPIPS(vgg, spatial=True). Returns a callable that takes two [B,3,H,W]
    tensors in [0,1] and returns a numpy [H,W] distance map."""
    import lpips as _lpips
    model = _lpips.LPIPS(net="vgg", spatial=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def _compute(img_a: torch.Tensor, img_b: torch.Tensor, size: int = 512) -> np.ndarray:
        if img_a.shape[-1] != size or img_a.shape[-2] != size:
            img_a = F.interpolate(img_a, size=(size, size), mode="bilinear", align_corners=False)
        if img_b.shape[-1] != size or img_b.shape[-2] != size:
            img_b = F.interpolate(img_b, size=(size, size), mode="bilinear", align_corners=False)
        # LPIPS expects [-1, 1]
        a = (img_a.to(device) * 2.0 - 1.0)
        b = (img_b.to(device) * 2.0 - 1.0)
        dist = model(a, b)  # [1, 1, H, W]
        dist_np = dist[0, 0].cpu().numpy()
        from scipy import ndimage as _nd
        dist_np = _nd.gaussian_filter(dist_np, sigma=4.0)
        return dist_np

    return _compute


def _mask_to_512(m: np.ndarray, legacy_nearest: bool = False) -> np.ndarray:
    """Binary float mask -> TARGET_SIZE x TARGET_SIZE {0,1}. MAX-POOL by default (never drops mask pixels, same as the
    generator); legacy_nearest reproduces the old NEAREST resize exactly."""
    if legacy_nearest:
        pil = Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
        return (np.array(pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    t = torch.from_numpy((m > 0.5).astype(np.float32))[None, None]
    return (downsample_mask_maxpool(t, TARGET_SIZE)[0, 0].numpy() > 0.5).astype(np.float32)


def _alpha_map_512(placed_bin_512: np.ndarray, band_mode: int = 2) -> torch.Tensor:
    """Build the same soft alpha map the generator used (from placed mask).

    Mirrors generate.py: mask downsampled to 64, create_latent_band_mask(band_mode),
    then nearest upsample to 512. Returns [1,1,512,512] float tensor.
    """
    mask_t = torch.from_numpy(placed_bin_512).unsqueeze(0).float()  # [1, 512, 512]
    m64 = downsample_mask_maxpool(mask_t, 64).unsqueeze(0)  # [1, 1, 64, 64]
    _, alpha_map_64, _, _ = create_latent_band_mask(m64, band_mode=band_mode)
    alpha_512 = F.interpolate(alpha_map_64, size=(512, 512), mode="nearest")
    return alpha_512  # [1, 1, 512, 512]


@torch.no_grad()
def _vae_roundtrip(vae, img_01: torch.Tensor, device: str) -> torch.Tensor:
    """Encode [0,1] RGB tensor through VAE → decode → back to [0,1].

    Uses the same scaling factor as the training pipeline (sample * scaling_factor).
    """
    x = img_01.to(device) * 2.0 - 1.0  # [-1, 1]
    x = x.to(next(vae.parameters()).dtype)
    latents = vae.encode(x).latent_dist.mean
    latents = latents * vae.config.scaling_factor
    decoded = vae.decode(latents / vae.config.scaling_factor).sample
    decoded = (decoded.clamp(-1, 1) + 1.0) / 2.0
    return decoded.float().cpu()


def main():
    # ── DEFAULTS (changed 2026-04-10 after human visual inspection) ─────────
    # Default pipeline: DINOv3-CNX backbone + α-blend canvas roundtrip +
    # factor=0.6 + σ=4. See module docstring for rationale.
    # Legacy pipeline (pre-2026-04-10): VGG + no roundtrip + factor=0.65.
    # To reproduce legacy results, pass --use-legacy-vgg.
    # ─────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Compute perceptually-refined masks for cashew 100. "
                    "Default: DINOv3-CNX + α-blend roundtrip + σ=4 + factor=0.6."
    )
    parser.add_argument("--gen-dir", type=str, required=True,
                        help="Path to generated images (e.g. <run>/cashew_100_rp/generated/cfg7_vis)")
    parser.add_argument("--prep-dir", type=str, required=True,
                        help="Path to prep directory with canvas/mask/meta per sample")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for refined masks")
    parser.add_argument("--fg-dir", type=str, required=True,
                        help="Path to birefnet Normal FG masks directory")
    parser.add_argument("--legacy-nearest", action="store_true",
                        help="Reproduce pre-2026-09-25 masks byte for byte: NEAREST downsampling of the FG mask, the "
                             "hard-case placed mask and the alpha-blend placed mask. Default is MAX-POOL (never drops "
                             "mask pixels; matches the generator, which max-pools the placed mask).")
    parser.add_argument("--min-blob-px", type=int, default=0,
                        help="Drop refined-mask blobs smaller than this (8-connected). Sampler label default: 16. 0 = off.")
    parser.add_argument("--abs-threshold", type=float, default=None,
                        help="ABSOLUTE refinement (x100 DINO units, ring-relative): keep pixels whose distance exceeds "
                             "the surroundings by more than this. Empty mask when nothing was painted. Calibrated on "
                             "blank repaints (max 7.0, results/SYNTHESIS_STATE.md). Default: relative --factor rule.")
    parser.add_argument("--factor", type=float, default=0.25,
                        help="Per-component threshold factor. "
                             "Default 0.25 (2026-09-30, mask ladder: scripts/inpaint_region_test.py; was 0.6 from 2026-04-10, before that 0.65 — "
                             "catches faint anomaly edges).")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Gaussian blur sigma on the distance map. "
                             "Default 3.0 (changed 2026-04-10 from 4.0 — tighter contours without "
                             "the ragged per-pixel artifacts of σ=0/1). Set to 0 to disable.")

    # Backbone selection — mutually exclusive, default = DINOv3-CNX
    backbone = parser.add_mutually_exclusive_group()
    backbone.add_argument("--use-dinov3-cnx", action="store_true",
                          help="(DEFAULT) DINOv3 ConvNeXt-Base distilled (timm convnext_base.dinov3_lvd1689m). "
                               "Same weights UniNet uses downstream.")
    backbone.add_argument("--use-dinov3-vit", action="store_true",
                          help="DINOv3 ViT-B/16 (timm vit_base_patch16_dinov3.lvd1689m). "
                               "32x32 patch grid at 512 input — semantic features, coarser spatial resolution.")
    backbone.add_argument("--use-dinov2", action="store_true",
                          help="DINOv2 ViT-B/14 patch cosine distance. Legacy DINOv2 reference.")
    backbone.add_argument("--use-lpips", action="store_true",
                          help="Real LPIPS(net='vgg', spatial=True). Applies unit-normalization + "
                               "learned channel weights.")
    backbone.add_argument("--use-legacy-vgg", action="store_true",
                          help="Legacy hand-written VGG16 perceptual distance (L2/C at 4 layers). "
                               "Used for pre-2026-04-10 results. Pair with --no-alpha-blend for the "
                               "exact original pipeline.")

    # Canvas calibration — default is α-blend roundtrip ON
    parser.add_argument("--no-alpha-blend", action="store_true",
                        help="Disable the default α-blend VAE roundtrip on the canvas. Use for legacy "
                             "reproducibility. Default behavior: canvas is VAE-roundtripped and blended "
                             "with the raw canvas using the same alpha map as the generator (see "
                             "generate.py:272). Cancels VAE bias inside the dilated band.")
    parser.add_argument("--vae-roundtrip-canvas", action="store_true",
                        help="(Experimental) Full VAE roundtrip of canvas without alpha-blend compositing. "
                             "Not recommended — introduces asymmetric VAE bias in the background.")
    parser.add_argument("--canvas-ref", choices=["sd15", "raw"], default="sd15",
                        help="Canvas reference. 'sd15' (default) = SD1.5 VAE calibration controlled by "
                             "--no-alpha-blend / --vae-roundtrip-canvas. 'raw' = generator-agnostic: raw "
                             "canvas, no VAE (for FLUX Kontext etc.). Pair with --canvas-resize-via.")
    parser.add_argument("--canvas-resize-via", type=int, default=0,
                        help="(--canvas-ref raw only) Resize the canvas to NxN with LANCZOS before the "
                             "LANCZOS resize to 512, mirroring a generator that works at NxN and saves a "
                             "512 copy (FLUX Kontext backend: N = --flux-res). 0 = direct resize to 512.")

    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.canvas_ref == "raw" and (args.vae_roundtrip_canvas or args.no_alpha_blend):
        parser.error("--canvas-ref raw already disables the VAE; drop --vae-roundtrip-canvas/--no-alpha-blend")
    if args.canvas_resize_via and args.canvas_ref != "raw":
        parser.error("--canvas-resize-via requires --canvas-ref raw")

    # Resolve backbone: default to DINOv3-CNX if no flag given
    if not (args.use_dinov3_cnx or args.use_dinov3_vit or args.use_dinov2
            or args.use_lpips or args.use_legacy_vgg):
        args.use_dinov3_cnx = True  # new default

    # Resolve canvas calibration: default to α-blend unless explicitly disabled
    # or the user passed the experimental full-roundtrip flag
    args.vae_roundtrip_alpha_blend = (args.canvas_ref == "sd15"
                                      and not args.no_alpha_blend and not args.vae_roundtrip_canvas)

    project_root = Path(__file__).parent.parent
    gen_dir = Path(args.gen_dir)
    if not gen_dir.is_absolute():
        gen_dir = project_root / gen_dir
    prep_dir = Path(args.prep_dir)
    if not prep_dir.is_absolute():
        prep_dir = project_root / prep_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    fg_dir = Path(args.fg_dir)
    if not fg_dir.is_absolute():
        fg_dir = project_root / fg_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    # Build distance model once. New default: DINOv3-CNX.
    lpips_fn = None
    dinov2_fn = None
    dinov3_fn = None
    dinov3_vit_fn = None
    vgg = None
    if args.use_dinov3_cnx:
        print("Loading DINOv3 ConvNeXt-Base (timm: convnext_base.dinov3_lvd1689m) — default backbone")
        dinov3_fn = _load_dinov3_cnx(device)
    elif args.use_dinov3_vit:
        print("Loading DINOv3 ViT-B/16 (timm: vit_base_patch16_dinov3.lvd1689m)")
        dinov3_vit_fn = _load_dinov3_vit(device)
    elif args.use_dinov2:
        print("Loading DINOv2 ViT-B/14")
        dinov2_fn = _load_dinov2(device)
    elif args.use_lpips:
        print("Loading real LPIPS (net='vgg', spatial=True)")
        lpips_fn = _load_lpips(device)
    elif args.use_legacy_vgg:
        print("Loading legacy hand-written VGG16 feature extractor (pre-2026-04-10 default)")
        vgg = build_vgg_features(device)

    vae = None
    if args.vae_roundtrip_canvas or args.vae_roundtrip_alpha_blend:
        print(f"Loading SD1.5 VAE for canvas roundtrip: {VAE_MODEL_ID}")
        vae = _load_vae(device)

    all_stats = {}
    n_done = 0
    n_skipped = 0

    for idx in range(100):
        s = f"{idx:03d}"
        out_path = output_dir / f"{s}.png"

        # Check idempotency
        if out_path.exists():
            n_skipped += 1
            continue

        meta_path = prep_dir / s / "meta.json"
        if not meta_path.exists():
            print(f"  [{s}] SKIP — no meta.json in prep")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        is_hard = meta.get("is_hard", False)
        normal_stem = Path(meta["normal"]).stem

        # Load birefnet FG mask
        fg_path = fg_dir / f"{normal_stem}_binary.png"
        fg_mask = None
        if fg_path.exists():
            fg_raw = np.array(Image.open(fg_path).convert("L")).astype(np.float32) / 255.0
            fg_mask = _mask_to_512((fg_raw > 0.5).astype(np.float32), args.legacy_nearest)

        if is_hard:
            # Hard cases: placed_mask ∩ FG (no LPIPS thresholding)
            if not (gen_dir / f"{s}.png").exists():
                print(f"  [{s}] SKIP — no generated image")
                continue
            placed_path = prep_dir / s / "placed_mask.png"
            placed_bin = _mask_to_512(
                (np.array(Image.open(placed_path).convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32),
                args.legacy_nearest)
            if fg_mask is not None:
                refined = placed_bin * fg_mask
            else:
                refined = placed_bin
            stats = {"hard": True, "n_refined_total": int(refined.sum())}
        else:
            # Easy cases: full LPIPS pipeline
            gen_path = gen_dir / f"{s}.png"
            canvas_path = prep_dir / s / "canvas.png"
            placed_path = prep_dir / s / "placed_mask.png"

            if not gen_path.exists():
                print(f"  [{s}] SKIP — no generated image")
                continue

            # Load images as [1,3,S,S] in [0,1]
            gen_pil = Image.open(gen_path).convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            canvas_pil = Image.open(canvas_path).convert("RGB")
            if args.canvas_resize_via:
                via = args.canvas_resize_via
                canvas_pil = canvas_pil.resize((via, via), Image.LANCZOS)
            canvas_pil = canvas_pil.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            gen_t = torch.from_numpy(np.array(gen_pil).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            canvas_t = torch.from_numpy(np.array(canvas_pil).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

            # Load placed mask at native resolution for dilation pipeline
            placed_raw = np.array(Image.open(placed_path).convert("L")).astype(np.float32) / 255.0
            placed_bin = (placed_raw > 0.5).astype(np.float32)

            if args.vae_roundtrip_alpha_blend:
                canvas_rt = _vae_roundtrip(vae, canvas_t, device)
                # 512 version of the placed mask for the alpha map (max-pool, as the generator; --legacy-nearest = old NEAREST)
                placed_512 = _mask_to_512(placed_bin, args.legacy_nearest)
                alpha_512 = _alpha_map_512(placed_512, band_mode=BAND_MODE)
                canvas_t = canvas_rt * alpha_512 + canvas_t * (1.0 - alpha_512)
            elif args.vae_roundtrip_canvas:
                canvas_t = _vae_roundtrip(vae, canvas_t, device)

            # Perceptual distance
            if dinov3_fn is not None:
                dist = dinov3_fn(gen_t, canvas_t, size=TARGET_SIZE, sigma=args.sigma)
            elif dinov3_vit_fn is not None:
                dist = dinov3_vit_fn(gen_t, canvas_t, size=TARGET_SIZE, sigma=args.sigma)
            elif dinov2_fn is not None:
                dist = dinov2_fn(gen_t, canvas_t, size=TARGET_SIZE, sigma=args.sigma)
            elif lpips_fn is not None:
                dist = lpips_fn(gen_t, canvas_t, size=TARGET_SIZE)
            else:
                dist = compute_perceptual_distance(
                    vgg, gen_t, canvas_t, device=device, size=TARGET_SIZE, sigma=args.sigma
                )

            # Roundtripped dilated mask
            dilated = compute_dilated_mask_512(placed_bin, band_mode=BAND_MODE, target_size=TARGET_SIZE)

            # Per-component refinement
            refined, stats = build_refined_mask(dist, dilated, factor=args.factor, fg_mask=fg_mask,
                                                 abs_threshold=None if args.abs_threshold is None else args.abs_threshold / 100.0)
            refined = drop_small_blobs(refined, args.min_blob_px)

            if args.canvas_ref == "raw":
                # Sanity check of the reference: far from the dilated mask a generator that
                # pastes back raw canvas pixels must reproduce the reference (up to resampling).
                from scipy import ndimage as _nd
                far = ~_nd.binary_dilation(dilated > 0.5, iterations=16)
                diff = (gen_t - canvas_t).abs().mean(dim=1)[0].numpy() * 255.0
                stats["bg_mae_255"] = float(diff[far].mean()) if far.any() else None

        # Save
        refined_uint8 = (refined * 255).astype(np.uint8)
        Image.fromarray(refined_uint8).save(out_path)
        all_stats[s] = stats
        n_done += 1

        kind = "hard" if is_hard else "easy"
        n_px = int(refined.sum())
        bg = f"  bg_mae_255={stats['bg_mae_255']:.2f}" if stats.get("bg_mae_255") is not None else ""
        print(f"  [{s}] {kind:4s} — {n_px:5d} px refined{bg}")

    # Save summary
    backbone = (
        "dinov3_cnx" if args.use_dinov3_cnx
        else "dinov3_vit" if args.use_dinov3_vit
        else "dinov2" if args.use_dinov2
        else "lpips" if args.use_lpips
        else "vgg_legacy"
    )
    summary = {
        "factor": args.factor,
        "sigma": args.sigma,
        "band_mode": BAND_MODE,
        "backbone": backbone,
        "canvas_ref": args.canvas_ref,
        "canvas_resize_via": args.canvas_resize_via,
        "vae_roundtrip_canvas": bool(args.vae_roundtrip_canvas),
        "vae_roundtrip_alpha_blend": bool(args.vae_roundtrip_alpha_blend),
        "use_lpips": bool(args.use_lpips),
        "n_processed": n_done,
        "n_skipped": n_skipped,
        "per_sample": all_stats,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! {n_done} processed, {n_skipped} skipped (already exist)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
