#!/usr/bin/env python3
"""In-process copy of compute_refined_masks.py's easy-case path with --canvas-ref raw (+ optional --abs-threshold),
so a sampler can refine every attempt without reloading DINOv3 per call. Must stay byte-identical to the script:
run this file to verify against the script's outputs.

  python scripts/refine_inprocess.py        # verification
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.utils.perceptual_mask_utils import build_refined_mask, compute_dilated_mask_512, drop_small_blobs  # noqa: E402

S, BAND_MODE, SIGMA = 512, 2, 3.0          # compute_refined_masks defaults (TARGET_SIZE, BAND_MODE, --sigma)
FG_DIR = ROOT / "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/birefnet_masks/Normal"


def load_fg(prep_case: Path, legacy_nearest: bool = False) -> np.ndarray | None:
    """BiRefNet foreground of the case's canvas, loaded with compute_refined_masks' own helper (max-pool by default)."""
    from compute_refined_masks import _mask_to_512
    stem = Path(json.loads((prep_case / "meta.json").read_text())["normal"]).stem
    p = FG_DIR / f"{stem}_binary.png"
    if not p.exists():
        return None
    raw = np.array(Image.open(p).convert("L")).astype(np.float32) / 255.0
    return _mask_to_512((raw > 0.5).astype(np.float32), legacy_nearest)


_VAE = None


def refine(dino_fn, gen: Path | Image.Image, prep_case: Path, abs_threshold: float | None, factor: float = 0.25,
           vae_reference: bool = False, thesis_reference: bool = False, min_blob_px: int = 0,
           level_on_footprint: bool = False, percentile: float = 90.0) -> np.ndarray:
    """Refined mask [512,512] bool for one generation, exactly as the script with --canvas-ref raw, or with
    --vae-roundtrip-canvas when vae_reference=True (reference = canvas VAE round-tripped, the pipeline's no-op output).
    abs_threshold in x100 DINO units (None = relative factor rule). level_on_footprint: take each piece's p90 over
    the piece's footprint pixels (on the cashew) instead of the whole piece; the mask is still cut over the whole piece."""
    global _VAE
    gen_pil = (gen if isinstance(gen, Image.Image) else Image.open(gen)).convert("RGB").resize((S, S), Image.LANCZOS)
    can_pil = Image.open(prep_case / "canvas.png").convert("RGB").resize((S, S), Image.LANCZOS)
    t = lambda p: torch.from_numpy(np.array(p).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    can_t = t(can_pil)
    if thesis_reference:
        # compute_refined_masks default (--canvas-ref sd15): VAE round-trip INSIDE the paint band (alpha-blended),
        # raw canvas outside - copied from the script's vae_roundtrip_alpha_blend branch
        from compute_refined_masks import _load_vae, _vae_roundtrip, _alpha_map_512, _mask_to_512
        if _VAE is None:
            _VAE = _load_vae("cuda")
        placed_bin = (np.array(Image.open(prep_case / "placed_mask.png").convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        alpha_512 = _alpha_map_512(_mask_to_512(placed_bin), band_mode=BAND_MODE)
        can_t = _vae_roundtrip(_VAE, can_t, "cuda") * alpha_512 + can_t * (1.0 - alpha_512)
    elif vae_reference:
        from compute_refined_masks import _load_vae, _vae_roundtrip
        if _VAE is None:
            _VAE = _load_vae("cuda")
        can_t = _vae_roundtrip(_VAE, can_t, "cuda")
    dist = dino_fn(t(gen_pil), can_t, size=S, sigma=SIGMA)
    placed = (np.array(Image.open(prep_case / "placed_mask.png").convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    dilated = compute_dilated_mask_512(placed, band_mode=BAND_MODE, target_size=S)
    fg = load_fg(prep_case)
    level = None
    if level_on_footprint:
        from sheet_utils import mask_resize
        level = mask_resize(placed > 0.5, S) & ((fg > 0.5) if fg is not None else True)
    refined, _ = build_refined_mask(dist, dilated, factor=factor, fg_mask=fg,
                                    abs_threshold=None if abs_threshold is None else abs_threshold / 100.0, level_mask=level,
                                    percentile=percentile)
    return drop_small_blobs(refined, min_blob_px) > 0.5


def main() -> None:
    from compute_refined_masks import _load_dinov3_cnx
    dino = _load_dinov3_cnx("cuda")
    prep = ROOT / "results/matrix_cashew_20k_single/core-50k_at_20k/prep"
    R = ROOT / "results/refine_compare"
    for label, gdir, ref_dir in (("generations", ROOT / "results/matrix_cashew_20k_single/_gen_all", R / "gen_abs10"),
                                 ("blank repaints", ROOT / "results/proxy_null/n2", R / "null_abs10")):
        same = 0
        for i in range(20):
            m = refine(dino, gdir / f"{i:03d}.png", prep / f"{i:03d}", 10.0)
            same += np.array_equal(m, np.array(Image.open(ref_dir / f"{i:03d}.png").convert("L")) > 127)
        print(f"{label}: in-process == compute_refined_masks.py --abs-threshold 10 in {same}/20")


if __name__ == "__main__":
    main()
