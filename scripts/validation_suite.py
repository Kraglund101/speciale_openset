"""6-Panel Reproducible Validation Suite.

Produces deterministic validation outputs across checkpoints:
  Panel A — Easy ID Sanity (texture placement)
  Panel B — ID Constrained Hard (anomaly-as-canvas)
  Panel C — ID Counterfactual Cross-Object (native vs swapped)
  Panel D — OOD Counterfactual Cross-Domain (cashew + PCB)
  Panel E — OOD Hard Deformation (cashew hard)
  Panel F — OOD Easy Cashew (small surface defects)

Two entry points:
  1. run_validation_suite()  — called from training loop
  2. __main__               — standalone CLI with checkpoint loading

CFG variants (3 per example):
  nocfg  — cfg_mode="visual", guidance_scale=1.0
  ip_35  — cfg_mode="visual", guidance_scale=3.5
  ip_70  — cfg_mode="visual", guidance_scale=7.0
"""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.generate import generate_anomagic_single
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from src.utils.crop_utils import clip_crop_multi
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool, unet_roundtrip_masks

logger = logging.getLogger(__name__)

# ── CFG Variants ────────────────────────────────────────────────────────────

CFG_VARIANTS = [
    {"name": "nocfg", "cfg_mode": "visual", "guidance_scale": 1.0},
    {"name": "ip_35", "cfg_mode": "visual", "guidance_scale": 3.5},
    {"name": "ip_70", "cfg_mode": "visual", "guidance_scale": 7.0},
]

NUM_STEPS = 30

# ── Seed Strategy ───────────────────────────────────────────────────────────

PANEL_SEEDS = {"A": 42, "B": 142, "C": 242, "D": 342, "E": 442, "F": 542}

# ── Panel Definitions ───────────────────────────────────────────────────────
# Single source of truth for ALL example specs.
#
# Fields used by the validation suite at inference time:
#   id, image_path, mask_path, noise_strength
#
# Fields used by the prep script (prep_validation_data.py):
#   needs_prep  — True if a canvas + placed_mask must be generated offline
#   normal_dir  — directory of good/normal images to pick as canvas
#   normal_index — which sorted image to pick (deterministic)
#   fg_mode     — "ones" (full image is surface) or "birefnet" (segment object)
#
# The prep script iterates ALL examples, skips needs_prep=False ones, and
# processes the rest generically regardless of which panel they belong to.

PANEL_A_EXAMPLES = [
    {
        "id": "leather_fold_016",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/leather/test/fold/016.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/leather/ground_truth/fold/016_mask.png",
        "needs_prep": True,
        "normal_dir": "AnomVerse_data_filtered/mvtec/mvtec/leather/train/good",
        "normal_index": 0,
        "fg_mode": "birefnet",
    },
    {
        "id": "carpet_thread_011",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/carpet/test/thread/011.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/carpet/ground_truth/thread/011_mask.png",
        "needs_prep": True,
        "normal_dir": "AnomVerse_data_filtered/mvtec/mvtec/carpet/train/good",
        "normal_index": 0,
        "fg_mode": "birefnet",
    },
    {
        "id": "hazelnut_cut_011",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/test/cut/011.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/ground_truth/cut/011_mask.png",
        "needs_prep": True,
        "normal_dir": "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/train/good",
        "normal_index": 0,
        "fg_mode": "birefnet",
    },
]

PANEL_B_EXAMPLES = [
    {
        "id": "bottle_broken_large_011",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/bottle/test/broken_large/011.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/bottle/ground_truth/broken_large/011_mask.png",
        "noise_strength": 1.0,
        "needs_prep": False,
    },
    {
        "id": "transistor_cut_lead_005",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/transistor/test/cut_lead/005.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/transistor/ground_truth/cut_lead/005_mask.png",
        "noise_strength": 1.0,
        "needs_prep": False,
    },
    {
        "id": "mint_0027_QS",
        "image_path": "realiad_1024/mint/NG/QS/S0027/mint_0027_NG_QS_C2_20230910101534.jpg",
        "mask_path": "realiad_1024/mint/NG/QS/S0027/mint_0027_NG_QS_C2_20230910101534.png",
        "noise_strength": 1.0,
        "needs_prep": False,
    },
    {
        "id": "capsule_faulty_imprint_016",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/capsule/test/faulty_imprint/016.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/capsule/ground_truth/faulty_imprint/016_mask.png",
        "noise_strength": 1.0,
        "needs_prep": False,
    },
]

PANEL_C_EXAMPLES = [
    {
        "id": "bracket_black_scratch_026",
        "image_path": "AnomVerse_data_filtered/MPDD/MPDD/bracket_black/test/scratches/026.png",
        "mask_path": "AnomVerse_data_filtered/MPDD/MPDD/bracket_black/ground_truth/scratches/026_mask.png",
        "needs_prep": True,
        "normal_dir": "AnomVerse_data_filtered/MPDD/MPDD/bracket_black/train/good",
        "normal_index": 0,
        "fg_mode": "birefnet",
    },
    {
        "id": "hazelnut_cut_005",
        "image_path": "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/test/cut/005.png",
        "mask_path": "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/ground_truth/cut/005_mask.png",
        "needs_prep": True,
        "normal_dir": "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/train/good",
        "normal_index": 0,
        "fg_mode": "birefnet",
    },
]

# Base directories for VisA data (relative to project root)
_CASHEW_BASE = "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew"
_PCB_BASE = "anomverse_extension/datasets/VisA_validation_dataset/datasets/hard_test/pcb2"

PANEL_D_EXAMPLES = [
    {
        "id": "cashew_easy_002_086",
        "image_path": "experiment_ResNet/source/reference_anomalies/easy/imgs/086.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/easy/masks/086.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 2,  # → 002.JPG
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
    },
    {
        "id": "pcb2_094",
        "image_path": "Data/Images/Anomaly/094.JPG",
        "mask_path": "Data/Masks/Anomaly/094.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 0,  # → 0000.JPG
        "fg_mode": "birefnet",
        "base_dir": _PCB_BASE,
        "fg_masks_dir": "Data/Masks/Normal",
        "fg_mask_suffix": "",
        "mask_is_label": True,  # VisA PCB uses 0/1/2 label values, not 0/255
        "scale_range": [0.3, 0.8],  # Elongated mask needs downscaling to fit in FG
    },
]

PANEL_E_EXAMPLES = [
    {
        "id": "cashew_hard_029_096",
        "image_path": "experiment_ResNet/source/reference_anomalies/hard/imgs/096.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/hard/masks/096.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 29,  # → 029.JPG
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
        "noise_strength": 1.0,
        "placement_mode": "hard",  # Anomaly mask covers entire object
    },
    {
        "id": "cashew_hard_293_098",
        "image_path": "experiment_ResNet/source/reference_anomalies/hard/imgs/098.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/hard/masks/098.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 293,  # → 293.JPG
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
        "noise_strength": 1.0,
        "placement_mode": "hard",  # Anomaly mask covers entire object
    },
]

PANEL_F_EXAMPLES = [
    {
        "id": "cashew_easy_005_001",
        "image_path": "experiment_ResNet/source/reference_anomalies/easy/imgs/001.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/easy/masks/001.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 5,
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
    },
    {
        "id": "cashew_easy_010_029",
        "image_path": "experiment_ResNet/source/reference_anomalies/easy/imgs/029.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/easy/masks/029.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 10,
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
    },
    {
        "id": "cashew_easy_015_044",
        "image_path": "experiment_ResNet/source/reference_anomalies/easy/imgs/044.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/easy/masks/044.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 15,
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
    },
    {
        "id": "cashew_easy_020_079",
        "image_path": "experiment_ResNet/source/reference_anomalies/easy/imgs/079.JPG",
        "mask_path": "experiment_ResNet/source/reference_anomalies/easy/masks/079.png",
        "needs_prep": True,
        "normal_dir": "Data/Images/Normal",
        "normal_index": 20,
        "fg_mode": "birefnet",
        "base_dir": _CASHEW_BASE,
        "fg_masks_dir": "birefnet_masks/Normal",
        "fg_mask_suffix": "_binary",
    },
]

# Flat list of every example that needs offline prep (used by prep script).
ALL_PREP_EXAMPLES = [
    ex for panel in (PANEL_A_EXAMPLES, PANEL_B_EXAMPLES, PANEL_C_EXAMPLES,
                     PANEL_D_EXAMPLES, PANEL_E_EXAMPLES, PANEL_F_EXAMPLES)
    for ex in panel if ex.get("needs_prep", False)
]

# Hand-crafted captions for placement panels (input ≠ reference).
# Training captions describe the original image context, but placement panels
# paste the anomaly onto a different normal canvas, so we need captions that
# describe the reference anomaly as-if placed on the target surface.
VALIDATION_CAPTIONS: Dict[str, str] = {
    # Panel A — Easy ID (same-product placement)
    "leather_fold_016": "Leather surface has a horizontal crease forming a shallow fold across the grain.",
    "carpet_thread_011": "Woven fabric has a thin dark thread lying diagonally across the surface.",
    "hazelnut_cut_011": "Hazelnut shell has a small cut with a light-colored linear nick on the surface.",
    # Panel C — ID Counterfactual (same-product placement)
    "bracket_black_scratch_026": "Black metal bracket has a small scratch mark on its surface.",
    "hazelnut_cut_005": "Hazelnut shell has a scratch with a light-colored linear mark on its surface.",
    # Panel D — OOD Counterfactual (cross-domain placement)
    "cashew_easy_002_086": "Cashew nut has a thin scratch-like mark on its lower surface.",
    "pcb2_094": "Circuit board has white scratch marks across the blue solder mask in the lower section.",
    # Panel E — OOD Hard Deformation
    "cashew_hard_029_096": "Cashew nut has a misshapen, multi-lobed form with an abnormal split shape.",
    "cashew_hard_293_098": "Cashew nut has multiple overlapping pieces forming an irregular stacked mass.",
    # Panel F — OOD Easy Cashew
    "cashew_easy_005_001": "Cashew nut has a small dark blemish on its lower curved surface.",
    "cashew_easy_010_029": "Cashew nut has a tiny dark spot near the upper left of its surface.",
    "cashew_easy_015_044": "Cashew nut has a discolored patch on the right side of its body.",
    "cashew_easy_020_079": "Cashew nut has a thin diagonal scratch across its lower surface.",
}


# ═════════════════════════════════════════════════════════════════════════════
# Shared Helpers
# ═════════════════════════════════════════════════════════════════════════════


def _load_image_tensor(path: Path, size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Load image → [1, 3, H, W] in [-1, 1]."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _load_mask_tensor(path: Path, size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Load mask → [1, 1, H, W] binary (maxpool downsample)."""
    pil = Image.open(path).convert("L")
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy((arr > 0.5).astype(np.float32)).unsqueeze(0)  # [1, H, W]
    t = downsample_mask_maxpool(t, size)  # [1, size, size]
    return t.unsqueeze(0).float().to(device)  # [1, 1, size, size]


def _prepare_clip_crops(
    ref_img_path: Path,
    ref_mask_path: Path,
    band_mode: int,
    clip_align: bool,
    device: str,
) -> Dict[str, Any]:
    """Load reference + mask, run clip_crop_multi, return dict of tensors.

    Returns dict with keys: clip_ref, clip_mask_1, clip_core_1,
    clip_ref_2, clip_mask_2, clip_core_2, group_valid, crop_np, crop_mask_np.
    """
    ref_pil = Image.open(ref_img_path).convert("RGB")
    ref_mask_np = (
        np.array(Image.open(ref_mask_path).convert("L")).astype(np.float32) / 255.0 > 0.5
    ).astype(np.float32)
    ref_tensor = torch.from_numpy(
        np.array(ref_pil).astype(np.float32) / 255.0
    ).permute(2, 0, 1)  # [3, H, W] in [0, 1]
    ref_mask_tensor = torch.from_numpy(ref_mask_np).unsqueeze(0).float()  # [1, H, W]

    if clip_align:
        core_native, dil_native = unet_roundtrip_masks(ref_mask_tensor, band_mode)
        crops, crop_masks, valid, extra = clip_crop_multi(
            ref_tensor, ref_mask_tensor, n_groups=2,
            clip_masks=[dil_native, core_native],
        )
        result = {
            "clip_ref": (crops[0] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_1": (extra[0][0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_1": (extra[0][1] > 0.5).float().unsqueeze(0).to(device),
            "clip_ref_2": (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_2": (extra[1][0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_2": (extra[1][1] > 0.5).float().unsqueeze(0).to(device),
        }
    else:
        crops, crop_masks, valid = clip_crop_multi(
            ref_tensor, ref_mask_tensor, n_groups=2,
        )
        result = {
            "clip_ref": (crops[0] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_1": (crop_masks[0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_1": None,
            "clip_ref_2": (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_2": (crop_masks[1] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_2": None,
        }

    result["group_valid"] = torch.tensor(
        [[float(valid[0]), float(valid[1])]], device=device,
    )
    # For visualisation (both crops)
    result["crop_np"] = (crops[0].permute(1, 2, 0).numpy()).clip(0, 1)
    result["crop_mask_np"] = crop_masks[0][0].numpy()
    result["crop_np_2"] = (crops[1].permute(1, 2, 0).numpy()).clip(0, 1)
    result["crop_mask_np_2"] = crop_masks[1][0].numpy()
    return result


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [1, 3, H, W] in [-1, 1] to PIL RGB."""
    arr = ((t[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _compute_diff_map(a: torch.Tensor, b: torch.Tensor) -> Image.Image:
    """|A - B| per-channel mean → heatmap PIL."""
    diff = (a - b).abs().mean(dim=1, keepdim=True)  # [1, 1, H, W]
    diff_np = diff[0, 0].cpu().float().numpy()
    # Normalise to [0, 255]
    if diff_np.max() > 0:
        diff_np = diff_np / diff_np.max()
    return Image.fromarray((diff_np * 255).astype(np.uint8))


def _save_pil(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _generate_and_save(
    pipeline,
    ip_adapter,
    canvas_t: torch.Tensor,
    mask_t: torch.Tensor,
    clip_data: Dict[str, Any],
    anomaly_type: str,
    caption: str,
    noise_strength: float,
    inference_mode: str,
    band_mode: int,
    t2i_adapter,
    seed: int,
    out_dir: Path,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Run 3 CFG variants, save PNGs + metadata. Returns dict of output tensors."""
    outputs = {}

    # Save canvas + mask + reference crop
    _save_pil(_tensor_to_pil(canvas_t), out_dir / "canvas.png")
    mask_np = mask_t[0, 0].cpu().numpy()
    _save_pil(Image.fromarray((mask_np * 255).astype(np.uint8)), out_dir / "mask.png")

    # Save reference crop 1
    crop_overlay = clip_data["crop_np"].copy()
    m = clip_data["crop_mask_np"][..., None]
    tint = np.array([1.0, 0.2, 0.2])
    crop_overlay = (crop_overlay * (1 - 0.3 * m) + tint * 0.3 * m).clip(0, 1)
    _save_pil(
        Image.fromarray((crop_overlay * 255).astype(np.uint8)),
        out_dir / "reference.png",
    )

    # Save reference crop 2
    crop_overlay_2 = clip_data["crop_np_2"].copy()
    m2 = clip_data["crop_mask_np_2"][..., None]
    crop_overlay_2 = (crop_overlay_2 * (1 - 0.3 * m2) + tint * 0.3 * m2).clip(0, 1)
    _save_pil(
        Image.fromarray((crop_overlay_2 * 255).astype(np.uint8)),
        out_dir / "reference_2.png",
    )

    gen_kwargs = dict(
        num_steps=NUM_STEPS,
        noise_strength=noise_strength,
        reference_mode="crop",
        band_mode=band_mode,
        t2i_adapter=t2i_adapter,
        clip_mask=clip_data["clip_mask_1"],
        clip_core_mask=clip_data["clip_core_1"],
        reference_2=clip_data["clip_ref_2"],
        clip_mask_2=clip_data["clip_mask_2"],
        clip_core_mask_2=clip_data["clip_core_2"],
        group_valid=clip_data["group_valid"],
        inference_mode=inference_mode,
    )

    for variant in CFG_VARIANTS:
        with torch.no_grad():
            out = generate_anomagic_single(
                pipeline, ip_adapter,
                canvas_t, mask_t, clip_data["clip_ref"],
                anomaly_type, caption,
                guidance_scale=variant["guidance_scale"],
                cfg_mode=variant["cfg_mode"],
                seed=seed,
                **gen_kwargs,
            )
        torch.cuda.synchronize()
        outputs[variant["name"]] = out
        _save_pil(_tensor_to_pil(out), out_dir / f"{variant['name']}.png")

    # Metadata
    meta = {
        "anomaly_type": anomaly_type,
        "caption": caption,
        "noise_strength": noise_strength,
        "inference_mode": inference_mode,
        "seed": seed,
        "cfg_variants": [v["name"] for v in CFG_VARIANTS],
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return outputs


def _get_caption(captions: Dict[str, str], image_path: str, fallback: str = "a defect",
                 example_id: str = "") -> str:
    """Look up caption: hand-crafted → training → fallback."""
    # 1. Hand-crafted validation captions (placement panels)
    if example_id and example_id in VALIDATION_CAPTIONS:
        return VALIDATION_CAPTIONS[example_id]
    # 2. Training captions (same-image panels like B)
    if captions and image_path in captions:
        return captions[image_path]
    return f"a photo of a {fallback}"


# ═════════════════════════════════════════════════════════════════════════════
# Panel Plot Helper
# ═════════════════════════════════════════════════════════════════════════════


def _create_panel_plot(
    panel_name: str,
    examples: List[Dict],
    band_mode: int,
    out_path: Path,
) -> None:
    """Create a combined matplotlib figure for a validation panel.

    Args:
        panel_name: Title for the figure (e.g. "Panel A — Easy ID Sanity").
        examples: List of dicts, each with:
            - dir: Path to example output directory (canvas.png, mask.png, etc.)
            - label: Column title string
            - is_swapped: bool — if True, include diff row
        band_mode: Band dilation mode (1 or 2) for alpha_map_64 computation.
        out_path: Where to save the combined PNG.
    """
    if not examples:
        return

    has_diff = any(ex.get("is_swapped", False) for ex in examples)
    ROW_CANVAS, ROW_MASK = 0, 1
    ROW_REF1, ROW_REF2 = 2, 3
    ROW_NOCFG, ROW_IP35, ROW_IP70 = 4, 5, 6
    base_rows = [
        "Canvas", "Mask @512",
        "Reference 1", "Reference 2",
        "nocfg", "ip_35", "ip_70",
    ]
    if has_diff:
        ROW_DIFF = len(base_rows)
        base_rows.append("Diff (ip_70)")
    n_rows = len(base_rows)
    n_cols = len(examples)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    # Ensure axes is always 2D
    if n_cols == 1:
        axes = axes[:, np.newaxis]
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(panel_name, fontsize=14, fontweight="bold", y=0.995)

    overlay_cells: List[Tuple[int, np.ndarray]] = []  # (col, overlay) for second pass
    last_diff_im = None  # for colorbar

    for col, ex in enumerate(examples):
        ex_dir = Path(ex["dir"])
        label = ex.get("label", ex_dir.name)
        is_swapped = ex.get("is_swapped", False)

        axes[ROW_CANVAS, col].set_title(label, fontsize=9, pad=4)

        # Row 0: Canvas
        canvas_path = ex_dir / "canvas.png"
        if canvas_path.exists():
            axes[ROW_CANVAS, col].imshow(np.array(Image.open(canvas_path).convert("RGB")))

        # Row 1: Mask @512 (gray)
        mask_path = ex_dir / "mask.png"
        if mask_path.exists():
            mask_pil = Image.open(mask_path).convert("L")
            axes[ROW_MASK, col].imshow(np.array(mask_pil), cmap="gray", vmin=0, vmax=255)

        # Precompute contour overlay for generation rows (red=core edge, yellow=band edge)
        alpha_overlay = None
        if mask_path.exists():
            from scipy import ndimage
            mask_t = _load_mask_tensor(mask_path, size=512, device="cpu")
            kernel = 512 // 64  # = 8
            core_64 = F.max_pool2d(mask_t, kernel_size=kernel)
            core_64 = (core_64 > 0.5).float()
            _, alpha_64, _, band_64 = create_latent_band_mask(core_64, band_mode)
            core_512 = F.interpolate(core_64, size=(512, 512), mode="nearest")[0, 0].numpy() > 0.5
            band_512 = F.interpolate(band_64, size=(512, 512), mode="nearest")[0, 0].numpy() > 0.5
            dilated_512 = core_512 | band_512
            # Extract edges: erode then XOR to get 1px contour
            core_edge = core_512 ^ ndimage.binary_erosion(core_512, iterations=1)
            dilated_edge = dilated_512 ^ ndimage.binary_erosion(dilated_512, iterations=1)
            band_edge = dilated_edge & ~core_512  # outer contour minus core interior
            # RGBA overlay: only edges visible
            overlay = np.zeros((512, 512, 4), dtype=np.float32)
            overlay[band_edge] = [1.0, 0.85, 0.0, 0.8]   # yellow edge
            overlay[core_edge] = [1.0, 0.0, 0.0, 0.8]     # red edge
            alpha_overlay = overlay

        # Row 2: Reference crop 1 (red overlay)
        ref_path = ex_dir / "reference.png"
        if ref_path.exists():
            axes[ROW_REF1, col].imshow(np.array(Image.open(ref_path).convert("RGB")))

        # Row 3: Reference crop 2 (red overlay)
        ref2_path = ex_dir / "reference_2.png"
        if ref2_path.exists():
            axes[ROW_REF2, col].imshow(np.array(Image.open(ref2_path).convert("RGB")))
        else:
            axes[ROW_REF2, col].text(
                0.5, 0.5, "(single crop)", ha="center", va="center",
                transform=axes[ROW_REF2, col].transAxes, fontsize=9, color="gray",
            )

        # Rows 4-6: nocfg, ip_35, ip_70
        for row_offset, variant_name in enumerate(["nocfg", "ip_35", "ip_70"]):
            gen_path = ex_dir / f"{variant_name}.png"
            if gen_path.exists():
                axes[ROW_NOCFG + row_offset, col].imshow(
                    np.array(Image.open(gen_path).convert("RGB")),
                )
        # Store overlay for second pass
        if alpha_overlay is not None:
            overlay_cells.append((col, alpha_overlay))

        # Diff row (only for swapped examples)
        if has_diff:
            if is_swapped:
                diff_path = ex_dir / "diff_ip_70.png"
                if diff_path.exists():
                    _diff_im = axes[ROW_DIFF, col].imshow(
                        np.array(Image.open(diff_path).convert("L")),
                        cmap="hot", vmin=0, vmax=255,
                    )
                    last_diff_im = _diff_im
                else:
                    axes[ROW_DIFF, col].text(
                        0.5, 0.5, "no diff", ha="center", va="center",
                        transform=axes[ROW_DIFF, col].transAxes, fontsize=9, color="gray",
                    )
            else:
                axes[ROW_DIFF, col].text(
                    0.5, 0.5, "(native)", ha="center", va="center",
                    transform=axes[ROW_DIFF, col].transAxes, fontsize=9, color="gray",
                )

    # Row labels
    for row_idx, row_label in enumerate(base_rows):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=10, rotation=90, labelpad=8)

    # Add colorbar for diff row (hot: black=0, red=low, yellow=high, white=max)
    if has_diff and last_diff_im is not None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[ROW_DIFF, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(last_diff_im, cax=cax)
        cbar.set_ticks([0, 128, 255])
        cbar.set_ticklabels(["0", "med", "max"])
        cbar.ax.tick_params(labelsize=7)

    # Turn off all axes
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save clean version (no overlay) for seam quality inspection
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved panel plot: %s", out_path)

    # Save overlay version (core/band contour edges on generated rows)
    if overlay_cells:
        overlay_artists = []
        for col, alpha_overlay in overlay_cells:
            for row_offset in range(3):  # nocfg, ip_35, ip_70
                art = axes[ROW_NOCFG + row_offset, col].imshow(alpha_overlay)
                overlay_artists.append(art)
        overlay_path = out_path.parent / (out_path.stem + "_overlay" + out_path.suffix)
        fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
        logger.info("Saved overlay plot: %s", overlay_path)
        # Remove overlay artists so they don't leak if figure is reused
        for art in overlay_artists:
            art.remove()

    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# Panel Implementations
# ═════════════════════════════════════════════════════════════════════════════


def _run_panel_a(
    pipeline, ip_adapter, device: str, band_mode: int, t2i_adapter,
    clip_align: bool, val_data_dir: Path, captions: Dict[str, str],
    out_dir: Path,
    noise_strength: float = 0.7,
    inference_mode: str = "different",
) -> Dict[str, Any]:
    """Panel A — Easy ID Sanity: 3 examples × 3 CFG = 9 runs."""
    panel_dir = out_dir / "panel_A_easy_id"
    results = {"panel": "A", "examples": []}

    # PANEL_A_EXAMPLES defined at module level
    base_seed = PANEL_SEEDS["A"]

    for i, ex in enumerate(PANEL_A_EXAMPLES):
        ex_prep = val_data_dir / ex["id"]
        if not ex_prep.exists():
            logger.warning("Panel A: %s not prepped, skipping", ex["id"])
            results["examples"].append({"id": ex["id"], "status": "skipped"})
            continue

        random.seed(base_seed + i)
        ex_out = panel_dir / ex["id"]

        canvas_t = _load_image_tensor(ex_prep / "canvas.png", device=device)
        mask_t = _load_mask_tensor(ex_prep / "placed_mask.png", device=device)

        clip_data = _prepare_clip_crops(
            ex_prep / "ref_image.png", ex_prep / "ref_mask.png",
            band_mode, clip_align, device,
        )

        caption = _get_caption(captions, ex["image_path"], example_id=ex["id"])

        _generate_and_save(
            pipeline, ip_adapter, canvas_t, mask_t, clip_data,
            ex["id"].split("_")[0], caption,  # anomaly_type = product name
            noise_strength=noise_strength, inference_mode=inference_mode,
            band_mode=band_mode, t2i_adapter=t2i_adapter,
            seed=base_seed + i, out_dir=ex_out, device=device,
        )
        results["examples"].append({"id": ex["id"], "status": "ok"})

    # Panel plot
    plot_examples = [
        {"dir": panel_dir / ex["id"], "label": ex["id"], "is_swapped": False}
        for ex in PANEL_A_EXAMPLES
        if (panel_dir / ex["id"]).exists()
    ]
    if plot_examples:
        _create_panel_plot(
            "Panel A \u2014 Easy ID Sanity", plot_examples, band_mode,
            panel_dir / "panel_plot.png",
        )

    results["status"] = "ok"
    return results


def _run_panel_b(
    pipeline, ip_adapter, device: str, band_mode: int, t2i_adapter,
    clip_align: bool, data_root: Path, captions: Dict[str, str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Panel B — ID Constrained Hard: 4 examples × 3 CFG = 12 runs.

    Anomaly image IS the canvas. CLIP crop from the anomaly image itself.
    """
    panel_dir = out_dir / "panel_B_id_constrained"
    results = {"panel": "B", "examples": []}
    base_seed = PANEL_SEEDS["B"]

    for i, ex in enumerate(PANEL_B_EXAMPLES):
        image_path = data_root / ex["image_path"]
        mask_path = data_root / ex["mask_path"]

        if not image_path.exists() or not mask_path.exists():
            logger.warning("Panel B: %s missing files, skipping", ex["id"])
            results["examples"].append({"id": ex["id"], "status": "skipped"})
            continue

        random.seed(base_seed + i)
        ex_out = panel_dir / ex["id"]

        canvas_t = _load_image_tensor(image_path, device=device)
        mask_t = _load_mask_tensor(mask_path, device=device)

        clip_data = _prepare_clip_crops(
            image_path, mask_path,
            band_mode, clip_align, device,
        )

        caption = _get_caption(captions, ex["image_path"], example_id=ex["id"])

        _generate_and_save(
            pipeline, ip_adapter, canvas_t, mask_t, clip_data,
            ex["id"].rsplit("_", 1)[0], caption,
            noise_strength=ex["noise_strength"], inference_mode="same",
            band_mode=band_mode, t2i_adapter=t2i_adapter,
            seed=base_seed + i, out_dir=ex_out, device=device,
        )
        results["examples"].append({"id": ex["id"], "status": "ok"})

    # Panel plot
    plot_examples = [
        {"dir": panel_dir / ex["id"], "label": ex["id"], "is_swapped": False}
        for ex in PANEL_B_EXAMPLES
        if (panel_dir / ex["id"]).exists()
    ]
    if plot_examples:
        _create_panel_plot(
            "Panel B \u2014 ID Constrained Hard", plot_examples, band_mode,
            panel_dir / "panel_plot.png",
        )

    results["status"] = "ok"
    return results


def _run_panel_c(
    pipeline, ip_adapter, device: str, band_mode: int, t2i_adapter,
    clip_align: bool, val_data_dir: Path, data_root: Path,
    captions: Dict[str, str], out_dir: Path,
) -> Dict[str, Any]:
    """Panel C — ID Counterfactual Cross-Object: 4 pairs × 3 CFG = 12 runs.

    Two scratch examples (bracket_black, nut). For each CFG:
    - Native: bracket→bracket, nut→nut
    - Swapped: bracket→nut, nut→bracket
    Then diff = |native - swapped| masked to anomaly region.
    """
    panel_dir = out_dir / "panel_C_id_counterfactual"
    results = {"panel": "C", "examples": []}
    base_seed = PANEL_SEEDS["C"]

    # PANEL_C_EXAMPLES defined at module level

    # Load all prepped canvases + CLIP data
    canvas_data = {}
    for i, ex in enumerate(PANEL_C_EXAMPLES):
        ex_prep = val_data_dir / ex["id"]
        if not ex_prep.exists():
            logger.warning("Panel C: %s not prepped, skipping", ex["id"])
            continue

        random.seed(base_seed + i)
        canvas_t = _load_image_tensor(ex_prep / "canvas.png", device=device)
        mask_t = _load_mask_tensor(ex_prep / "placed_mask.png", device=device)

        clip_data = _prepare_clip_crops(
            ex_prep / "ref_image.png", ex_prep / "ref_mask.png",
            band_mode, clip_align, device,
        )

        caption = _get_caption(captions, ex["image_path"], example_id=ex["id"])
        canvas_data[ex["id"]] = {
            "canvas_t": canvas_t, "mask_t": mask_t,
            "clip_data": clip_data, "caption": caption, "ex": ex,
        }

    if len(canvas_data) < 2:
        logger.warning("Panel C: need 2 examples, got %d, skipping", len(canvas_data))
        results["status"] = "skipped"
        return results

    ids = list(canvas_data.keys())

    # Generate native and swapped for all combinations
    pair_idx = 0
    for target_idx in range(len(ids)):
        for ref_idx in range(len(ids)):
            target_id = ids[target_idx]
            ref_id = ids[ref_idx]
            is_native = (target_idx == ref_idx)
            label = f"{target_id}_from_{ref_id}"
            pair_dir = panel_dir / label

            target = canvas_data[target_id]
            ref = canvas_data[ref_id]

            random.seed(base_seed + pair_idx)
            outputs = _generate_and_save(
                pipeline, ip_adapter,
                target["canvas_t"], target["mask_t"],
                ref["clip_data"],  # Use ref's CLIP crops
                ref_id.split("_")[0], ref["caption"],
                noise_strength=0.7, inference_mode="different",
                band_mode=band_mode, t2i_adapter=t2i_adapter,
                seed=base_seed + pair_idx, out_dir=pair_dir, device=device,
            )

            results["examples"].append({
                "id": label,
                "native": is_native,
                "target": target_id,
                "ref": ref_id,
                "status": "ok",
            })
            pair_idx += 1

    # Compute diff maps: |native - swapped| for each target
    for target_idx in range(len(ids)):
        target_id = ids[target_idx]
        native_dir = panel_dir / f"{target_id}_from_{target_id}"
        for ref_idx in range(len(ids)):
            if ref_idx == target_idx:
                continue
            ref_id = ids[ref_idx]
            swapped_dir = panel_dir / f"{target_id}_from_{ref_id}"

            for variant in CFG_VARIANTS:
                native_path = native_dir / f"{variant['name']}.png"
                swapped_path = swapped_dir / f"{variant['name']}.png"
                if native_path.exists() and swapped_path.exists():
                    native_t = _load_image_tensor(native_path, device="cpu")
                    swapped_t = _load_image_tensor(swapped_path, device="cpu")
                    diff_img = _compute_diff_map(native_t, swapped_t)
                    _save_pil(diff_img, swapped_dir / f"diff_{variant['name']}.png")

    # Panel plot: column order [A→A, A→B, B→B, B→A]
    if len(ids) == 2:
        a, b = ids[0], ids[1]
        plot_examples = [
            {"dir": panel_dir / f"{a}_from_{a}", "label": f"{a}\u2192{a}", "is_swapped": False},
            {"dir": panel_dir / f"{a}_from_{b}", "label": f"{a}\u2192{b}", "is_swapped": True},
            {"dir": panel_dir / f"{b}_from_{b}", "label": f"{b}\u2192{b}", "is_swapped": False},
            {"dir": panel_dir / f"{b}_from_{a}", "label": f"{b}\u2192{a}", "is_swapped": True},
        ]
        plot_examples = [ex for ex in plot_examples if Path(ex["dir"]).exists()]
        if plot_examples:
            _create_panel_plot(
                "Panel C \u2014 ID Counterfactual Cross-Object", plot_examples, band_mode,
                panel_dir / "panel_plot.png",
            )

    results["status"] = "ok"
    return results


def _run_panel_d(
    pipeline, ip_adapter, device: str, band_mode: int, t2i_adapter,
    clip_align: bool, val_data_dir: Path, captions: Dict[str, str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Panel D — OOD Counterfactual Cross-Domain: cashew ↔ PCB.

    Two domains (cashew easy, PCB). For each CFG:
    - Native: cashew→cashew, pcb→pcb
    - Swapped: cashew→pcb, pcb→cashew
    Then diff = |native - swapped| masked to anomaly region.
    """
    panel_dir = out_dir / "panel_D_ood_counterfactual"
    results = {"panel": "D", "examples": []}
    base_seed = PANEL_SEEDS["D"]

    # Load all prepped canvases + CLIP data
    canvas_data = {}
    for i, ex in enumerate(PANEL_D_EXAMPLES):
        ex_prep = val_data_dir / ex["id"]
        if not ex_prep.exists():
            logger.warning("Panel D: %s not prepped, skipping", ex["id"])
            continue

        random.seed(base_seed + i)
        canvas_t = _load_image_tensor(ex_prep / "canvas.png", device=device)
        mask_t = _load_mask_tensor(ex_prep / "placed_mask.png", device=device)

        clip_data = _prepare_clip_crops(
            ex_prep / "ref_image.png", ex_prep / "ref_mask.png",
            band_mode, clip_align, device,
        )

        caption = _get_caption(captions, ex.get("image_path", ""), example_id=ex["id"])
        canvas_data[ex["id"]] = {
            "canvas_t": canvas_t, "mask_t": mask_t,
            "clip_data": clip_data, "caption": caption, "ex": ex,
        }

    if len(canvas_data) < 2:
        logger.warning("Panel D: need 2 examples, got %d, skipping", len(canvas_data))
        results["status"] = "skipped"
        return results

    ids = list(canvas_data.keys())

    # Generate native and swapped for all combinations
    pair_idx = 0
    for target_idx in range(len(ids)):
        for ref_idx in range(len(ids)):
            target_id = ids[target_idx]
            ref_id = ids[ref_idx]
            is_native = (target_idx == ref_idx)
            label = f"{target_id}_from_{ref_id}"
            pair_dir = panel_dir / label

            target = canvas_data[target_id]
            ref = canvas_data[ref_id]

            random.seed(base_seed + pair_idx)
            _generate_and_save(
                pipeline, ip_adapter,
                target["canvas_t"], target["mask_t"],
                ref["clip_data"],
                ref_id.split("_")[0], ref["caption"],
                noise_strength=0.7, inference_mode="different",
                band_mode=band_mode, t2i_adapter=t2i_adapter,
                seed=base_seed + pair_idx, out_dir=pair_dir, device=device,
            )

            results["examples"].append({
                "id": label, "native": is_native,
                "target": target_id, "ref": ref_id, "status": "ok",
            })
            pair_idx += 1

    # Compute diff maps: |native - swapped| for each target
    for target_idx in range(len(ids)):
        target_id = ids[target_idx]
        native_dir = panel_dir / f"{target_id}_from_{target_id}"
        for ref_idx in range(len(ids)):
            if ref_idx == target_idx:
                continue
            ref_id = ids[ref_idx]
            swapped_dir = panel_dir / f"{target_id}_from_{ref_id}"

            for variant in CFG_VARIANTS:
                native_path = native_dir / f"{variant['name']}.png"
                swapped_path = swapped_dir / f"{variant['name']}.png"
                if native_path.exists() and swapped_path.exists():
                    native_t = _load_image_tensor(native_path, device="cpu")
                    swapped_t = _load_image_tensor(swapped_path, device="cpu")
                    diff_img = _compute_diff_map(native_t, swapped_t)
                    _save_pil(diff_img, swapped_dir / f"diff_{variant['name']}.png")

    # Panel plot: column order [A→A, A→B, B→B, B→A]
    if len(ids) == 2:
        a, b = ids[0], ids[1]
        plot_examples = [
            {"dir": panel_dir / f"{a}_from_{a}", "label": f"{a}\u2192{a}", "is_swapped": False},
            {"dir": panel_dir / f"{a}_from_{b}", "label": f"{a}\u2192{b}", "is_swapped": True},
            {"dir": panel_dir / f"{b}_from_{b}", "label": f"{b}\u2192{b}", "is_swapped": False},
            {"dir": panel_dir / f"{b}_from_{a}", "label": f"{b}\u2192{a}", "is_swapped": True},
        ]
        plot_examples = [ex for ex in plot_examples if Path(ex["dir"]).exists()]
        if plot_examples:
            _create_panel_plot(
                "Panel D \u2014 OOD Counterfactual Cross-Domain", plot_examples, band_mode,
                panel_dir / "panel_plot.png",
            )

    results["status"] = "ok"
    return results


def _run_panel_e(
    pipeline, ip_adapter, device: str, band_mode: int, t2i_adapter,
    clip_align: bool, val_data_dir: Path, captions: Dict[str, str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Panel E — OOD Hard Deformation: 2 cashew hard × 3 CFG = 6 runs."""
    panel_dir = out_dir / "panel_E_ood_hard_deformation"
    results = {"panel": "E", "examples": []}
    base_seed = PANEL_SEEDS["E"]

    for i, ex in enumerate(PANEL_E_EXAMPLES):
        ex_prep = val_data_dir / ex["id"]
        if not ex_prep.exists():
            logger.warning("Panel E: %s not prepped, skipping", ex["id"])
            results["examples"].append({"id": ex["id"], "status": "skipped"})
            continue

        random.seed(base_seed + i)
        ex_out = panel_dir / ex["id"]

        canvas_t = _load_image_tensor(ex_prep / "canvas.png", device=device)
        mask_t = _load_mask_tensor(ex_prep / "placed_mask.png", device=device)

        clip_data = _prepare_clip_crops(
            ex_prep / "ref_image.png", ex_prep / "ref_mask.png",
            band_mode, clip_align, device,
        )

        caption = _get_caption(captions, ex.get("image_path", ""), example_id=ex["id"])
        noise_strength = ex.get("noise_strength", 1.0)

        _generate_and_save(
            pipeline, ip_adapter, canvas_t, mask_t, clip_data,
            "cashew", caption,
            noise_strength=noise_strength, inference_mode="different",
            band_mode=band_mode, t2i_adapter=t2i_adapter,
            seed=base_seed + i, out_dir=ex_out, device=device,
        )
        results["examples"].append({"id": ex["id"], "status": "ok"})

    # Panel plot
    plot_examples = [
        {"dir": panel_dir / ex["id"], "label": ex["id"], "is_swapped": False}
        for ex in PANEL_E_EXAMPLES
        if (panel_dir / ex["id"]).exists()
    ]
    if plot_examples:
        _create_panel_plot(
            "Panel E \u2014 OOD Hard Deformation", plot_examples, band_mode,
            panel_dir / "panel_plot.png",
        )

    results["status"] = "ok"
    return results


def _run_panel_f(
    pipeline, ip_adapter, device: str, band_mode: int, t2i_adapter,
    clip_align: bool, val_data_dir: Path, captions: Dict[str, str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Panel F — OOD Easy Cashew: 4 easy cashew × 3 CFG = 12 runs."""
    panel_dir = out_dir / "panel_F_ood_easy_cashew"
    results = {"panel": "F", "examples": []}
    base_seed = PANEL_SEEDS["F"]

    for i, ex in enumerate(PANEL_F_EXAMPLES):
        ex_prep = val_data_dir / ex["id"]
        if not ex_prep.exists():
            logger.warning("Panel F: %s not prepped, skipping", ex["id"])
            results["examples"].append({"id": ex["id"], "status": "skipped"})
            continue

        random.seed(base_seed + i)
        ex_out = panel_dir / ex["id"]

        canvas_t = _load_image_tensor(ex_prep / "canvas.png", device=device)
        mask_t = _load_mask_tensor(ex_prep / "placed_mask.png", device=device)

        clip_data = _prepare_clip_crops(
            ex_prep / "ref_image.png", ex_prep / "ref_mask.png",
            band_mode, clip_align, device,
        )

        caption = _get_caption(captions, ex.get("image_path", ""), example_id=ex["id"])

        _generate_and_save(
            pipeline, ip_adapter, canvas_t, mask_t, clip_data,
            "cashew", caption,
            noise_strength=0.7, inference_mode="different",
            band_mode=band_mode, t2i_adapter=t2i_adapter,
            seed=base_seed + i, out_dir=ex_out, device=device,
        )
        results["examples"].append({"id": ex["id"], "status": "ok"})

    # Panel plot
    plot_examples = [
        {"dir": panel_dir / ex["id"], "label": ex["id"], "is_swapped": False}
        for ex in PANEL_F_EXAMPLES
        if (panel_dir / ex["id"]).exists()
    ]
    if plot_examples:
        _create_panel_plot(
            "Panel F \u2014 OOD Easy Cashew", plot_examples, band_mode,
            panel_dir / "panel_plot.png",
        )

    results["status"] = "ok"
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═════════════════════════════════════════════════════════════════════════════


def run_validation_suite(
    pipeline,
    ip_adapter,
    step: int,
    output_dir: Path,
    device: str,
    band_mode: int = 2,
    t2i_adapter=None,
    clip_align: bool = True,
    data_root: Path = None,
    val_data_dir: Path = None,
    captions_file: Path = None,
    panels: List[str] = None,
) -> Dict[str, Any]:
    """Run the 6-panel validation suite.

    Args:
        pipeline: Loaded SD inpainting pipeline.
        ip_adapter: Loaded IP-Adapter (trained weights).
        step: Current training step (for folder naming).
        output_dir: Base output directory (val_suite subfolder created).
        device: "cuda" or "cpu".
        band_mode: Band dilation mode (1 or 2).
        t2i_adapter: T2I-Adapter instance or None.
        clip_align: Whether to use CLIP-UNet roundtrip alignment.
        data_root: Root for training data (for Panel B image paths).
        val_data_dir: Directory with prepped validation data (for A, C).
        captions_file: Path to captions JSON.
        panels: Which panels to run (default all: ["A","B","C","D","E","F"]).

    Returns:
        Summary dict with per-panel results.
    """
    if panels is None:
        panels = ["A", "B", "C", "D", "E", "F"]

    suite_dir = output_dir / "val_suite" / f"checkpoint_{step:05d}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    # Load captions
    captions: Dict[str, str] = {}
    if captions_file and Path(captions_file).exists():
        with open(captions_file, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            captions = {entry["image_path"]: entry["caption"] for entry in raw}
        elif isinstance(raw, dict):
            captions = raw

    summary: Dict[str, Any] = {"step": step, "panels": {}}
    t0 = time.time()

    ip_adapter.eval()

    for panel in panels:
        panel_t0 = time.time()
        print(f"  [val_suite] Panel {panel}...", end=" ", flush=True)

        try:
            if panel == "A":
                if val_data_dir is None:
                    logger.warning("Panel A requires --val-data-dir, skipping")
                    summary["panels"]["A"] = {"status": "skipped"}
                    print("skipped (no val_data_dir)")
                    continue
                result = _run_panel_a(
                    pipeline, ip_adapter, device, band_mode, t2i_adapter,
                    clip_align, Path(val_data_dir), captions, suite_dir,
                )
            elif panel == "B":
                if data_root is None:
                    logger.warning("Panel B requires --data-root, skipping")
                    summary["panels"]["B"] = {"status": "skipped"}
                    print("skipped (no data_root)")
                    continue
                result = _run_panel_b(
                    pipeline, ip_adapter, device, band_mode, t2i_adapter,
                    clip_align, Path(data_root), captions, suite_dir,
                )
            elif panel == "C":
                if val_data_dir is None or data_root is None:
                    logger.warning("Panel C requires --val-data-dir and --data-root, skipping")
                    summary["panels"]["C"] = {"status": "skipped"}
                    print("skipped (missing dirs)")
                    continue
                result = _run_panel_c(
                    pipeline, ip_adapter, device, band_mode, t2i_adapter,
                    clip_align, Path(val_data_dir), Path(data_root), captions, suite_dir,
                )
            elif panel == "D":
                if val_data_dir is None:
                    logger.warning("Panel D requires --val-data-dir, skipping")
                    summary["panels"]["D"] = {"status": "skipped"}
                    print("skipped (no val_data_dir)")
                    continue
                result = _run_panel_d(
                    pipeline, ip_adapter, device, band_mode, t2i_adapter,
                    clip_align, Path(val_data_dir), captions, suite_dir,
                )
            elif panel == "E":
                if val_data_dir is None:
                    logger.warning("Panel E requires --val-data-dir, skipping")
                    summary["panels"]["E"] = {"status": "skipped"}
                    print("skipped (no val_data_dir)")
                    continue
                result = _run_panel_e(
                    pipeline, ip_adapter, device, band_mode, t2i_adapter,
                    clip_align, Path(val_data_dir), captions, suite_dir,
                )
            elif panel == "F":
                if val_data_dir is None:
                    logger.warning("Panel F requires --val-data-dir, skipping")
                    summary["panels"]["F"] = {"status": "skipped"}
                    print("skipped (no val_data_dir)")
                    continue
                result = _run_panel_f(
                    pipeline, ip_adapter, device, band_mode, t2i_adapter,
                    clip_align, Path(val_data_dir), captions, suite_dir,
                )
            else:
                logger.warning("Unknown panel: %s", panel)
                continue
        except Exception as e:
            logger.error("Panel %s failed: %s", panel, e, exc_info=True)
            result = {"panel": panel, "status": "error", "error": str(e)}

        elapsed = time.time() - panel_t0
        result["elapsed_s"] = round(elapsed, 1)
        summary["panels"][panel] = result
        print(f"{result.get('status', '?')} ({elapsed:.1f}s)")

        torch.cuda.empty_cache()

    summary["total_elapsed_s"] = round(time.time() - t0, 1)

    # Write summary
    with open(suite_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  [val_suite] Done in {summary['total_elapsed_s']:.1f}s -> {suite_dir}")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ═════════════════════════════════════════════════════════════════════════════


def _load_checkpoint(checkpoint_dir: Path, device: str):
    """Load pipeline + IP-Adapter + T2I-Adapter from a training checkpoint.

    Loads EMA weights if available (training validation uses EMA).
    T2I-Adapter weights are inside training_state.pt, not a separate file.
    """
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter
    from src.models.t2i_adapter import T2IAdapter
    from src.utils.optim_utils import build_norm_param_id_set, split_decay_no_decay

    # Load config from checkpoint
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Load pipeline
    sd_version = config.get("sd_version", "sd_1.5")
    pipeline = create_pipeline(sd_version, device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()
    pipeline.vae.float()  # VAE loads as fp16, needs fp32 for decode

    # Find checkpoint files
    ip_ckpt = checkpoint_dir / "ip_adapter.pt"
    if not ip_ckpt.exists():
        raise FileNotFoundError(f"No ip_adapter.pt in {checkpoint_dir}")

    # Detect num_layers from state dict
    state = torch.load(ip_ckpt, map_location="cpu", weights_only=True)
    sa_layers = 0
    for k in state.keys():
        if k.startswith("masked_self_attn.layers."):
            layer_idx = int(k.split(".")[2])
            sa_layers = max(sa_layers, layer_idx + 1)
    if sa_layers == 0:
        sa_layers = 3

    # Detect force_gates from config or state dict
    # Old checkpoints don't save force_gates in config.json — detect from state dict:
    # if no gate params exist in state dict, it was forced gates
    force_gates = config.get("force_gates", False)
    learnable_gates = config.get("learnable_gates", True)
    if not force_gates and "force_gates" not in config:
        # Heuristic: check if any gate param exists in saved weights
        has_gate_params = any("gate.gate" in k for k in state.keys())
        if not has_gate_params:
            force_gates = True
            learnable_gates = False
            print("  Detected force_gates=True (no gate params in checkpoint)")

    # Create IP-Adapter from config
    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=config.get("adapter_type", "plus"),
        num_tokens=config.get("num_tokens", 16),
        scale=config.get("scale", 1.0),
        load_pretrained=False,
        anomaly_residual=config.get("anomaly_residual", False),
        mask_visual=config.get("mask_visual", True),
        visual_mode=config.get("visual_mode", 3),
        sa_num_layers=sa_layers,
        force_gates=force_gates,
        learnable_gates=learnable_gates,
    )
    ip_adapter.load_state_dict(state, strict=False)
    ip_adapter.to(device)

    # Ensure trainable modules are fp32 (matches training setup)
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # Load training_state.pt for T2I-Adapter + EMA
    t2i_adapter = None
    training_state_path = checkpoint_dir / "training_state.pt"
    training_state = None
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)

    # T2I-Adapter: load from training_state.pt (not a separate file)
    if training_state and "t2i_adapter" in training_state:
        t2i_state = training_state["t2i_adapter"]
        in_ch = 2
        for k, v in t2i_state.items():
            if "conv_in" in k and v.dim() == 4:
                in_ch = v.shape[1]
                break
        t2i_adapter = T2IAdapter(in_channels=in_ch).to(device).float()
        t2i_adapter.load_state_dict(t2i_state)
        print("  T2I-Adapter loaded from training_state.pt")

    # Apply EMA weights if available (matches training validation behavior)
    if training_state and "ema" in training_state:
        ema_state = training_state["ema"]
        shadow = ema_state["shadow"]

        # Reconstruct trainable param list in same order as training:
        # A_decay, A_no_decay, B_decay, B_no_decay, C_gates
        group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
        group_b_modules = []
        if hasattr(ip_adapter, 'masked_self_attn'):
            group_b_modules.append(ip_adapter.masked_self_attn)
        if t2i_adapter is not None:
            group_b_modules.append(t2i_adapter)

        norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)
        a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

        group_c_params = []
        if hasattr(ip_adapter, 'masked_self_attn') and ip_adapter.masked_self_attn.learnable_gates:
            for layer in ip_adapter.masked_self_attn.layers:
                group_c_params.append(layer["attn_gate"].gate)
                if "ff_gate" in layer:
                    group_c_params.append(layer["ff_gate"].gate)
        if t2i_adapter is not None:
            for s in t2i_adapter.scales:
                group_c_params.append(s)
        group_c_ids = {id(p) for p in group_c_params}

        b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
        b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
        b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

        trainable_params = a_decay + a_no_decay + b_decay + b_no_decay + group_c_params

        if len(trainable_params) == len(shadow):
            # Shape sanity check
            shapes_match = all(
                p.shape == s.shape for p, s in zip(trainable_params, shadow)
            )
            if shapes_match:
                for p, s in zip(trainable_params, shadow):
                    p.data.copy_(s.to(p.device))
                print(f"  EMA weights applied ({len(shadow)} params, decay={ema_state.get('decay', 0):.6f})")
            else:
                print("  WARNING: EMA shape mismatch — using online weights")
        else:
            print(f"  WARNING: EMA param count mismatch: model={len(trainable_params)}, shadow={len(shadow)} — using online weights")

    ip_adapter.eval()
    if t2i_adapter is not None:
        t2i_adapter.eval()

    return pipeline, ip_adapter, t2i_adapter


def main():
    import argparse
    parser = argparse.ArgumentParser(description="6-Panel Validation Suite (standalone)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for validation results")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root for training data (Panel B)")
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Prepped validation data directory (Panels A, C)")
    parser.add_argument("--captions-file", type=str, default=None,
                        help="Path to captions JSON")
    parser.add_argument("--panels", type=str, nargs="+", default=["A", "B", "C", "D", "E", "F"],
                        help="Which panels to run")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--band-mode", type=int, default=2, choices=[1, 2])
    parser.add_argument("--no-clip-align", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    val_data_dir = Path(args.val_data_dir) if args.val_data_dir else None
    if val_data_dir and not val_data_dir.is_absolute():
        val_data_dir = project_root / val_data_dir

    captions_file = Path(args.captions_file) if args.captions_file else None
    if captions_file and not captions_file.is_absolute():
        captions_file = project_root / captions_file

    # Detect step from checkpoint dir name
    step = 0
    if checkpoint_dir.stem.startswith("checkpoint_"):
        try:
            step = int(checkpoint_dir.stem.split("_", 1)[1])
        except ValueError:
            pass

    print(f"Loading checkpoint from {checkpoint_dir}...")
    pipeline, ip_adapter, t2i_adapter = _load_checkpoint(checkpoint_dir, args.device)

    run_validation_suite(
        pipeline=pipeline,
        ip_adapter=ip_adapter,
        step=step,
        output_dir=output_dir,
        device=args.device,
        band_mode=args.band_mode,
        t2i_adapter=t2i_adapter,
        clip_align=not args.no_clip_align,
        data_root=data_root,
        val_data_dir=val_data_dir,
        captions_file=captions_file,
        panels=args.panels,
    )


if __name__ == "__main__":
    main()
