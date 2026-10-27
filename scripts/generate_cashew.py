"""Generate synthetic cashew anomalies using trained Anomagic checkpoint.

Reads placed masks + manifests, loads a trained checkpoint, and generates
anomaly images with comparison panels.

Supports both experiment layouts via --experiment flag:
- ResNet (default): anomaly/masks/, source/reference_anomalies/, source/canvas_normals/
- UniNet: synthetic/placed_masks/, source/anomalies/, source/normals/canvas/

Two inference modes (matching the two placement strategies):
- Easy (surface): noise_strength=0.7 — partial denoising preserves canvas context
- Hard (deformation): noise_strength=1.0 — pure noise, full generation within mask
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.crop_utils import clip_crop_multi
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool, unet_roundtrip_masks
from src.utils.placement_utils import place_easy_mask, place_hard_mask
from src.inference.generate import generate_anomagic_single


# ── Paths ─────────────────────────────────────────────────────────────────
CASHEW_ROOT = (Path(__file__).resolve().parent.parent / "anomverse_extension" / "datasets" / "VisA_validation_dataset"
               / "datasets" / "easy_test" / "cashew")   # project-relative (was a hard-coded Windows path until 2026-10-03)
EXTRA_NORMALS = CASHEW_ROOT / "Data" / "Images" / "Normal"
IMAGE_ANNO_CSV = CASHEW_ROOT / "image_anno.csv"
VIZ_SIZE = 512

# Experiment-specific paths (set in main() via --experiment flag)
EXPERIMENT = "ResNet"
EXP = None
CANVAS_IMGS = None


def setup_experiment(experiment: str):
    """Set experiment-specific global paths."""
    global EXPERIMENT, EXP, CANVAS_IMGS
    EXPERIMENT = experiment
    EXP = CASHEW_ROOT / f"experiment_{experiment}"
    if experiment == "ResNet":
        CANVAS_IMGS = EXP / "source" / "canvas_normals" / "imgs"
    else:  # UniNet
        CANVAS_IMGS = EXP / "source" / "normals" / "canvas" / "imgs"


# ── Utilities ─────────────────────────────────────────────────────────────

def overlay_mask_rgba(
    img: Image.Image, mask: np.ndarray, color: tuple, alpha: float,
) -> Image.Image:
    """Overlay a binary mask on an image with color + alpha."""
    base = img.convert("RGBA")
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    overlay[:, :, 3] = (mask * alpha * 255).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(overlay)).convert("RGB")


def mask_boundary_overlay(
    mask_arr: np.ndarray, color_rgba: tuple, width: int = 2,
) -> Image.Image:
    """Create RGBA overlay with only the boundary of a binary mask.

    Uses morphological erosion: boundary = mask - eroded(mask), then
    dilates the boundary by ``width`` for visibility.
    """
    from scipy.ndimage import binary_erosion, binary_dilation
    binary = mask_arr > 0.5
    eroded = binary_erosion(binary, iterations=1, border_value=0)
    boundary = binary & ~eroded
    if width > 1:
        boundary = binary_dilation(boundary, iterations=width - 1)
    overlay = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
    overlay[boundary, 0] = color_rgba[0]
    overlay[boundary, 1] = color_rgba[1]
    overlay[boundary, 2] = color_rgba[2]
    overlay[boundary, 3] = color_rgba[3]
    return Image.fromarray(overlay)


def _build_clip_grid(
    crop_pil: Image.Image,
    crop_mask_t: torch.Tensor,
    dilate: bool,
    mask_16: np.ndarray,
    dilated_16: np.ndarray,
    viz_size: int,
) -> Image.Image:
    """Build a CLIP 16×16 grid visualization with boundary-only mask outlines.

    Args:
        crop_pil: Crop image as PIL (any size, will be resized)
        crop_mask_t: Crop mask tensor [1, H, W]
        dilate: Whether to show dilation ring (hard mode)
        mask_16: Original mask at 16×16 as numpy
        dilated_16: Dilated mask at 16×16 as numpy
        viz_size: Output size in pixels

    Returns:
        RGB PIL image of the grid visualization
    """
    base = crop_pil.resize((viz_size, viz_size), Image.LANCZOS).convert("RGBA")
    cell_sz = viz_size // 16
    draw = ImageDraw.Draw(base)

    # Light grid lines
    for k in range(17):
        pos = k * cell_sz
        draw.line([(pos, 0), (pos, viz_size - 1)], fill=(255, 255, 255, 50), width=1)
        draw.line([(0, pos), (viz_size - 1, pos)], fill=(255, 255, 255, 50), width=1)

    def _cell_in(mask_arr, rr, cc):
        if 0 <= rr < 16 and 0 <= cc < 16:
            return mask_arr[rr, cc] > 0.5
        return False

    def _draw_region_boundary(mask_arr, color, lw=2):
        """Draw boundary edges of a 16×16 binary mask region."""
        for r in range(16):
            for c in range(16):
                if not _cell_in(mask_arr, r, c):
                    continue
                y0 = r * cell_sz
                x0 = c * cell_sz
                y1 = (r + 1) * cell_sz - 1
                x1 = (c + 1) * cell_sz - 1
                if not _cell_in(mask_arr, r - 1, c):
                    draw.line([(x0, y0), (x1, y0)], fill=color, width=lw)
                if not _cell_in(mask_arr, r + 1, c):
                    draw.line([(x0, y1), (x1, y1)], fill=color, width=lw)
                if not _cell_in(mask_arr, r, c - 1):
                    draw.line([(x0, y0), (x0, y1)], fill=color, width=lw)
                if not _cell_in(mask_arr, r, c + 1):
                    draw.line([(x1, y0), (x1, y1)], fill=color, width=lw)

    if dilate:
        # Two boundaries: cyan outer (dilation ring) then red inner (core)
        dilation_only = ((dilated_16 > 0.5) & (mask_16 < 0.5)).astype(np.float32)
        _draw_region_boundary(dilation_only, (0, 220, 255, 220), lw=2)
        _draw_region_boundary(mask_16, (255, 40, 40, 220), lw=2)
    else:
        # Single boundary: red (core only, no dilation)
        _draw_region_boundary(mask_16, (255, 40, 40, 220), lw=2)

    return base.convert("RGB")


def add_label(
    img: Image.Image, text: str, font_size: int = 18,
) -> Image.Image:
    """Add a text label bar on top of an image."""
    bar_h = font_size + 10
    out = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 3), text, fill=(255, 255, 200), font=font)
    return out


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a grayscale mask and binarize at ANY non-zero pixel.

    VisA ground-truth masks store defect pixels with small values (1, 2, 5), so the old 0.5 threshold on the
    /255 image read them as empty — a reference mask loaded that way silently produced a black generation.
    Placed/refined masks are 0/255, for which this is identical.
    """
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.float32)


def parse_defect_types(csv_path: Path) -> dict[str, str]:
    """Parse image_anno.csv → {anomaly_id: defect_type_string}.

    Uses csv.reader to handle quoted fields like "burnt,corner or edge breakage".
    """
    defect_map = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            image_path, label, _mask = row[0], row[1], row[2] if len(row) > 2 else ""
            if label == "normal":
                continue
            # Extract anomaly ID from path like "cashew/Data/Images/Anomaly/000.JPG"
            stem = Path(image_path).stem  # "000"
            defect_map[stem] = label
    return defect_map


def resolve_anomaly_paths(ref_id: str, difficulty: str) -> tuple[Path, Path]:
    """Resolve paths to the original anomaly image and mask in Data/."""
    subdir = "easy_imgs" if difficulty == "easy" else "hard_imgs"
    img_path = CASHEW_ROOT / "Data" / "Images" / "Anomaly" / subdir / f"{ref_id}.JPG"
    mask_path = CASHEW_ROOT / "Data" / "Masks" / "Anomaly" / f"{ref_id}.png"
    return img_path, mask_path


def resolve_canvas_image(canvas_id: str) -> Path:
    """Find canvas normal image — standard dir first, then extra normals pool."""
    standard = CANVAS_IMGS / f"{canvas_id}.JPG"
    if standard.exists():
        return standard
    extra = EXTRA_NORMALS / f"{canvas_id}.JPG"
    if extra.exists():
        return extra
    raise FileNotFoundError(
        f"Canvas {canvas_id} not found in {CANVAS_IMGS} or {EXTRA_NORMALS}"
    )


# ── Model loading ─────────────────────────────────────────────────────────

def load_models(checkpoint_dir: Path, device: str = "cuda"):
    """Load pipeline + IP-Adapter + T2I-Adapter from a training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)

    # Load config from checkpoint
    config_path = checkpoint_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    adapter_type = config["adapter_type"]
    num_tokens = config["num_tokens"]
    scale = config.get("scale", 1.0)
    mask_visual = config.get("mask_visual", True)
    visual_mode = config.get("visual_mode", 0)

    # Auto-detect self-attention layer count from checkpoint weights
    ip_state = torch.load(checkpoint_dir / "ip_adapter.pt", map_location="cpu", weights_only=False)
    sa_layers = set()
    for k in ip_state:
        if "masked_self_attn.layers." in k:
            idx = k.split("masked_self_attn.layers.")[1].split(".")[0]
            sa_layers.add(int(idx))
    sa_num_layers = max(sa_layers) + 1 if sa_layers else 1
    del ip_state

    print(f"Checkpoint config: type={adapter_type}, K={num_tokens}, "
          f"scale={scale}, mask_visual={mask_visual}, visual_mode={visual_mode}, "
          f"sa_layers={sa_num_layers}")

    # Pipeline
    from src.models.base import create_pipeline
    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()
    # Match training: explicit fp32 for VAE + text encoder (AMP handles UNet internally)
    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # IP-Adapter (init with pretrained, then overwrite with finetuned)
    from src.models.ip_adapter import create_ip_adapter
    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=adapter_type,
        num_tokens=num_tokens,
        scale=scale,
        load_pretrained=True,
        mask_visual=mask_visual,
        visual_mode=visual_mode,
        sa_num_layers=sa_num_layers,
    )
    ip_adapter.freeze_image_encoder()
    ip_adapter.load_finetuned(checkpoint_dir)
    # Re-cast to fp32 after loading
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()
    print(f"Loaded IP-Adapter from {checkpoint_dir / 'ip_adapter.pt'}")

    # T2I-Adapter (from training_state.pt)
    t2i_adapter = None
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        band_mode_saved = state.get("band_mode", 2)
        if "t2i_adapter" in state:
            from src.models.t2i_adapter import T2IAdapter
            t2i_adapter = T2IAdapter(in_channels=2).to(device)
            t2i_adapter.load_state_dict(state["t2i_adapter"])
            t2i_adapter.eval()
            print(f"Loaded T2I-Adapter from training_state.pt")
        step = state.get("step", "?")
        print(f"Checkpoint step: {step}, band_mode: {band_mode_saved}")
        del state
        torch.cuda.empty_cache()

    # Set all trainable parts to eval mode
    ip_adapter.masked_self_attn.eval()
    ip_adapter.image_projection.eval()
    for proc in ip_adapter.attn_processors.values():
        proc.eval()

    return pipeline, ip_adapter, t2i_adapter


# ── Generation ────────────────────────────────────────────────────────────

LAST_STATE: dict | None = None   # set by generate_one: this pass's state for latent-carry multi-edit


def generate_one(
    pipeline, ip_adapter, t2i_adapter,
    canvas_id: str, ref_id: str, difficulty: str,
    defect_map: dict,
    noise_strength: float, num_steps: int, guidance_scale: float,
    band_mode: int, seed: int,
    device: str = "cuda",
    layout: str = "A",
    force_no_dilate: bool = False,
    ref_img_override: Path | None = None,
    ref_mask_override: Path | None = None,
    placed_mask_override: Path | None = None,
    save_raw: bool = False,
    canvas_override: Path | None = None,
    context: dict | None = None,
    raw_background: bool = False,
    cfg_mode: str = "visual",
    clip_align: str = "raw",
    caption_override: str | None = None,
    even_steps: bool = False,
) -> Path | None:
    """Generate one synthetic anomaly and save comparison panel.

    canvas_override lets the caller paint into an intermediate image (e.g. the output of a previous pass)
    instead of the clean canvas resolved from canvas_id. That RE-ENCODES the previous output, so every extra pass
    compresses the background once more - prefer ``context``.

    context: latent-carry multi-edit. {"latents": state latent, "pixels": [1,3,512,512] in [-1,1]} of the previous
    pass, exactly as left in ``LAST_STATE``. The background is then decoded once however many passes run.
    After every call ``LAST_STATE`` holds this pass's state for the next pass.
    """
    global LAST_STATE
    # 1. Load canvas normal image
    if context is not None:
        canvas_t = context["pixels"].to(device)
        canvas_512 = Image.fromarray(((canvas_t[0].cpu().clamp(-1, 1) + 1) / 2 * 255).round().byte()
                                     .permute(1, 2, 0).numpy())
    else:
        canvas_path = canvas_override or resolve_canvas_image(canvas_id)
        canvas_pil = Image.open(canvas_path).convert("RGB")
        canvas_512 = canvas_pil.resize((512, 512), Image.LANCZOS)
        canvas_t = torch.from_numpy(
            np.array(canvas_512).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, 512, 512] in [-1, 1]

    # 2. Load placed mask (image resolution → 512)
    if placed_mask_override is not None:
        placed_mask_path = placed_mask_override
    elif EXPERIMENT == "ResNet":
        placed_mask_path = EXP / "anomaly" / "masks" / difficulty / f"{canvas_id}.png"
    else:
        placed_mask_path = EXP / "synthetic" / "placed_masks" / difficulty / f"{canvas_id}.png"
    if not placed_mask_path.exists():
        print(f"  WARNING: placed mask not found: {placed_mask_path}")
        return None
    placed_pil = Image.open(placed_mask_path).convert("L")
    # Max-pool, not NEAREST: subsampling a native-resolution mask (1274 px) down to 512 drops pixels of small
    # defects before they ever reach the latent grid, and the refiner max-pools the same mask (see
    # perceptual_mask_utils.compute_dilated_mask_512), so NEAREST here made the two disagree.
    mask_np = downsample_mask_maxpool(
        torch.from_numpy((np.array(placed_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32))[None, None],
        512,
    )[0, 0].numpy()
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(device)  # [1, 1, 512, 512]

    # 3. Load reference anomaly image + its own mask → CLIP crop
    if ref_img_override is not None:
        ref_img_path = ref_img_override
        ref_mask_path = ref_mask_override
    elif EXPERIMENT == "ResNet":
        ref_imgs_dir = EXP / "source" / "reference_anomalies" / difficulty / "imgs"
        ref_masks_dir = EXP / "source" / "reference_anomalies" / difficulty / "masks"
        ref_img_path = ref_imgs_dir / f"{ref_id}.JPG"
        ref_mask_path = ref_masks_dir / f"{ref_id}.png"
    else:
        ref_imgs_dir = EXP / "source" / "anomalies" / difficulty / "imgs"
        ref_masks_dir = EXP / "source" / "anomalies" / difficulty / "masks"
        ref_img_path = ref_imgs_dir / f"{ref_id}.JPG"
        ref_mask_path = ref_masks_dir / f"{ref_id}.png"
    if not ref_img_path.exists() or not ref_mask_path.exists():
        print(f"  WARNING: ref image/mask not found for {ref_id}")
        return None

    ref_img_pil = Image.open(ref_img_path).convert("RGB")
    ref_mask_np = load_binary_mask(ref_mask_path)
    # a reference mask must belong to its image: same size and non-empty (an empty or foreign mask silently
    # conditions the IP-Adapter on the wrong region)
    if ref_mask_np.shape != (ref_img_pil.size[1], ref_img_pil.size[0]):
        raise ValueError(f"reference mask {ref_mask_path} {ref_mask_np.shape[::-1]} != image {ref_img_pil.size}")
    if ref_mask_np.sum() == 0:
        raise ValueError(f"reference mask {ref_mask_path} is empty")

    ref_tensor = torch.from_numpy(
        np.array(ref_img_pil).astype(np.float32) / 255.0
    ).permute(2, 0, 1)  # [3, H, W] in [0, 1]
    ref_mask_tensor = torch.from_numpy(ref_mask_np).unsqueeze(0).float()  # [1, H, W]

    # Multi-crop: 2 groups matching training --multi-crop
    # Seed global random for reproducible crop selection
    random.seed(seed)
    if clip_align == "raw":
        crops, crop_masks, valid = clip_crop_multi(
            ref_tensor, ref_mask_tensor, crop_size=224, n_groups=2,
        )
        clip_img_t = crops[0]       # primary crop [3, 224, 224]
        clip_mask_t = crop_masks[0]  # primary mask [1, 224, 224]

        # Convert CLIP crops to [-1, 1] for generate_anomagic_single
        clip_ref = (clip_img_t * 2.0 - 1.0).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        clip_mask_for_attn = clip_mask_t.unsqueeze(0).to(device)  # [1, 1, 224, 224]
        clip_core_for_attn = clip_mask_for_attn  # no roundtrip dilation → crop mask IS core

        # Second crop
        clip_ref_2 = (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        clip_mask_2 = crop_masks[1].unsqueeze(0).to(device)  # [1, 1, 224, 224]
        clip_core_2 = clip_mask_2  # no roundtrip dilation → crop mask IS core
    else:
        # Training-aligned CLIP masks (anomaly_dataset clip_align=True): group + crop on the UNet-roundtripped mask;
        # "core" = roundtripped core (training clip_core_only=True), "ring" = core + band (clip_core_only=False).
        core_native, dil_native = unet_roundtrip_masks(ref_mask_tensor, 2, image_size=512)
        attn_mask = core_native if clip_align == "core" else dil_native
        crops, crop_masks, valid, extra = clip_crop_multi(
            ref_tensor, attn_mask, crop_size=224, n_groups=2, clip_masks=[attn_mask, core_native],
        )
        clip_img_t = crops[0]                        # [3, 224, 224], used by the panel code below
        clip_ref = (clip_img_t * 2.0 - 1.0).unsqueeze(0).to(device)
        clip_mask_t = (extra[0][0] > 0.5).float()   # [1, 224, 224], used by the panel code below
        clip_mask_for_attn = clip_mask_t.unsqueeze(0).to(device)
        clip_core_for_attn = (extra[0][1] > 0.5).float().unsqueeze(0).to(device)
        clip_ref_2 = (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device)
        clip_mask_2 = (extra[1][0] > 0.5).float().unsqueeze(0).to(device)
        clip_core_2 = (extra[1][1] > 0.5).float().unsqueeze(0).to(device)

    # Group validity flags [1, 2]
    group_valid = torch.tensor(
        [[float(valid[0]), float(valid[1])]], device=device,
    )

    # 4. Build caption from defect type
    defect_type = defect_map.get(ref_id, "defect")
    caption = caption_override if caption_override else f"a photo of a {defect_type} defect on a cashew"

    # 5. Generate
    dilate = (difficulty == "hard") and not force_no_dilate
    g1_tag = "valid" if valid[0] else "NULL"
    g2_tag = "valid" if valid[1] else "NULL"
    print(f"  [{difficulty}] canvas={canvas_id} ref={ref_id} "
          f"type='{defect_type}' noise={noise_strength:.1f} "
          f"crops=[{g1_tag},{g2_tag}] dilate_clip={dilate}")

    with torch.no_grad():
        output = generate_anomagic_single(
            pipeline, ip_adapter,
            normal_image=canvas_t,
            mask=mask_t,
            reference=clip_ref,
            anomaly_type=defect_type,
            caption=caption,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            noise_strength=noise_strength,
            reference_mode="crop",
            band_mode=band_mode,
            t2i_adapter=t2i_adapter,
            seed=seed,
            clip_mask=clip_mask_for_attn,
            clip_core_mask=clip_core_for_attn,
            reference_2=clip_ref_2,
            clip_mask_2=clip_mask_2,
            clip_core_mask_2=clip_core_2,
            group_valid=group_valid,
            inference_mode="same" if clip_align == "ring" else "different",
            context_latents=None if context is None else context["latents"],
            raw_background=raw_background,
            even_steps=even_steps,
            return_latents=True,
            cfg_mode=cfg_mode,
        )
    output, state = output
    LAST_STATE = {"latents": state, "pixels": output}

    # Decode output tensor to PIL
    gen_np = ((output[0].cpu().clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
    gen_pil = Image.fromarray(gen_np)

    # Save raw 512×512 generated image (keyed by ref_id for 1:1 mapping)
    if save_raw:
        raw_dir = EXP / "anomaly" / "imgs" / difficulty
        raw_dir.mkdir(parents=True, exist_ok=True)
        gen_pil.save(raw_dir / f"{ref_id}.png")

    # 6. Compute 64×64 roundtrip masks (same as placement viz + what UNet actually sees)
    core_64 = downsample_mask_maxpool(mask_t.cpu(), 64)
    dilated_binary, _, _, band_mask = create_latent_band_mask(
        core_64, band_mode=band_mode,
    )
    # Roundtrip to 512×512 — this is what the UNet actually operates on
    core_up = F.interpolate(core_64, size=(512, 512), mode="nearest")[0, 0].numpy()
    band_up = F.interpolate(band_mask, size=(512, 512), mode="nearest")[0, 0].numpy()

    # 7. Compute CLIP 16×16 masks: original + dilated (matches MaskedAnomalySelfAttention)
    clip_mask_16_t = downsample_mask_maxpool(clip_mask_t.unsqueeze(0), 16)  # [1, 1, 16, 16]
    clip_mask_16 = (clip_mask_16_t[0, 0].numpy() > 0.5).astype(np.float32)
    # +1px dilation at 16×16 (same as MaskedAnomalySelfAttention.forward line 337)
    clip_dilated_16_t = F.max_pool2d(clip_mask_16_t, kernel_size=3, stride=1, padding=1)
    clip_dilated_16 = (clip_dilated_16_t[0, 0].numpy() > 0.5).astype(np.float32)
    clip_dilation_only = ((clip_dilated_16 > 0.5) & (clip_mask_16 < 0.5)).astype(np.float32)
    n_orig = int(clip_mask_16.sum())
    n_dilated = int(clip_dilated_16.sum())
    n_new = n_dilated - n_orig

    # 8. Build 2-row viz panel
    #    Row 1: Original ref | Canvas normal | Placed mask | CLIP 16x16
    #    Row 2: Crop 1       | Crop 2        | Generated   | Generated+mask
    clip_crop_pil = Image.fromarray(
        (clip_img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    clip_mask_np = clip_mask_t.squeeze(0).numpy()
    clip_crop_rgba = clip_crop_pil.convert("RGBA")
    clip_crop_with_mask = Image.alpha_composite(
        clip_crop_rgba, mask_boundary_overlay(clip_mask_np, (255, 40, 40, 220), width=2)
    ).convert("RGB")

    coverage = mask_np.sum() / (512 * 512) * 100
    dil_tag = "+1px" if dilate else "no"

    # --- Row 1 ---
    # R1C1: Original anomaly reference (full image + mask overlay)
    ref_img_512 = ref_img_pil.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS)
    ref_mask_512 = np.array(
        Image.fromarray((ref_mask_np * 255).astype(np.uint8)).resize(
            (VIZ_SIZE, VIZ_SIZE), Image.NEAREST,
        )
    ).astype(np.float32) / 255.0
    ref_rgba = ref_img_512.convert("RGBA")
    ref_rgba = Image.alpha_composite(
        ref_rgba, mask_boundary_overlay(ref_mask_512, (255, 40, 40, 220), width=2))
    r1c1 = add_label(ref_rgba.convert("RGB"), f"Original ref ({ref_id}) [{defect_type[:25]}]")

    # R1C2: Canvas normal
    r1c2 = add_label(canvas_512.copy(), f"Canvas normal ({canvas_id})")

    # R1C3: Placed mask on canvas — boundary-only (red=core, yellow=band)
    rt_canvas = canvas_512.copy().convert("RGBA")
    # Band boundary first (yellow), then core on top (red)
    band_only = ((band_up > 0.5) & (core_up < 0.5)).astype(np.float32)
    rt_canvas = Image.alpha_composite(
        rt_canvas, mask_boundary_overlay(band_only, (255, 220, 0, 220), width=2))
    rt_canvas = Image.alpha_composite(
        rt_canvas, mask_boundary_overlay(core_up, (255, 40, 40, 220), width=2))
    r1c3 = add_label(rt_canvas.convert("RGB"),
                      f"Placed [{difficulty}] cov={coverage:.2f}%")

    # R1C4: Generated image
    r1c4 = add_label(gen_pil.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS), "Generated")

    # --- Row 2 ---
    # R2C1: CLIP crop 1 — 16×16 grid with mask boundary
    r2c1 = _build_clip_grid(clip_crop_pil, clip_mask_t, dilate, clip_mask_16,
                            clip_dilated_16, VIZ_SIZE)
    n1_orig = int(clip_mask_16.sum())
    dil_label = " +dil" if dilate else ""
    r2c1 = add_label(r2c1, f"CLIP crop 1 ({n1_orig} cells{dil_label})")

    # R2C2: CLIP crop 2 — 16×16 grid (or dark placeholder if NULL)
    if valid[1]:
        crop2_pil = Image.fromarray(
            (crops[1].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        crop2_mask_16_t = downsample_mask_maxpool(crop_masks[1].unsqueeze(0), 16)
        crop2_mask_16 = (crop2_mask_16_t[0, 0].numpy() > 0.5).astype(np.float32)
        crop2_dilated_16_t = F.max_pool2d(crop2_mask_16_t, kernel_size=3, stride=1, padding=1)
        crop2_dilated_16 = (crop2_dilated_16_t[0, 0].numpy() > 0.5).astype(np.float32)
        r2c2 = _build_clip_grid(crop2_pil, crop_masks[1], dilate, crop2_mask_16,
                                crop2_dilated_16, VIZ_SIZE)
        n2_orig = int(crop2_mask_16.sum())
        r2c2 = add_label(r2c2, f"CLIP crop 2 ({n2_orig} cells{dil_label})")
    else:
        null_img = Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (30, 30, 30))
        draw_null = ImageDraw.Draw(null_img)
        try:
            font_null = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            font_null = ImageFont.load_default()
        draw_null.text((VIZ_SIZE // 2 - 50, VIZ_SIZE // 2 - 12), "NULL group",
                       fill=(100, 100, 100), font=font_null)
        r2c2 = add_label(null_img, "CLIP crop 2 [NULL]")

    # R2C3: Generated + roundtripped mask boundary (red=core, yellow=band)
    gen_512 = gen_pil.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS)
    gen_rgba = gen_512.convert("RGBA")
    band_only_gen = ((band_up > 0.5) & (core_up < 0.5)).astype(np.float32)
    gen_rgba = Image.alpha_composite(
        gen_rgba, mask_boundary_overlay(band_only_gen, (255, 220, 0, 220), width=2))
    gen_rgba = Image.alpha_composite(
        gen_rgba, mask_boundary_overlay(core_up, (255, 40, 40, 220), width=2))
    r2c3 = add_label(gen_rgba.convert("RGB"), f"Generated + mask ({coverage:.2f}%)")

    # --- Assemble 2-row panel ---
    if layout == "A":
        # Layout A — 2×3 (compact)
        #   Row 1: Original ref | Placed mask    | Generated+mask
        #   Row 2: CLIP crop 1  | CLIP crop 2    | Generated (clean)
        row1 = [r1c1, r1c3, r2c3]
        row2 = [r2c1, r2c2, r1c4]
    else:
        # Layout D — 2×3 (story flow)
        #   Row 1: Original ref | Placed mask    | CLIP crop 1
        #   Row 2: CLIP crop 2  | Generated      | Generated+mask
        row1 = [r1c1, r1c3, r2c1]
        row2 = [r2c2, r1c4, r2c3]

    n_cols = max(len(row1), len(row2))
    row_h = r1c1.height
    panel = Image.new("RGB", (VIZ_SIZE * n_cols, row_h * 2), (20, 20, 20))
    for j, col in enumerate(row1):
        panel.paste(col, (j * VIZ_SIZE, 0))
    for j, col in enumerate(row2):
        panel.paste(col, (j * VIZ_SIZE, row_h))

    # Save panel to viz/inference_test
    viz_dir = CASHEW_ROOT / "viz" / f"inference_{EXPERIMENT}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    dil_tag = f"_dil{1 if dilate else 0}"
    panel_path = viz_dir / f"{difficulty}_{canvas_id}_ref{ref_id}_layout{layout}{dil_tag}.png"
    panel.save(panel_path, quality=95)
    return panel_path


# ── Full manifest builder ─────────────────────────────────────────────────

def build_full_manifest(seed: int = 42) -> list[dict]:
    """Build manifest for all 100 anomalies (reference + train_real + test).

    For the 40 reference-split entries: uses existing manifest pairings.
    For the 60 train_real + test entries: assigns canvases from the experiment
    canvas pool and places masks on-the-fly using placement_utils.

    Returns:
        List of 100 manifest entries, each with keys:
        canvas_id, ref_id, difficulty, and optionally placed_mask_name (for
        new entries with non-standard naming).
    """
    splits_path = EXP / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    # Load existing manifests (reference split — 38 easy + 2 hard = 40)
    easy_manifest_path = EXP / "anomaly" / "masks" / "easy" / "manifest.json"
    hard_manifest_path = EXP / "anomaly" / "masks" / "hard" / "manifest.json"

    existing_easy = []
    if easy_manifest_path.exists():
        with open(easy_manifest_path) as f:
            existing_easy = json.load(f)
    existing_hard = []
    if hard_manifest_path.exists():
        with open(hard_manifest_path) as f:
            existing_hard = json.load(f)

    # Reference entries keep their original placed mask naming ({canvas_id}.png)
    all_entries = []
    for e in existing_easy + existing_hard:
        all_entries.append(dict(e))

    # Collect the 60 non-reference anomalies
    new_anomalies = []
    for split_name in ["train_real", "test"]:
        for diff in ["easy", "hard"]:
            for aid in splits[split_name].get(diff, []):
                new_anomalies.append({"ref_id": aid, "difficulty": diff})

    new_easy = [a for a in new_anomalies if a["difficulty"] == "easy"]
    new_hard = [a for a in new_anomalies if a["difficulty"] == "hard"]
    print(f"Full manifest: {len(all_entries)} reference + "
          f"{len(new_easy)} new easy + {len(new_hard)} new hard = "
          f"{len(all_entries) + len(new_anomalies)} total")

    # Load canvas FG masks (from the 50 designated experiment canvases)
    canvas_mask_dir = EXP / "source" / "canvas_normals" / "masks"
    canvas_ids = splits["normals"]["canvas"]
    canvas_fg: dict[str, np.ndarray] = {}
    for cid in canvas_ids:
        mask_path = canvas_mask_dir / f"{cid}.png"
        if mask_path.exists():
            canvas_fg[cid] = load_binary_mask(mask_path)
    print(f"Loaded {len(canvas_fg)} canvas FG masks")

    placed_mask_dir_easy = EXP / "anomaly" / "masks" / "easy"
    placed_mask_dir_easy.mkdir(parents=True, exist_ok=True)
    placed_mask_dir_hard = EXP / "anomaly" / "masks" / "hard"
    placed_mask_dir_hard.mkdir(parents=True, exist_ok=True)

    # Assign canvases to new easy entries (cycle through shuffled pool)
    rng = random.Random(seed)
    canvas_pool = list(canvas_fg.keys())
    rng.shuffle(canvas_pool)
    pool_idx = 0

    for i, entry in enumerate(new_easy):
        aid = entry["ref_id"]
        canvas_id = canvas_pool[pool_idx % len(canvas_pool)]
        pool_idx += 1

        placed_name = f"{aid}_on_{canvas_id}.png"
        placed_path = placed_mask_dir_easy / placed_name

        if placed_path.exists():
            print(f"  Easy {aid}: placed mask exists, skipping placement")
        else:
            anomaly_mask = load_binary_mask(
                CASHEW_ROOT / "Data" / "Masks" / "Anomaly" / f"{aid}.png"
            )
            fg_mask = canvas_fg[canvas_id]
            placed = place_easy_mask(anomaly_mask, fg_mask, seed=seed + i)

            if placed is None:
                # Try other canvases as fallback
                for alt_cid in canvas_fg:
                    if alt_cid == canvas_id:
                        continue
                    placed = place_easy_mask(
                        anomaly_mask, canvas_fg[alt_cid], seed=seed + i,
                    )
                    if placed is not None:
                        canvas_id = alt_cid
                        placed_name = f"{aid}_on_{canvas_id}.png"
                        placed_path = placed_mask_dir_easy / placed_name
                        break
                if placed is None:
                    print(f"  ERROR: could not place easy {aid}, skipping")
                    continue

            Image.fromarray(
                (placed * 255).astype(np.uint8)
            ).save(placed_path)
            print(f"  Placed easy {aid} on canvas {canvas_id}")

        entry["canvas_id"] = canvas_id
        entry["placed_mask_name"] = placed_name
        all_entries.append(entry)

    # Place hard anomalies (exhaustive search across all canvases)
    hard_used: set[str] = set()
    for i, entry in enumerate(new_hard):
        aid = entry["ref_id"]

        # Check if already placed from a previous run
        existing = list(placed_mask_dir_hard.glob(f"{aid}_on_*.png"))
        if existing:
            placed_path = existing[0]
            canvas_id = placed_path.stem.split("_on_")[1]
            placed_name = placed_path.name
            print(f"  Hard {aid}: placed mask exists ({placed_name})")
        else:
            anomaly_mask = load_binary_mask(
                CASHEW_ROOT / "Data" / "Masks" / "Anomaly" / f"{aid}.png"
            )
            result = place_hard_mask(
                anomaly_mask, canvas_fg, used=hard_used, seed=seed + 100 + i,
            )
            if result is None:
                print(f"  ERROR: could not place hard {aid}, skipping")
                continue

            canvas_id, placed, angle, scale, cov = result
            hard_used.add(canvas_id)
            placed_name = f"{aid}_on_{canvas_id}.png"
            placed_path = placed_mask_dir_hard / placed_name
            Image.fromarray(
                (placed * 255).astype(np.uint8)
            ).save(placed_path)
            print(f"  Placed hard {aid} on canvas {canvas_id} "
                  f"(angle={angle:.1f}, scale={scale:.2f}, cov={cov:.2f}%)")

        entry["canvas_id"] = canvas_id
        entry["placed_mask_name"] = placed_name
        all_entries.append(entry)

    # Save combined manifest
    manifest_path = EXP / "anomaly" / "manifest_all.json"
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)
    print(f"Saved manifest_all.json with {len(all_entries)} entries")

    return all_entries


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate cashew anomalies")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to checkpoint dir")
    parser.add_argument("--experiment", type=str, default="ResNet",
                        choices=["ResNet", "UniNet"],
                        help="Experiment layout (default: ResNet)")
    parser.add_argument("--all", action="store_true",
                        help="Generate for ALL 100 anomalies (reference + "
                             "train_real + test)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip entries where raw image already exists "
                             "(for resume after interruption)")
    parser.add_argument("--num-easy", type=int, default=5)
    parser.add_argument("--num-hard", type=int, default=1)
    parser.add_argument("--noise-easy", type=float, default=0.7)
    parser.add_argument("--noise-hard", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--band-mode", type=int, default=2)
    parser.add_argument("--layout", type=str, default="A", choices=["A", "D"],
                        help="Panel layout: A=compact, D=story flow")
    parser.add_argument("--no-clip-dilate", action="store_true",
                        help="Disable CLIP mask dilation for hard mode")
    args = parser.parse_args()

    setup_experiment(args.experiment)

    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_dir}")
        sys.exit(1)

    viz_base = CASHEW_ROOT / "viz"
    print(f"Viz output: {viz_base}")

    # Parse defect types from image_anno.csv
    defect_map = parse_defect_types(IMAGE_ANNO_CSV)
    print(f"Loaded {len(defect_map)} defect type entries from image_anno.csv")

    if args.all:
        # ── Full generation mode: all 100 anomalies ──────────────────────
        all_entries = build_full_manifest(seed=args.seed)
        total = len(all_entries)
        print(f"\nGenerating ALL {total} samples")
    else:
        # ── Subset mode: random selection from reference manifests ────────
        if EXPERIMENT == "ResNet":
            easy_manifest_path = EXP / "anomaly" / "masks" / "easy" / "manifest.json"
            hard_manifest_path = EXP / "anomaly" / "masks" / "hard" / "manifest.json"
        else:
            easy_manifest_path = EXP / "synthetic" / "placed_masks" / "easy" / "manifest.json"
            hard_manifest_path = EXP / "synthetic" / "placed_masks" / "hard" / "manifest.json"

        easy_entries = []
        if easy_manifest_path.exists():
            with open(easy_manifest_path) as f:
                easy_entries = json.load(f)
            print(f"Easy manifest: {len(easy_entries)} entries")
        else:
            print(f"WARNING: {easy_manifest_path} not found")

        hard_entries = []
        if hard_manifest_path.exists():
            with open(hard_manifest_path) as f:
                hard_entries = json.load(f)
            print(f"Hard manifest: {len(hard_entries)} entries")
        else:
            print(f"WARNING: {hard_manifest_path} not found")

        rng = random.Random(args.seed)
        easy_selected = (rng.sample(easy_entries, args.num_easy)
                         if len(easy_entries) > args.num_easy
                         else easy_entries[:args.num_easy])
        hard_selected = (rng.sample(hard_entries, args.num_hard)
                         if len(hard_entries) > args.num_hard
                         else hard_entries[:args.num_hard])

        all_entries = easy_selected + hard_selected
        total = len(all_entries)
        print(f"\nGenerating {len(easy_selected)} easy + "
              f"{len(hard_selected)} hard = {total} samples")

    if total == 0:
        print("Nothing to generate. Check manifests.")
        sys.exit(0)

    # Load models
    pipeline, ip_adapter, t2i_adapter = load_models(checkpoint_dir)

    # Generate
    panels = []
    skipped = 0
    for i, entry in enumerate(all_entries):
        difficulty = entry["difficulty"]
        noise = args.noise_easy if difficulty == "easy" else args.noise_hard

        # Skip existing if requested
        if args.skip_existing:
            raw_path = (EXP / "anomaly" / "imgs" / difficulty
                        / f"{entry['ref_id']}.png")
            if raw_path.exists():
                print(f"\n[{i + 1}/{total}] Skipping {entry['ref_id']} "
                      f"(raw exists)")
                skipped += 1
                continue

        # Resolve paths for non-reference entries (those with placed_mask_name)
        ref_img = ref_mask = placed_mask = None
        if "placed_mask_name" in entry:
            ref_img, ref_mask = resolve_anomaly_paths(
                entry["ref_id"], difficulty,
            )
            placed_mask = (EXP / "anomaly" / "masks" / difficulty
                           / entry["placed_mask_name"])

        print(f"\n[{i + 1}/{total}]", end="")
        panel_path = generate_one(
            pipeline, ip_adapter, t2i_adapter,
            canvas_id=entry["canvas_id"],
            ref_id=entry["ref_id"],
            difficulty=difficulty,
            defect_map=defect_map,
            noise_strength=noise,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            band_mode=args.band_mode,
            seed=args.seed + i,
            device="cuda",
            layout=args.layout,
            force_no_dilate=args.no_clip_dilate,
            ref_img_override=ref_img,
            ref_mask_override=ref_mask,
            placed_mask_override=placed_mask,
            save_raw=args.all,
        )
        if panel_path:
            panels.append(panel_path)

    print(f"\n{'=' * 60}")
    skipped_msg = f" (skipped {skipped})" if skipped else ""
    print(f"Generated {len(panels)}/{total} samples{skipped_msg}")
    print(f"Viz: {CASHEW_ROOT / 'viz'}")
    if args.all:
        print(f"Raw images: {EXP / 'anomaly' / 'imgs'}")
    for p in panels:
        print(f"  {p}")


if __name__ == "__main__":
    main()
