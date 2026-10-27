"""
Shared anomaly dataset for joint training across anomaly types.

Used by Anomagic (train_anomagic.py) training script.
"""
import os
import json
import random
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import numpy as np

from src.utils.crop_utils import clip_crop, clip_crop_multi, lanczos_resize_tensor
from src.utils.mask_utils import downsample_mask_maxpool, unet_roundtrip_masks


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _apply_unet_augmentation(
    image: Image.Image,
    mask: Image.Image,
    target_size: int = 512,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """UNet-path augmentation: color jitter + LANCZOS-3 downsample to target_size.

    No flips, rotations, or crops — captions contain spatial references that must
    match the image orientation and framing. Decorrelation is achieved by
    augmenting the CLIP path independently.

    Args:
        image: PIL RGB image at original resolution (e.g. 1024×1024)
        mask: PIL grayscale mask at original resolution
        target_size: Output size (512 for SD 1.5)

    Returns:
        (image_tensor [3, target_size, target_size] in [-1,1],
         mask_tensor [1, target_size, target_size] in {0,1})
    """
    # Brightness/contrast jitter ±15%
    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(b)
    image = ImageEnhance.Contrast(image).enhance(c)

    # LANCZOS-3 resize in PIL space (sharper + properly antialiased).
    image = image.resize((target_size, target_size), Image.LANCZOS)
    img_t = TF.to_tensor(image)                    # [3, H, W] in [0, 1]
    mask_t = TF.to_tensor(mask.convert("L"))        # [1, H, W]
    mask_t = (mask_t > 0.5).float()
    mask_t = downsample_mask_maxpool(mask_t, target_size)  # [1, H, W]

    # Normalize to [-1, 1]
    img_t = img_t * 2.0 - 1.0

    return img_t, mask_t


def _clip_augmentation_transforms(
    image: Image.Image,
    mask: Image.Image,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
    return_meta: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CLIP-path augmentation transforms (flip, jitter, rotate) to PIL images.

    Returns full-resolution tensors BEFORE cropping, so callers can apply
    either single-crop (clip_crop) or multi-crop (clip_crop_multi).

    Args:
        image: PIL RGB image at original resolution
        mask: PIL grayscale mask at original resolution
        return_meta: If True, return augmentation metadata dict as third element.

    Returns:
        (image_tensor [3, H, W] in [0,1],
         mask_tensor [1, H, W] in {0,1})
        If return_meta: additionally returns dict with flip_h, flip_v, rotation.
    """
    # Horizontal flip p=0.5
    flip_h = random.random() < 0.5
    if flip_h:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Vertical flip p=0.5
    flip_v = random.random() < 0.5
    if flip_v:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Brightness/contrast jitter ±15%
    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(b)
    image = ImageEnhance.Contrast(image).enhance(c)

    # Rotate by 0/90/180/270 (artifact-free discrete rotations)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        image = TF.rotate(image, angle, expand=False)
        mask = TF.rotate(mask, angle, expand=False)

    # Convert to tensors
    img_t = TF.to_tensor(image)              # [3, H, W] in [0, 1]
    mask_t = TF.to_tensor(mask.convert("L")) # [1, H, W]
    mask_t = (mask_t > 0.5).float()

    if return_meta:
        meta = {"flip_h": flip_h, "flip_v": flip_v, "rotation": angle}
        return img_t, mask_t, meta
    return img_t, mask_t


def _apply_clip_augmentation(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int = 224,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CLIP-path augmentation: flip, jitter, discrete rotate, crop_utils crop.

    Order: flip → jitter → rotate {0,90,180,270} → crop_utils → 224×224

    Independent draws from UNet augmentation.

    Args:
        image: PIL RGB image at original resolution
        mask: PIL grayscale mask at original resolution
        crop_size: Output size (224 for CLIP)

    Returns:
        (image_tensor [3, crop_size, crop_size] in [0,1],
         mask_tensor [1, crop_size, crop_size] in {0,1})
    """
    img_t, mask_t = _clip_augmentation_transforms(
        image, mask, brightness_range, contrast_range,
    )

    # crop_utils: component grouping + anomaly-aware crop → 224×224
    crop_img, crop_mask = clip_crop(img_t, mask_t, crop_size=crop_size, pad_frac=0.0)
    crop_mask = (crop_mask > 0.5).float()

    return crop_img, crop_mask


def _serialize_clip_meta(
    result: Dict,
    aug_meta: Dict,
    crop_metas: List[Dict],
    img_shape: Tuple[int, ...],
) -> None:
    """Serialize CLIP augmentation metadata as individual tensors for DataLoader.

    Args:
        result: Dict to populate with tensor fields.
        aug_meta: {"flip_h": bool, "flip_v": bool, "rotation": int}
        crop_metas: List of {"crop_top", "crop_left", "crop_h", "crop_w", "resized"} dicts.
        img_shape: (C, H, W) shape of the augmented image (after flips/rotation).
    """
    result["clip_flip_h"] = torch.tensor(float(aug_meta["flip_h"]))
    result["clip_flip_v"] = torch.tensor(float(aug_meta["flip_v"]))
    result["clip_rotation"] = torch.tensor(float(aug_meta["rotation"]))
    result["orig_h"] = torch.tensor(float(img_shape[1]))
    result["orig_w"] = torch.tensor(float(img_shape[2]))

    # Crop 1
    cm1 = crop_metas[0]
    result["clip_crop_top"] = torch.tensor(float(cm1["crop_top"]))
    result["clip_crop_left"] = torch.tensor(float(cm1["crop_left"]))
    result["clip_crop_h"] = torch.tensor(float(cm1["crop_h"]))
    result["clip_crop_w"] = torch.tensor(float(cm1["crop_w"]))
    result["clip_resized"] = torch.tensor(float(cm1["resized"]))

    # Crop 2 (multi-crop only; zeros if not available)
    if len(crop_metas) > 1:
        cm2 = crop_metas[1]
        result["clip_crop_top_2"] = torch.tensor(float(cm2["crop_top"]))
        result["clip_crop_left_2"] = torch.tensor(float(cm2["crop_left"]))
        result["clip_crop_h_2"] = torch.tensor(float(cm2["crop_h"]))
        result["clip_crop_w_2"] = torch.tensor(float(cm2["crop_w"]))
        result["clip_resized_2"] = torch.tensor(float(cm2["resized"]))
    else:
        result["clip_crop_top_2"] = torch.tensor(0.0)
        result["clip_crop_left_2"] = torch.tensor(0.0)
        result["clip_crop_h_2"] = torch.tensor(0.0)
        result["clip_crop_w_2"] = torch.tensor(0.0)
        result["clip_resized_2"] = torch.tensor(0.0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnomalyDataset(Dataset):
    """Dataset that loads anomaly types from split JSONs for joint training."""

    def __init__(
        self,
        splits_dir: Path,
        image_size: int = 512,
        anomaly_types: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        data_root: Optional[Path] = None,
        captions_file: Optional[Path] = None,
        return_reference: bool = False,
        reference_mode: str = "full",
        reference_crop_size: int = 224,
        augment: bool = False,
        multi_crop: bool = False,
        band_mode: int = 2,
        clip_align: bool = True,
        clip_core_only: bool = False,
        return_clip_meta: bool = False,
    ):
        """
        Args:
            splits_dir: Directory containing per-anomaly-type JSON files
            image_size: Target image size for SD (default: 512 for SD 1.5)
            anomaly_types: Specific types to load (default: all)
            exclude_sources: Dataset sources to skip
            data_root: Root for resolving relative image paths
            captions_file: Path to captions.json
            return_reference: Whether to include reference image for CLIP
            reference_mode: 'full' = downscale entire image, 'crop' = anomaly bbox crop
            reference_crop_size: Target size for reference image (224 for CLIP)
            augment: Enable data augmentation (UNet + CLIP paths)
            multi_crop: Return 2 independent CLIP crops from different component groups
            band_mode: Latent-space band dilation mode (1 or 2) for UNet roundtrip masks
            clip_align: Use UNet-roundtripped dilated masks for CLIP self-attention + role embeddings.
                When False, reverts to raw cropped masks with no role embedding distinction.
            clip_core_only: When True (and clip_align=True), use core-only roundtripped mask
                (no band dilation) for CLIP attention. Core = anomaly pixels only after
                latent quantization roundtrip.
            return_clip_meta: If True, return CLIP augmentation metadata (flip, rotation,
                crop coordinates) as individual tensor fields for DataLoader compatibility.
        """
        self.image_size = image_size
        self.splits_dir = Path(splits_dir)
        self.band_mode = band_mode
        self.exclude_sources: Set[str] = set(exclude_sources or [])
        self.data_root = Path(data_root) if data_root else None
        self.return_reference = return_reference
        self.reference_mode = reference_mode
        self.reference_crop_size = reference_crop_size
        self.augment = augment
        self.multi_crop = multi_crop
        self.clip_align = clip_align
        self.clip_core_only = clip_core_only
        self.return_clip_meta = return_clip_meta

        # Load captions if provided (keyed by image_path for uniqueness)
        self.captions: Dict[str, str] = {}
        if captions_file and Path(captions_file).exists():
            with open(captions_file, encoding="utf-8") as f:
                captions_data = json.load(f)
            for entry in captions_data:
                key = entry.get("image_path") or entry.get("filename", "")
                self.captions[key] = entry["caption"]
            print(f"Loaded {len(self.captions)} captions from {captions_file}")

        # Load all anomaly types
        self.samples: List[Dict] = []
        self.type_to_samples: Dict[str, List[int]] = defaultdict(list)

        json_files = list(self.splits_dir.glob("*.json"))

        if anomaly_types:
            json_files = [f for f in json_files if f.stem in anomaly_types]

        if self.exclude_sources:
            print(f"Excluding sources: {self.exclude_sources}")

        print(f"Loading {len(json_files)} anomaly types...")

        skipped_source = 0
        skipped_duplicate = 0
        for json_path in sorted(json_files):
            anomaly_type = json_path.stem

            try:
                with open(json_path) as f:
                    data = json.load(f)

                # Support both AnomVerse format ("images") and RealIAD format ("samples")
                entries = data.get("images") or data.get("samples", [])
                loaded = 0

                for img_data in entries:
                    # Skip bad data sources (AnomVerse only)
                    source = img_data.get("source_dataset", "")
                    if source and source in self.exclude_sources:
                        skipped_source += 1
                        continue

                    image_path = img_data.get("image_path_full") or img_data.get("image_path_abs") or img_data.get("image_path")
                    mask_path = img_data.get("mask_path_full") or img_data.get("mask_path_abs") or img_data.get("mask_path")

                    # Resolve relative paths against data_root if provided
                    if self.data_root and image_path and not os.path.isabs(image_path):
                        image_path = str(self.data_root / image_path)
                    if self.data_root and mask_path and not os.path.isabs(mask_path):
                        mask_path = str(self.data_root / mask_path)

                    # Skip entries where image == mask (e.g. MTD dataset bug)
                    if image_path and mask_path and image_path == mask_path:
                        skipped_duplicate += 1
                        continue

                    # Look up caption by relative image_path (fallback to filename)
                    raw_path = img_data.get("image_path_full") or img_data.get("image_path_abs") or img_data.get("image_path", "")
                    caption = self.captions.get(raw_path, "")
                    if not caption:
                        caption = self.captions.get(os.path.basename(raw_path), "")

                    sample = {
                        "anomaly_type": anomaly_type,
                        "product": img_data.get("product", ""),
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "caption": caption,
                    }
                    self.samples.append(sample)
                    self.type_to_samples[anomaly_type].append(len(self.samples) - 1)
                    loaded += 1

                if loaded > 0:
                    print(f"  {anomaly_type}: {loaded} samples (skipped {len(entries) - loaded})")

            except Exception as e:
                print(f"  Error loading {json_path}: {e}")

        print(f"\nTotal: {len(self.samples)} samples across {len(self.type_to_samples)} types")
        if skipped_source > 0:
            print(f"Skipped {skipped_source} samples from excluded sources")
        if skipped_duplicate > 0:
            print(f"Skipped {skipped_duplicate} samples where image_path == mask_path (MTD bug)")

        self.anomaly_types = sorted(self.type_to_samples.keys())

        # Count captions
        n_captions = sum(1 for s in self.samples if s.get("caption"))
        print(f"Samples with captions: {n_captions}/{len(self.samples)}")

        # Non-augmented transforms (fallback when augment=False)
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, _retries: int = 0) -> Dict:
        sample = self.samples[idx]

        try:
            image_pil = Image.open(sample["image_path"])
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
        except Exception:
            image_pil = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        try:
            mask_pil = Image.open(sample["mask_path"]).convert("L")
        except Exception:
            mask_pil = Image.new("L", (self.image_size, self.image_size), 0)

        if self.augment:
            # --- Deterministic epoch-aware seeding for reproducibility ---
            # With persistent_workers, worker_init_fn only runs once. Re-seed per
            # (epoch, index) so augmentation varies across epochs but is reproducible.
            _epoch = getattr(self, '_epoch', 0)
            _local_seed = 42 + _epoch * len(self) + idx
            random.seed(_local_seed)
            np.random.seed(_local_seed)

            # --- Augmented path ---
            # UNet: flip, 90° rotate, jitter, anomaly-aware 512 crop
            image, mask = _apply_unet_augmentation(
                image_pil, mask_pil, target_size=self.image_size,
            )

            result = {
                "image": image,
                "mask": mask,
                "anomaly_type": sample["anomaly_type"],
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "caption": sample.get("caption", ""),
            }

            # CLIP: independent flip, jitter, continuous rotate, crop_utils crop
            if self.return_reference:
                _rcm = self.return_clip_meta
                if self.multi_crop:
                    # Multi-crop: apply transforms then get 2 separate group crops
                    if _rcm:
                        img_t, mask_t, aug_meta = _clip_augmentation_transforms(
                            image_pil, mask_pil, return_meta=True,
                        )
                    else:
                        img_t, mask_t = _clip_augmentation_transforms(image_pil, mask_pil)
                    if self.clip_align:
                        # UNet-roundtripped masks for CLIP alignment.
                        # Group on the roundtripped mask (core or dilated per clip_core_only).
                        core_native, dil_native = unet_roundtrip_masks(mask_t, self.band_mode, image_size=self.image_size)
                        attn_mask = core_native if self.clip_core_only else dil_native
                        mc_result = clip_crop_multi(
                            img_t, attn_mask, crop_size=self.reference_crop_size,
                            clip_masks=[attn_mask, core_native],
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crops, crop_masks, valid, extra, crop_metas = mc_result
                        else:
                            crops, crop_masks, valid, extra = mc_result
                        result["reference"] = crops[0] * 2.0 - 1.0
                        result["clip_mask"] = (extra[0][0] > 0.5).float()       # attn (dilated or core per clip_core_only)
                        result["clip_core_mask"] = (extra[0][1] > 0.5).float()  # core (role embeddings)
                        result["reference_2"] = crops[1] * 2.0 - 1.0
                        result["clip_mask_2"] = (extra[1][0] > 0.5).float()     # attn (matches clip_mask)
                        result["clip_core_mask_2"] = (extra[1][1] > 0.5).float()
                        result["group_valid"] = torch.tensor(
                            [float(valid[0]), float(valid[1])],
                        )
                    else:
                        # Old behavior: raw cropped masks, no role embeddings
                        mc_result = clip_crop_multi(
                            img_t, mask_t, crop_size=self.reference_crop_size,
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crops, crop_masks, valid, crop_metas = mc_result
                        else:
                            crops, crop_masks, valid = mc_result
                        result["reference"] = crops[0] * 2.0 - 1.0
                        result["clip_mask"] = (crop_masks[0] > 0.5).float()
                        result["reference_2"] = crops[1] * 2.0 - 1.0
                        result["clip_mask_2"] = (crop_masks[1] > 0.5).float()
                        result["group_valid"] = torch.tensor(
                            [float(valid[0]), float(valid[1])],
                        )
                    if _rcm:
                        _serialize_clip_meta(result, aug_meta, crop_metas, img_t.shape)
                else:
                    # Single crop: apply augmentation then crop
                    if _rcm:
                        img_t, mask_t, aug_meta = _clip_augmentation_transforms(
                            image_pil, mask_pil, return_meta=True,
                        )
                    else:
                        img_t, mask_t = _clip_augmentation_transforms(
                            image_pil, mask_pil,
                        )
                    if self.clip_align:
                        # Group on the roundtripped mask (core or dilated per clip_core_only).
                        core_native, dil_native = unet_roundtrip_masks(mask_t, self.band_mode, image_size=self.image_size)
                        attn_mask = core_native if self.clip_core_only else dil_native
                        sc_result = clip_crop(
                            img_t, attn_mask, crop_size=self.reference_crop_size,
                            clip_masks=[attn_mask, core_native],
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crop_img, crop_mask, extra, crop_meta_1 = sc_result
                        else:
                            crop_img, crop_mask, extra = sc_result
                        crop_mask = (crop_mask > 0.5).float()
                        result["reference"] = crop_img * 2.0 - 1.0
                        result["clip_mask"] = (extra[0] > 0.5).float()       # attn (dilated or core per clip_core_only)
                        result["clip_core_mask"] = (extra[1] > 0.5).float()  # core
                    else:
                        sc_result = clip_crop(
                            img_t, mask_t, crop_size=self.reference_crop_size,
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crop_img, crop_mask, crop_meta_1 = sc_result
                        else:
                            crop_img, crop_mask = sc_result
                        crop_mask = (crop_mask > 0.5).float()
                        result["reference"] = crop_img * 2.0 - 1.0
                        result["clip_mask"] = crop_mask
                    if _rcm:
                        _serialize_clip_meta(result, aug_meta, [crop_meta_1], img_t.shape)

        else:
            # --- Non-augmented path (original behavior) ---
            image = self.transform(image_pil)
            mask = self.mask_transform(mask_pil)
            mask = (mask > 0.5).float()
            mask = downsample_mask_maxpool(mask, self.image_size)  # maxpool to target size

            result = {
                "image": image,
                "mask": mask,
                "anomaly_type": sample["anomaly_type"],
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "caption": sample.get("caption", ""),
            }

            if self.return_reference:
                _rcm = self.return_clip_meta
                if self.multi_crop:
                    # Multi-crop: get 2 separate group crops from non-augmented image
                    img_t = TF.to_tensor(image_pil)  # [3, H, W] in [0, 1]
                    mask_t = TF.to_tensor(mask_pil.convert("L"))
                    mask_t = (mask_t > 0.5).float()
                    if self.clip_align:
                        # UNet-roundtripped masks for CLIP alignment.
                        # Group on the roundtripped mask (core or dilated per clip_core_only).
                        core_native, dil_native = unet_roundtrip_masks(mask_t, self.band_mode, image_size=self.image_size)
                        attn_mask = core_native if self.clip_core_only else dil_native
                        mc_result = clip_crop_multi(
                            img_t, attn_mask, crop_size=self.reference_crop_size,
                            clip_masks=[attn_mask, core_native],
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crops, crop_masks, valid, extra, crop_metas = mc_result
                        else:
                            crops, crop_masks, valid, extra = mc_result
                        result["reference"] = crops[0] * 2.0 - 1.0
                        result["clip_mask"] = (extra[0][0] > 0.5).float()       # dilated
                        result["clip_core_mask"] = (extra[0][1] > 0.5).float()  # core
                        result["reference_2"] = crops[1] * 2.0 - 1.0
                        result["clip_mask_2"] = (extra[1][0] > 0.5).float()     # attn (matches clip_mask)
                        result["clip_core_mask_2"] = (extra[1][1] > 0.5).float()
                        result["group_valid"] = torch.tensor(
                            [float(valid[0]), float(valid[1])],
                        )
                    else:
                        # Old behavior: raw cropped masks, no role embeddings
                        mc_result = clip_crop_multi(
                            img_t, mask_t, crop_size=self.reference_crop_size,
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crops, crop_masks, valid, crop_metas = mc_result
                        else:
                            crops, crop_masks, valid = mc_result
                        result["reference"] = crops[0] * 2.0 - 1.0
                        result["clip_mask"] = (crop_masks[0] > 0.5).float()
                        result["reference_2"] = crops[1] * 2.0 - 1.0
                        result["clip_mask_2"] = (crop_masks[1] > 0.5).float()
                        result["group_valid"] = torch.tensor(
                            [float(valid[0]), float(valid[1])],
                        )
                    if _rcm:
                        no_aug_meta = {"flip_h": False, "flip_v": False, "rotation": 0}
                        _serialize_clip_meta(result, no_aug_meta, crop_metas, img_t.shape)
                else:
                    # Single crop (non-augmented)
                    img_t = TF.to_tensor(image_pil)  # [3, H, W] in [0, 1]
                    mask_t = TF.to_tensor(mask_pil.convert("L"))
                    mask_t = (mask_t > 0.5).float()
                    if self.clip_align:
                        # Group on the roundtripped mask (core or dilated per clip_core_only).
                        core_native, dil_native = unet_roundtrip_masks(mask_t, self.band_mode, image_size=self.image_size)
                        attn_mask = core_native if self.clip_core_only else dil_native
                        sc_result = clip_crop(
                            img_t, attn_mask, crop_size=self.reference_crop_size,
                            clip_masks=[attn_mask, core_native],
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crop_img, crop_mask, extra_crops, crop_meta_1 = sc_result
                        else:
                            crop_img, crop_mask, extra_crops = sc_result
                        result["reference"] = crop_img * 2.0 - 1.0
                        result["clip_mask"] = (extra_crops[0] > 0.5).float()       # attn (dilated or core per clip_core_only)
                        result["clip_core_mask"] = (extra_crops[1] > 0.5).float()  # core
                    else:
                        sc_result = clip_crop(
                            img_t, mask_t, crop_size=self.reference_crop_size,
                            return_crop_meta=_rcm,
                        )
                        if _rcm:
                            crop_img, crop_mask, crop_meta_1 = sc_result
                        else:
                            crop_img, crop_mask = sc_result
                        crop_mask = (crop_mask > 0.5).float()
                        result["reference"] = crop_img * 2.0 - 1.0
                        result["clip_mask"] = crop_mask
                    if _rcm:
                        no_aug_meta = {"flip_h": False, "flip_v": False, "rotation": 0}
                        _serialize_clip_meta(result, no_aug_meta, [crop_meta_1], img_t.shape)

        # Skip samples where mask is empty (tiny anomalies vanish)
        if result["mask"].sum() == 0 and _retries < 10:
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx, _retries=_retries + 1)

        return result


def _prepare_reference_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "full",
    crop_size: int = 224,
) -> torch.Tensor:
    """Prepare reference image for CLIP (non-augmented fallback).

    Args:
        image: [C, H, W] in [-1, 1]
        mask: [1, H, W] in {0, 1}
        mode: 'full' or 'crop'
        crop_size: Target size
    """
    if mode == "crop":
        return _extract_anomaly_crop(image, mask, crop_size)
    else:
        return lanczos_resize_tensor(image, crop_size, value_range="-11")


def _extract_anomaly_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 224,
) -> torch.Tensor:
    """Extract anomaly bbox crop (non-augmented fallback)."""
    mask_2d = mask.squeeze(0)
    nonzero = torch.nonzero(mask_2d, as_tuple=False)

    if len(nonzero) == 0:
        _, h, w = image.shape
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        crop = image[:, top:top + crop_size, left:left + crop_size]
        return lanczos_resize_tensor(crop, crop_size, value_range="-11")

    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values
    _, h, w = image.shape

    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    pad_y = max(1, int(bbox_h * 0.1))
    pad_x = max(1, int(bbox_w * 0.1))
    y_min = max(0, y_min - pad_y)
    x_min = max(0, x_min - pad_x)
    y_max = min(h, y_max + pad_y + 1)
    x_max = min(w, x_max + pad_x + 1)

    crop_h = y_max - y_min
    crop_w = x_max - x_min
    if crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_min + crop_h)
        if x_max - x_min < crop_h:
            x_min = max(0, x_max - crop_h)
    elif crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_min + crop_w)
        if y_max - y_min < crop_w:
            y_min = max(0, y_max - crop_w)

    crop = image[:, y_min:y_max, x_min:x_max]
    return lanczos_resize_tensor(crop, crop_size, value_range="-11")


# Backward compatibility alias
MultiAnomalyDataset = AnomalyDataset
