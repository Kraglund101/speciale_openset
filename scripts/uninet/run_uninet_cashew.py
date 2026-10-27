"""
UniNet Cashew Experiment: Baseline vs Supervised vs Extension.

Mode 1 — Baseline:
  Original UniNet (unsupervised KD). Train on normals only.

Mode 2 — Supervised:
  Original UniNet with synthetic anomaly data. Same loss as Mode 1 but with
  spatial masks passed to UniNetLoss supervised branch. Includes BCE head.

Mode 3 — Extension:
  New objective with two-sided L_KD + L_SC (no margin, no BCE head),
  plus canvas-normal alignment:
    Student(x_norm) = Teacher(x_norm)
    Student(x_syn)  = Teacher(x_norm_canvas)      [canvas alignment]
    Student(x_syn)  ≠ Teacher(x_syn)               [dis-align anomaly, align normal]

Usage:
  python run_uninet_cashew.py --placement rp --cfg-variant cfg7_vis
"""

import argparse
import json
import os
import random
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import roc_auc_score

from anomalib.models.image.uninet.torch_model import UniNetModel
from anomalib.models.image.uninet.components.loss import UniNetLoss
from anomalib.models.image.uninet.components.attention_bottleneck import (
    AttentionBottleneck,
    conv3x3 as _bn_conv3x3,
    conv1x1 as _bn_conv1x1,
)
from anomalib.models.image.uninet.components.dfs import DomainRelatedFeatureSelection
from anomalib.data import InferenceBatch
import timm as _timm

# --- Portable project-root resolution -------------------------------------
# Resolves the repository root so this
# script runs from any checkout. Override with SPECIALE_ROOT if needed.
def _find_project_root(start: Path) -> Path:
    env = os.environ.get("SPECIALE_ROOT")
    if env:
        return Path(env).resolve()
    for p in [start] + list(start.parents):
        if (p / "scripts").is_dir() and (p / "anomverse_extension").is_dir():
            return p
    return start.parents[2]  # scripts/uninet/run_uninet_cashew.py -> repository root

PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
CASHEW_ROOT = (PROJECT_ROOT / "anomverse_extension" / "datasets"
               / "VisA_validation_dataset" / "datasets" / "easy_test" / "cashew")
# --------------------------------------------------------------------------


# ── ConvNeXt-Base backbone classes ─────────────────────────────────────

class ConvNeXtTeachers(nn.Module):
    """Dual ConvNeXt-Base teachers: source (frozen) + target (trainable).

    Mirrors the WRN-50-2 Teachers class but uses timm features_only
    to extract 3 feature levels: [128@64, 256@32, 512@16].
    """

    def __init__(self) -> None:
        super().__init__()
        self.source_teacher = _timm.create_model(
            "convnext_base.dinov3_lvd1689m", pretrained=True,
            features_only=True, out_indices=[0, 1, 2],
        ).eval()
        for p in self.source_teacher.parameters():
            p.requires_grad_(False)
        self.target_teacher = _timm.create_model(
            "convnext_base.dinov3_lvd1689m", pretrained=True,
            features_only=True, out_indices=[0, 1, 2],
        )

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        with torch.no_grad():
            source_feats = self.source_teacher(images)  # [128@64, 256@32, 512@16]
        target_feats = self.target_teacher(images)

        # Match WRN Teachers return format:
        # source_target_features = [s0, s1, s2, t0, t1, t2]
        source_target_features = list(source_feats) + list(target_feats)
        if globals().get("_STUDENT_SKIPS", "off") != "off":
            # U-Net-style skips for the student (ablation: AD folklore says these let the decoder copy the input).
            # source teacher features per scale: [128@64, 256@32, 512@16]; detached (skips must not train the teacher).
            globals()["_SKIP_FEATS"] = [f.detach() for f in source_feats]
        # bottleneck_inputs = [cat(t, s)] for each level — doubles batch dim
        bottleneck_inputs = [
            torch.cat([t, s], dim=0) for t, s in zip(target_feats, source_feats)
        ]
        return source_target_features, bottleneck_inputs


class ConvNeXtBottleneckLayer(nn.Module):
    """BottleneckLayer for ConvNeXt-Base: [128, 256, 512] → [1024, 8, 8].

    Mirrors the WRN BottleneckLayer but with half channel dimensions.
    """

    def __init__(self, block: type = AttentionBottleneck, layers: int = 3, halve: int = 2) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = 1
        self.base_width = 64
        self.halve = halve

        # inplanes tracks current channel count for _make_layer
        self.inplanes = 128 * block.channel_expansion  # 512

        # Main bottleneck: planes=256 → output = 256*4 = 1024, stride=2 → 8×8
        self.bn_layer = nn.Sequential(self._make_layer(block, 256, layers, stride=2))

        # Projection convs for x[0]: 128@64 → 256@32 → 512@16
        self.conv1 = _bn_conv3x3(128, 256, 2)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _bn_conv3x3(256, 512, 2)
        self.bn2 = norm_layer(512)

        # Projection conv for x[1]: 256@32 → 512@16
        self.conv3 = _bn_conv3x3(256, 512, 2)
        self.bn3 = norm_layer(512)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif hasattr(module, "merge_kernel") and self.halve == 2:
                module.merge_kernel()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.channel_expansion:
            downsample = nn.Sequential(
                _bn_conv1x1(self.inplanes * 3, planes * block.channel_expansion, stride),
                norm_layer(planes * block.channel_expansion),
            )
        layers = []
        layers.append(block(
            self.inplanes * 3, planes, stride, downsample,
            self.groups, self.base_width, norm_layer, halve=self.halve,
        ))
        self.inplanes = planes * block.channel_expansion
        layers.extend(
            block(self.inplanes, planes, groups=self.groups,
                  base_width=self.base_width, norm_layer=norm_layer, halve=self.halve)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        # x[0]: [2B, 128, 64, 64]  x[1]: [2B, 256, 32, 32]  x[2]: [2B, 512, 16, 16]
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
        l2 = self.relu(self.bn3(self.conv3(x[1])))
        feature = torch.cat([l1, l2, x[2]], 1)  # [2B, 1536, 16, 16]
        output = self.bn_layer(feature)  # [2B, 1024, 8, 8]
        return output.contiguous()


# ── ConvNeXt-flavored student decoder ──────────────────────────────────
class ConvNeXtBlock(nn.Module):
    """Native ConvNeXt block: 7×7 depthwise → channels-last LN → 4× MLP → GELU → LayerScale → residual.

    Matches the block in Liu et al. 2022 (`A ConvNet for the 2020s`). Operates on
    [B, C, H, W] tensors but permutes to [B, H, W, C] for the LN+MLP path so that
    LayerNorm normalizes over the channel dim (much faster than channels-first LN).
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, layer_scale_init: float = 1e-6):
        super().__init__()
        hidden = int(mlp_ratio * dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)              # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)       # → [B, H, W, C] (channels-last)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)       # → [B, C, H, W]
        return residual + x


class ConvNeXtUpsample(nn.Module):
    """ConvNeXt-style upsample: channels-last LN → learned upsample → (halve ch, ↑2).

    Mirrors the encoder's downsample stem (LN → 2×2 conv stride=2) but inverted.
    LN is channels-last for consistency with the blocks.

    Three upsample modes (set once per run via the decoder's `upsample_mode` kwarg):

    - "transpose" (default): 2×2 ConvTranspose2d stride=2.
        Single learned kernel does upsample + channel projection. ~4× fewer params
        than the resize+conv variants. Can produce checkerboard artifacts if the
        kernel doesn't align cleanly (not usually visible at feature resolutions).

    - "bilinear": F.interpolate(bilinear) + 3×3 Conv2d stride=1.
        Fixed bilinear upsample produces smooth features; learnable 3×3 conv projects
        channels and blends spatial neighborhood. No checkerboard. Slightly heavier
        (3×3 vs 2×2 kernel) but more stable.

    - "nearest": F.interpolate(nearest) + 3×3 Conv2d stride=1.
        Fixed nearest-neighbor upsample gives block-constant outputs; the following
        3×3 conv has to do all smoothing. Cheapest upsample op but the conv has
        more work to do. Least common choice.
    """

    def __init__(self, in_dim: int, out_dim: int, mode: str = "transpose"):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        if mode == "transpose":
            self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)
        elif mode in ("bilinear", "nearest"):
            interp_kwargs = {"scale_factor": 2, "mode": mode}
            if mode == "bilinear":
                interp_kwargs["align_corners"] = False
            self.up = nn.Sequential(
                nn.Upsample(**interp_kwargs),
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown upsample mode: {mode!r}. "
                             "Expected one of 'transpose', 'bilinear', 'nearest'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return self.up(x)


class ConvNeXtDecoder(nn.Module):
    """ConvNeXt-flavored student decoder mirroring ConvNeXt-Base structurally.

    Input:  [B, 1024, 8, 8] from ConvNeXtBottleneckLayer
    Output: [feature_c, feature_b, feature_a] with shapes
            [B, 128, 64, 64], [B, 256, 32, 32], [B, 512, 16, 16]
            (matches ConvNeXtTeachers' returned ordering)

    Architecture (inverse of timm's `convnext_base.dinov3_lvd1689m` encoder):
        in [1024, 8, 8]
        → ConvNeXtUpsample(1024 → 512, ↑2)            [512, 16, 16]
        → N₁ × ConvNeXtBlock(dim=512)                  ← matches teacher level 2
        → ConvNeXtUpsample(512 → 256, ↑2)             [256, 32, 32]
        → N₂ × ConvNeXtBlock(dim=256)                  ← matches teacher level 1
        → ConvNeXtUpsample(256 → 128, ↑2)             [128, 64, 64]
        → N₃ × ConvNeXtBlock(dim=128)                  ← matches teacher level 0

    Default depths [3, 3, 3] keep the student under ~9M params. Strict mirror of
    ConvNeXt-Base encoder depths [3, 3, 27, 3] would be [27, 3, 3] (~60M params) —
    too heavy for a KD student; Reverse Distillation convention is for the student
    to be smaller than the teacher.
    """

    def __init__(
        self,
        depths: list[int] = (3, 3, 3),
        dims: list[int] = (512, 256, 128),
        in_dim: int = 1024,
        layer_scale_init: float = 1e-6,
        upsample_mode: str = "transpose",
    ):
        super().__init__()
        assert len(depths) == 3 and len(dims) == 3

        in_dims = [in_dim, dims[0], dims[1]]  # [1024, 512, 256]
        self.upsamples = nn.ModuleList([
            ConvNeXtUpsample(i, o, mode=upsample_mode) for i, o in zip(in_dims, dims)
        ])
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i], layer_scale_init=layer_scale_init)
                for _ in range(depths[i])
            ])
            for i in range(3)
        ])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, batch: torch.Tensor) -> list[torch.Tensor]:
        sk = globals().get("_SKIP_FEATS") if globals().get("_STUDENT_SKIPS", "off") != "off" else None

        def _add(y, i):
            if sk is None:
                return y
            f = sk[i]
            if f.shape[0] != y.shape[0]:           # student runs on the doubled batch (target || source)
                f = f.repeat(y.shape[0] // f.shape[0], 1, 1, 1)
            return y + f

        x = self.upsamples[0](batch)         # [B, 512, 16, 16]
        feature_a = _add(self.stages[0](x), 2)
        x = self.upsamples[1](feature_a)     # [B, 256, 32, 32]
        feature_b = _add(self.stages[1](x), 1)
        x = self.upsamples[2](feature_b)     # [B, 128, 64, 64]
        feature_c = _add(self.stages[2](x), 0)
        return [feature_c, feature_b, feature_a]


# ConvNeXt-decoder config, set once in main() from CLI args. Kept as a module-level
# global so the knob is scoped to the ConvNeXt path only — it doesn't need to thread
# through run_mode() / make_model() which are backbone-agnostic.
_CNX_UPSAMPLE_MODE: str = "transpose"


def _swap_to_convnext(model: nn.Module) -> None:
    """Replace WRN-50-2 components with ConvNeXt-Base equivalents in-place.

    Swaps: teachers, bottleneck, student (decoder), DFS, FC head. Uses a
    ConvNeXt-flavored decoder (depthwise 7×7 + channels-last LN + GELU +
    LayerScale) that structurally mirrors the encoder. Upsample mode is
    read from the module-level ``_CNX_UPSAMPLE_MODE`` so that the knob stays
    scoped to the ConvNeXt path.
    """
    model.teachers = ConvNeXtTeachers()
    model.bottleneck = ConvNeXtBottleneckLayer()
    model.student = ConvNeXtDecoder(upsample_mode=_CNX_UPSAMPLE_MODE)
    model.dfs = DomainRelatedFeatureSelection(num_channels=128)
    model.fc = nn.Linear(128, 1)


# ── Config ──────────────────────────────────────────────────────────────
SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
CASHEW = CASHEW_ROOT  # .../easy_test/cashew/
UNINET_EXP = CASHEW / "experiment_UniNet"
SPLITS = PROJECT_ROOT / "data" / "splits.json"

N_TRAIN_EASY = 44
N_TRAIN_HARD = 0  # hard cases → test only
N_EPOCHS = 20  # T2bs4 default (was 30 in older runs)
BATCH_SIZE = 4  # T2bs4 default (was 16 in March 2026 runs)
IMAGE_SIZE = 256

torch.set_float32_matmul_precision("medium")


def _worker_init_fn(worker_id):
    """Seed Python random + numpy in DataLoader workers.

    Each worker derives its seed from torch.initial_seed(), which the parent
    DataLoader sets per-worker based on its `generator` arg. So multiple workers
    get distinct random streams while staying reproducible across runs.
    Mirrors scripts/train_anomagic.py:_worker_init_fn (diffusion training).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.LANCZOS)
TO_TENSOR = transforms.ToTensor()
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class CashewAugmentation:
    """Single augmentation point: all ops happen in [0, 1] tensor space, post-resize, pre-Normalize.

    Pipeline order (image): geometric → CJ → fixed photometric → blur → noise.
    Geometric (flip, 90° rot) is applied jointly to image+mask+canvas with the same params.
    Photometric (CJ, blur, noise) is applied independently per image (NEVER to mask).

    All ops commute with /255 and with the LANCZOS resize, so placing them here gives the
    same result a pre-resize chain would (modulo FP/quantization noise) while keeping blur σ
    and noise std defined in pixels at model-input scale.
    """

    def __init__(
        self,
        geometric: bool = False,
        cj: float = 0.0,
        blur_sigma: float = 0.0,
        blur_range: bool = False,
        noise_std: float = 0.0,
        fixed_brightness: float = 0.0,
        fixed_contrast: float = 0.0,
        fixed_saturation: float = 0.0,
        dust_n: int = 0,
    ):
        self.geometric = geometric
        self.cj = (
            transforms.ColorJitter(brightness=cj, contrast=cj, saturation=cj)
            if cj > 0 else None
        )
        if blur_sigma > 0:
            ks = max(3, int(blur_sigma * 6) | 1)
            sigma = (0.01, blur_sigma) if blur_range else (blur_sigma, blur_sigma)
            self.blur = transforms.GaussianBlur(kernel_size=ks, sigma=sigma)
        else:
            self.blur = None
        self.noise_std = noise_std
        self.fixed_b = fixed_brightness
        self.fixed_c = fixed_contrast
        self.fixed_s = fixed_saturation
        # Dust: n speckles per image. Radius scales by level:
        # n≤15 → r=2 (light), 15<n≤80 → r=3 (medium/heavy), n>80 → r=4 (extreme).
        self.dust_n = int(dust_n)
        if self.dust_n <= 15:
            self.dust_r = 2
        elif self.dust_n <= 80:
            self.dust_r = 3
        else:
            self.dust_r = 4

    def __call__(self, image, mask, canvas, paired_photometric: bool = False):
        """All inputs are float tensors in [0, 1] at model resolution.

        image:  [3, H, W]
        mask:   [1, H, W] or None  (binary, NEVER touched by photometric ops)
        canvas: [3, H, W] or None

        paired_photometric: if True AND canvas is not None, apply the SAME photometric
            params (cj, blur, noise, dust) to image and canvas. Default False (legacy
            behavior: independent photometric per side, used for CAAO training).
        """
        if self.geometric:
            image, mask, canvas = self._joint_geometric(image, mask, canvas)
        if paired_photometric and canvas is not None:
            image, canvas = self._joint_photometric(image, canvas)
        else:
            image = self._photometric(image)
            if canvas is not None:
                canvas = self._photometric(canvas)
        return image, mask, canvas

    def _joint_photometric(self, image, canvas):
        """Apply IDENTICAL photometric params to image and canvas (paired aug).

        Same ColorJitter factors, same blur sigma, same noise tensor, same dust positions.
        """
        # ColorJitter: sample factors once via torchvision's get_params, apply to both
        if self.cj is not None:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
                transforms.ColorJitter.get_params(
                    self.cj.brightness, self.cj.contrast,
                    self.cj.saturation, self.cj.hue)
            )
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    image = TF.adjust_brightness(image, brightness_factor)
                    canvas = TF.adjust_brightness(canvas, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    image = TF.adjust_contrast(image, contrast_factor)
                    canvas = TF.adjust_contrast(canvas, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    image = TF.adjust_saturation(image, saturation_factor)
                    canvas = TF.adjust_saturation(canvas, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    image = TF.adjust_hue(image, hue_factor)
                    canvas = TF.adjust_hue(canvas, hue_factor)
        # Fixed B/C/S (already deterministic, identical for both)
        if self.fixed_b != 0.0:
            image = TF.adjust_brightness(image, 1.0 + self.fixed_b)
            canvas = TF.adjust_brightness(canvas, 1.0 + self.fixed_b)
        if self.fixed_c != 0.0:
            image = TF.adjust_contrast(image, 1.0 + self.fixed_c)
            canvas = TF.adjust_contrast(canvas, 1.0 + self.fixed_c)
        if self.fixed_s != 0.0:
            image = TF.adjust_saturation(image, 1.0 + self.fixed_s)
            canvas = TF.adjust_saturation(canvas, 1.0 + self.fixed_s)
        # Blur: sample sigma ONCE if range, apply same sigma to both
        if self.blur is not None:
            kernel_size = self.blur.kernel_size
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            sigma_range = self.blur.sigma
            if isinstance(sigma_range, (list, tuple)) and sigma_range[0] != sigma_range[1]:
                sigma = float(torch.empty(1).uniform_(float(sigma_range[0]), float(sigma_range[1])).item())
            elif isinstance(sigma_range, (list, tuple)):
                sigma = float(sigma_range[0])
            else:
                sigma = float(sigma_range)
            image = TF.gaussian_blur(image, kernel_size, [sigma, sigma])
            canvas = TF.gaussian_blur(canvas, kernel_size, [sigma, sigma])
        # Noise: sample tensor ONCE, add to both
        if self.noise_std > 0:
            noise = torch.randn_like(image) * self.noise_std
            image = (image + noise).clamp(0, 1)
            canvas = (canvas + noise).clamp(0, 1)
        # Dust: same positions
        if self.dust_n > 0:
            n = self.dust_n; r = self.dust_r
            h, w = image.shape[-2:]
            ys = torch.randint(r, max(r + 1, h - r), (n,))
            xs = torch.randint(r, max(r + 1, w - r), (n,))
            for y, x_ in zip(ys.tolist(), xs.tolist()):
                image[..., y - r:y + r + 1, x_ - r:x_ + r + 1] = 1.0
                canvas[..., y - r:y + r + 1, x_ - r:x_ + r + 1] = 1.0
        return image, canvas

    def _joint_geometric(self, image, mask, canvas):
        if random.random() > 0.5:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)
            if canvas is not None:
                canvas = TF.hflip(canvas)
        if random.random() > 0.5:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)
            if canvas is not None:
                canvas = TF.vflip(canvas)
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            image = torch.rot90(image, k, dims=(-2, -1))
            if mask is not None:
                mask = torch.rot90(mask, k, dims=(-2, -1))
            if canvas is not None:
                canvas = torch.rot90(canvas, k, dims=(-2, -1))
        return image, mask, canvas

    def _photometric(self, x):
        if self.cj is not None:
            x = self.cj(x)
        if self.fixed_b != 0.0:
            x = TF.adjust_brightness(x, 1.0 + self.fixed_b)
        if self.fixed_c != 0.0:
            x = TF.adjust_contrast(x, 1.0 + self.fixed_c)
        if self.fixed_s != 0.0:
            x = TF.adjust_saturation(x, 1.0 + self.fixed_s)
        if self.blur is not None:
            x = self.blur(x)
        if self.noise_std > 0:
            x = (x + torch.randn_like(x) * self.noise_std).clamp(0, 1)
        if self.dust_n > 0:
            x = self._add_dust(x)
        return x

    def _add_dust(self, x):
        """Scatter n white speckles of radius r onto x [C, H, W] in [0, 1]."""
        n = self.dust_n
        r = self.dust_r
        h, w = x.shape[-2:]
        ys = torch.randint(r, max(r + 1, h - r), (n,))
        xs = torch.randint(r, max(r + 1, w - r), (n,))
        for y, x_ in zip(ys.tolist(), xs.tolist()):
            x[..., y - r:y + r + 1, x_ - r:x_ + r + 1] = 1.0
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="UniNet cashew: baseline vs supervised vs extension")
    parser.add_argument("--placement", choices=["rp", "mo"], required=True,
                        help="Placement mode: rp (random position) or mo (mimic original)")
    parser.add_argument("--cfg-variant",
                        default="cfg7_vis",
                        help="Subfolder of <cashew-100-dir>/generated/ to train on "
                             "(any variant dir name, e.g. flux_kontext / a kontext_sweep config)")
    parser.add_argument("--cashew-100-dir", type=Path, default=None,
                        help="Override cashew_100 base dir")
    parser.add_argument("--prep-dir", type=Path, default=None,
                        help="Override prep dir (default: <cashew-100-dir>/prep)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--modes", type=int, nargs="+", default=[4],
                        help="Which modes to run (e.g. --modes 1 4). Default: 4 (Mode 4 Extension only). "
                             "Pass '--modes 1 2 3 4' to run all modes.")
    parser.add_argument("--w-canvas", type=float, default=1.0,
                        help="Weight for canvas-normal alignment term (Mode 4)")
    parser.add_argument("--w-twosided", type=float, default=1.0,
                        help="Weight for two-sided syn→teacher term (Mode 4)")
    parser.add_argument("--weight-push-only", action="store_true",
                        help="Scale KD push by per-position teacher importance WITHOUT changing pull "
                             "to weighted MSE. Pull stays cosine. Tests push weighting in isolation.")
    parser.add_argument("--lw-align-only", action="store_true",
                        help="When combined with --loss-form weighted_mse, only apply weighted MSE to "
                             "the alignment terms (Term 1 normal→normal, Term 2 synth→canvas). "
                             "Term 3 (synth→synth two-sided) stays cosine. Tests whether the "
                             "restoration mapping alone benefits from weighted MSE.")
    parser.add_argument("--scale-kd-push", action="store_true",
                        help="When combined with --loss-form weighted_mse, also scale KD push (anomaly "
                             "dis-alignment) by per-position teacher importance. Default OFF.")
    parser.add_argument("--scale-push-contrastive", action="store_true",
                        help="When combined with --loss-form weighted_mse, also scale contrastive L_SC "
                             "losses by per-position teacher importance. Default OFF. "
                             "Implies --scale-kd-push.")
    parser.add_argument("--loss-form", type=str, default="cos", choices=["cos","weighted_mse"],
                        help="Local-cosine loss form. 'cos' (default) = 1-cos(S, T). "
                             "'weighted_mse' = sum_c a_c (S_c - T_c)^2 on F.normalize'd features, "
                             "with a_c = teacher channel magnitude. Ablation for DFS train-test asymmetry. "
                             "Pair with --no-train-apply-dfs to remove the DFS modulation entirely.")
    parser.add_argument("--lambda-norm-cos", type=float, default=1.0,
                        help="Scale applied to the FULL normal-side pull (cosine + contrastive + margin) "
                             "in _lkd_lsc_twosided. 1.0 = unchanged. Mathematically equivalent to "
                             "preserve_outside (Term 3b) when lkd_mode='local': λ_norm = 1 + w_preserve. "
                             "1.25 ≡ preserve025, 1.50 ≡ preserve050.")
    parser.add_argument("--lambda-push", type=float, default=1.0,
                        help="Multiplier on anomaly-pixel (push) branch of Term 3 only. "
                             "1.0 = current 3:1 align:push. 3.0 = 1:1.")
    parser.add_argument("--ct-twosided", choices=["off", "full", "push_only"], default="off",
                        help="exp2: two-sided T(synth)->T(canvas) on the ADAPTIVE teacher only "
                             "(target stages). 'full' = pull T_synth->T_canvas on normal pixels + push "
                             "apart on anomaly pixels. 'push_only' = anomaly push only. No SG, grad to "
                             "adaptive teacher. Sharpens the teacher's own anomaly signal vs the canvas.")
    parser.add_argument("--w-ct-twosided", type=float, default=1.0,
                        help="Weight on the ct-twosided term (per-term lever; default 1.0).")
    parser.add_argument("--tproto", choices=["off", "push", "full", "intra"], default="off",
                        help="M2/M3 teacher-prototype contrastive (canvas-free analog of ct-twosided): "
                             "on the ADAPTIVE teacher (target stages), within each anomaly image push "
                             "anomaly-pixel feats off the image's own normal-pixel prototype (orthogonal-"
                             "stop, no SG). 'push'=push_only; 'full' adds the (near-trivial) normal pull.")
    parser.add_argument("--w-tproto", type=float, default=1.0,
                        help="Weight on the tproto term (default 1.0).")
    parser.add_argument("--real-tproto", choices=["off", "push", "full", "intra"], default="off",
                        help="Mode 5 only: teacher same-object push on the REAL source anomalies "
                             "(the disjoint-44 'ones used to inpaint'). Within each real anomaly image, "
                             "on the adaptive teacher (target stages), push anomaly-pixel feats off the "
                             "image's own normal-region prototype (orthogonal-stop, no SG). 'push'=push "
                             "only; 'full' adds normal-pixel pull. Reuses --target-teacher-lr.")
    parser.add_argument("--w-real-tproto", type=float, default=1.0,
                        help="Weight on the real-tproto term (default 1.0).")
    parser.add_argument("--no-supervised-bce", action="store_true",
                        help="Mode 2/3: pass label=None so the base UniNet classification BCE term is SKIPPED (M2 without BCE).")
    parser.add_argument("--no-real-student-sup", action="store_true",
                        help="Mode 5: do NOT supervise the student on real anomalies (skip the real student<->teacher two-sided). Reals then shape ONLY the adaptive teacher via --real-tproto (teacher-only variant).")
    parser.add_argument("--target-teacher-lr", type=float, default=1e-6,
                        help="LR for the adaptive (target) teacher param group. Default 1e-6. "
                             "Raise (e.g. 1e-5) to give teacher-shaping terms (ct-twosided) a real lever.")
    parser.add_argument("--lambda-weight", type=float, default=None,
                        help="Override the UniNet loss split: loss = λ·cosine (L_KD) + (1−λ)·contrastive (L_SC). "
                             "Default None = 0.7 (UniNet). 1.0 removes L_SC entirely (minimal-UniNet ablation).")
    parser.add_argument("--student-skips", choices=["off", "add"], default="off",
                        help="Ablation (ConvNeXt student only): add U-Net skip connections from the frozen source "
                             "teacher's features into the student decoder at each scale. AD folklore avoids these "
                             "because the decoder can copy the input and reconstruct anomalies too.")
    parser.add_argument("--save-best", action="store_true",
                        help="Also save the checkpoint of the epoch with the best composite image AUROC (mean over the "
                             "evaluated shift levels, primary agg) as <name>_best.pt, with its epoch in <name>_best.json. "
                             "NOTE: the epoch is chosen on the test set — the same oracle selection the tables' 'max' uses.")
    parser.add_argument("--single-teacher", action="store_true",
                        help="Single ADAPTIVE teacher: tie the frozen source-teacher branch to the adaptive target "
                             "teacher (same module), so there is no frozen teacher at all. Equivalent to one trainable "
                             "teacher with each image passed twice. Use with --target-teacher-lr.")
    parser.add_argument("--no-uninet-bce", action="store_true",
                        help="Modes 1-3: zero anomalib's image-level BCE on the avgpool+fc head of student stage 0 "
                             "(minimal-UniNet ablation). Mode 4 has no such BCE unless --native-bce.")
    parser.add_argument("--margin-mode", choices=["off", "on"], default="off",
                        help="Paper-style margin loss term. 'off' (current default) = no margin; "
                             "'on' = add per-group margin: pull normals toward similarity ≥ margin "
                             "(relu(margin - diag_sim)), push anomalies below margin/2 "
                             "(relu(diag_sim - margin/2)). Applies to both Modes 1-3 (UniNetLoss) "
                             "and Mode 4 (ExtensionLoss): Term 1 (normal align) and Term 2 (synth→canvas) "
                             "use pull only; Term 3 (synth two-sided) uses pull on normals, push on anomalies.")
    parser.add_argument("--canvas-align-mode", choices=["uniform", "anomaly-only", "50-50"],
                        default="uniform",
                        help="Mode 4 only: position weighting for Term 2 (canvas-align on synth). "
                             "'uniform' (default) = mean over all positions (~5%% effective weight "
                             "on anomaly pixels, 95%% on normal — diluted). 'anomaly-only' = ~100%% "
                             "on anomaly positions (concentrates Term 2 where it adds new info). "
                             "'50-50' = balanced split between normal and anomaly groups. "
                             "L_KD falls back to local for the L_KD component when not 'uniform' "
                             "(global L_KD can't compose with position masking).")
    parser.add_argument("--proj-consistency", action="store_true",
                        help="Group D: add projection-consistency aux loss. "
                             "S(T(synth)) ≈ stopgrad(S(T(canvas))) at anomaly positions. "
                             "Costs ~1 extra student forward on the synth subset per step.")
    parser.add_argument("--lambda-proj", type=float, default=0.1,
                        help="Weight on projection-consistency loss. Default 0.1.")
    parser.add_argument("--residual-direction-alignment", action="store_true",
                        help="Group F: align the detector residual direction with the teacher's "
                             "synthetic→canvas counterfactual residual at anomaly positions. "
                             "r_det = normalize(T_s) - normalize(S_s); r_cf = normalize(T_s) - normalize(T_c). "
                             "L = mean_in_mask(1 - cos(r_det_hat, stopgrad(r_cf_hat))). "
                             "No extra forward — reuses already-computed sf_s, tf_s, ctf_flat.")
    parser.add_argument("--lambda-resdir", type=float, default=0.05,
                        help="Weight on residual-direction-alignment loss. Default 0.05.")
    parser.add_argument("--pairrank", action="store_true",
                        help="Group G: pairwise rank loss on (synth, canvas) summed-map scores. "
                             "L = mean(softplus(margin - (s_s - s_c))), score = top-frac mean of "
                             "the per-stage cosine-distance maps accumulated to 64x64. "
                             "Requires --proj-consistency (uses canvas_student_features).")
    parser.add_argument("--lambda-pairrank", type=float, default=0.05,
                        help="Weight on pairwise rank loss. Default 0.05.")
    parser.add_argument("--pairrank-margin", type=float, default=0.1,
                        help="Pairrank margin: penalize when s_s - s_c < margin. Default 0.1.")
    parser.add_argument("--reltail", action="store_true",
                        help="Group G: relative-tail suppression on canvas. "
                             "L = mean(relu(s_canvas_tail - rho * stopgrad(s_anom_ref))) where "
                             "s_canvas_tail = top-frac of A_c, s_anom_ref = mean of A_s inside "
                             "anomaly mask. Requires --proj-consistency.")
    parser.add_argument("--lambda-reltail", type=float, default=0.05,
                        help="Weight on relative-tail loss. Default 0.05.")
    parser.add_argument("--reltail-rho", type=float, default=0.5,
                        help="Reltail ratio: canvas top-frac suppressed below rho * anom-ref. Default 0.5.")
    parser.add_argument("--reltail-frac", type=float, default=0.01,
                        help="Top-fraction for pairrank/reltail score (e.g. 0.01 = top 1%%). Default 0.01.")
    parser.add_argument("--mask-rank-loss", action="store_true",
                        help="Group G: intra-image localization rank — s_in (mean of A_synth inside "
                             "mask) should exceed s_out (top-1%% of A_synth outside mask) by margin. "
                             "L = mean(softplus(maskrank_margin - (s_in - s_out))). "
                             "Requires --proj-consistency (uses shared accumulator).")
    parser.add_argument("--lambda-maskrank", type=float, default=0.05,
                        help="Weight on mask-rank loss. Default 0.05.")
    parser.add_argument("--maskrank-margin", type=float, default=0.1,
                        help="Mask-rank margin: penalize when s_in - s_out < margin. Default 0.1.")
    parser.add_argument("--adaptive-triplet", action="store_true",
                        help="Group E: adaptive counterfactual triplet at anomaly positions. "
                             "L_triplet = relu(d(S_s, T_c) - d(S_s, T_s) + alpha * stopgrad(d(T_s, T_c))). "
                             "Designed to replace two-sided disalignment (use with --w-twosided 0.0).")
    parser.add_argument("--triplet-alpha", type=float, default=0.5,
                        help="Margin scale on the teacher gap d(T_s, T_c) in adaptive triplet. Default 0.5.")
    parser.add_argument("--gap-gated-outside-canvas", action="store_true",
                        help="Variant 3: apply soft-gated canvas-align OUTSIDE anomaly mask. "
                             "Gate g_p = exp(-gap_p / tau), tau = stopgrad(median outside gap). "
                             "Only fires where T_s and T_c agree. Used on top of CAAO.")
    parser.add_argument("--lambda-gated-outcanvas", type=float, default=0.25,
                        help="Weight for gap-gated outside canvas-align term.")
    parser.add_argument("--stop-gradient", action="store_true",
                        help="Detach teacher features in loss (paper SG()). Prevents gradients "
                             "flowing through adaptive target teacher in Terms 1/3 (Extension) "
                             "and all terms (Modes 1/2/3).")
    parser.add_argument("--lambda-idem", type=float, default=0.0,
                        help="Idempotency loss weight. When >0, feeds student output back through "
                             "bottleneck+decoder and penalizes deviation from the single-pass output. "
                             "Default 0.0 (off). Mode 4 only.")
    parser.add_argument("--idem-caao", action="store_true",
                        help="Restrict idempotency loss to anomaly pixels only (CAAO-style). "
                             "Default: all pixels of synthetic images.")
    parser.add_argument("--no-twosided-push", action="store_true",
                        help="Disable push (dis-align) in two-sided loss. Keep pull on normal pixels, "
                             "skip push on anomaly pixels. Tests whether push is needed.")
    parser.add_argument("--no-detach-any", action="store_true",
                        help="Disable ALL detach/stopgrad everywhere — canvas teacher features, "
                             "proj-consistency target, everything. Full gradient flow. "
                             "Overrides --stop-gradient if both set.")
    # ── Teacher-BCE ablation (image-level BCE on ADAPTIVE TARGET teacher features only) ──
    parser.add_argument("--use-teacher-bce", action="store_true",
                        help="Add an auxiliary image-level BCE on the adaptive target teacher "
                             "features (NOT student). Head: AdaptivePool -> Flatten -> Linear(C,1) "
                             "-> BCEWithLogitsLoss. Grad flows to head + target_teacher only.")
    parser.add_argument("--teacher-bce-weight", type=float, default=0.0,
                        help="Weight for the teacher-BCE term. total = orig + w * teacher_bce.")
    parser.add_argument("--teacher-bce-feature-layer", type=int, default=0,
                        help="Which target-teacher feature layer to attach the BCE head to (0/1/2).")
    parser.add_argument("--teacher-bce-pool", type=str, default="avg", choices=["avg", "max"],
                        help="Spatial pooling for the teacher-BCE head.")
    parser.add_argument("--teacher-bce-loss-type", type=str, default="bce", choices=["bce", "focal"],
                        help="'bce' = BCEWithLogitsLoss; 'focal' = focal loss from logits.")
    parser.add_argument("--teacher-bce-focal-gamma", type=float, default=2.0,
                        help="Focal-loss gamma (only used when --teacher-bce-loss-type focal).")
    parser.add_argument("--bce-pool-max", action="store_true",
                        help="Replace the UniNet image-BCE head's AdaptiveAvgPool2d with "
                             "AdaptiveMaxPool2d (modes 2/3 supervised BCE + mode-4 native-BCE). "
                             "Max-pool matches the AD prior (one strong region) vs avg dilution.")
    parser.add_argument("--recon-concat-bce", action="store_true",
                        help="Discriminative BCE head over [source-teacher input ⊕ student "
                             "reconstruction] (concat on channels) → conv stack → spatial max-pool "
                             "→ image logit. Grad to head + student. Reads the T-S discrepancy "
                             "directly (does NOT force the student to self-classify like native-BCE).")
    parser.add_argument("--recon-bce-weight", type=float, default=0.0,
                        help="Weight for the recon-concat BCE term.")
    parser.add_argument("--recon-head", type=str, default="grouped", choices=["grouped","grouped_pmax","grouped_pmax_cos","grouped_pmax_cos_avg","grouped_cosrep","fc_mix_lin","f_mix_lin","fc_per_gridmax","fc_mix_gridmax","grouped_cos","cosonly","tsrd_dilated","tiny","dense"],
                        help="recon-concat-BCE head: grouped (per-channel grouped 1x1, ~5C params, default), "
                             "tsrd_dilated (T,S,|T-S|,T-S views → dilated conv, gate champion port), "
                             "tiny (single 1x1 over 4C), dense (3x3 conv stack, ~332k, overfit-prone).")
    parser.add_argument("--recon-l2", action="store_true",
                        help="L2-normalize each feature map (channel dim) before the recon-BCE head "
                             "→ magnitude-free directional features aligned with the cosine geometry "
                             "the detector scores on. Helped the post-hoc gate (esp. TSRD-dilated).")
    parser.add_argument("--push-to-max", action="store_true",
                        help="Anomaly cosine (L_KD) push: replace relu(cos) [stops at orthogonal "
                             "cos=0] with (1+cos) [pushes to full anti-alignment cos=-1]. L_SC "
                             "contrastive push unchanged (temperature makes it ~equivalent).")
    parser.add_argument("--teacher-bce-real-only", action="store_true",
                        help="Real-anomaly-counterpart teacher-BCE: BCE positives are the REAL "
                             "anomaly counterparts of the in-batch synthetic samples (run through "
                             "the teacher, excluded from restoration); negatives are normals. "
                             "Synthetic samples are excluded from the BCE (restoration only).")
    parser.add_argument("--balanced-sampler", action="store_true",
                        help="Use a WeightedRandomSampler that balances normal vs synthetic "
                             "exposure 50/50 per epoch. Without this flag, sampling is uniform "
                             "(reflects 100 normals : 44 synth = ~70/30 ratio). With this flag, "
                             "weights are set so E[#normals] = E[#synth] per epoch. "
                             "Does NOT change loss definitions — pure sampling/exposure change.")
    parser.add_argument("--n-dynamic-synth", type=int, default=0,
                        help="N extra training-synth slots generated ON-THE-FLY per __getitem__ call. "
                             "Each call: random shape (ellipse/blob/multi/scratch) + random texture from "
                             "DTD pool + random alpha + random position inside FG mask. "
                             "Max diversity — fresh sample every batch. Used in addition to existing "
                             "diffusion synthetics (anomagic). Default 0 = disabled.")
    parser.add_argument("--dtd-train-dir", type=Path,
                        default=PROJECT_ROOT / "datasets" / "dtd" / "images",
                        help="Root of DTD textures used for dynamic synth generation.")
    parser.add_argument("--exclude-dtd-from-json", type=Path,
                        default=PROJECT_ROOT / "results" / "ood_v2_small_used_textures.json",
                        help="JSON file with list of DTD relative paths (category/file.jpg) to EXCLUDE "
                             "from training pool (held out for OOD test). Default points at ood_v2_small.")
    parser.add_argument("--dynamic-canvas-pool", type=str, default="train_plus_canvas",
                        choices=["train_only", "train_plus_canvas", "train_plus_unseen100", "diff_canvases"],
                        help="Which normals to use as canvases for dynamic synth generation. "
                             "'diff_canvases' = the 44 cashew_100 canvases that have paired diffusion synths.")
    # K=2 grouped multi-corruption setup (each grouped slot returns K views sharing one canvas)
    parser.add_argument("--n-grouped-slots", type=int, default=0,
                        help="N grouped slots. Each returns K=multi-k views on same canvas (custom collate flattens). "
                             "Used for K=2 multi-corruption CAAO experiment.")
    parser.add_argument("--grouped-type", default=None, choices=[None, "mix", "texture", "diffusion"],
                        help="K=2 view composition: mix=diff+texture, texture=texture+texture, "
                             "diffusion=diff+diff (needs multiple diff/canvas).")
    parser.add_argument("--grouped-multi-k", type=int, default=2,
                        help="K = number of views per grouped slot.")
    parser.add_argument("--alpha-jitter-log", type=float, default=0.0,
                        help="Multiplicative jitter on dynamic-synth alpha: alpha *= exp(U(-j, +j)). "
                             "Set 0.15 to apply ±15%% jitter (per K=2 grouped spec).")
    parser.add_argument("--paired-photometric-aug", dest="paired_photometric_aug",
                        action="store_true", default=True,
                        help="DEFAULT ON. Apply IDENTICAL photometric params (ColorJitter, blur, "
                             "noise, dust) to both x_synth and x_canvas. Required for valid CAAO "
                             "counterfactual: only anomaly content should differ between synth and "
                             "canvas. Geometric ops were already coupled. Use --no-paired-photometric-aug "
                             "to revert to OLD broken behavior (independent photometric on each side).")
    parser.add_argument("--no-paired-photometric-aug", dest="paired_photometric_aug",
                        action="store_false",
                        help="Explicit opt-OUT: independent photometric per side. ONLY for "
                             "reproducing old broken-aug behavior. NOT recommended.")
    parser.add_argument("--preserve-outside-mask", action="store_true",
                        help="Add outside-mask preservation term: align S(synth) -> T(synth) at "
                             "NON-anomaly positions of synth samples. Fills the coverage gap left "
                             "by CAAO + w_twosided=0, where synth-image normal positions are "
                             "otherwise unconstrained (causing pAUC-full drift on cnx).")
    parser.add_argument("--w-preserve", type=float, default=1.0,
                        help="Weight on the outside-mask preservation term. Default 1.0.")
    parser.add_argument("--canvas-align-anomaly-only", action="store_true",
                        help="DEPRECATED alias for --canvas-align-mode anomaly-only. "
                             "Kept for backward compatibility with running sweeps.")
    parser.add_argument("--tau", type=float, default=None,
                        help="Symmetric margin threshold around 0.5. When set AND --margin-mode=on, "
                             "replaces legacy asymmetric margin: P_align = 0.5 + 0.2*tau, "
                             "P_disalign = 0.5 - 0.2*tau (sum to 1). tau=1 → (0.7, 0.3); "
                             "tau=2 → (0.9, 0.1). Clamped to [0, 2.5]. Applies to both "
                             "UniNetLossControllable (Modes 1/2/3) and ExtensionLoss (Mode 4).")
    parser.add_argument("--two-sided", action="store_true",
                        help="Modes 1/2/3 only: enable two-sided spatial L_KD+L_SC at anomaly positions. "
                             "When set AND mask provided, anomaly positions get explicit dis-align "
                             "(relu(cos) push for L_KD, -log(1-diag) push for L_SC) in addition to "
                             "L_SC's existing margin push. Default OFF preserves current Mode 1/2/3 "
                             "behavior. Mode 1 has no mask so flag is inert. Mode 4 already two-sided.")
    parser.add_argument("--lkd-mode", choices=["local", "global", "both"], default="local",
                        help="L_KD formulation in ExtensionLoss (Mode 4 / Mode 3 extension). "
                             "'local' (default) = Anomalib-style: per-spatial-position cosine, averaged. "
                             "'global' = paper-faithful (Eq. 1): flatten C×H×W per sample, single cosine. "
                             "'both' = average (0.5×local + 0.5×global) — same total magnitude as single-mode "
                             "so lambda_weight balance with L_SC stays comparable. Local supervises "
                             "per-position alignment, global supervises overall feature distribution. "
                             "Two-sided spatial term is always per-position.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training augmentation (flips, 90/180/270 rot, color jitter)")
    parser.add_argument("--train-normal-dir", type=Path, default=None,
                        help="read TRAIN normals (same file names as splits.json) from this folder instead of "
                             "Data/Images/Normal. Test normals are always read from Data/Images/Normal.")
    parser.add_argument("--train-normals", type=int, default=None,
                        help="Subsample training normals (e.g. 10, 25, 50). Default: use all.")
    parser.add_argument("--no-baseline-extend", action="store_true",
                        help="Disable extending Mode 1 (Baseline) train set with extra normals "
                             "to match the sample count of Modes 2/3/4. By default Mode 1 sees "
                             "len(train_normals) + len(real_anomaly_train) samples (=144 typically), "
                             "matching the total seen by other modes. Pass this flag to keep Mode 1 "
                             "at the original len(train_normals) count.")
    parser.add_argument("--train-cj", type=float, default=0.1,
                        help="Training ColorJitter range for brightness/contrast/saturation (default 0.1)")
    parser.add_argument("--train-blur", type=float, default=0.5,
                        help="Training Gaussian blur sigma (default 0.5)")
    parser.add_argument("--test-cj", type=float, default=0.2,
                        help="Test ColorJitter range for brightness/contrast/saturation (default 0.2)")
    parser.add_argument("--test-blur", type=float, default=1.0,
                        help="Test Gaussian blur sigma (default 1.0)")
    parser.add_argument("--train-noise", type=float, default=0.0,
                        help="Training Gaussian noise std (default 0)")
    parser.add_argument("--test-noise", type=float, default=0.0,
                        help="Test Gaussian noise std (default 0)")
    parser.add_argument("--train-dust", type=int, default=0,
                        help="Training dust particle count per image (default 0 = off). "
                             "Levels: 15=light(r=2), 40=medium(r=3), 80=heavy(r=3), 150=extreme(r=4)")
    parser.add_argument("--test-dust", type=int, default=0,
                        help="Test dust particle count per image (default 0 = off). Same scale as --train-dust.")
    parser.add_argument("--test-fixed-brightness", type=float, default=0.0,
                        help="Test fixed brightness factor offset (e.g. 0.15 → 1.15x). Default 0.")
    parser.add_argument("--test-fixed-contrast", type=float, default=0.0,
                        help="Test fixed contrast factor offset (e.g. 0.15 → 1.15x). Default 0.")
    parser.add_argument("--test-fixed-saturation", type=float, default=0.0,
                        help="Test fixed saturation factor offset (e.g. 0.15 → 1.15x). Default 0.")
    parser.add_argument("--head-grad-to-student", action="store_true",
                        help="Mode 4 + --score-head only: un-detach the head's input from "
                             "student/teacher distance maps so BCE gradient flows back into "
                             "student features. Default (off) = head learns calibration alone; "
                             "features are unaffected. On = BCE shapes features like Mode 2.")
    parser.add_argument("--bce-all-normal", action="store_true",
                        help="Override BCE labels to zero for ALL samples (treat every input as "
                             "normal). Sanity ablation for --native-bce on Mode 4. Tests whether "
                             "globally-suppressing-anomaly pressure adds anything beyond Term 2 "
                             "restoration. If head collapses to always-normal, gradient signal is "
                             "trivial and this confirms the loss is redundant. Default OFF.")
    parser.add_argument("--native-bce", action="store_true",
                        help="Mode 4 only: enable anomalib's NATIVE BCE mechanism — avgpool + "
                             "fc on student_features[0] → BCE on per-image predictions vs labels. "
                             "Same mechanism Mode 2 uses, applied on top of ExtensionLoss. "
                             "Gradient flows back through fc → avgpool → student backbone. "
                             "Default OFF preserves Mode 4 hero behavior (no BCE).")
    parser.add_argument("--bce-variant", choices=["summed", "native64", "per_stage", "per_path", "final_plus_stage",
                                                    "tiny_spatial_sum", "tiny_spatial_native64", "tiny_spatial_perstage",
                                                    "tiny_spatial_final_plus_stage",
                                                    "direct_score_bce", "direct_score_rank", "direct_score_pairwise",
                                                    "asymmetric_normal"],
                        default="summed",
                        help="Mode 4 + --score-head only: which BCE-on-distance-maps family. "
                             "'summed' (B0, default) = single StatisticsHead on sum of all 6 maps. "
                             "'native64' (B1) = single head on sum of natively-64x64 maps only. "
                             "'per_stage' (B2) = 3 heads, one per stage (source+target paths summed), avg BCE. "
                             "'per_path' (B4) = 6 heads, one per individual map, avg BCE. "
                             "'final_plus_stage' (B3) = B0 + lambda_stage*B2_mean. "
                             "'tiny_spatial_sum' (B6) = TinySpatialHead on summed map (training-only). "
                             "'tiny_spatial_native64' (B7) = TinySpatialHead on native-64 sum (training-only). "
                             "'tiny_spatial_perstage' (B8) = 3 TinySpatialHeads per stage, avg BCE (training-only). "
                             "'tiny_spatial_final_plus_stage' (B12) = B3 with TinySpatialHead replacing StatisticsHead: "
                             "1 final + 3 per-stage TinySpatialHeads, L_bce = lambda_bce*(BCE_final + lambda_stage*mean_l BCE_stage_l). "
                             "'direct_score_bce' (B9) = no learned map reader; top-1%% mean of A_sum, "
                             "learnable scale+bias → BCE vs label. Calibrates the actual inference-like score. "
                             "'direct_score_rank' (B10) = top-1%% mean of A_sum, pairwise softplus rank loss "
                             "with margin=0.1. Directly optimizes image-level ranking (iAUC proxy). "
                             "'asymmetric_normal' (B11) = StatisticsHead BCE applied ONLY to normal samples "
                             "(push toward label=0); no loss on anomaly samples. Suppresses false positives "
                             "without amplifying anomaly signal.")
    parser.add_argument("--lambda-bce", type=float, default=1.0,
                        help="Outer weight on BCE block in any variant. Default 1.0.")
    parser.add_argument("--lambda-stage", type=float, default=0.1,
                        help="Inner weight on the stage block in B3 (final_plus_stage). "
                             "L_bce = lambda_bce * (BCE_final + lambda_stage * mean_l(BCE_stage_l)). Default 0.1.")
    parser.add_argument("--native-bce-per-stage", action="store_true",
                        help="Mode 4 only: enable PER-STAGE native BCE — separate avgpool+fc head "
                             "per feature stage (3 stages × source/target = 6 BCE terms). Each "
                             "stage gets its own LazyLinear head. Distributes image-level BCE "
                             "supervision across all feature scales instead of just the deepest. "
                             "Default OFF.")
    parser.add_argument("--score-head", choices=["1ch", "6ch", "stats"], default=None,
                        help="Learned anomaly score head for Mode 4 (1ch=summed, 6ch=per-layer). Default: off.")
    parser.add_argument("--head-dropout", type=float, default=0.5,
                        help="Dropout probability for score head (default 0.5)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory for results JSON (default: script dir)")
    parser.add_argument("--mask-dir", type=Path, default=None,
                        help="Override mask source for synthetic samples (use LPIPS refined masks)")
    parser.add_argument("--synthetic-epochs", type=Path, default=None,
                        help="JSON {'epochs': [[{'image','mask','canvas'}, ...], ...]}: a DIFFERENT synthetic pool per "
                             "epoch (epoch e trains on all train normals + epochs[e]). Replaces the 44 fixed thesis "
                             "synthetics; the test split (real anomalies) is unchanged. Default None = thesis behaviour.")
    parser.add_argument("--branch-balance", action="store_true",
                        help="Mode 4: put the normal branch (Term 1) and the synthetic branch (all other terms) on the "
                             "same scale and weight them by their share of the batch. Default off.")
    parser.add_argument("--grad-balance", action="store_true",
                        help="Mode 4: like --branch-balance, but the synthetic branch is scaled so its GRADIENT norm "
                             "(w.r.t. all trainable parameters) matches the normal branch's; EMA, measured every 5 steps.")
    parser.add_argument("--random-test-shifts", dest="fixed_test_shifts", action="store_false",
                        help="Legacy: draw test jitter/noise at random per epoch/seed (before 2026-10-01 13:05).")
    parser.add_argument("--fixed-test-shifts", dest="fixed_test_shifts", action="store_true", default=True,
                        help="Fix the photometric draw (jitter factors/order, noise) of every TEST image per shift level, "
                             "seeded by filename + level: identical across epochs, seeds, folds and methods. DEFAULT ON since 2026-10-01 13:05.")
    parser.add_argument("--keep-leaky-synthetics", action="store_true",
                        help="Legacy (before 2026-10-02): keep training synthetics whose canvas is a TEST normal (067 on 472).")
    parser.add_argument("--train-normal-list", type=Path, default=None,
                        help="Text file of normal image names (one per line) that REPLACE the splits.json train normals.")
    parser.add_argument("--test-anomaly-list", type=Path, default=None,
                        help="JSON list of {image, gt_mask}: REPLACES the thesis-split test anomalies (open-set folds).")
    parser.add_argument("--baseline-extra-list", type=Path, default=None,
                        help="Text file of normal image names (one per line) that Mode 1 gets on top of the train "
                             "normals (e.g. the clean canvases). Default None = random extras matched to the 44.")
    parser.add_argument("--fg-anomaly-dir", type=Path, default=None,
                        help="BiRefNet FG masks for the REAL anomaly images (e.g. cashew/birefnet_masks_anomaly). "
                             "Default None = legacy stem lookup, which gives real anomaly NNN the FG of NORMAL NNN.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override training seed (model init, dataloader, training). "
                             "Data split always uses SEED=42 for consistency.")
    parser.add_argument("--shift-levels", type=str, default=None,
                        help="Custom shift levels as JSON string, e.g. "
                             "'{\"1x\":{\"cj\":0.1,\"blur\":1.0,\"noise\":0.005}}'. "
                             "Default: 0x + 4x. Use --all-shifts for 0x+1x+2x+3x+4x preset.")
    parser.add_argument("--all-shifts", action="store_true",
                        help="Convenience: evaluate at all 5 shift levels (0x+1x+2x+3x+4x). "
                             "Overrides --shift-levels.")
    parser.add_argument("--backbone", type=str, default="wrn50",
                        choices=["wrn50", "convnext_b"],
                        help="Backbone: wrn50 (Wide-ResNet-50-2) or convnext_b (DINOv3 ConvNeXt-Base)")
    parser.add_argument("--cnx-upsample", type=str, default="transpose",
                        choices=["transpose", "bilinear", "nearest"],
                        help="ConvNeXt decoder upsample mode (only used when --backbone convnext_b). "
                             "'transpose' (default) = 2×2 ConvTranspose2d stride=2. "
                             "'bilinear' = F.interpolate(bilinear) + 3×3 Conv2d. "
                             "'nearest' = F.interpolate(nearest) + 3×3 Conv2d.")
    parser.add_argument("--score-mode", type=str, default="max",
                        choices=["max", "paper"],
                        help="Image-level score aggregation. 'max' (default, backward-compat): max pixel of "
                             "blurred summed map. 'paper': mean of adaptive top-k pixels as intended by "
                             "the UniNet paper's weighted_decision_mechanism.")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip discrepancy map visualization after training")
    parser.add_argument("--viz-samples", type=int, default=10,
                        help="Number of anomaly samples to visualize (default 10)")
    parser.add_argument("--decoder", choices=["weights", "concat", "residual"], default=None,
                        help="Train a learned anomaly map decoder after UniNet training. "
                             "Freezes UniNet, trains decoder on distance maps with dice loss.")
    parser.add_argument("--decoder-epochs", type=int, default=50,
                        help="Epochs to train the decoder (default 50)")
    parser.add_argument("--decoder-lr", type=float, default=1e-3,
                        help="Decoder learning rate (default 1e-3)")
    parser.add_argument("--blur-pixel-map", action="store_true",
                        help="Apply GaussianBlur2d(sigma=4, k=5) to pixel anomaly map before "
                             "pAUROC evaluation and discrepancy visualization.")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="ExtensionLoss temperature τ (used by L_KD/L_SC cosine softmax). "
                             "Default 2.0 (paper-faithful, T2bs4). Lower = sharper similarity distribution. "
                             "Set to 0.1 to reproduce older anomalib-default behavior.")
    parser.add_argument("--lsc-pre-relu", action="store_true",
                        help="Apply F.relu to the raw cosine similarity matrix BEFORE dividing by "
                             "temperature in ExtensionLoss._lkd_lsc_align and _lkd_lsc_twosided. "
                             "Mode 4 ablation: clamps negative similarities to 0 before softmax.")
    # Dinomaly-style hard mining (Mode 4 / Extension only). Paper-faithful: gradient-reduction
    # via detach trick. Default off (p=0.0). Paper default: --dinomaly-p 0.9 --dinomaly-factor 0.1
    # (ramp default tuned for our ~360-step training: 100 instead of paper's 1000 for ~10k-step runs).
    parser.add_argument("--dinomaly-p", type=float, default=0.0,
                        help="Fraction of EASY positions (low per-position loss) to downweight. "
                             "0.0 = disabled. Paper default: 0.9 (downweight easiest 90%%, keep "
                             "hardest 10%% at full gradient). Ramps linearly from 0 to this value "
                             "over --dinomaly-ramp-iters steps.")
    parser.add_argument("--dinomaly-factor", type=float, default=0.1,
                        help="Gradient multiplier for easy positions when Dinomaly mining is on. "
                             "Default 0.1. Forward loss values unchanged; only the gradient to "
                             "the student through easy positions is scaled.")
    parser.add_argument("--dinomaly-ramp-iters", type=int, default=100,
                        help="Linear ramp length (in training steps) for p. Default 100. "
                             "Tuned for our ~360-step training (20 ep × 18 batches): p reaches "
                             "its final value at step ~100 (≈ epoch 5-6), then full mining for "
                             "the remaining ~14 epochs. Paper default was 1000 for ~10k-step runs.")
    # DFS ablation flags (Mode 4 via the DFS-ablate class; Modes 1-3 via a _feature_selection override). Defaults reproduce current behavior.
    parser.add_argument("--train-apply-dfs", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply DFS (Domain-related Feature Selection) at training. "
                             "Default True. Use --no-train-apply-dfs to disable DFS during training.")
    parser.add_argument("--eval-apply-dfs", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply DFS at inference (before cosine_similarity in score computation). "
                             "Default False (anomalib's default behavior). Use --eval-apply-dfs to enable.")
    parser.add_argument("--infer-aggs", type=str,
                        default="standard_paper,standard_sum_max",
                        help="Comma-separated list of inference aggregations. Default = all 9 (3×3 grid). "
                             "Each name is <teacher>_<xstage> where teacher ∈ {standard, min, max} and "
                             "xstage ∈ {sum_max, softmax_max_mix, paper}. 'standard' uses 6 maps as-is, "
                             "'min'/'max' do element-wise teacher-pair reduction per layer → 3 maps. "
                             "First element is the 'primary' agg for top-level result keys; full breakdown saved under "
                             "results['Mode 4 — Extension']['per_agg'][<agg_name>].")
    parser.add_argument("--mask-downsample", type=str, default="max", choices=["max", "nearest"],
                        help="How to downsample anomaly masks to feature-map resolution (16/32/64) "
                             "during loss computation. 'max' = adaptive_max_pool2d (default, preserves "
                             "small anomalies, dilates at coarse scales). 'nearest' = F.interpolate "
                             "(spatially honest, can drop small anomalies at 16x16).")
    # ── Geometry modules (CAAO extensions) ─────────────────────────────────
    # Module 1 — Upstream teacher anomaly repulsion
    parser.add_argument("--teacher-repulsion", action="store_true",
                        help="Enable teacher-space margin repulsion: push T_synth away from T_canvas "
                             "at anomaly positions (target stages only). Default off.")
    parser.add_argument("--teacher-repulsion-weight", type=float, default=1.0,
                        help="Weight for the teacher repulsion loss term. Default 1.0.")
    parser.add_argument("--teacher-repulsion-margin", type=float, default=0.5,
                        help="Cosine-distance margin for teacher repulsion: "
                             "loss = relu(margin - D_cos(T_s, T_n)) on Omega_anom. Default 0.5.")
    parser.add_argument("--teacher-repulsion-clean-sg", action="store_true",
                        help="Stop-gradient through T_n (canvas) in the teacher repulsion loss. "
                             "Default off (no extra SG beyond the existing no_grad context).")
    parser.add_argument("--teacher-repulsion-stages", type=str, default="trainable_only",
                        choices=["trainable_only", "all_available"],
                        help="Which teacher stages to apply repulsion. "
                             "'trainable_only' = target teacher stages only (i >= n//2, default). "
                             "'all_available' = all 6 stages (source gives zero gradient anyway).")
    # Module 2 — Paired bottleneck alignment
    parser.add_argument("--bottleneck-pair", action="store_true",
                        help="Enable paired bottleneck alignment: pull B_synth toward B_canvas "
                             "at anomaly positions (8x8 bottleneck resolution). Default off.")
    parser.add_argument("--bottleneck-pair-weight", type=float, default=1.0,
                        help="Weight for the bottleneck pair alignment loss. Default 1.0.")
    parser.add_argument("--bottleneck-pair-metric", type=str, default="cosine",
                        choices=["cosine", "layernorm_smooth_l1", "layernorm_mse"],
                        help="Distance metric for bottleneck pair loss. "
                             "'cosine' = 1 - cos(B_s, B_n). "
                             "'layernorm_smooth_l1' = SmoothL1(LN(B_s), LN(B_n)). "
                             "'layernorm_mse' = MSE(LN(B_s), LN(B_n)). Default cosine.")
    parser.add_argument("--bottleneck-pair-clean-sg", action="store_true",
                        help="Stop-gradient through B_n (canvas bottleneck) in the pair loss. "
                             "Default off.")
    # Module 3 — Bottleneck prototype organization
    parser.add_argument("--bottleneck-protos", action="store_true",
                        help="Enable bottleneck prototype organization: maintain K learnable normal "
                             "prototypes on the unit sphere; pull B_n and B_s (anomaly, using clean "
                             "assignment) toward their respective prototypes. Default off.")
    parser.add_argument("--bottleneck-protos-k", type=int, default=8,
                        help="Number of prototypes K for bottleneck prototype organization. Default 8.")
    parser.add_argument("--bottleneck-protos-weight", type=float, default=1.0,
                        help="Weight for the prototype organization loss. Default 1.0.")
    parser.add_argument("--bottleneck-protos-assignment", type=str, default="nearest",
                        choices=["nearest", "soft", "sinkhorn"],
                        help="Prototype assignment mode. "
                             "'nearest' = hard argmax (default). "
                             "'soft' = softmax-weighted argmax. "
                             "'sinkhorn' = balanced optimal-transport assignment.")
    parser.add_argument("--bottleneck-protos-temperature", type=float, default=0.1,
                        help="Temperature for soft/sinkhorn prototype assignment. Default 0.1.")
    parser.add_argument("--bottleneck-protos-ema", type=float, default=0.99,
                        help="EMA decay for prototype momentum update. Default 0.99.")
    parser.add_argument("--bottleneck-protos-sinkhorn-iters", type=int, default=3,
                        help="Number of Sinkhorn iterations for balanced assignment. Default 3.")
    parser.add_argument("--bottleneck-protos-update-mode", type=str, default="gradient",
                        choices=["gradient", "ema"],
                        help="Prototype update mode. "
                             "'gradient' = nn.Parameter, updated by optimizer (default). "
                             "'ema' = register_buffer, updated by EMA only — NOT in optimizer.")
    parser.add_argument("--bottleneck-protos-lr", type=float, default=5e-3,
                        help="Learning rate for prototype parameters in gradient mode. Default 5e-3.")
    # Geometry diagnostics
    parser.add_argument("--no-geom-diagnostics", action="store_true",
                        help="Disable geometry diagnostics (V, N, R, A) for speed. "
                             "Default: always compute in Mode 4 when paired data is present.")
    # ── Geom2: SPM (ArcAD-style spatial prototype memory) ──────────────
    parser.add_argument("--spm", action="store_true",
                        help="Enable SPM loss (ArcAD-style K=500 prototype memory on bottleneck). Default off.")
    parser.add_argument("--spm-k", type=int, default=500,
                        help="Number of SPM prototypes. Default 500.")
    parser.add_argument("--spm-weight", type=float, default=1.0,
                        help="Weight for SPM loss term. Default 1.0.")
    parser.add_argument("--spm-tau", type=float, default=0.05,
                        help="Temperature for SPM Sinkhorn assignment and cross-entropy prediction. Default 0.05.")
    parser.add_argument("--spm-eps", type=float, default=0.05,
                        help="Regularization epsilon for Sinkhorn OT in SPM assignment. Default 0.05.")
    parser.add_argument("--spm-sinkhorn-iters", type=int, default=3,
                        help="Number of Sinkhorn iterations in SPM. Default 3.")
    parser.add_argument("--spm-ema", type=float, default=0.99,
                        help="EMA decay for SPM prototype update. Default 0.99.")
    parser.add_argument("--spm-init-file", type=str, default=None,
                        help="Path to .npy file with K-means initialized SPM prototypes. "
                             "If not given, prototypes are random-init on unit sphere.")
    parser.add_argument("--spm-init-n-samples", type=int, default=5000,
                        help="Number of bottleneck tokens to collect for K-means SPM init. Default 5000.")
    parser.add_argument("--spm-init-seed", type=int, default=42,
                        help="Random seed for K-means SPM init. Default 42.")
    parser.add_argument("--compute-spm-init", action="store_true",
                        help="Compute K-means SPM init, save .npy, then exit (no training).")
    # ── Geom2: paired prototype loss ────────────────────────────────────
    parser.add_argument("--paired-proto", action="store_true",
                        help="Enable paired prototype loss: synth token inherits canvas prototype assignment. Default off.")
    parser.add_argument("--paired-proto-weight", type=float, default=0.1,
                        help="Weight for paired prototype loss. Default 0.1.")
    parser.add_argument("--paired-proto-no-sg-target", action="store_true",
                        help="Do NOT stop-gradient through normal canvas target in paired proto loss. Default: sg on.")
    # ── Geom2: soft bottleneck pair ─────────────────────────────────────
    parser.add_argument("--soft-pair", action="store_true",
                        help="Enable soft bottleneck pair loss (fractional occupancy weighting). Default off.")
    parser.add_argument("--soft-pair-weight", type=float, default=0.01,
                        help="Weight for soft pair loss. Default 0.01.")
    # ── Geom2: new teacher repulsion (trep2) ────────────────────────────
    parser.add_argument("--trep", action="store_true",
                        help="Enable new teacher repulsion (trep2) with configurable/adaptive margin. Default off.")
    parser.add_argument("--trep-weight", type=float, default=0.05,
                        help="Weight for trep2 loss. Default 0.05.")
    parser.add_argument("--trep-adaptive-margin", action="store_true",
                        help="Set trep2 margin from baseline V q50 loaded from file. Default: use --trep-margin.")
    parser.add_argument("--trep-margin", type=float, default=0.0,
                        help="Fixed cosine-distance margin for trep2. Only used when --trep-adaptive-margin is off.")
    parser.add_argument("--trep-margin-file", type=str, default=None,
                        help="Path to file containing baseline V q50 value for adaptive margin. "
                             "Written by baseline A run automatically.")
    return parser.parse_args()


# ── Data loading ────────────────────────────────────────────────────────
def load_data(args):
    """Load cashew data: normals, synthetics (with canvas normals), real anomalies."""
    # Open-set runs give the synthetics (--synthetic-epochs) or the Mode-1 extra normals (--baseline-extra-list) and
    # the test anomalies (--test-anomaly-list) explicitly; the earlier fixed synthetic set is then not used and not loaded.
    open_set = getattr(args, "test_anomaly_list", None) is not None and (
        getattr(args, "synthetic_epochs", None) is not None or getattr(args, "baseline_extra_list", None) is not None)
    if open_set:
        assert not args.teacher_bce_real_only, "--teacher-bce-real-only needs the fixed set's real train anomalies"
        hard_indices, train_syn_indices, test_anom_indices, synthetic_triples = [], set(), set(), []
        print("  open-set run: the fixed synthetic set is not loaded")
    else:
        if args.cashew_100_dir is not None:
            c100_dir = args.cashew_100_dir
        else:
            c100_dir = PROJECT_ROOT / "results" / "baseline_learnable" / f"cashew_100_{args.placement}"

        prep_dir = args.prep_dir if args.prep_dir is not None else c100_dir / "prep"
        gen_dir = c100_dir / "generated" / args.cfg_variant

        assert prep_dir.exists(), f"Prep dir not found: {prep_dir}"
        assert gen_dir.exists(), f"Generated dir not found: {gen_dir}"

        easy_indices, hard_indices = [], []
        for idx_dir in sorted(prep_dir.iterdir()):
            if not idx_dir.is_dir():
                continue
            meta = json.load(open(idx_dir / "meta.json"))
            idx = meta["idx"]
            (hard_indices if meta["is_hard"] else easy_indices).append(idx)

        print(f"Cashew_100: {len(easy_indices)} easy + {len(hard_indices)} hard")
        print(f"  Placement: {args.placement} | CFG: {args.cfg_variant}")
        print(f"  Source: {c100_dir}")

        rng = random.Random(SEED)
        easy_shuffled, hard_shuffled = easy_indices[:], hard_indices[:]
        rng.shuffle(easy_shuffled)
        rng.shuffle(hard_shuffled)

        train_syn_indices = set(easy_shuffled[:N_TRAIN_EASY] + hard_shuffled[:N_TRAIN_HARD])
        test_anom_indices = set(easy_shuffled[N_TRAIN_EASY:] + hard_shuffled[N_TRAIN_HARD:])

        print(f"  Train indices: {len(train_syn_indices)} | Test anomalies: {len(test_anom_indices)}")

        # Synthetic triples: (generated_image, mask, canvas_normal)
        # If --mask-dir provided, use refined masks; otherwise fall back to placed_mask
        mask_dir = args.mask_dir
        synthetic_triples = []
        # LEAK GUARD (2026-10-02): never train on a synthetic whose canvas is a TEST normal (thesis placements put
        # synthetic 067 on test normal 472.JPG; CAAO also uses that clean canvas as its target). Legacy: --keep-leaky-synthetics.
        _test_stems = {Path(n).stem for n in json.load(open(SPLITS))["normals"]["test"]}
        _dropped, _replaced = [], []
        for idx in sorted(train_syn_indices):
            s = f"{idx:03d}"
            _cv = Path(json.load(open(prep_dir / s / "meta.json"))["normal"]).stem
            if _cv in _test_stems and not getattr(args, "keep_leaky_synthetics", False):
                # preferred: a re-synthesis on a clean canvas (scripts/fix_thesis_leak_067.py) -> the set keeps its size
                _fx = c100_dir / "leak_fix" / s
                _fm = _fx / f"{mask_dir.name}.png" if mask_dir is not None else _fx / "placed_mask.png"
                if (_fx / "image.png").exists() and _fm.exists():
                    _fc = Path(json.load(open(_fx / "meta.json"))["normal"]).stem
                    assert _fc not in _test_stems, f"leak_fix canvas {_fc} is a test normal"
                    synthetic_triples.append((_fx / "image.png", _fm, _fx / "canvas.png")); _replaced.append((s, _cv, _fc)); continue
                _dropped.append((s, _cv)); continue
            syn_img = gen_dir / f"{s}.png"
            # Try refined mask first, then placed_mask
            if mask_dir is not None and (mask_dir / f"{s}.png").exists():
                mask = mask_dir / f"{s}.png"
            else:
                mask = prep_dir / s / "placed_mask.png"
            canvas = prep_dir / s / "canvas.png"
            for p in (syn_img, mask, canvas):
                assert p.exists(), f"Not found: {p}"
            synthetic_triples.append((syn_img, mask, canvas))

        print(f"  LEAK GUARD: replaced {_replaced} (idx, test normal, new canvas) by leak_fix re-syntheses; dropped {_dropped}")
    if getattr(args, "synthetic_epochs", None) is not None:
        global SYNTH_EPOCH_SLICES
        ep = json.load(open(args.synthetic_epochs))["epochs"]
        synthetic_triples, SYNTH_EPOCH_SLICES = [], []
        for e_list in ep:
            a = len(synthetic_triples)
            for t in e_list:
                trip = (Path(t["image"]), Path(t["mask"]), Path(t["canvas"]))
                for p in trip:
                    assert p.exists(), f"Not found: {p}"
                synthetic_triples.append(trip)
            SYNTH_EPOCH_SLICES.append((a, len(synthetic_triples)))
        print(f"  --synthetic-epochs {args.synthetic_epochs}: {len(ep)} epochs x {[b - a for a, b in SYNTH_EPOCH_SLICES][:3]}... "
              f"synthetics ({len(synthetic_triples)} total) replace the {len(train_syn_indices)} fixed thesis synthetics")

    # Real anomaly TRAIN pairs (same indices as synthetic, but real images + real masks)
    real_anomaly_train_pairs = []
    for idx in sorted(train_syn_indices):
        s = f"{idx:03d}"
        subdir = "hard_imgs" if idx in hard_indices else "easy_imgs"
        img = CASHEW / "Data" / "Images" / "Anomaly" / subdir / f"{s}.JPG"
        mask = CASHEW / "Data" / "Masks" / "Anomaly" / f"{s}.png"
        assert img.exists(), f"Not found: {img}"
        assert mask.exists(), f"Not found: {mask}"
        real_anomaly_train_pairs.append((img, mask))

    # Real anomaly TEST pairs (distinct indices)
    real_anomaly_test_pairs = []
    for idx in sorted(test_anom_indices):
        s = f"{idx:03d}"
        subdir = "hard_imgs" if idx in hard_indices else "easy_imgs"
        img = CASHEW / "Data" / "Images" / "Anomaly" / subdir / f"{s}.JPG"
        mask = CASHEW / "Data" / "Masks" / "Anomaly" / f"{s}.png"
        assert img.exists(), f"Not found: {img}"
        assert mask.exists(), f"Not found: {mask}"
        real_anomaly_test_pairs.append((img, mask))
    if getattr(args, "test_anomaly_list", None) is not None:
        # open-set folds: the test anomalies are given explicitly (JSON list of {image, gt_mask | mask})
        _ta = json.load(open(args.test_anomaly_list))
        real_anomaly_test_pairs = [(Path(x["image"]), Path(x.get("gt_mask", x.get("mask")))) for x in _ta]
        for _i, _m in real_anomaly_test_pairs:
            assert _i.exists() and _m.exists(), f"Not found: {_i} / {_m}"
        print(f"  --test-anomaly-list: {len(real_anomaly_test_pairs)} test anomalies from {args.test_anomaly_list}")

    # Normals from splits.json
    with open(SPLITS) as f:
        splits = json.load(f)
    normal_dir = CASHEW / "Data" / "Images" / "Normal"
    train_normals = [normal_dir / n for n in splits["normals"]["train"]]
    test_normals = [normal_dir / n for n in splits["normals"]["test"]]
    if getattr(args, "train_normal_list", None) is not None:
        # replace the splits.json train normals by an explicit list (normal-set draw tests); never a test normal
        _names = [l.strip() for l in open(args.train_normal_list) if l.strip()]
        assert not set(_names) & set(splits["normals"]["test"]), "--train-normal-list contains test normals"
        assert len(set(_names)) == len(_names), "--train-normal-list has duplicates"
        train_normals = [normal_dir / n for n in _names]
        print(f"  --train-normal-list: {len(train_normals)} train normals from {args.train_normal_list}")
    if getattr(args, "train_normal_dir", None) is not None:
        train_normals = [args.train_normal_dir / n for n in splits["normals"]["train"]]
        print(f"  TRAIN normals redirected to {args.train_normal_dir} (test normals unchanged)")
    for p in train_normals + test_normals:
        assert p.exists(), f"Not found: {p}"

    # Subsample or extend training normals if requested
    if args.train_normals is not None:
        rng_n = random.Random(SEED)
        if args.train_normals <= len(train_normals):
            train_normals_shuffled = train_normals[:]
            rng_n.shuffle(train_normals_shuffled)
            train_normals = train_normals_shuffled[:args.train_normals]
        else:
            # Pull extra normals from disk (exclude test + already used)
            test_names = {p.name for p in test_normals}
            train_names = {p.name for p in train_normals}
            all_on_disk = sorted(normal_dir.iterdir())
            extra_pool = [p for p in all_on_disk
                          if p.name not in test_names and p.name not in train_names
                          and p.suffix.upper() in (".JPG", ".PNG", ".JPEG")]
            rng_n.shuffle(extra_pool)
            need = args.train_normals - len(train_normals)
            train_normals = train_normals + extra_pool[:need]
            if len(train_normals) < args.train_normals:
                print(f"  WARNING: only {len(train_normals)} normals available "
                      f"(requested {args.train_normals})")

    # Extra normals for Mode 1 (Baseline) — the other 3 modes see 100 normals + 44 anomaly
    # samples each = 144 total per epoch. Mode 1 saw only 100 (no anomaly counterpart).
    # Load len(real_anomaly_train_pairs) additional normals so Mode 1 sees a matched-size
    # train set. Sampled from the disk pool excluding test normals and the train normals
    # already used. Reproducible via SEED+1 RNG (avoids collision with --train-normals RNG).
    extra_baseline_normals = []
    if not getattr(args, "no_baseline_extend", False):
        n_extra = len(real_anomaly_train_pairs)
        if n_extra > 0:
            test_names = {p.name for p in test_normals}
            train_names = {p.name for p in train_normals}
            all_on_disk = sorted(normal_dir.iterdir())
            extra_pool = [p for p in all_on_disk
                          if p.name not in test_names and p.name not in train_names
                          and p.suffix.upper() in (".JPG", ".PNG", ".JPEG")]
            rng_extra = random.Random(SEED + 1)
            rng_extra.shuffle(extra_pool)
            extra_baseline_normals = extra_pool[:n_extra]
            if len(extra_baseline_normals) < n_extra:
                print(f"  WARNING: only {len(extra_baseline_normals)} extra normals available "
                      f"for baseline-extend (wanted {n_extra}); baseline will train on "
                      f"{len(train_normals) + len(extra_baseline_normals)} samples")

    if getattr(args, "baseline_extra_list", None) is not None:
        names = [l.strip() for l in open(args.baseline_extra_list) if l.strip()]
        extra_baseline_normals = [normal_dir / n for n in names]
        clash = {p.name for p in extra_baseline_normals} & ({p.name for p in train_normals} | {p.name for p in test_normals})
        assert not clash, f"--baseline-extra-list overlaps train/test normals: {sorted(clash)[:5]}"
        for p in extra_baseline_normals:
            assert p.exists(), f"Not found: {p}"
        print(f"  --baseline-extra-list: Mode 1 extras = {len(extra_baseline_normals)} listed normals")

    print(f"  Train normals: {len(train_normals)} | Test normals: {len(test_normals)}")
    print(f"  Real anomaly train: {len(real_anomaly_train_pairs)} | Synthetic triples: {len(synthetic_triples)}")
    if extra_baseline_normals:
        print(f"  Baseline-extend: +{len(extra_baseline_normals)} extra normals "
              f"(Mode 1 will see {len(train_normals) + len(extra_baseline_normals)} samples to match other modes)")
    return train_normals, test_normals, synthetic_triples, real_anomaly_train_pairs, real_anomaly_test_pairs, extra_baseline_normals


# ── Mask downsampling (loss-time) ───────────────────────────────────────
# Controls how anomaly masks are downsampled to match feature-map resolutions
# during loss computation (contrastive, L_KD term-2, undershoot logging).
# - "max": F.adaptive_max_pool2d — preserves small anomalies but dilates at coarse scales.
# - "nearest": F.interpolate(mode="nearest") — spatially honest but can lose small
#   anomalies entirely at 16x16. Default "max" matches the diffusion-side convention.
MASK_DOWNSAMPLE_MODE = "max"

# ── Gaussian blur kernel size for image-score aggregation ──────────────
# Controls the kernel size of the σ=4 Gaussian blur used inside _xstage_sum_max and
# _xstage_paper at scoring time. The anomalib reference uses kernel_size=5 which
# truncates σ=4 so heavily that the kernel weights are nearly uniform (max/min ≈ 1.28)
# — effectively a 5x5 box filter with no real Gaussian smoothing. Setting this to 25
# (≥ 6σ+1) makes the blur behave like an actual σ=4 Gaussian.
BLUR_KERNEL_SIZE = 5


def _downsample_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Downsample a binary mask (NCHW or NHW) to (h, w) using the global mode."""
    needs_squeeze = (mask.dim() == 4 and mask.shape[1] == 1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # → NCHW with C=1
    if MASK_DOWNSAMPLE_MODE == "max":
        out = F.adaptive_max_pool2d(mask.float(), (h, w))
    else:  # "nearest"
        out = F.interpolate(mask.float(), size=(h, w), mode="nearest")
    return out.squeeze(1) if needs_squeeze else out


def _tproto_fg_flat(fg, h: int, w: int):
    """Downsample a foreground mask to (h,w) and flatten → bool [N, h*w], or None.
    For the on-object tproto reference (restrict normal pixels to foreground).
    Uses AVG-pool + >0.5 so a feature cell counts as foreground only if mostly object."""
    if fg is None:
        return None
    f = fg.float()
    if f.dim() == 3:
        f = f.unsqueeze(1)              # → N1HW
    fd = F.adaptive_avg_pool2d(f, (h, w))
    return (fd.view(fd.size(0), -1) > 0.5)


# ── Dataset ─────────────────────────────────────────────────────────────
def maxpool_resize_mask(mask_pil: Image.Image, size: int) -> torch.Tensor:
    """Resize mask using maxpool to preserve small anomalies. Returns [1, size, size]."""
    mask_np = np.array(mask_pil).astype(np.float32)
    # Binarize first: any nonzero pixel is anomaly (handles VisA masks with values 1/2)
    mask_np = (mask_np > 0).astype(np.float32)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_t = F.adaptive_max_pool2d(mask_t, (size, size))
    return (mask_t.squeeze(0) > 0.5).float()


# Foreground (BiRefNet) mask sources beyond the stem lookup. Both off by default = legacy behaviour.
# FG_ANOMALY_DIR: masks for REAL anomaly images (Data/Images/Anomaly/*_imgs/NNN.JPG -> FG_ANOMALY_DIR/NNN_binary.png);
#   real anomalies then never fall back to birefnet_masks/Normal/NNN (a different cashew with the same number).
# FG_OVERRIDES: explicit {image path str: mask path} (e.g. a synthetic -> the BiRefNet mask of its canvas normal).
FG_ANOMALY_DIR = None
FG_OVERRIDES = {}
# --synthetic-epochs: [(start, end)) slices into synthetic_triples, one per epoch (set in load_data); None = off.
SYNTH_EPOCH_SLICES = None
# --branch-balance: normal vs synthetic loss branches on the same scale + sample-share weights (ExtensionLoss).
BRANCH_BALANCE = False
# --grad-balance: like BRANCH_BALANCE but c from GRADIENT norms ||dL_N|| / ||dL_S|| (measured every GB_EVERY steps)
GRAD_BALANCE = False
GB_EVERY = 5
# per-batch loss terms of the current epoch (filled by train_one_epoch, flushed into results each epoch)
TERM_LOG: dict = {}


class EpochPoolSampler(torch.utils.data.Sampler):
    """Per epoch e: all normal indices + the dataset indices of synthetic slice e, shuffled (seeded per epoch).
    Lives in the main process, so it works with persistent workers; set_epoch(e) is called before each epoch."""

    def __init__(self, normal_idx: list, synth_offset: int, slices: list, seed: int):
        self.normal_idx, self.off, self.slices, self.seed, self.epoch = list(normal_idx), synth_offset, slices, seed, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def _indices(self) -> list:
        a, b = self.slices[self.epoch % len(self.slices)]
        return self.normal_idx + list(range(self.off + a, self.off + b))

    def __iter__(self):
        idx = self._indices(); g = torch.Generator().manual_seed(self.seed * 1000 + self.epoch)
        return iter([idx[j] for j in torch.randperm(len(idx), generator=g).tolist()])

    def __len__(self) -> int:
        return len(self._indices())


class CashewDataset(Dataset):
    """Cashew dataset with normals, synthetics (with canvas), and real anomalies.

    Pipeline (image/canvas):  load → JPEG-recompress → Resize(LANCZOS) → ToTensor →
                              CashewAugmentation (single point) → Normalize
    Pipeline (mask):          load → maxpool resize → joint geometric (via aug)
    """

    def __init__(self, normal_paths, synthetic_triples=None, real_anomaly_pairs=None,
                 aug: "CashewAugmentation" = None,
                 n_dynamic_synth: int = 0, dynamic_canvas_pool=None, dtd_pool=None,
                 dynamic_size_range=(6, 20),
                 dynamic_alpha_choices=(0.025, 0.05, 0.1, 0.2, 0.4, 0.8),
                 dynamic_alpha_weights=(0.30, 0.25, 0.20, 0.15, 0.07, 0.03),
                 alpha_jitter_log: float = 0.0,
                 n_grouped_slots: int = 0, grouped_type: str = None,
                 grouped_multi_k: int = 2,
                 grouped_canvas_paths: list = None,
                 paired_photometric: bool = False,
                 synth_real_pairs: list = None):
        self.samples = []
        self.aug = aug
        # Real-counterpart lookup (canvas_path_str -> real anomaly image Path) for teacher-BCE-real-only.
        # synth_real_pairs is parallel to synthetic_triples: real image path per synth sample.
        self._real_lookup = {}
        for p in normal_paths:
            self.samples.append((p, None, 0, None))
        if synthetic_triples:
            for i, (syn_img, mask, canvas) in enumerate(synthetic_triples):
                self.samples.append((syn_img, mask, 1, canvas))
                if synth_real_pairs is not None and i < len(synth_real_pairs):
                    self._real_lookup[str(canvas)] = synth_real_pairs[i]
        if real_anomaly_pairs:
            for img, mask in real_anomaly_pairs:
                self.samples.append((img, mask, 1, None))

        # Dynamic-texture-blend synth slots: re-generated on every __getitem__ call.
        # Each slot has a "DYNAMIC" marker and triggers on-the-fly generation.
        self.paired_photometric = bool(paired_photometric)
        self.n_dynamic_synth = int(n_dynamic_synth)
        if self.n_dynamic_synth > 0:
            assert dynamic_canvas_pool is not None and len(dynamic_canvas_pool) > 0, \
                "n_dynamic_synth > 0 requires dynamic_canvas_pool (list of Path)"
            assert dtd_pool is not None and len(dtd_pool) > 0, \
                "n_dynamic_synth > 0 requires dtd_pool (list of Path)"
            self._dynamic_canvas_pool = list(dynamic_canvas_pool)
            self._dtd_pool = list(dtd_pool)
            self._dynamic_size_range = dynamic_size_range
            self._dynamic_alpha_choices = list(dynamic_alpha_choices)
            assert len(dynamic_alpha_weights) == len(dynamic_alpha_choices), \
                "dynamic_alpha_weights must have same length as dynamic_alpha_choices"
            self._dynamic_alpha_weights = list(dynamic_alpha_weights)
            self._alpha_jitter_log = float(alpha_jitter_log)
            for _ in range(self.n_dynamic_synth):
                self.samples.append(("__DYNAMIC__", None, 1, "__DYNAMIC__"))

        # K=2 grouped multi-corruption: each grouped slot returns a LIST of K=multi_k samples
        # sharing the same canvas. Custom collate flattens. Used for "normalizing operator" experiment.
        self.n_grouped_slots = int(n_grouped_slots)
        self.grouped_type = grouped_type
        self.grouped_multi_k = int(grouped_multi_k)
        if self.n_grouped_slots > 0:
            assert grouped_type in ("mix", "texture", "diffusion"), \
                f"n_grouped_slots > 0 requires grouped_type in mix|texture|diffusion (got {grouped_type})"
            assert self.grouped_multi_k >= 2, "grouped_multi_k must be >= 2"
            # For grouped slots we always need DTD pool (texture views) and the canvas paths.
            if dtd_pool is not None:
                self._dtd_pool = list(dtd_pool)
            self._dynamic_size_range = dynamic_size_range
            self._dynamic_alpha_choices = list(dynamic_alpha_choices)
            self._dynamic_alpha_weights = list(dynamic_alpha_weights)
            self._alpha_jitter_log = float(alpha_jitter_log)
            # Canvas pool: for mix/diffusion, must be the diffusion canvases (have paired synths).
            if grouped_type == "mix":
                # Build lookup from canvas → (diff_synth, diff_mask)
                assert synthetic_triples is not None and len(synthetic_triples) > 0, \
                    "grouped_type='mix' requires synthetic_triples (canvas→diff_synth pairs)"
                self._diff_canvas_lookup = {str(c): (s, m) for s, m, c in synthetic_triples}
                self._grouped_canvas_paths = list(self._diff_canvas_lookup.keys())
            elif grouped_type == "texture":
                # texture+texture: use grouped_canvas_paths if provided, else fall back to diff canvases.
                if grouped_canvas_paths is not None and len(grouped_canvas_paths) > 0:
                    self._grouped_canvas_paths = [str(p) for p in grouped_canvas_paths]
                elif synthetic_triples is not None and len(synthetic_triples) > 0:
                    self._grouped_canvas_paths = [str(c) for _, _, c in synthetic_triples]
                else:
                    raise ValueError("grouped_type='texture' requires grouped_canvas_paths or synthetic_triples")
            for slot_i in range(self.n_grouped_slots):
                # Bind each slot to a specific canvas (cyclic if more slots than canvases).
                canvas_idx = slot_i % len(self._grouped_canvas_paths)
                self.samples.append(("__GROUPED__", None, 1, ("__GROUPED__", canvas_idx)))

        # Build FG mask lookup: stem → BiRefNet binary mask path
        self._fg_mask_paths = {}
        biref_root = Path(__file__).resolve().parent.parent.parent / "birefnet_masks"
        for p, _, label, _ in self.samples:
            if p == "__DYNAMIC__": continue   # dynamic slots — FG looked up by canvas path
            stem = Path(p).stem
            if str(p) in FG_OVERRIDES:
                self._fg_mask_paths[str(p)] = FG_OVERRIDES[str(p)]
                continue
            if FG_ANOMALY_DIR is not None and Path(p).parent.parent.name == "Anomaly":
                bp = Path(FG_ANOMALY_DIR) / f"{stem}_binary.png"
                if bp.exists():
                    self._fg_mask_paths[str(p)] = bp
                continue   # never fall back to a normal's mask with the same number
            # Try Anomaly/ then Normal/ subdirs
            for subdir in ("Anomaly", "Normal"):
                bp = biref_root / subdir / f"{stem}_binary.png"
                if bp.exists():
                    self._fg_mask_paths[str(p)] = bp
                    break
        # Add dynamic canvas pool paths so per-call lookup works.
        if self.n_dynamic_synth > 0:
            for cp in self._dynamic_canvas_pool:
                if str(cp) in self._fg_mask_paths: continue
                stem = Path(cp).stem
                for subdir in ("Normal", "Anomaly"):
                    bp = biref_root / subdir / f"{stem}_binary.png"
                    if bp.exists():
                        self._fg_mask_paths[str(cp)] = bp
                        break
        # Also add grouped canvas paths.
        if self.n_grouped_slots > 0:
            for cp_str in self._grouped_canvas_paths:
                if cp_str in self._fg_mask_paths: continue
                stem = Path(cp_str).stem
                for subdir in ("Normal", "Anomaly"):
                    bp = biref_root / subdir / f"{stem}_binary.png"
                    if bp.exists():
                        self._fg_mask_paths[cp_str] = bp
                        break

    def __len__(self):
        return len(self.samples)

    def _generate_dynamic_synth(self):
        """Per-call: fresh texture-blend synth using random shape/size/alpha/texture/canvas.
        Returns (synth_pil, mask_pil, canvas_pil)."""
        import sys as _sys
        if "_dyn_synth_helpers_loaded" not in CashewDataset.__dict__:
            # Lazy import shape helpers from build_ood_texture_dataset
            _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from build_ood_texture_dataset import make_shape_inside_fg, feather_mask, blend_texture
            CashewDataset._make_shape_inside_fg = staticmethod(make_shape_inside_fg)
            CashewDataset._feather_mask = staticmethod(feather_mask)
            CashewDataset._blend_texture = staticmethod(blend_texture)
            CashewDataset._dyn_synth_helpers_loaded = True

        # Use module-level random (seeded by _worker_init_fn from torch.initial_seed()).
        # With shuffle=True + seeded torch generator, call order is deterministic →
        # synth sequence is reproducible per (train_seed, worker_id, call_order).
        rng = random
        # 1. Sample canvas + load
        canvas_path = rng.choice(self._dynamic_canvas_pool)
        canvas_pil = Image.open(canvas_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        # 2. FG mask
        fg_path_str = self._fg_mask_paths.get(str(canvas_path))
        if fg_path_str is not None:
            fg_mask_np = (np.array(Image.open(fg_path_str).convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)) > 127).astype(np.uint8)
        else:
            fg_mask_np = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        # 3. Shape
        shape_type = rng.randint(0, 3)
        size_range = self._dynamic_size_range
        shape_mask = CashewDataset._make_shape_inside_fg(
            fg_mask_np, shape_type=shape_type, size_range=size_range, rng=rng)
        if shape_mask is None:
            # No valid placement — return canvas as-is (acts like a normal sample)
            return canvas_pil, Image.fromarray(np.zeros_like(fg_mask_np)), canvas_pil
        # 4. Texture
        tex_path = rng.choice(self._dtd_pool)
        try:
            texture_pil = Image.open(tex_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        except Exception:
            return canvas_pil, Image.fromarray(np.zeros_like(fg_mask_np)), canvas_pil
        # 5. Soft mask + alpha + blend
        soft = CashewDataset._feather_mask(shape_mask, sigma=2.5)
        # Subtle-skewed weighted sample: low alphas get more probability
        alpha = rng.choices(self._dynamic_alpha_choices, weights=self._dynamic_alpha_weights, k=1)[0]
        # Optional multiplicative jitter: alpha *= exp(U(-j, +j)), clipped to bucket range.
        if getattr(self, "_alpha_jitter_log", 0.0) > 0:
            import math as _math
            j = self._alpha_jitter_log
            alpha = alpha * _math.exp(rng.uniform(-j, j))
            lo, hi = min(self._dynamic_alpha_choices), max(self._dynamic_alpha_choices)
            alpha = max(lo, min(hi, alpha))
        canvas_np = np.array(canvas_pil, dtype=np.uint8)
        texture_np = np.array(texture_pil, dtype=np.uint8)
        synth_np = CashewDataset._blend_texture(canvas_np, texture_np, soft, alpha)
        synth_pil = Image.fromarray(synth_np)
        mask_pil = Image.fromarray((shape_mask * 255).astype(np.uint8))
        return synth_pil, mask_pil, canvas_pil

    def _generate_dynamic_synth_on(self, canvas_path_str):
        """Generate one dynamic texture-blend synth using the given canvas. Used for grouped views.
        Replicates the body of _generate_dynamic_synth but with a fixed canvas (no swap)."""
        import sys as _sys
        if "_dyn_synth_helpers_loaded" not in CashewDataset.__dict__:
            _sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from build_ood_texture_dataset import make_shape_inside_fg, feather_mask, blend_texture
            CashewDataset._make_shape_inside_fg = staticmethod(make_shape_inside_fg)
            CashewDataset._feather_mask = staticmethod(feather_mask)
            CashewDataset._blend_texture = staticmethod(blend_texture)
            CashewDataset._dyn_synth_helpers_loaded = True

        rng = random
        canvas_pil = Image.open(canvas_path_str).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        # FG mask
        fg_path_str = self._fg_mask_paths.get(canvas_path_str)
        if fg_path_str is not None:
            fg_mask_np = (np.array(Image.open(fg_path_str).convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)) > 127).astype(np.uint8)
        else:
            fg_mask_np = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        # Shape
        shape_type = rng.randint(0, 3)
        shape_mask = CashewDataset._make_shape_inside_fg(
            fg_mask_np, shape_type=shape_type, size_range=self._dynamic_size_range, rng=rng)
        if shape_mask is None:
            return canvas_pil, Image.fromarray(np.zeros_like(fg_mask_np)), canvas_pil
        # Texture
        tex_path = rng.choice(self._dtd_pool)
        try:
            texture_pil = Image.open(tex_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        except Exception:
            return canvas_pil, Image.fromarray(np.zeros_like(fg_mask_np)), canvas_pil
        # Soft mask + alpha + blend
        soft = CashewDataset._feather_mask(shape_mask, sigma=2.5)
        alpha = rng.choices(self._dynamic_alpha_choices, weights=self._dynamic_alpha_weights, k=1)[0]
        if getattr(self, "_alpha_jitter_log", 0.0) > 0:
            import math as _math
            j = self._alpha_jitter_log
            alpha = alpha * _math.exp(rng.uniform(-j, j))
            lo, hi = min(self._dynamic_alpha_choices), max(self._dynamic_alpha_choices)
            alpha = max(lo, min(hi, alpha))
        synth_np = CashewDataset._blend_texture(
            np.array(canvas_pil, dtype=np.uint8),
            np.array(texture_pil, dtype=np.uint8), soft, alpha)
        return Image.fromarray(synth_np), Image.fromarray((shape_mask * 255).astype(np.uint8)), canvas_pil

    def _build_sample_dict(self, image_pil, mask_pil, canvas_pil, img_path, canvas_path, label, pair_id=-1):
        """Common processing: JPEG-recompress → tensor → aug → normalize → FG mask → dict."""
        import io
        # Mask-only JPEG compression at q=75 (see original comment block).
        if mask_pil is not None and not str(img_path).lower().endswith((".jpg", ".jpeg")):
            image_arr = np.array(image_pil)
            buf = io.BytesIO()
            image_pil.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            jpeg_arr = np.array(Image.open(buf).convert("RGB"))
            if mask_pil.size != image_pil.size:
                m = mask_pil.resize(image_pil.size, Image.NEAREST)
            else:
                m = mask_pil
            mask_bool = np.array(m) > 127
            image_pil = Image.fromarray(
                np.where(mask_bool[..., None], jpeg_arr, image_arr).astype(np.uint8)
            )

        image = TO_TENSOR(RESIZE(image_pil))
        if mask_pil is not None:
            mask = maxpool_resize_mask(mask_pil, IMAGE_SIZE)
        else:
            mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        canvas = TO_TENSOR(RESIZE(canvas_pil)) if canvas_pil is not None else None

        if self.aug is not None:
            image, mask, canvas = self.aug(image, mask, canvas,
                                            paired_photometric=self.paired_photometric)

        image = NORMALIZE(image)
        has_canvas = canvas is not None
        canvas = NORMALIZE(canvas) if has_canvas else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

        # FG mask: for dynamic/grouped, lookup by canvas path; else by img path.
        is_synthetic_img_path = str(img_path) == "__DYNAMIC__"
        _fg_lookup_key = str(canvas_path) if (is_synthetic_img_path and canvas_path is not None) else str(img_path)
        fg_path = self._fg_mask_paths.get(_fg_lookup_key)
        if fg_path is not None:
            fg_mask = maxpool_resize_mask(Image.open(fg_path).convert("L"), IMAGE_SIZE)
        else:
            fg_mask = torch.ones(1, IMAGE_SIZE, IMAGE_SIZE)

        # Real anomaly counterpart (teacher-BCE-real-only): for synth samples with a known
        # canvas, load the real anomaly image; else a zero placeholder (keeps collate uniform).
        real_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        if self._real_lookup and canvas_path is not None:
            rp = self._real_lookup.get(str(canvas_path))
            if rp is not None:
                real_pil = Image.open(rp).convert("RGB")
                real_image = NORMALIZE(TO_TENSOR(RESIZE(real_pil)))

        return {
            "image": image,
            "gt_mask": mask,
            "gt_label": torch.tensor(label, dtype=torch.long),
            "canvas_normal": canvas,
            "has_canvas": torch.tensor(has_canvas, dtype=torch.bool),
            "fg_mask": fg_mask,
            "pair_id": torch.tensor(pair_id, dtype=torch.long),
            "real_image": real_image,
        }

    def __getitem__(self, idx):
        img_path, mask_path, label, canvas_path = self.samples[idx]
        # --fixed-test-shifts: identical photometric draw for a given (test image, shift level) in every epoch,
        # seed, fold and method - seeded by the image FILENAME (not idx), so it is fold-independent too.
        if getattr(self, "fixed_shift_seed", None) is not None:
            import zlib
            _sd = (zlib.crc32(Path(str(img_path)).name.encode()) + 1000003 * self.fixed_shift_seed) % (2 ** 31 - 1)
            torch.manual_seed(_sd); random.seed(_sd); np.random.seed(_sd % (2 ** 32 - 1))

        # GROUPED slot — return LIST of K view dicts (same canvas, K different corruptions).
        # Custom collate (cashew_collate_grouped) flattens lists into batch dim.
        if isinstance(canvas_path, tuple) and canvas_path[0] == "__GROUPED__":
            _, canvas_idx_int = canvas_path
            canvas_p_str = self._grouped_canvas_paths[canvas_idx_int]
            views = []
            # Unique pair_id per __getitem__ call so paired-loss can find them in the batch.
            pair_id = (hash((id(self), canvas_idx_int, random.random())) & 0x7FFFFFFF)
            for view_idx in range(self.grouped_multi_k):
                if self.grouped_type == "mix" and view_idx == 0:
                    # View 0 of mix variant: load the fixed diffusion synth
                    diff_synth_p, diff_mask_p = self._diff_canvas_lookup[canvas_p_str]
                    img_pil = Image.open(diff_synth_p).convert("RGB")
                    msk_pil = Image.open(diff_mask_p).convert("L")
                    cvs_pil = Image.open(canvas_p_str).convert("RGB")
                    view_dict = self._build_sample_dict(
                        img_pil, msk_pil, cvs_pil,
                        img_path=diff_synth_p, canvas_path=canvas_p_str,
                        label=1, pair_id=pair_id)
                else:
                    # Dynamic texture view on the fixed canvas
                    img_pil, msk_pil, cvs_pil = self._generate_dynamic_synth_on(canvas_p_str)
                    view_dict = self._build_sample_dict(
                        img_pil, msk_pil, cvs_pil,
                        img_path="__DYNAMIC__", canvas_path=canvas_p_str,
                        label=1, pair_id=pair_id)
                views.append(view_dict)
            return views

        # Dynamic synth slot — generate fresh on every call.
        if canvas_path == "__DYNAMIC__":
            image_pil, mask_pil, canvas_pil = self._generate_dynamic_synth()
            return self._build_sample_dict(
                image_pil, mask_pil, canvas_pil,
                img_path="__DYNAMIC__", canvas_path=None, label=label)

        # Standard sample (normal or pre-baked synth/real)
        image_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L") if mask_path is not None else None
        canvas_pil = Image.open(canvas_path).convert("RGB") if canvas_path is not None else None
        return self._build_sample_dict(
            image_pil, mask_pil, canvas_pil,
            img_path=img_path, canvas_path=canvas_path, label=label)


# ── Controllable UniNetLoss (Modes 1, 2, 3) ────────────────────────────
class _ZeroLoss(nn.Module):
    """Stand-in for anomalib's bce_loss that contributes nothing (keeps the graph connected)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.sum() * 0.0


class UniNetLossControllable(nn.Module):
    """Local copy of anomalib's UniNetLoss with a flag to disable the margin term.

    Anomalib's `UniNetLoss` always includes the margin terms; this version skips
    them when `margin_enabled=False` so we can ablate margin on/off uniformly
    across modes.

    margin_enabled=True semantics (paper):
      - Normal positions: margin_loss_n = relu(margin - diag_sim).mean() (pull toward 1)
      - Anomaly positions: margin_loss_a = relu(diag_sim - margin/2).mean() (push below 0.5)
    """

    def __init__(self, lambda_weight: float = 0.7, temperature: float = 2.0,
                 margin_enabled: bool = True, lkd_mode: str = "local",
                 two_sided: bool = False, tau: float | None = None,
                 loss_form: str = "cos", stop_gradient: bool = False,
                 tproto: str = "off", w_tproto: float = 1.0) -> None:
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.margin_enabled = margin_enabled
        self.loss_form = loss_form
        self.stop_gradient = bool(stop_gradient)
        # M2/M3 teacher-prototype contrastive (canvas-free analog of M4 ct-twosided):
        # within each anomaly image, on the ADAPTIVE teacher (target stages), push anomaly-pixel
        # features off the image's own normal-pixel prototype. "push" = push_only; "full" adds the
        # (near-trivial, self-referential) normal-compactness pull. No SG (grad to adaptive teacher).
        assert tproto in ("off", "push", "full", "intra"), tproto
        self.tproto = tproto
        self.w_tproto = w_tproto
        self.scale_push_contrastive = False  # set externally when --scale-push-contrastive is passed
        assert lkd_mode in ("local", "global", "both"), \
            f"lkd_mode must be 'local', 'global', or 'both', got {lkd_mode}"
        self.lkd_mode = lkd_mode
        # two_sided: when True AND a mask is provided, add explicit dis-align at anomaly
        # positions for BOTH L_KD (per-position cos→0 push) and L_SC (per-position
        # -log(1-diag) push). Mirrors ExtensionLoss._lkd_lsc_twosided. Default False
        # preserves current Mode 1/2/3 behavior exactly. Mode 1 has no mask so this
        # flag is inert for normal-only training.
        self.two_sided = two_sided
        # tau: symmetric margin threshold around 0.5. When None, use legacy asymmetric
        # margin (relu(1 - diag) pull, relu(diag - 0.5) push). When set, replaces those
        # with symmetric pair:
        #   P_align    = 0.5 + 0.2 * tau   (pull diag up to this, then stop)
        #   P_disalign = 0.5 - 0.2 * tau   (push diag down to this, then stop)
        # tau=1 → (0.7, 0.3); tau=2 → (0.9, 0.1). Always requires margin_enabled=True.
        # Independent of two_sided. Clamped to [0, 2.5] so thresholds stay in [0.5, 1] / [0, 0.5].
        self.tau = tau

    def forward(self, student_features, teacher_features,
                margin: int = 1, mask=None, stop_gradient: bool | None = None):
        # Bugfix 2026-05-28: anomalib's UniNetModel._compute_loss passes stop_gradient=False
        # explicitly, which previously overrode self.stop_gradient set by --stop-gradient flag.
        # Instance setting always wins now.
        stop_gradient = bool(stop_gradient) or self.stop_gradient
        loss = 0.0
        for idx in range(len(student_features)):
            sf = student_features[idx]
            tf = teacher_features[idx]
            n, c, h, w = sf.shape
            sf = sf.view(n, c, -1).transpose(1, 2)
            tf = tf.view(n, c, -1).transpose(1, 2)
            sf_n = F.normalize(sf, p=2, dim=2)
            tf_n = F.normalize(tf, p=2, dim=2)
            tf_n_kd = tf_n.detach() if stop_gradient else tf_n

            # Compute spatial masks once (used by both L_KD two-sided and L_SC).
            normal_mask = None
            abnormal_mask = None
            if mask is not None:
                if mask.dim() < 3:
                    normal_mask = mask == 0
                    abnormal_mask = mask == 1
                else:
                    mask_ = _downsample_mask(mask, h, w)
                    mask_flat = mask_.view(mask_.size(0), -1)
                    normal_mask = mask_flat == 0
                    abnormal_mask = mask_flat == 1

            # L_KD: local (per-position cosine or weighted MSE), global, or both.
            # tf_n_kd for KD (may be detached), tf_n for L_SC (always has grad).
            cos_per_pos = F.cosine_similarity(sf_n, tf_n_kd, dim=2)  # L_KD cosine
            if self.loss_form == "weighted_mse":
                a = tf_n_kd.detach().abs().mean(dim=1, keepdim=True)  # [n, 1, c]
                a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
                wmse_per_pos = (a * (sf_n - tf_n_kd).pow(2)).sum(dim=-1) # [n, h*w]
            # Compute w_pos for push weighting (when scale_push_contrastive OR weight_push_only)
            _spc = self.loss_form == "weighted_mse" and self.scale_push_contrastive
            _wpo = getattr(self, 'weight_push_only', False)
            w_pos = None
            if _spc or _wpo:
                a_push = tf_n_kd.detach().abs().mean(dim=1, keepdim=True)
                a_push = a_push / (a_push.sum(dim=-1, keepdim=True) + 1e-8)
                w_pos = (a_push * tf_n_kd.detach().pow(2)).sum(dim=-1)
                w_pos = w_pos / (w_pos.mean(dim=-1, keepdim=True) + 1e-8)
            if self.two_sided and abnormal_mask is not None and abnormal_mask.sum() > 0:
                if self.loss_form == "weighted_mse":
                    align_term = wmse_per_pos[normal_mask].mean() if normal_mask.sum() > 0 \
                        else torch.tensor(0.0, device=sf.device)
                else:
                    align_term = (1 - cos_per_pos[normal_mask]).mean() if normal_mask.sum() > 0 \
                        else torch.tensor(0.0, device=sf.device)
                # Anomaly cosine push: relu(cos) stops at orthogonal; push_to_max uses (1+cos)
                # to drive to full anti-alignment (cos=-1). Matches ExtensionLoss behavior.
                def _push(c):
                    return (1.0 + c) if getattr(self, 'push_to_max', False) else F.relu(c)
                if w_pos is not None:
                    push_term = (w_pos[abnormal_mask] * _push(cos_per_pos[abnormal_mask])).mean()
                else:
                    push_term = _push(cos_per_pos[abnormal_mask]).mean()
                local_kd = align_term + push_term
            else:
                if self.loss_form == "weighted_mse":
                    local_kd = wmse_per_pos.mean()
                else:
                    local_kd = (1 - cos_per_pos).mean()
            if self.lkd_mode in ("global", "both"):
                sf_g = F.normalize(sf.reshape(n, -1), p=2, dim=1)
                tf_kd_raw = tf.detach() if stop_gradient else tf
                tf_g = F.normalize(tf_kd_raw.reshape(n, -1), p=2, dim=1)
                if self.loss_form == "weighted_mse":
                    a_g = tf_g.detach().abs()
                    a_g = a_g / (a_g.sum(dim=1, keepdim=True) + 1e-8)
                    global_kd = (a_g * (sf_g - tf_g).pow(2)).sum(dim=1).mean()
                else:
                    global_kd = (1 - (sf_g * tf_g).sum(dim=1)).mean()
            else:
                global_kd = None
            if self.lkd_mode == "local":
                cosine_loss = local_kd
            elif self.lkd_mode == "global":
                cosine_loss = global_kd
            else:  # both
                cosine_loss = 0.5 * local_kd + 0.5 * global_kd

            sim = torch.matmul(sf_n, tf_n.transpose(1, 2)) / self.temperature
            sim = torch.exp(sim)
            sim = sim / (sim.sum(dim=2, keepdim=True) + 1e-8)
            diag_sum = torch.diagonal(sim, dim1=1, dim2=2)

            # Resolve margin thresholds: tau (symmetric) takes precedence over legacy.
            if self.tau is not None:
                tau_clamped = max(0.0, min(2.5, self.tau))
                p_align = 0.5 + 0.2 * tau_clamped       # tau=1 → 0.7
                p_disalign = 0.5 - 0.2 * tau_clamped    # tau=1 → 0.3
            else:
                p_align = float(margin)                 # legacy: 1.0
                p_disalign = float(margin) / 2.0        # legacy: 0.5

            margin_loss_n = torch.tensor(0.0, device=sf.device)
            margin_loss_a = torch.tensor(0.0, device=sf.device)
            if mask is None:
                if _spc and w_pos is not None:
                    contrastive_loss = (w_pos * -torch.log(diag_sum + 1e-8)).mean()
                else:
                    contrastive_loss = -torch.log(diag_sum + 1e-8).mean()
                if self.margin_enabled:
                    margin_loss_n = F.relu(p_align - diag_sum).mean()
            else:
                contrastive_loss = torch.tensor(0.0, device=sf.device)
                if normal_mask.sum() > 0:
                    diag_n = diag_sum[normal_mask]
                    if _spc and w_pos is not None:
                        contrastive_loss = (w_pos[normal_mask] * -torch.log(diag_n + 1e-8)).mean()
                    else:
                        contrastive_loss = -torch.log(diag_n + 1e-8).mean()
                    if self.margin_enabled:
                        margin_loss_n = F.relu(p_align - diag_n).mean()
                if abnormal_mask.sum() > 0:
                    diag_a = diag_sum[abnormal_mask].clamp(max=1.0 - 1e-6)
                    if self.two_sided:
                        if _spc and w_pos is not None:
                            contrastive_loss = contrastive_loss + (w_pos[abnormal_mask] * -torch.log(1.0 - diag_a + 1e-8)).mean()
                        else:
                            contrastive_loss = contrastive_loss + (-torch.log(1.0 - diag_a + 1e-8)).mean()
                    if self.margin_enabled:
                        margin_loss_a = F.relu(diag_a - p_disalign).mean()

            margin_loss = margin_loss_n + margin_loss_a
            loss = loss + cosine_loss * self.lambda_weight + contrastive_loss * (1 - self.lambda_weight) + margin_loss

            # M2/M3 teacher-prototype contrastive — target stages only (adaptive teacher).
            # Per image: normal prototype m_n = mean(adaptive-teacher normal-pixel feats); push
            # anomaly-pixel feats off m_n (relu(cos)→orthogonal, gentle; never anti-aligns). No SG
            # → grad to the adaptive teacher (push moves anomaly off m_n AND m_n off anomaly).
            if (self.tproto != "off" and abnormal_mask is not None
                    and idx >= len(student_features) // 2):
                tproto_acc = torch.tensor(0.0, device=sf.device); cnt = 0
                # On-object reference: restrict normal pixels to foreground (mask==0 & fg>0.5).
                fg_flat = _tproto_fg_flat(getattr(self, "_batch_fg", None), h, w)
                for bi in range(n):
                    nm = normal_mask[bi]; am = abnormal_mask[bi]
                    if fg_flat is not None:
                        nm = nm & fg_flat[bi]
                    if self.tproto == "intra":
                        # spread anomaly pixels APART from each other: push each off the
                        # anomaly-region centroid (NO normal reference). Diversifies the defect
                        # cluster; non-trivial (anomaly pixels start similar). Orthogonal to the
                        # anomaly-vs-normal axis the detector scores on.
                        if am.sum() < 2:
                            continue
                        m_a = F.normalize(tf_n[bi][am].mean(dim=0), p=2, dim=0)    # anomaly centroid
                        tproto_acc = tproto_acc + F.relu((tf_n[bi][am] * m_a.unsqueeze(0)).sum(dim=-1)).mean()
                        cnt += 1
                    else:
                        if am.sum() == 0 or nm.sum() == 0:
                            continue
                        m_n = F.normalize(tf_n[bi][nm].mean(dim=0), p=2, dim=0)        # [c]
                        push = F.relu((tf_n[bi][am] * m_n.unsqueeze(0)).sum(dim=-1)).mean()
                        if self.tproto == "full":
                            pull = (1.0 - (tf_n[bi][nm] * m_n.unsqueeze(0)).sum(dim=-1)).mean()
                            tproto_acc = tproto_acc + push + pull
                        else:
                            tproto_acc = tproto_acc + push
                        cnt += 1
                if cnt > 0:
                    loss = loss + self.w_tproto * (tproto_acc / cnt)
        return loss


# ── Extension Loss (Mode 3) ────────────────────────────────────────────
class ExtensionLoss(nn.Module):
    """Extension loss with 3 terms per feature layer, NO margin, NO BCE.

    For each feature layer, given student_features, teacher_features,
    canvas_teacher_features (for synthetics), syn_idx, and spatial mask:

      1. Normal samples: L_KD + L_SC align (Student(x_norm), Teacher(x_norm))
      2. Synthetic → canvas: L_KD + L_SC align (Student(x_syn), Teacher(x_canvas)) everywhere
      3. Synthetic → same: L_KD + L_SC two-sided (Student(x_syn), Teacher(x_syn))
         - normal positions: align
         - anomaly positions: dis-align
    """

    def __init__(self, lambda_weight: float = 0.7, temperature: float = 2.0,
                 w_canvas: float = 1.0, w_twosided: float = 1.0,
                 lambda_push: float = 1.0,
                 lambda_norm_cos: float = 1.0,
                 loss_form: str = "cos",
                 lkd_mode: str = "local",
                 margin_mode: str = "off", margin: float = 1.0,
                 lsc_pre_relu: bool = False,
                 dinomaly_p: float = 0.0,
                 dinomaly_factor: float = 0.1,
                 dinomaly_ramp_iters: int = 100,
                 tau: float | None = None,
                 canvas_align_mode: str = "uniform",
                 adaptive_triplet: bool = False, triplet_alpha: float = 0.5,
                 proj_consistency: bool = False, lambda_proj: float = 0.1,
                 residual_direction_alignment: bool = False, lambda_resdir: float = 0.05,
                 pairrank: bool = False, lambda_pairrank: float = 0.05, pairrank_margin: float = 0.1,
                 reltail: bool = False, lambda_reltail: float = 0.05, reltail_rho: float = 0.5,
                 reltail_frac: float = 0.01,
                 maskrank: bool = False, lambda_maskrank: float = 0.05, maskrank_margin: float = 0.1,
                 preserve_outside: bool = False, w_preserve: float = 1.0,
                 gap_gated_outside_canvas: bool = False, lambda_gated_outcanvas: float = 0.25,
                 ct_twosided: str = "off", w_ct_twosided: float = 1.0,
                 stop_gradient: bool = False,
                 # Module 1 — teacher repulsion
                 teacher_repulsion: bool = False,
                 teacher_repulsion_weight: float = 1.0,
                 teacher_repulsion_margin: float = 0.5,
                 teacher_repulsion_clean_sg: bool = False,
                 teacher_repulsion_stages: str = "trainable_only",
                 # Module 2 — bottleneck pair alignment
                 bottleneck_pair: bool = False,
                 bottleneck_pair_weight: float = 1.0,
                 bottleneck_pair_metric: str = "cosine",
                 bottleneck_pair_clean_sg: bool = False,
                 # Module 3 — bottleneck prototype organization
                 bottleneck_protos: bool = False,
                 bottleneck_protos_k: int = 8,
                 bottleneck_protos_weight: float = 1.0,
                 bottleneck_protos_assignment: str = "nearest",
                 bottleneck_protos_temperature: float = 0.1,
                 bottleneck_protos_ema: float = 0.99,
                 bottleneck_protos_sinkhorn_iters: int = 3,
                 bottleneck_protos_update_mode: str = "gradient",
                 # Bottleneck channel dimension (1024 for convnext_b, 2048 for wrn50)
                 bottleneck_dim: int = 1024,
                 # Geometry diagnostics
                 geom_diagnostics: bool = True,
                 # Geom2: SPM
                 spm: bool = False,
                 spm_k: int = 500,
                 spm_weight: float = 1.0,
                 spm_tau: float = 0.05,
                 spm_eps: float = 0.05,
                 spm_sinkhorn_iters: int = 3,
                 spm_ema: float = 0.99,
                 spm_init_file: str | None = None,
                 # Geom2: paired prototype loss
                 paired_proto: bool = False,
                 paired_proto_weight: float = 0.1,
                 paired_proto_sg_target: bool = True,
                 # Geom2: soft bottleneck pair
                 soft_pair: bool = False,
                 soft_pair_weight: float = 0.01,
                 # Geom2: new teacher repulsion (trep2)
                 trep: bool = False,
                 trep_weight: float = 0.05,
                 trep_adaptive_margin: bool = False,
                 trep_margin: float = 0.0) -> None:
        """
        lkd_mode:
          'local'  — Anomalib-style: per-spatial-position cosine, averaged.
                     L_KD becomes a per-position alignment term.
          'global' — Paper-faithful (Eq. 1): flatten the entire C×H×W feature
                     map per sample, single cosine per (sample, stage).
                     Only affects the align-everywhere helper; the two-sided
                     spatial discrimination stays per-position because it is
                     inherently spatial (no global analog).
        """
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.w_canvas = w_canvas
        self.w_twosided = w_twosided
        self.lambda_push = lambda_push
        # exp2: two-sided on the ADAPTIVE teacher's (T_synth, T_canvas) pairing (target stages only).
        assert ct_twosided in ("off", "full", "push_only"), ct_twosided
        self.ct_twosided = ct_twosided
        self.w_ct_twosided = w_ct_twosided
        # real_tproto: teacher same-object push on real-anomaly samples (Mode 5). off/push/full.
        self.real_tproto = "off"
        self.w_real_tproto = 1.0
        # tproto on SYNTHETIC samples (M4 CAAO): push synth-anomaly off synth FG-normal mean. off/push/full.
        self.tproto = "off"
        self.w_tproto = 1.0
        # Mode 5: whether reals also supervise the STUDENT (student↔teacher two-sided). Default True.
        # False = teacher-only variant (reals shape only the adaptive teacher via real_tproto).
        self.real_student_sup = True
        self.lambda_norm_cos = lambda_norm_cos
        self.loss_form = loss_form
        assert lkd_mode in ("local", "global", "both"), \
            f"lkd_mode must be 'local', 'global', or 'both', got {lkd_mode}"
        self.lkd_mode = lkd_mode
        # Margin loss (paper-style): 'on' adds relu(margin - diag_sim) for align (pull) and
        # relu(diag_sim - margin/2) for anomaly two-sided (push). 'off' = no margin term (current default).
        assert margin_mode in ("off", "on"), f"margin_mode must be 'off' or 'on', got {margin_mode}"
        self.margin_mode = margin_mode
        self.margin = float(margin)
        # tau: symmetric margin around 0.5. When None, legacy asymmetric (pull @ margin,
        # push @ margin/2). When set: P_align = 0.5 + 0.2*tau, P_disalign = 0.5 - 0.2*tau.
        # tau=1 → (0.7, 0.3). Clamped to [0, 2.5]. Only active when margin_mode='on'.
        self.tau = tau
        # canvas_align_mode: controls Term 2 (canvas-align on synthetic samples) position weighting.
        #   'uniform'      = default, uniform mean over all positions (~5% effective weight on anomaly).
        #   'anomaly-only' = only anomaly mask positions of synthetic samples (~100% on anomaly).
        #   '50-50'        = 0.5 * mean(normal positions) + 0.5 * mean(anomaly positions). Normal
        #                    positions get equal weight to anomaly positions (anomaly is ~5% of pixels
        #                    so this is ~10× concentration; normal positions still get some Term 2 signal).
        assert canvas_align_mode in ("uniform", "anomaly-only", "50-50"), \
            f"canvas_align_mode must be uniform/anomaly-only/50-50, got {canvas_align_mode}"
        self.canvas_align_mode = canvas_align_mode
        # Group D: projection consistency — S(T(synth)) ≈ stopgrad(S(T(canvas))) at anomaly positions.
        # Needs canvas_student_features passed in. If True but no canvas_student_features provided,
        # term is skipped silently.
        self.proj_consistency = bool(proj_consistency)
        self.lambda_proj = float(lambda_proj)
        # Group F: residual-direction alignment — make the detector residual (T_s - S_s) point in
        # the same direction as the teacher's synthetic→canvas counterfactual residual (T_s - T_c).
        # Pure additional loss; reuses already-normalized sf_s, tf_s, ctf_flat from CAAO.
        self.residual_direction_alignment = bool(residual_direction_alignment)
        self.lambda_resdir = float(lambda_resdir)
        # Per-step stats for residual-direction alignment (populated when term is active).
        # Read externally for logging; reset on each forward() call.
        self.last_resdir_stats = None
        # Group G: pairrank — explicit (synth, canvas) pair ordering on summed-map scores.
        # Reuses already-computed sf_s, tf_s, ctf_flat + canvas_student_features. Needs proj_consistency.
        self.pairrank = bool(pairrank)
        self.lambda_pairrank = float(lambda_pairrank)
        self.pairrank_margin = float(pairrank_margin)
        self.last_pairrank_stats = None
        # Group G: reltail — suppress canvas top-fraction peaks relative to the anomaly-region
        # activation. Stopgrad on s_anom_ref so gradient pushes ONLY canvas tail down.
        self.reltail = bool(reltail)
        self.lambda_reltail = float(lambda_reltail)
        self.reltail_rho = float(reltail_rho)
        self.reltail_frac = float(reltail_frac)
        self.last_reltail_stats = None
        # Group G: maskrank — intra-image localization rank between in-mask and out-of-mask pixels.
        # Reuses dist_acc_synth + mask_64_syn from the shared accumulator.
        self.maskrank = bool(maskrank)
        self.lambda_maskrank = float(lambda_maskrank)
        self.maskrank_margin = float(maskrank_margin)
        self.last_maskrank_stats = None
        # Group E: adaptive counterfactual triplet — relative-geometry push using teacher gap as margin.
        # L_triplet = relu(d(S_s, T_c) - d(S_s, T_s) + alpha * stopgrad(d(T_s, T_c))) at anomaly positions.
        # Designed to replace/complement w_twosided (typically used with --w-twosided 0.0).
        self.adaptive_triplet = bool(adaptive_triplet)
        self.triplet_alpha = float(triplet_alpha)
        # Outside-mask preservation: align Student(synth) → Teacher(synth) at non-anomaly positions
        # of synthetic samples. Designed to fill the coverage gap when w_twosided=0 + CAAO together
        # leave synth-image normal positions unconstrained, causing pAUC-full drift.
        self.preserve_outside = bool(preserve_outside)
        self.w_preserve = float(w_preserve)
        # Variant 3: gap-gated outside canvas-align — softgate canvas-align outside the mask
        # by exp(-gap/tau) where gap = 1 - cos(T_s, T_c), tau = stopgrad median outside gap.
        # Only applies canvas-align outside mask where T_s and T_c agree.
        self.gap_gated_outside_canvas = bool(gap_gated_outside_canvas)
        self.lambda_gated_outcanvas = float(lambda_gated_outcanvas)
        self.last_gated_outcanvas_stats = None
        self.lsc_pre_relu = bool(lsc_pre_relu)
        # Dinomaly-style hard mining (paper-faithful: gradient-downweighting via detach trick).
        # All points contribute to the forward loss value unchanged, but the EASY p fraction
        # (lowest per-position loss) gets its gradient scaled down by `factor` (typically 0.1).
        # This makes the student focus learning effort on the HARDEST 1-p fraction of points.
        # `dinomaly_p` ramps linearly from 0 to its final value over `dinomaly_ramp_iters` steps.
        # Default off (p=0.0). Paper default: p=0.9, factor=0.1, ramp=1000 (we use 100 for our ~360-step training).
        assert 0.0 <= dinomaly_p < 1.0, f"dinomaly_p in [0, 1), got {dinomaly_p}"
        self.stop_gradient = bool(stop_gradient)
        self.no_detach_any = False  # set externally by model when --no-detach-any is passed
        self.dinomaly_p = float(dinomaly_p)
        self.dinomaly_factor = float(dinomaly_factor)
        self.dinomaly_ramp_iters = max(1, int(dinomaly_ramp_iters))
        # Step counter, incremented each forward(). Used for the linear ramp of p.
        self._dinomaly_step = 0
        # Per-batch logs of student×DFS vs teacher activation magnitudes, per stage and term.
        # Each entry: {epoch (int|None), stage, term, undershoot_frac, s_norm_mean, t_norm_mean,
        # ratio_mean, ratio_p10, ratio_p50, ratio_p90}. Aggregated by run_mode.
        self.shoot_log: list[dict] = []
        self._current_epoch: int | None = None  # set by run_mode each epoch

        # ── Module 1: teacher repulsion ────────────────────────────────────
        self.teacher_repulsion = bool(teacher_repulsion)
        self.teacher_repulsion_weight = float(teacher_repulsion_weight)
        self.teacher_repulsion_margin = float(teacher_repulsion_margin)
        self.teacher_repulsion_clean_sg = bool(teacher_repulsion_clean_sg)
        assert teacher_repulsion_stages in ("trainable_only", "all_available"), \
            f"teacher_repulsion_stages must be trainable_only/all_available, got {teacher_repulsion_stages}"
        self.teacher_repulsion_stages = teacher_repulsion_stages

        # ── Module 2: bottleneck pair alignment ────────────────────────────
        self.bottleneck_pair = bool(bottleneck_pair)
        self.bottleneck_pair_weight = float(bottleneck_pair_weight)
        assert bottleneck_pair_metric in ("cosine", "layernorm_smooth_l1", "layernorm_mse"), \
            f"bottleneck_pair_metric must be cosine/layernorm_smooth_l1/layernorm_mse, got {bottleneck_pair_metric}"
        self.bottleneck_pair_metric = bottleneck_pair_metric
        self.bottleneck_pair_clean_sg = bool(bottleneck_pair_clean_sg)

        # ── Module 3: bottleneck prototype organization ────────────────────
        self.bottleneck_protos = bool(bottleneck_protos)
        self.bottleneck_protos_k = int(bottleneck_protos_k)
        self.bottleneck_protos_weight = float(bottleneck_protos_weight)
        assert bottleneck_protos_assignment in ("nearest", "soft", "sinkhorn"), \
            f"bottleneck_protos_assignment must be nearest/soft/sinkhorn, got {bottleneck_protos_assignment}"
        self.bottleneck_protos_assignment = bottleneck_protos_assignment
        self.bottleneck_protos_temperature = float(bottleneck_protos_temperature)
        self.bottleneck_protos_ema = float(bottleneck_protos_ema)
        self.bottleneck_protos_sinkhorn_iters = int(bottleneck_protos_sinkhorn_iters)
        assert bottleneck_protos_update_mode in ("gradient", "ema"), \
            f"bottleneck_protos_update_mode must be gradient/ema, got {bottleneck_protos_update_mode}"
        self.bottleneck_protos_update_mode = bottleneck_protos_update_mode
        self.bottleneck_dim = int(bottleneck_dim)  # actual channel dim of bottleneck output
        if self.bottleneck_protos:
            _proto_init = F.normalize(
                torch.randn(self.bottleneck_protos_k, self.bottleneck_dim), p=2, dim=1)
            if self.bottleneck_protos_update_mode == "ema":
                # Buffer: EMA drives prototype positions; NOT in optimizer (no gradient update).
                self.register_buffer("prototypes", _proto_init.clone())
                self.register_buffer("_proto_ema", _proto_init.clone().detach())
            else:
                # gradient mode: nn.Parameter updated by optimizer; EMA NOT used.
                self.prototypes = nn.Parameter(_proto_init)
        else:
            self.prototypes = None  # no param; optimizer must guard with isinstance check

        # ── Geometry diagnostics ────────────────────────────────────────────
        self.geom_diagnostics = bool(geom_diagnostics)
        # Populated each forward(); read externally for per-epoch logging.
        self.last_geometry_stats: dict | None = None
        self.last_proto_stats: dict | None = None

        # ── Geom2: SPM (ArcAD-style spatial prototype memory) ──────────────
        self.spm = bool(spm)
        self.spm_k = int(spm_k)
        self.spm_weight = float(spm_weight)
        self.spm_tau = max(float(spm_tau), 1e-8)
        self.spm_eps = float(spm_eps)
        self.spm_sinkhorn_iters = int(spm_sinkhorn_iters)
        self.spm_ema = float(spm_ema)
        if self.spm:
            if spm_init_file is not None:
                import numpy as np
                _init = torch.from_numpy(np.load(spm_init_file)).float()  # [K, C]
                _init = F.normalize(_init, p=2, dim=1)
            else:
                _init = F.normalize(torch.randn(self.spm_k, self.bottleneck_dim), p=2, dim=1)
            # SPM uses EMA-only updates (not nn.Parameter — not in optimizer)
            self.register_buffer("_spm_protos", _init.clone())
        else:
            self._spm_protos = None

        # ── Geom2: paired prototype loss ────────────────────────────────────
        self.paired_proto = bool(paired_proto)
        self.paired_proto_weight = float(paired_proto_weight)
        self.paired_proto_sg_target = bool(paired_proto_sg_target)  # True = sg on normal canvas

        # ── Geom2: soft bottleneck pair ─────────────────────────────────────
        self.soft_pair = bool(soft_pair)
        self.soft_pair_weight = float(soft_pair_weight)

        # ── Geom2: new teacher repulsion (trep2) ────────────────────────────
        self.trep = bool(trep)
        self.trep_weight = float(trep_weight)
        self.trep_adaptive_margin = bool(trep_adaptive_margin)
        self._trep_margin = float(trep_margin)  # set at runtime from baseline V q50 when adaptive

        # ── Flag: does any module need bottleneck features? ────────────────
        # Also True when geom_diagnostics enabled so N (bottleneck cosine) metrics are always logged.
        self._need_bottleneck_features = (self.bottleneck_pair or self.bottleneck_protos
                                          or self.spm or self.paired_proto or self.soft_pair
                                          or self.geom_diagnostics)

    def _dinomaly_downweight(self, loss_per_pos: torch.Tensor) -> torch.Tensor:
        """Apply Dinomaly's hard-mining gradient-reduction trick.

        For each position, if its loss is in the EASY p-fraction (below the p-th percentile),
        scale its gradient by `factor` (typically 0.1) while leaving the forward value unchanged.
        Forward: returned tensor has same numerical values as input.
        Backward: easy positions' gradients are multiplied by `factor`; hard positions unchanged.

        Implementation uses the standard "detach scaling" trick:
            modified = x * factor + (x - x * factor).detach()
            forward = factor + (1 - factor) * x = x  ✓
            backward = factor * dx/dθ                ✓
        """
        if self.dinomaly_p <= 0:
            return loss_per_pos
        flat = loss_per_pos.detach().flatten()
        n = flat.numel()
        if n < 2:
            return loss_per_pos
        # Linear ramp: p_current = min(p_final * step / ramp_iters, p_final)
        p_now = min(
            self.dinomaly_p * (self._dinomaly_step / self.dinomaly_ramp_iters),
            self.dinomaly_p,
        )
        if p_now <= 0:
            return loss_per_pos
        # Threshold = the p_now-th percentile of the loss distribution. Easy = below threshold.
        thresh_idx = max(0, min(n - 1, int(n * p_now)))
        threshold = torch.kthvalue(flat, thresh_idx + 1).values
        easy_mask = loss_per_pos < threshold
        # Detach trick: same forward, gradient × factor on easy positions.
        downweighted = loss_per_pos * self.dinomaly_factor + (
            loss_per_pos - loss_per_pos * self.dinomaly_factor
        ).detach()
        return torch.where(easy_mask, downweighted, loss_per_pos)

    # ── Prototype helpers (Module 3) ────────────────────────────────────────

    def _assign_prototypes(self, tokens: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        """Assign each token to prototypes as a full distribution over K clusters.

        Args:
            tokens: [N, D] L2-normalised token vectors.
            proto:  [K, D] L2-normalised prototype vectors.

        Returns:
            float tensor [N, K] — rows sum to 1 (one-hot for nearest, soft dist otherwise).
        """
        sim = tokens @ proto.T                                         # [N, K]
        if self.bottleneck_protos_assignment == "sinkhorn":
            return self._sinkhorn_assign(sim)
        if self.bottleneck_protos_assignment == "soft":
            tau = max(self.bottleneck_protos_temperature, 1e-8)
            return (sim / tau).softmax(dim=1)                          # [N, K] soft dist
        # nearest: one-hot at argmax (default)
        idx = sim.argmax(dim=1)                                        # [N]
        return torch.zeros_like(sim).scatter_(1, idx.unsqueeze(1), 1.0)  # [N, K] one-hot

    def _sinkhorn_assign(self, sim: torch.Tensor) -> torch.Tensor:
        """Log-domain Sinkhorn for balanced prototype assignment.

        Args:
            sim: [N, K] raw cosine similarities.

        Returns:
            float tensor [N, K] — doubly-normalised assignment distribution.
        """
        n_iters = self.bottleneck_protos_sinkhorn_iters
        tau = max(self.bottleneck_protos_temperature, 1e-8)
        log_Q = sim / tau                                              # [N, K] unnorm log-probs
        # Sinkhorn in log domain: alternately normalise rows then columns.
        for _ in range(n_iters):
            log_Q = log_Q - log_Q.logsumexp(dim=1, keepdim=True)     # row normalise
            log_Q = log_Q - log_Q.logsumexp(dim=0, keepdim=True)     # col normalise
        return log_Q.exp()                                             # [N, K] doubly-normalised

    @torch.no_grad()
    def _update_proto_ema(self, tokens: torch.Tensor, hard_assignment: torch.Tensor) -> None:
        """EMA update of each prototype from its hard-assigned tokens (no grad).

        Only called in 'ema' update_mode. Both self._proto_ema and self.prototypes are
        registered buffers in that mode — .copy_() is correct (no .data needed).

        Args:
            tokens:          [N, D] L2-normalised token vectors (already detached).
            hard_assignment: [N]   int indices from argmax(q_n) — always hard regardless of mode.
        """
        K = self._proto_ema.shape[0]
        new_proto = self._proto_ema.clone()
        alpha = self.bottleneck_protos_ema
        for k in range(K):
            mask_k = (hard_assignment == k)
            if mask_k.sum() > 0:
                mean_k = tokens[mask_k].mean(dim=0)
                new_proto[k] = alpha * new_proto[k] + (1.0 - alpha) * mean_k
        new_proto = F.normalize(new_proto, p=2, dim=1)
        self._proto_ema.copy_(new_proto)
        self.prototypes.copy_(new_proto)   # buffer copy (not .data — prototypes is a buffer here)

    # ── Geom2 SPM helpers ────────────────────────────────────────────────────

    def _spm_sinkhorn_assign(self, sim: torch.Tensor) -> torch.Tensor:
        """Balanced Sinkhorn assignment for SPM (log-domain, regularized OT).

        Args:
            sim: [N, K] cosine similarities / self.spm_tau.

        Returns:
            [N, K] doubly-normalised assignment distribution.
        """
        log_Q = sim  # already divided by tau by caller
        eps = self.spm_eps
        for _ in range(self.spm_sinkhorn_iters):
            log_Q = log_Q - torch.log(log_Q.exp().sum(dim=1, keepdim=True) + eps)
            log_Q = log_Q - torch.log(log_Q.exp().sum(dim=0, keepdim=True) + eps)
        return log_Q.exp()

    @torch.no_grad()
    def _spm_update_ema(self, tokens: torch.Tensor, Q: torch.Tensor) -> None:
        """EMA update of SPM prototypes using soft assignment Q.

        Args:
            tokens: [N, D] L2-normalised bottleneck tokens.
            Q:      [N, K] soft assignment distribution from Sinkhorn.
        """
        # Weighted mean of tokens per prototype: [K, D] = Q^T @ tokens / (sum_Q + eps)
        weighted = Q.T @ tokens  # [K, D]
        counts = Q.sum(dim=0).unsqueeze(1).clamp(min=1e-8)  # [K, 1]
        new_proto = weighted / counts  # [K, D]
        new_proto = F.normalize(new_proto, p=2, dim=1)
        alpha = self.spm_ema
        updated = alpha * self._spm_protos + (1.0 - alpha) * new_proto
        self._spm_protos.copy_(F.normalize(updated, p=2, dim=1))

    # ── Existing helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _log_shoot(
        self,
        sf_raw: torch.Tensor,
        tf_raw: torch.Tensor,
        stage: int,
        term: str,
        anomaly_mask: torch.Tensor | None,
    ) -> None:
        """Element-wise student×DFS vs teacher comparison, restricted to anomaly pixels.

        anomaly_mask: [N, H, W] with 1 at anomaly spatial positions. If None or no
        anomaly pixels in the batch, the call is a no-op (we only care about the
        anomaly region).
        """
        if anomaly_mask is None or sf_raw.numel() == 0:
            return
        # Build per-element selection mask broadcast across channels: [N, 1, H, W] → [N, C, H, W]
        m_bool = (anomaly_mask > 0.5).unsqueeze(1).expand_as(sf_raw)
        if not m_bool.any():
            return
        sf_anom = sf_raw[m_bool]  # 1-D tensor of anomaly-only elements
        tf_anom = tf_raw[m_bool]
        diff = sf_anom - tf_anom
        self.shoot_log.append({
            "epoch": self._current_epoch,
            "stage": int(stage),
            "term": term,
            "n_elements": int(sf_anom.numel()),
            "elem_under_frac": float((sf_anom < tf_anom).float().mean()),
            "signed_diff_mean": float(diff.mean()),
            "signed_diff_p10": float(diff.quantile(0.10)),
            "signed_diff_p50": float(diff.quantile(0.50)),
            "signed_diff_p90": float(diff.quantile(0.90)),
            "s_mean": float(sf_anom.mean()),
            "t_mean": float(tf_anom.mean()),
        })

    def _lkd_lsc_align(self, sf_flat, tf_flat, sf_raw=None, tf_raw=None,
                       pos_mask: torch.Tensor | None = None):
        """L_KD + L_SC: align (optionally masked to specific positions).

        L_KD uses self.lkd_mode:
          'local'  — per-position cosine averaged (Anomalib-style).
          'global' — flatten C×H×W per sample, single cosine per (sample, stage). Paper Eq. 1.
          'both'   — sum of local + global L_KD.
        L_SC always per-position (spatial contrastive).

        pos_mask: optional boolean [n, h*w]. When provided, all per-position means
        are restricted to True positions only. Used by Term 2 (canvas-align) when
        --canvas-align-anomaly-only is set to concentrate gradient on anomaly pixels
        instead of diluting it across all positions. Global L_KD path falls back to
        local when pos_mask is set (no clean way to mask the flatten op).
        """
        # Local cosine (always computed; cheap)
        # Use weighted MSE here? Depends on loss_form + lw_align_only.
        # lw_align_only=True → only use weighted MSE when pos_mask is set (= Term 2 restoration).
        # Term 1 (normal→normal, no pos_mask) stays cosine.
        _use_wmse_here = (self.loss_form == "weighted_mse"
                          and not (getattr(self, 'lw_align_only', False) and pos_mask is None))
        tf_kd = tf_flat.detach() if self.stop_gradient else tf_flat
        if _use_wmse_here:
            a = tf_kd.detach().abs().mean(dim=1, keepdim=True)            # [n, 1, c]
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
            diff_sq = (sf_flat - tf_kd).pow(2)                            # [n, h*w, c]
            loss_per_pos = (a * diff_sq).sum(dim=-1)                      # [n, h*w]
        else:
            cos_sim_local = F.cosine_similarity(sf_flat, tf_kd, dim=2)
            loss_per_pos = (1 - cos_sim_local)                             # [n, h*w]
        loss_per_pos = self._dinomaly_downweight(loss_per_pos)
        if pos_mask is not None and pos_mask.any():
            local_kd = loss_per_pos[pos_mask].mean()
        else:
            local_kd = loss_per_pos.mean()

        # Global cosine if requested (only when pos_mask is None — global doesn't compose with pos masking)
        global_kd = None
        if pos_mask is None and self.lkd_mode in ("global", "both") and sf_raw is not None and tf_raw is not None:
            n = sf_raw.shape[0]
            sf_g = F.normalize(sf_raw.reshape(n, -1), p=2, dim=1)
            tf_raw_kd = tf_raw.detach() if self.stop_gradient else tf_raw
            tf_g = F.normalize(tf_raw_kd.reshape(n, -1), p=2, dim=1)
            if self.loss_form == "weighted_mse":
                a_g = tf_g.detach().abs()
                a_g = a_g / (a_g.sum(dim=1, keepdim=True) + 1e-8)
                global_kd = (a_g * (sf_g - tf_g).pow(2)).sum(dim=1).mean()
            else:
                global_kd = (1 - (sf_g * tf_g).sum(dim=1)).mean()

        # Select / combine. 'both' uses 0.5 * each so total magnitude matches single-mode runs
        # → lambda_weight balance with L_SC contrastive term stays comparable across modes.
        if self.lkd_mode == "local" or pos_mask is not None:
            cosine_loss = local_kd
        elif self.lkd_mode == "global":
            cosine_loss = global_kd if global_kd is not None else local_kd  # fallback if raw missing
        else:  # 'both'
            if global_kd is not None:
                cosine_loss = 0.5 * local_kd + 0.5 * global_kd
            else:
                cosine_loss = local_kd

        sim_raw = torch.matmul(sf_flat, tf_flat.transpose(1, 2))
        if self.lsc_pre_relu:
            sim_raw = F.relu(sim_raw)
        sim_matrix = sim_raw / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        sim_matrix = sim_matrix / (sim_matrix.sum(dim=2, keepdim=True) + 1e-8)
        diag_sim = torch.diagonal(sim_matrix, dim1=1, dim2=2)

        # Per-position weight for scaling contrastive + margin when loss_form='weighted_mse'
        if pos_mask is not None and pos_mask.any():
            contrastive_loss = -torch.log(diag_sim[pos_mask] + 1e-8).mean()
        else:
            contrastive_loss = -torch.log(diag_sim + 1e-8).mean()

        margin_loss = 0.0
        if self.margin_mode == "on":
            if self.tau is not None:
                tau_c = max(0.0, min(2.5, self.tau))
                p_align = 0.5 + 0.2 * tau_c
            else:
                p_align = self.margin
            if pos_mask is not None and pos_mask.any():
                margin_loss = F.relu(p_align - diag_sim[pos_mask]).mean()
            else:
                margin_loss = F.relu(p_align - diag_sim).mean()

        return cosine_loss * self.lambda_weight + contrastive_loss * (1 - self.lambda_weight) + margin_loss

    def _tk(self, key: str, v: torch.Tensor) -> torch.Tensor:
        """Track a loss term (detached, summed over stages) for per-epoch logging; returns v unchanged."""
        self._tkacc[key] = self._tkacc.get(key, 0.0) + v.detach()
        return v

    def _lkd_lsc_twosided(self, sf_flat, tf_flat, normal_mask, abnormal_mask):
        """L_KD + L_SC: two-sided spatial (align normal, dis-align anomaly). With optional margin."""
        tf_kd = tf_flat.detach() if self.stop_gradient else tf_flat
        cos_sim = F.cosine_similarity(sf_flat, tf_kd, dim=2)

        sim_raw = torch.matmul(sf_flat, tf_flat.transpose(1, 2))
        if self.lsc_pre_relu:
            sim_raw = F.relu(sim_raw)
        sim_matrix = sim_raw / self.temperature
        sim_matrix = torch.exp(sim_matrix)
        sim_matrix = sim_matrix / (sim_matrix.sum(dim=2, keepdim=True) + 1e-8)
        diag_sim = torch.diagonal(sim_matrix, dim1=1, dim2=2)

        cosine_loss = torch.tensor(0.0, device=sf_flat.device)
        contrastive_loss = torch.tensor(0.0, device=sf_flat.device)

        # tau (when set) overrides legacy asymmetric margin: pull @ 0.5+0.2*tau, push @ 0.5-0.2*tau.
        if self.tau is not None:
            tau_c = max(0.0, min(2.5, self.tau))
            p_align = 0.5 + 0.2 * tau_c
            p_disalign = 0.5 - 0.2 * tau_c
        else:
            p_align = self.margin
            p_disalign = self.margin / 2.0

        margin_loss = torch.tensor(0.0, device=sf_flat.device)
        # L_weighted ablation: compute per-position weighted-MSE on already-L2-normed features,
        # then mask to normal positions. Only used when loss_form='weighted_mse'.
        per_pos_wmse = None
        w_pos = None
        _twosided_wmse = self.loss_form == "weighted_mse" and not getattr(self, 'lw_align_only', False)
        _use_push_wpos = _twosided_wmse and getattr(self, 'scale_kd_push', False)
        _use_sc_wpos = _twosided_wmse and getattr(self, 'scale_push_contrastive', False)
        if _twosided_wmse:
            a = tf_kd.detach().abs().mean(dim=1, keepdim=True)
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
            per_pos_wmse = (a * (sf_flat - tf_kd).pow(2)).sum(dim=-1)     # [n, h*w]
            if _use_push_wpos or _use_sc_wpos:
                w_pos = (a * tf_kd.detach().pow(2)).sum(dim=-1)
                w_pos = w_pos / (w_pos.mean(dim=-1, keepdim=True) + 1e-8)
        if normal_mask.sum() > 0:
            if per_pos_wmse is not None:
                norm_loss = self._dinomaly_downweight(per_pos_wmse[normal_mask])
            else:
                norm_loss = self._dinomaly_downweight(1 - cos_sim[normal_mask])
            cosine_loss = cosine_loss + self.lambda_norm_cos * norm_loss.mean()
            if _use_sc_wpos and w_pos is not None:
                contrastive_loss = self.lambda_norm_cos * (w_pos[normal_mask] * -torch.log(diag_sim[normal_mask] + 1e-8)).mean()
            else:
                contrastive_loss = self.lambda_norm_cos * (-torch.log(diag_sim[normal_mask] + 1e-8)).mean()
            if self.margin_mode == "on":
                if _use_sc_wpos and w_pos is not None:
                    margin_loss = margin_loss + self.lambda_norm_cos * (w_pos[normal_mask] * F.relu(p_align - diag_sim[normal_mask])).mean()
                else:
                    margin_loss = margin_loss + self.lambda_norm_cos * F.relu(p_align - diag_sim[normal_mask]).mean()
        if abnormal_mask.sum() > 0 and not getattr(self, 'no_twosided_push', False):
            # Anomaly cosine push: default relu(cos) stops at orthogonal (cos=0); push_to_max
            # uses (1+cos) which keeps pushing to full anti-alignment (cos=-1).
            def _push(c):
                return (1.0 + c) if getattr(self, 'push_to_max', False) else F.relu(c)
            if _use_push_wpos and w_pos is not None:
                anom_loss = self._dinomaly_downweight(w_pos[abnormal_mask] * _push(cos_sim[abnormal_mask]))
            else:
                anom_loss = self._dinomaly_downweight(_push(cos_sim[abnormal_mask]))
            cosine_loss = cosine_loss + self.lambda_push * anom_loss.mean()
            diag_anom = diag_sim[abnormal_mask].clamp(max=1.0 - 1e-6)
            if _use_sc_wpos and w_pos is not None:
                contrastive_loss = contrastive_loss + self.lambda_push * (w_pos[abnormal_mask] * -torch.log(1.0 - diag_anom + 1e-8)).mean()
            else:
                contrastive_loss = contrastive_loss + self.lambda_push * (-torch.log(1.0 - diag_anom + 1e-8)).mean()
            if self.margin_mode == "on":
                if _use_sc_wpos and w_pos is not None:
                    margin_loss = margin_loss + self.lambda_push * (w_pos[abnormal_mask] * F.relu(diag_anom - p_disalign)).mean()
                else:
                    margin_loss = margin_loss + self.lambda_push * F.relu(diag_anom - p_disalign).mean()

        return cosine_loss * self.lambda_weight + contrastive_loss * (1 - self.lambda_weight) + margin_loss

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        canvas_teacher_features: list[torch.Tensor] | None = None,
        syn_idx: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        canvas_student_features: list[torch.Tensor] | None = None,
        bottleneck_synth: torch.Tensor | None = None,
        bottleneck_canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=student_features[0].device)
        loss_N = torch.tensor(0.0, device=student_features[0].device)   # normal branch (only used with BRANCH_BALANCE)
        self._tkacc = {}   # per-term loss tracking (logging only; values detached)
        if not hasattr(self, "_bb_ema_n"):
            self._bb_ema_n, self._bb_ema_s = 0.0, 0.0
        has_syn = canvas_teacher_features is not None and syn_idx is not None and len(syn_idx) > 0
        # Step counter for the Dinomaly hard-mining ramp. Incremented once per training step
        # (forward() is only called during model.training in UniNetModelExtension).
        if self.training:
            self._dinomaly_step += 1

        # Mode 5 support: detect real-anomaly samples (have mask anomaly pixels but no canvas,
        # so not in syn_idx). Term 3 (two-sided spatial) is applied to them; term 2 (canvas-align)
        # is skipped because no canvas exists. For Modes 1-4 with no mixed real/synth batches,
        # real_anom_idx_global is empty and behavior is unchanged.
        real_anom_idx_global = None
        if mask is not None:
            m = mask.float()
            mb = m.view(m.size(0), -1)
            sample_has_anom = (mb.sum(dim=1) > 0)
            syn_set_global = set(syn_idx.tolist()) if (syn_idx is not None and len(syn_idx) > 0) else set()
            real_list = [j for j in range(mb.size(0))
                         if sample_has_anom[j].item() and j not in syn_set_global]
            if real_list:
                real_anom_idx_global = torch.tensor(real_list, device=m.device, dtype=torch.long)
        has_real_anom = real_anom_idx_global is not None and len(real_anom_idx_global) > 0

        # Group G accumulators (pairrank/reltail): summed cosine-distance maps at 64x64
        # over all stages, computed inside the loop on the synth subset. Both flags need
        # canvas_student_features (i.e., --proj-consistency). When neither flag is set,
        # the accumulators stay None and the post-loop blocks are skipped.
        need_dist_accum = ((self.pairrank or self.reltail or self.maskrank) and has_syn
                            and canvas_student_features is not None)
        dist_acc_synth = None    # [n_syn, 1, 64, 64]
        dist_acc_canvas = None   # [n_syn, 1, 64, 64]
        mask_64_syn = None       # [n_syn, 1, 64, 64] — mask resized once

        # Geometry diagnostics: accumulate per-target-stage values, averaged after the loop.
        need_geom_diag = self.geom_diagnostics and has_syn
        diag_acc: dict[str, list[float]] = {}   # key → list of per-stage scalar values

        for i in range(len(student_features)):
            sf = student_features[i]
            tf = teacher_features[i]
            n, c, h, w = sf.shape

            if not has_syn and not has_real_anom:
                # All normal samples: align everywhere — no anomaly mask, no shoot logging.
                sf_flat = F.normalize(sf.view(n, c, -1).transpose(1, 2), p=2, dim=2)
                tf_flat = F.normalize(tf.view(n, c, -1).transpose(1, 2), p=2, dim=2)
                if BRANCH_BALANCE or GRAD_BALANCE:
                    loss_N = loss_N + self._tk("T1_normal", self._lkd_lsc_align(sf_flat, tf_flat, sf_raw=sf, tf_raw=tf))
                else:
                    loss = loss + self._tk("T1_normal", self._lkd_lsc_align(sf_flat, tf_flat, sf_raw=sf, tf_raw=tf))
                continue

            anom_set = set()
            if has_syn:
                anom_set.update(syn_idx.tolist())
            if has_real_anom:
                anom_set.update(real_anom_idx_global.tolist())
            norm_idx = torch.tensor([j for j in range(n) if j not in anom_set],
                                    device=sf.device, dtype=torch.long)

            # Term 1: Normal samples — L_KD + L_SC align (no anomaly mask, no shoot logging)
            if len(norm_idx) > 0:
                sf_n_raw = sf[norm_idx]
                tf_n_raw = tf[norm_idx]
                sf_n = F.normalize(sf_n_raw.view(len(norm_idx), c, -1).transpose(1, 2), p=2, dim=2)
                tf_n = F.normalize(tf_n_raw.view(len(norm_idx), c, -1).transpose(1, 2), p=2, dim=2)
                if BRANCH_BALANCE or GRAD_BALANCE:
                    loss_N = loss_N + self._tk("T1_normal", self._lkd_lsc_align(sf_n, tf_n, sf_raw=sf_n_raw, tf_raw=tf_n_raw))
                else:
                    loss = loss + self._tk("T1_normal", self._lkd_lsc_align(sf_n, tf_n, sf_raw=sf_n_raw, tf_raw=tf_n_raw))

            # Terms 2 & 3 for synthetic samples (canvas-align + two-sided synth→synth)
            if has_syn:
                n_syn = len(syn_idx)
                sf_s_raw = sf[syn_idx]
                sf_s = F.normalize(sf_s_raw.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)
                tf_s_raw = tf[syn_idx]
                tf_s = F.normalize(tf_s_raw.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)
                ctf = canvas_teacher_features[i]
                ctf_flat = F.normalize(ctf.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)

                # Downsample anomaly mask once for stage's spatial grid (used for shoot log + Term 3 two-sided).
                mask_down_syn = None
                if mask is not None:
                    mask_syn_orig = mask[syn_idx].float()
                    if mask_syn_orig.dim() == 3:
                        mask_syn_orig = mask_syn_orig.unsqueeze(1)
                    mask_down_syn = _downsample_mask(mask_syn_orig, h, w)

                # Term 2: Student(x_syn) → Teacher(x_canvas) — log within anomaly pixels of syn
                self._log_shoot(sf_s_raw, ctf, stage=i, term="term2_canvas",
                                anomaly_mask=mask_down_syn)
                # canvas_align_mode controls position weighting of Term 2:
                #   uniform      = default (~5% on anomaly, 95% on normal — diluted)
                #   anomaly-only = ~100% on anomaly (concentrate gradient where Term 2 adds new info)
                #   50-50        = 0.5 * mean(normal) + 0.5 * mean(anomaly) (balanced split)
                if self.canvas_align_mode == "anomaly-only" and mask_down_syn is not None:
                    anom_mask = (mask_down_syn.view(n_syn, -1) > 0.5)
                    loss = loss + self._tk("T2_canvas", self.w_canvas * self._lkd_lsc_align(
                        sf_s, ctf_flat, sf_raw=sf_s_raw, tf_raw=ctf, pos_mask=anom_mask))
                elif self.canvas_align_mode == "50-50" and mask_down_syn is not None:
                    anom_mask = (mask_down_syn.view(n_syn, -1) > 0.5)
                    norm_mask = ~anom_mask
                    loss_anom = self._lkd_lsc_align(
                        sf_s, ctf_flat, sf_raw=sf_s_raw, tf_raw=ctf, pos_mask=anom_mask)
                    loss_norm = self._lkd_lsc_align(
                        sf_s, ctf_flat, sf_raw=sf_s_raw, tf_raw=ctf, pos_mask=norm_mask)
                    loss = loss + self._tk("T2_canvas", self.w_canvas * 0.5 * (loss_anom + loss_norm))
                else:
                    # uniform / legacy
                    loss = loss + self._tk("T2_canvas", self.w_canvas * self._lkd_lsc_align(
                        sf_s, ctf_flat, sf_raw=sf_s_raw, tf_raw=ctf))

                # exp2: T(synth) → T(canvas) two-sided on the ADAPTIVE teacher only.
                # Applied on TARGET stages (second half: i >= n_stage/2) so the gradient reaches
                # ONLY the adaptive teacher — both tf_s and ctf_flat are target-teacher feats there;
                # source stages are frozen (would add a constant). No SG (hero setting): grad flows
                # to T_synth and T_canvas. Pull T_synth→T_canvas on normal pixels (full only),
                # push apart on anomaly pixels. Sharpens the teacher's own anomaly signal.
                if (self.ct_twosided != "off" and mask_down_syn is not None
                        and i >= len(student_features) // 2):
                    mf_tc = mask_down_syn.view(n_syn, -1)
                    abn_tc = mf_tc > 0.5
                    nrm_tc = torch.zeros_like(abn_tc) if self.ct_twosided == "push_only" else (mf_tc <= 0.5)
                    loss = loss + self.w_ct_twosided * self._lkd_lsc_twosided(tf_s, ctf_flat, nrm_tc, abn_tc)

                # Variant 3 (gap-gated outside canvas-align):
                # In addition to standard CAAO at anomaly positions, apply a SOFT-GATED
                # canvas-align outside the mask where T_s ≈ T_c (teacher gap small).
                # Soft gate g_p = exp(-gap_p / tau), tau = stopgrad(median outside gap).
                # Stop-gradient through gate.
                if (self.gap_gated_outside_canvas
                        and mask_down_syn is not None and self.canvas_align_mode == "anomaly-only"):
                    with torch.no_grad():
                        # Per-pixel teacher gap on this stage [n_syn, h*w]
                        gap = 1.0 - (tf_s * ctf_flat).sum(dim=-1)  # cosine distance, [n_syn, h*w]
                        outside_mask = (mask_down_syn.view(n_syn, -1) <= 0.5)
                        if outside_mask.any():
                            outside_gaps = gap[outside_mask]
                            tau = outside_gaps.median().clamp(min=1e-6) + 1e-6
                            gate = (-gap / tau).exp()  # [n_syn, h*w]
                            # mask: keep only outside pixels
                            gate_outside = gate * outside_mask.float()
                            mean_gate = float(gate_outside[outside_mask].mean().item())
                            mean_gap = float(outside_gaps.mean().item())
                            median_gap = float(outside_gaps.median().item())
                        else:
                            gate_outside = None
                    if gate_outside is not None:
                        # Per-position cosine distance d_cos(S_s, T_c)
                        per_pos_dist = 1.0 - (sf_s * ctf_flat).sum(dim=-1)  # [n_syn, h*w]
                        # Apply gate, average over outside-mask positions
                        weighted = per_pos_dist * gate_outside.detach()
                        n_outside = outside_mask.sum().clamp(min=1).float()
                        L_gated = weighted.sum() / n_outside
                        loss = loss + self.lambda_gated_outcanvas * L_gated
                        self.last_gated_outcanvas_stats = {
                            "stage": i, "mean_gate": mean_gate,
                            "mean_outside_gap": mean_gap, "median_outside_gap": median_gap,
                            "L_gated_raw": float(L_gated.item()),
                            "L_gated_weighted": float(self.lambda_gated_outcanvas * L_gated.item()),
                        }

                # Module 1: Teacher repulsion — push T_s away from T_n at anomaly pixels.
                # trainable_only (default): applied at target stages only (i >= n_stages/2).
                # all_available: applied at all stages (source stages give zero gradient anyway).
                _apply_trep = (i >= len(student_features) // 2) \
                    if self.teacher_repulsion_stages == "trainable_only" else True
                if (self.teacher_repulsion and mask_down_syn is not None and _apply_trep):
                    anom_pos_tr = (mask_down_syn.view(n_syn, -1) > 0.5)
                    if anom_pos_tr.any():
                        ctf_rep = ctf_flat.detach() if self.teacher_repulsion_clean_sg else ctf_flat
                        cos_dist_tr = 1.0 - (tf_s * ctf_rep).sum(dim=-1)   # [n_syn, h*w]
                        margin_hinge = F.relu(self.teacher_repulsion_margin - cos_dist_tr)
                        L_Trep = margin_hinge[anom_pos_tr].mean()
                        loss = loss + self.teacher_repulsion_weight * L_Trep

                # Geom2: trep2 — new teacher repulsion with adaptive/configurable margin.
                # Applied at target stages only (i >= n_stages/2). Separate from Module 1.
                if (self.trep and mask_down_syn is not None and i >= len(student_features) // 2):
                    anom_pos_trep2 = (mask_down_syn.view(n_syn, -1) > 0.5)
                    if anom_pos_trep2.any():
                        cos_dist_trep2 = 1.0 - (tf_s * ctf_flat.detach()).sum(dim=-1)  # [n_syn, h*w]
                        margin_hinge_trep2 = F.relu(self._trep_margin - cos_dist_trep2)
                        L_trep2 = margin_hinge_trep2[anom_pos_trep2].mean()
                        loss = loss + self.trep_weight * L_trep2

                # Geometry diagnostics accumulation (target stages only, no grad).
                if need_geom_diag and i >= len(student_features) // 2 and mask_down_syn is not None:
                    with torch.no_grad():
                        mf_diag = mask_down_syn.view(n_syn, -1)
                        anom_diag = mf_diag > 0.5
                        norm_diag = ~anom_diag
                        if anom_diag.any():
                            V_val = (1.0 - (tf_s * ctf_flat).sum(dim=-1))[anom_diag].mean().item()
                            R_val = (1.0 - (sf_s * ctf_flat).sum(dim=-1))[anom_diag].mean().item()
                            A_val = (1.0 - (sf_s * tf_s).sum(dim=-1))[anom_diag].mean().item()
                            diag_acc.setdefault("V_anom", []).append(V_val)
                            diag_acc.setdefault("R_anom", []).append(R_val)
                            diag_acc.setdefault("A_anom", []).append(A_val)
                        if norm_diag.any():
                            diag_acc.setdefault("V_norm", []).append(
                                (1.0 - (tf_s * ctf_flat).sum(dim=-1))[norm_diag].mean().item())
                            diag_acc.setdefault("A_norm", []).append(
                                (1.0 - (sf_s * tf_s).sum(dim=-1))[norm_diag].mean().item())

                # Term 3: Student(x_syn) → Teacher(x_syn) — log within anomaly pixels of syn
                self._log_shoot(sf_s_raw, tf_s_raw, stage=i, term="term3_syn",
                                anomaly_mask=mask_down_syn)
                if mask_down_syn is not None:
                    mask_flat = mask_down_syn.view(n_syn, -1)
                    normal_mask = mask_flat == 0
                    abnormal_mask = mask_flat == 1
                    loss = loss + self._tk("T3_twosided", self.w_twosided * self._lkd_lsc_twosided(sf_s, tf_s, normal_mask, abnormal_mask))
                else:
                    loss = loss + self._tk("T3_twosided", self.w_twosided * self._lkd_lsc_align(sf_s, tf_s, sf_raw=sf_s_raw, tf_raw=tf_s_raw))

                # M4 CAAO tproto: push synth-anomaly teacher feats off the synth image's
                # FG-clean normal mean (on-object). Target stages, adaptive teacher, no-SG.
                # push = anomaly off normal-mean; full = + pull FG-normal pixels onto the mean.
                if self.tproto != "off" and mask_down_syn is not None and i >= len(student_features) // 2:
                    mf = mask_down_syn.view(n_syn, -1)
                    nm_all = mf <= 0.5; am_all = mf > 0.5
                    fg_flat_s = _tproto_fg_flat(
                        self._batch_fg[syn_idx] if getattr(self, "_batch_fg", None) is not None else None, h, w)
                    tp_acc = torch.tensor(0.0, device=sf.device); tp_cnt = 0
                    for bi in range(n_syn):
                        nm = nm_all[bi]; am = am_all[bi]
                        if fg_flat_s is not None:
                            nm = nm & fg_flat_s[bi]
                        if am.sum() == 0 or nm.sum() == 0:
                            continue
                        m_n = F.normalize(tf_s[bi][nm].mean(dim=0), p=2, dim=0)
                        push = F.relu((tf_s[bi][am] * m_n.unsqueeze(0)).sum(dim=-1)).mean()
                        if self.tproto == "full":
                            pull = (1.0 - (tf_s[bi][nm] * m_n.unsqueeze(0)).sum(dim=-1)).mean()
                            tp_acc = tp_acc + push + pull
                        else:
                            tp_acc = tp_acc + push
                        tp_cnt += 1
                    if tp_cnt > 0:
                        loss = loss + self.w_tproto * (tp_acc / tp_cnt)

                # Term 3b — outside-mask preservation:
                # Align S(synth) → T(synth) at NON-anomaly positions of synth samples.
                # Fills the coverage gap left by CAAO + w_twosided=0, where synth-image normal
                # positions are otherwise unconstrained → pAUC-full drift.
                if self.preserve_outside and mask_down_syn is not None:
                    norm_pos = (mask_down_syn.view(n_syn, -1) <= 0.5)  # inverse of anomaly mask
                    if norm_pos.any():
                        loss = loss + self.w_preserve * self._lkd_lsc_align(
                            sf_s, tf_s, sf_raw=sf_s_raw, tf_raw=tf_s_raw, pos_mask=norm_pos)

                # Term 4 (Group D — proj consistency):
                # S_s ≈ stopgrad(S_c) at anomaly positions, where S_c = student(T(canvas)).
                # Cosine distance per position, mean over anomaly pixels.
                if (self.proj_consistency and canvas_student_features is not None
                        and mask_down_syn is not None):
                    sc_raw = canvas_student_features[i]                      # [n_syn, c, h, w]
                    sc_flat = F.normalize(sc_raw.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)
                    anom_pos = (mask_down_syn.view(n_syn, -1) > 0.5)          # [n_syn, h*w]
                    if anom_pos.any():
                        sc_target = sc_flat if self.no_detach_any else sc_flat.detach()
                        cos_pp = (sf_s * sc_target).sum(dim=-1)               # [n_syn, h*w]
                        dist_pp = 1.0 - cos_pp
                        loss = loss + self._tk("T4_proj", self.lambda_proj * dist_pp[anom_pos].mean())

                # Term 6 (Group F — residual-direction alignment):
                # r_det = normalize(T_s) - normalize(S_s)  (detector residual)
                # r_cf  = normalize(T_s) - normalize(T_c)  (teacher synth→canvas counterfactual)
                # Both renormalized to unit length; stopgrad on r_cf_hat.
                # L = mean over anomaly positions of (1 - cos(r_det_hat, r_cf_hat)).
                # Reuses sf_s/tf_s/ctf_flat (already L2-normalized along channel dim).
                if self.residual_direction_alignment and mask_down_syn is not None:
                    anom_pos = (mask_down_syn.view(n_syn, -1) > 0.5)
                    if anom_pos.any():
                        eps = 1e-8
                        r_det = tf_s - sf_s                                       # [n_syn, h*w, c]
                        r_cf  = tf_s - ctf_flat                                   # [n_syn, h*w, c]
                        r_det_norm = r_det.norm(dim=-1, keepdim=True).clamp_min(eps)
                        r_cf_norm  = r_cf.norm(dim=-1, keepdim=True).clamp_min(eps)
                        r_det_hat = r_det / r_det_norm
                        r_cf_hat  = (r_cf / r_cf_norm).detach()                   # stopgrad
                        cos_pp = (r_det_hat * r_cf_hat).sum(dim=-1)               # [n_syn, h*w]
                        anom_cos = cos_pp[anom_pos]
                        l_resdir_raw = (1.0 - anom_cos).mean()
                        loss = loss + self.lambda_resdir * l_resdir_raw
                        # Per-step stats — overwritten each forward; readable externally.
                        with torch.no_grad():
                            anom_det_n = r_det_norm.squeeze(-1)[anom_pos]
                            anom_cf_n  = r_cf_norm.squeeze(-1)[anom_pos]
                            self.last_resdir_stats = {
                                "stage": i,
                                "l_resdir_raw": l_resdir_raw.item(),
                                "l_resdir_weighted": (self.lambda_resdir * l_resdir_raw).item(),
                                "mean_cos": anom_cos.mean().item(),
                                "mean_r_det_norm": anom_det_n.mean().item(),
                                "mean_r_cf_norm": anom_cf_n.mean().item(),
                                "near_zero_det_frac": (anom_det_n < eps * 10).float().mean().item(),
                                "near_zero_cf_frac": (anom_cf_n < eps * 10).float().mean().item(),
                                "n_anom_positions": int(anom_pos.sum().item()),
                            }

                # Term 5 (Group E — adaptive triplet):
                # L_triplet = relu(d(S_s, T_c) - d(S_s, T_s) + alpha * stopgrad(d(T_s, T_c)))
                # Applied at anomaly positions only. Designed to replace w_twosided (set w_twosided=0).
                if self.adaptive_triplet and mask_down_syn is not None:
                    anom_pos = (mask_down_syn.view(n_syn, -1) > 0.5)
                    if anom_pos.any():
                        # All three feature sets are already L2-normalized along the channel dim.
                        d_sc = 1.0 - (sf_s * ctf_flat).sum(dim=-1)              # student-to-canvas
                        d_ss = 1.0 - (sf_s * tf_s).sum(dim=-1)                  # student-to-synth
                        d_tt = 1.0 - (tf_s * ctf_flat).sum(dim=-1)              # teacher gap (margin)
                        d_tt = d_tt.detach()                                    # stopgrad on the margin
                        triplet_pp = F.relu(d_sc - d_ss + self.triplet_alpha * d_tt)
                        loss = loss + triplet_pp[anom_pos].mean()

                # Group G accumulation (pairrank/reltail): per-stage cosine-distance maps,
                # upsampled to 64x64 and summed across stages. Synth uses (sf_s, tf_s);
                # canvas uses (sc_norm, ctf_flat) with sc from canvas_student_features.
                if need_dist_accum:
                    sc_raw_g = canvas_student_features[i]                       # [n_syn, c, h, w]
                    sc_g = F.normalize(sc_raw_g.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)
                    d_syn_pp = 1.0 - (sf_s * tf_s).sum(dim=-1)                  # [n_syn, h*w]
                    d_can_pp = 1.0 - (sc_g * ctf_flat).sum(dim=-1)              # [n_syn, h*w]
                    d_syn_map = d_syn_pp.view(n_syn, 1, h, w)
                    d_can_map = d_can_pp.view(n_syn, 1, h, w)
                    if (h, w) != (64, 64):
                        d_syn_map = F.interpolate(d_syn_map, size=64, mode="bilinear", align_corners=False)
                        d_can_map = F.interpolate(d_can_map, size=64, mode="bilinear", align_corners=False)
                    dist_acc_synth = d_syn_map if dist_acc_synth is None else dist_acc_synth + d_syn_map
                    dist_acc_canvas = d_can_map if dist_acc_canvas is None else dist_acc_canvas + d_can_map
                    if mask_64_syn is None and mask is not None:
                        mask_syn_orig_g = mask[syn_idx].float()
                        if mask_syn_orig_g.dim() == 3:
                            mask_syn_orig_g = mask_syn_orig_g.unsqueeze(1)
                        mask_64_syn = _downsample_mask(mask_syn_orig_g, 64, 64)

            # Mode 5 addition: real-anomaly samples (no canvas) — apply Term 3 only.
            # Same two-sided spatial pull/push using student-teacher of the same image.
            # No Term 2 because no canvas teacher exists for real anomalies.
            if has_real_anom:
                n_real = len(real_anom_idx_global)
                sf_r_raw = sf[real_anom_idx_global]
                sf_r = F.normalize(sf_r_raw.view(n_real, c, -1).transpose(1, 2), p=2, dim=2)
                tf_r_raw = tf[real_anom_idx_global]
                tf_r = F.normalize(tf_r_raw.view(n_real, c, -1).transpose(1, 2), p=2, dim=2)

                mask_real_orig = mask[real_anom_idx_global].float()
                if mask_real_orig.dim() == 3:
                    mask_real_orig = mask_real_orig.unsqueeze(1)
                mask_down_real = _downsample_mask(mask_real_orig, h, w)
                mask_flat_r = mask_down_real.view(n_real, -1)
                normal_mask_r = mask_flat_r == 0
                abnormal_mask_r = mask_flat_r == 1
                # Student↔teacher two-sided on reals (= M2-style supervision of the STUDENT).
                # Default ON. Set real_student_sup=False for the "teacher-only" variant, where the
                # reals shape ONLY the adaptive teacher (via real_tproto), not the student/detector.
                if getattr(self, "real_student_sup", True):
                    loss = loss + self.w_twosided * self._lkd_lsc_twosided(sf_r, tf_r, normal_mask_r, abnormal_mask_r)

                # real_tproto: teacher same-object push on the REAL source anomalies (target stages,
                # adaptive teacher, no SG). Within each real anomaly image, push anomaly-pixel teacher
                # feats off the image's own normal-region prototype (orthogonal-stop). Same-object =
                # perfect common-mode cancellation, isolating the genuine defect signal. 'full' also
                # pulls normal pixels onto the prototype (compactness).
                if self.real_tproto != "off" and i >= len(student_features) // 2:
                    acc = torch.tensor(0.0, device=sf.device); cnt = 0
                    # On-object reference: restrict real-image normal pixels to foreground.
                    fg_flat_r = _tproto_fg_flat(
                        self._batch_fg[real_anom_idx_global] if getattr(self, "_batch_fg", None) is not None else None, h, w)
                    for bi in range(n_real):
                        nm = normal_mask_r[bi]; am = abnormal_mask_r[bi]
                        if fg_flat_r is not None:
                            nm = nm & fg_flat_r[bi]
                        if self.real_tproto == "intra":
                            # spread real-anomaly pixels apart (off their own centroid; no normal ref)
                            if am.sum() < 2:
                                continue
                            m_a = F.normalize(tf_r[bi][am].mean(dim=0), p=2, dim=0)
                            acc = acc + F.relu((tf_r[bi][am] * m_a.unsqueeze(0)).sum(dim=-1)).mean()
                            cnt += 1
                        else:
                            if am.sum() == 0 or nm.sum() == 0:
                                continue
                            m_n = F.normalize(tf_r[bi][nm].mean(dim=0), p=2, dim=0)        # [c]
                            push = F.relu((tf_r[bi][am] * m_n.unsqueeze(0)).sum(dim=-1)).mean()
                            if self.real_tproto == "full":
                                pull = (1.0 - (tf_r[bi][nm] * m_n.unsqueeze(0)).sum(dim=-1)).mean()
                                acc = acc + push + pull
                            else:
                                acc = acc + push
                            cnt += 1
                    if cnt > 0:
                        loss = loss + self.w_real_tproto * (acc / cnt)

        # Group G — pairrank: explicit (synth, canvas) pair ordering on top-frac of summed map.
        # L = mean(softplus(margin - (s_s - s_c))) with score = top-frac mean.
        if self.pairrank and dist_acc_synth is not None and dist_acc_canvas is not None:
            n_pairs = dist_acc_synth.shape[0]
            s_s = _topk_mean_score(dist_acc_synth, frac=self.reltail_frac)
            s_c = _topk_mean_score(dist_acc_canvas, frac=self.reltail_frac)
            diff = s_s - s_c
            l_pairrank_raw = F.softplus(self.pairrank_margin - diff).mean()
            loss = loss + self.lambda_pairrank * l_pairrank_raw
            with torch.no_grad():
                self.last_pairrank_stats = {
                    "l_pairrank_raw": l_pairrank_raw.item(),
                    "l_pairrank_weighted": (self.lambda_pairrank * l_pairrank_raw).item(),
                    "mean_s_s": s_s.mean().item(),
                    "mean_s_c": s_c.mean().item(),
                    "mean_diff": diff.mean().item(),
                    "frac_s_s_gt_s_c": (diff > 0).float().mean().item(),
                    "n_pairs": int(n_pairs),
                }

        # Group G — reltail: suppress canvas top-frac peaks below rho * stopgrad(s_anom_ref).
        # s_anom_ref = mean of A_synth pixels inside the resized anomaly mask (per sample).
        if (self.reltail and dist_acc_synth is not None
                and dist_acc_canvas is not None and mask_64_syn is not None):
            n_pairs = dist_acc_synth.shape[0]
            s_canvas_tail = _topk_mean_score(dist_acc_canvas, frac=self.reltail_frac)
            flat_synth = dist_acc_synth.view(n_pairs, -1)
            flat_mask = (mask_64_syn.view(n_pairs, -1) > 0.5).float()
            mask_pixel_counts = flat_mask.sum(dim=-1)
            has_anom_per_sample = mask_pixel_counts > 0
            skipped = int((~has_anom_per_sample).sum().item())
            if has_anom_per_sample.any():
                safe_counts = mask_pixel_counts.clamp_min(1.0)
                anom_sum_per = (flat_synth * flat_mask).sum(dim=-1)
                s_anom_ref_per = (anom_sum_per / safe_counts).detach()       # stopgrad
                threshold_per = self.reltail_rho * s_anom_ref_per
                violation_per = F.relu(s_canvas_tail - threshold_per)
                kept = has_anom_per_sample.float()
                l_reltail_raw = (violation_per * kept).sum() / kept.sum()
                loss = loss + self.lambda_reltail * l_reltail_raw
                with torch.no_grad():
                    k = has_anom_per_sample
                    self.last_reltail_stats = {
                        "l_reltail_raw": l_reltail_raw.item(),
                        "l_reltail_weighted": (self.lambda_reltail * l_reltail_raw).item(),
                        "mean_s_canvas_tail": s_canvas_tail[k].mean().item(),
                        "mean_s_anom_ref": s_anom_ref_per[k].mean().item(),
                        "mean_threshold": threshold_per[k].mean().item(),
                        "frac_violating": (violation_per[k] > 0).float().mean().item(),
                        "mean_violation_amount": violation_per[k].mean().item(),
                        "n_pairs_kept": int(kept.sum().item()),
                        "n_pairs_skipped": skipped,
                    }
            else:
                # All masks empty in this batch — skip and log.
                with torch.no_grad():
                    self.last_reltail_stats = {
                        "l_reltail_raw": 0.0, "l_reltail_weighted": 0.0,
                        "n_pairs_kept": 0, "n_pairs_skipped": skipped,
                    }

        # Group G — maskrank: intra-image rank between in-mask and out-of-mask pixels of A_synth.
        # s_in  = mean over mask pixels; s_out = top-1% mean over non-mask pixels.
        # Skip samples with no mask OR fewer than k_out non-mask pixels available.
        if self.maskrank and dist_acc_synth is not None and mask_64_syn is not None:
            n_pairs = dist_acc_synth.shape[0]
            flat_synth = dist_acc_synth.view(n_pairs, -1)                  # [n_pairs, 4096]
            mask_bool = (mask_64_syn.view(n_pairs, -1) > 0.5)               # [n_pairs, 4096]
            in_counts = mask_bool.sum(dim=-1)                               # [n_pairs]
            out_counts = (~mask_bool).sum(dim=-1)
            k_out = max(1, int(0.01 * 4096))                                # 40 pixels
            valid = (in_counts > 0) & (out_counts >= k_out)
            skipped = int((~valid).sum().item())
            if valid.any():
                # s_in: per-sample mean of A_synth inside the mask.
                in_mask_f = mask_bool.float()
                in_sums = (flat_synth * in_mask_f).sum(dim=-1)
                s_in_per = in_sums / in_counts.clamp_min(1).float()
                # s_out: top-1% mean of A_synth OUTSIDE the mask. Mask out inside positions with -1e9.
                very_neg = torch.full_like(flat_synth, -1e9)
                flat_out = torch.where(mask_bool, very_neg, flat_synth)
                topk_vals, _ = flat_out.topk(k_out, dim=-1)                 # [n_pairs, k_out]
                s_out_per = topk_vals.mean(dim=-1)
                diff = s_in_per - s_out_per
                violation = F.softplus(self.maskrank_margin - diff)
                kept = valid.float()
                l_maskrank_raw = (violation * kept).sum() / kept.sum()
                loss = loss + self.lambda_maskrank * l_maskrank_raw
                with torch.no_grad():
                    k_mask = valid
                    self.last_maskrank_stats = {
                        "l_maskrank_raw": l_maskrank_raw.item(),
                        "l_maskrank_weighted": (self.lambda_maskrank * l_maskrank_raw).item(),
                        "mean_s_in": s_in_per[k_mask].mean().item(),
                        "mean_s_out": s_out_per[k_mask].mean().item(),
                        "mean_diff": diff[k_mask].mean().item(),
                        "frac_in_gt_out": (diff[k_mask] > 0).float().mean().item(),
                        "frac_in_gt_out_plus_margin": (diff[k_mask] > self.maskrank_margin).float().mean().item(),
                        "n_pairs_kept": int(kept.sum().item()),
                        "n_pairs_skipped": skipped,
                    }
            else:
                with torch.no_grad():
                    self.last_maskrank_stats = {
                        "l_maskrank_raw": 0.0, "l_maskrank_weighted": 0.0,
                        "n_pairs_kept": 0, "n_pairs_skipped": skipped,
                    }

        # Module 2: Bottleneck pair alignment — pull B_s toward B_n at anomaly positions.
        if (self.bottleneck_pair and bottleneck_synth is not None
                and bottleneck_canvas is not None and mask is not None
                and syn_idx is not None and len(syn_idx) > 0):
            n_syn_bp = bottleneck_synth.shape[0]
            C_bn = bottleneck_synth.shape[1]                              # actual channel dim
            bn_h_bp, bn_w_bp = bottleneck_synth.shape[2:]                # actual spatial dims
            mask_syn_bp = mask[syn_idx].float()
            if mask_syn_bp.dim() == 3:
                mask_syn_bp = mask_syn_bp.unsqueeze(1)
            mask_bn_bp = _downsample_mask(mask_syn_bp, bn_h_bp, bn_w_bp) # [n_syn, 1, H, W]
            anom_8_bp = (mask_bn_bp.view(n_syn_bp, -1) > 0.5)            # [n_syn, H*W]
            if anom_8_bp.any():
                # permute+reshape: [n_syn, C, H, W] → [n_syn, H*W, C] (proper spatial tokens)
                B_s_bp = bottleneck_synth.permute(0, 2, 3, 1).reshape(n_syn_bp, -1, C_bn)
                B_n_bp = bottleneck_canvas
                if self.bottleneck_pair_clean_sg:
                    B_n_bp = B_n_bp.detach()
                B_n_bp = B_n_bp.permute(0, 2, 3, 1).reshape(n_syn_bp, -1, C_bn)
                if self.bottleneck_pair_metric == "cosine":
                    B_s_n2 = F.normalize(B_s_bp, p=2, dim=2)
                    B_n_n2 = F.normalize(B_n_bp, p=2, dim=2)
                    pair_loss_map = 1.0 - (B_s_n2 * B_n_n2).sum(dim=-1)  # [n_syn, H*W]
                elif self.bottleneck_pair_metric == "layernorm_smooth_l1":
                    B_s_ln = F.layer_norm(B_s_bp, [C_bn])
                    B_n_ln = F.layer_norm(B_n_bp, [C_bn])
                    pair_loss_map = F.smooth_l1_loss(B_s_ln, B_n_ln, reduction="none").mean(dim=-1)
                else:  # layernorm_mse
                    B_s_ln = F.layer_norm(B_s_bp, [C_bn])
                    B_n_ln = F.layer_norm(B_n_bp, [C_bn])
                    pair_loss_map = (B_s_ln - B_n_ln).pow(2).mean(dim=-1)
                L_Bpair = pair_loss_map[anom_8_bp].mean()
                loss = loss + self.bottleneck_pair_weight * L_Bpair

        # Module 3: Bottleneck prototype organization.
        if (self.bottleneck_protos and self.prototypes is not None
                and bottleneck_synth is not None and bottleneck_canvas is not None
                and mask is not None and syn_idx is not None and len(syn_idx) > 0):
            n_syn_p3 = bottleneck_synth.shape[0]
            C_bn_p3 = bottleneck_synth.shape[1]                          # actual channel dim
            bn_h_p3, bn_w_p3 = bottleneck_synth.shape[2:]               # actual spatial dims
            mask_syn_p3 = mask[syn_idx].float()
            if mask_syn_p3.dim() == 3:
                mask_syn_p3 = mask_syn_p3.unsqueeze(1)
            mask_bn_p3 = _downsample_mask(mask_syn_p3, bn_h_p3, bn_w_p3) # [n_syn, 1, H, W]
            anom_bn_p3 = (mask_bn_p3.view(n_syn_p3, -1) > 0.5).reshape(-1)  # [n_syn*HW] bool
            norm_bn_p3 = ~anom_bn_p3

            B_s_tok = bottleneck_synth.permute(0, 2, 3, 1).reshape(-1, C_bn_p3)  # [n_syn*HW, C_bn]
            B_n_tok = bottleneck_canvas.permute(0, 2, 3, 1).reshape(-1, C_bn_p3) # [n_syn*HW, C_bn]

            proto = F.normalize(self.prototypes, p=2, dim=1)                   # [K, 1024]
            # gradient mode: proto.requires_grad=True; ema mode: proto is a buffer (no grad)
            B_n_norm3 = F.normalize(B_n_tok, p=2, dim=1)                      # [n_syn*HW, 1024]
            tau_p3 = max(self.bottleneck_protos_temperature, 1e-8)

            # Compute assignment distribution from detached tokens/proto — no grad through assignment.
            # CE loss gradient flows through log_softmax(logits) only.
            with torch.no_grad():
                q_n = self._assign_prototypes(B_n_norm3.detach(), proto.detach())  # [n_syn*HW, K]

            # Pull normal canvas tokens toward their prototype via cross-entropy loss.
            # Grad: → B_n_norm3 (bottleneck) AND → proto (gradient mode only).
            L_proto_n3 = bottleneck_synth.new_zeros(())
            if norm_bn_p3.any():
                q_n_sel = q_n[norm_bn_p3]                                     # [M, K] target dist
                logits_n = (B_n_norm3[norm_bn_p3] @ proto.T) / tau_p3        # [M, K]
                log_p_n = F.log_softmax(logits_n, dim=1)
                L_proto_n3 = -(q_n_sel * log_p_n).sum(dim=1).mean()

            # Pull anomaly synth tokens toward paired clean token's prototype (CE, proto detached).
            # Synth inherits the FULL assignment distribution q_n from the paired canvas position.
            # Grad: → B_s_tok (bottleneck synth) only; proto.T.detach() prevents proto update via synth.
            L_proto_a3 = bottleneck_synth.new_zeros(())
            if anom_bn_p3.any():
                q_anom = q_n[anom_bn_p3]                                      # [M', K] inherited
                B_s_norm3 = F.normalize(B_s_tok, p=2, dim=1)
                B_s_anom3 = B_s_norm3[anom_bn_p3]                            # [M', 1024]
                logits_s = (B_s_anom3 @ proto.T.detach()) / tau_p3           # [M', K]; proto detached
                log_p_s = F.log_softmax(logits_s, dim=1)
                L_proto_a3 = -(q_anom * log_p_s).sum(dim=1).mean()

            L_proto = L_proto_n3 + L_proto_a3
            loss = loss + self.bottleneck_protos_weight * L_proto

            # EMA update only in ema mode. gradient mode uses the optimizer exclusively.
            if self.training and self.bottleneck_protos_update_mode == "ema":
                with torch.no_grad():
                    hard_assign = q_n.argmax(dim=1)                           # [n_syn*HW] int
                    self._update_proto_ema(B_n_norm3.detach(), hard_assign)

            # Prototype diagnostics
            with torch.no_grad():
                K = self.bottleneck_protos_k
                hard_assign_diag = q_n.argmax(dim=1).cpu()
                counts = torch.bincount(hard_assign_diag, minlength=K).float()
                total = counts.sum().clamp_min(1.0)
                dist_p = counts / total
                entropy = -(dist_p * (dist_p + 1e-10).log()).sum().item()
                effective_k = float(torch.exp(torch.tensor(entropy)).item())
                pairwise_sim = (proto.detach() @ proto.detach().T).cpu()
                off_diag = pairwise_sim[~torch.eye(K, dtype=torch.bool)]
                dead_thresh = 0.01 / K
                dead_count = int((dist_p < dead_thresh).sum().item())
                self.last_proto_stats = {
                    "proto/utilization_histogram": counts.tolist(),
                    "proto/effective_k": effective_k,
                    "proto/assignment_entropy": entropy,
                    "proto/mean_clean_sim_to_assigned": float(
                        (B_n_norm3 * proto.detach()[hard_assign_diag.to(B_n_norm3.device)]).sum(dim=-1).mean().item()),
                    "proto/pairwise_sim_mean": float(off_diag.mean().item()),
                    "proto/pairwise_sim_std": float(off_diag.std().item()),
                    "proto/pairwise_sim_min": float(off_diag.min().item()),
                    "proto/dead_proto_count": dead_count,
                    "proto/dead_proto_frac": dead_count / K,
                }

        # Geom2: Soft bottleneck pair — fractional occupancy weighting, no binarization.
        # L_pair = sum_p w_p * D_cos(B_s^p, B_n^p) / (sum_p w_p + eps)
        # where w_p = avg_pool(mask) in [0,1] at each bottleneck position.
        if (self.soft_pair and bottleneck_synth is not None
                and bottleneck_canvas is not None and mask is not None
                and syn_idx is not None and len(syn_idx) > 0):
            n_syn_sp = bottleneck_synth.shape[0]
            C_bn_sp = bottleneck_synth.shape[1]
            bn_h_sp, bn_w_sp = bottleneck_synth.shape[2:]
            mask_syn_sp = mask[syn_idx].float()
            if mask_syn_sp.dim() == 3:
                mask_syn_sp = mask_syn_sp.unsqueeze(1)
            # avg-pool gives fractional occupancy in [0,1] per position
            w_sp = F.adaptive_avg_pool2d(mask_syn_sp, (bn_h_sp, bn_w_sp))   # [n_syn,1,H,W]
            w_sp_flat = w_sp.view(n_syn_sp, -1)                              # [n_syn, H*W]
            total_w_sp = w_sp_flat.sum().clamp(min=1e-8)
            if total_w_sp > 0:
                B_s_sp = F.normalize(
                    bottleneck_synth.permute(0, 2, 3, 1).reshape(n_syn_sp, -1, C_bn_sp),
                    p=2, dim=2)  # [n_syn, H*W, C]
                B_n_sp = F.normalize(
                    bottleneck_canvas.permute(0, 2, 3, 1).reshape(n_syn_sp, -1, C_bn_sp),
                    p=2, dim=2)  # [n_syn, H*W, C]
                cos_dist_sp = 1.0 - (B_s_sp * B_n_sp).sum(dim=-1)          # [n_syn, H*W]
                L_softpair = (w_sp_flat * cos_dist_sp).sum() / total_w_sp
                loss = loss + self.soft_pair_weight * L_softpair

        # Geom2: SPM loss — ArcAD-style K=500 prototype memory on bottleneck (normal tokens only).
        # Assignment via Sinkhorn OT on canvas-normal tokens; loss = cross-entropy.
        # Prototypes updated by EMA (not in optimizer).
        # L_SPM = -sum_i w_i * sum_k q_i(k)*log p_i(k) / (sum_i w_i + eps)
        # where w_i = (1 - fractional_anomaly_occupancy) of each bottleneck token.
        if (self.spm and self._spm_protos is not None
                and bottleneck_synth is not None and bottleneck_canvas is not None
                and mask is not None and syn_idx is not None and len(syn_idx) > 0):
            n_syn_spm = bottleneck_synth.shape[0]
            C_bn_spm = bottleneck_synth.shape[1]
            bn_h_spm, bn_w_spm = bottleneck_synth.shape[2:]
            mask_syn_spm = mask[syn_idx].float()
            if mask_syn_spm.dim() == 3:
                mask_syn_spm = mask_syn_spm.unsqueeze(1)
            # fractional occupancy w_i = (1 - avg_pool(mask)) at each position
            occ_spm = F.adaptive_avg_pool2d(mask_syn_spm, (bn_h_spm, bn_w_spm))  # [n_syn,1,H,W]
            w_spm_flat = (1.0 - occ_spm).view(n_syn_spm, -1)                     # [n_syn, H*W]
            w_spm_1d = w_spm_flat.reshape(-1)                                      # [n_syn*H*W]
            # Canvas bottleneck tokens (L2-normalized)
            B_n_spm = F.normalize(
                bottleneck_canvas.permute(0, 2, 3, 1).reshape(-1, C_bn_spm),
                p=2, dim=1)  # [n_syn*H*W, C]
            # Clone protos to independent storage so the EMA copy_() later in this
            # forward pass does not invalidate the version counter used by autograd.
            proto_spm = self._spm_protos.clone().detach()                          # [K, C]
            # Sinkhorn assignment (no grad through assignment)
            with torch.no_grad():
                sim_spm = (B_n_spm.detach() @ proto_spm.T) / self.spm_tau        # [N, K]
                Q_spm = self._spm_sinkhorn_assign(sim_spm)                        # [N, K]
            # Cross-entropy loss: p = softmax((B_n @ proto^T) / tau)
            logits_spm = (B_n_spm @ proto_spm.T) / self.spm_tau                  # [N, K]
            log_p_spm = F.log_softmax(logits_spm, dim=1)
            ce_per_tok = -(Q_spm * log_p_spm).sum(dim=1)                          # [N]
            total_w_spm = w_spm_1d.sum().clamp(min=1e-8)
            L_spm = (w_spm_1d * ce_per_tok).sum() / total_w_spm
            loss = loss + self.spm_weight * L_spm
            # EMA update using normal-weighted tokens
            if self.training:
                with torch.no_grad():
                    self._spm_update_ema(B_n_spm.detach(), Q_spm)

        # Geom2: paired prototype loss — synth anomaly tokens inherit canvas prototype assignment.
        # Requires SPM to be active (shares SPM prototypes and Sinkhorn Q from normal tokens).
        # L_Q = -sum_p w_p * sum_k q_n(p,k)*log p_s(p,k) / (sum_p w_p + eps)
        # where w_p = fractional_anomaly_occupancy of each bottleneck token.
        if (self.paired_proto and self._spm_protos is not None
                and bottleneck_synth is not None and bottleneck_canvas is not None
                and mask is not None and syn_idx is not None and len(syn_idx) > 0):
            n_syn_pp = bottleneck_synth.shape[0]
            C_bn_pp = bottleneck_synth.shape[1]
            bn_h_pp, bn_w_pp = bottleneck_synth.shape[2:]
            mask_syn_pp = mask[syn_idx].float()
            if mask_syn_pp.dim() == 3:
                mask_syn_pp = mask_syn_pp.unsqueeze(1)
            occ_pp = F.adaptive_avg_pool2d(mask_syn_pp, (bn_h_pp, bn_w_pp))      # [n_syn,1,H,W]
            w_pp_flat = occ_pp.view(n_syn_pp, -1)                                 # anomaly weight
            w_pp_1d = w_pp_flat.reshape(-1)                                        # [n_syn*H*W]
            B_n_pp = F.normalize(
                bottleneck_canvas.permute(0, 2, 3, 1).reshape(-1, C_bn_pp),
                p=2, dim=1)  # [n_syn*H*W, C] — canvas (normal) tokens
            B_s_pp = F.normalize(
                bottleneck_synth.permute(0, 2, 3, 1).reshape(-1, C_bn_pp),
                p=2, dim=1)  # [n_syn*H*W, C] — synth tokens
            # Clone protos to independent storage (same reason as SPM block above).
            proto_pp = self._spm_protos.clone().detach()                           # [K, C]
            with torch.no_grad():
                # Normal canvas token assignment (SG, same as SPM)
                sim_n_pp = (B_n_pp.detach() @ proto_pp.T) / self.spm_tau
                Q_n_pp = self._spm_sinkhorn_assign(sim_n_pp)                      # [N, K]
            # Synth prediction: p_s = softmax((B_s @ proto^T) / tau)
            logits_s_pp = (B_s_pp @ proto_pp.T) / self.spm_tau                   # [N, K]
            log_p_s_pp = F.log_softmax(logits_s_pp, dim=1)
            ce_pp = -(Q_n_pp * log_p_s_pp).sum(dim=1)                             # [N]
            total_w_pp = w_pp_1d.sum().clamp(min=1e-8)
            L_paired_proto = (w_pp_1d * ce_pp).sum() / total_w_pp
            loss = loss + self.paired_proto_weight * L_paired_proto

        # Publish geometry diagnostics (averaged over target stages).
        if need_geom_diag and diag_acc:
            with torch.no_grad():
                gs: dict[str, float] = {}
                def _avg(key: str) -> float | None:
                    vals = diag_acc.get(key)
                    return float(sum(vals) / len(vals)) if vals else None
                for dst, src in [
                    ("geom/anom_teacher_clean_synth_cosdist", "V_anom"),
                    ("geom/anom_student_to_clean_cosdist", "R_anom"),
                    ("geom/anom_student_to_synth_cosdist", "A_anom"),
                    ("geom/normal_teacher_clean_synth_cosdist", "V_norm"),
                    ("geom/normal_student_to_synth_cosdist", "A_norm"),
                ]:
                    v = _avg(src)
                    if v is not None:
                        gs[dst] = v
                # Bottleneck-level N metrics (at actual bottleneck spatial resolution)
                if (bottleneck_synth is not None and bottleneck_canvas is not None
                        and mask is not None and syn_idx is not None and len(syn_idx) > 0):
                    n_syn_d = bottleneck_synth.shape[0]
                    C_bn_d = bottleneck_synth.shape[1]             # actual channel dim
                    bn_h_d, bn_w_d = bottleneck_synth.shape[2:]
                    mask_syn_d = mask[syn_idx].float()
                    if mask_syn_d.dim() == 3:
                        mask_syn_d = mask_syn_d.unsqueeze(1)
                    mask_bn_d = _downsample_mask(mask_syn_d, bn_h_d, bn_w_d)
                    # permute+reshape: [n_syn, C, H, W] → [n_syn, H*W, C] (proper spatial tokens)
                    B_s_d = F.normalize(
                        bottleneck_synth.permute(0, 2, 3, 1).reshape(n_syn_d, -1, C_bn_d),
                        p=2, dim=2)
                    B_n_d = F.normalize(
                        bottleneck_canvas.permute(0, 2, 3, 1).reshape(n_syn_d, -1, C_bn_d),
                        p=2, dim=2)
                    bn_cos = 1.0 - (B_s_d * B_n_d).sum(dim=-1)   # [n_syn, H*W]
                    anom_bn_d = (mask_bn_d.view(n_syn_d, -1) > 0.5)
                    norm_bn_d = ~anom_bn_d
                    if anom_bn_d.any():
                        N_anom = float(bn_cos[anom_bn_d].mean().item())
                        gs["geom/anom_bottleneck_clean_synth_cosdist"] = N_anom
                        V_anom = gs.get("geom/anom_teacher_clean_synth_cosdist", 0.0)
                        # _heuristic suffix: E_B = V - N and Q_B = V/N are heuristic indicators only
                        gs["geom/anom_bottleneck_removal_delta_heuristic"] = V_anom - N_anom
                        gs["geom/anom_bottleneck_removal_ratio_heuristic"] = V_anom / (N_anom + 1e-8)
                    if norm_bn_d.any():
                        gs["geom/normal_bottleneck_clean_synth_cosdist"] = float(
                            bn_cos[norm_bn_d].mean().item())
                self.last_geometry_stats = gs if gs else None
        else:
            self.last_geometry_stats = None

        if GRAD_BALANCE:
            B = student_features[0].shape[0]
            n_s = (len(syn_idx) if has_syn else 0) + (len(real_anom_idx_global) if has_real_anom else 0)
            w = (B - n_s) / B; cg = getattr(self, '_gn_ratio', 1.0)
            self._gb_parts = (loss_N, loss, w)          # read by train_one_epoch to measure gradient norms
            tot = w * loss_N + (1.0 - w) * cg * loss
            self.last_terms = dict(self._tkacc, total=tot.detach(), n_syn=float(n_s), batch=float(B), c_grad=float(cg))
            return tot
        if BRANCH_BALANCE:
            # --branch-balance: normal branch L_N (Term 1) and synthetic branch L_S (every other term) on the SAME
            # scale and weighted by their share of the batch: L = w L_N + (1-w) c L_S, w = n_normal / B,
            # c = stopgrad(EMA L_N / EMA L_S) -> a synthetic sample weighs as much as a normal one, regardless of how
            # many extra terms it carries. (AdamW: the overall loss scale does not matter, only this ratio.)
            B = student_features[0].shape[0]
            n_s = (len(syn_idx) if has_syn else 0) + (len(real_anom_idx_global) if has_real_anom else 0)
            w = (B - n_s) / B
            with torch.no_grad():
                if B - n_s > 0:
                    self._bb_ema_n = 0.99 * self._bb_ema_n + 0.01 * float(loss_N) if self._bb_ema_n else float(loss_N)
                if n_s > 0:
                    self._bb_ema_s = 0.99 * self._bb_ema_s + 0.01 * float(loss) if self._bb_ema_s else float(loss)
            cs = (self._bb_ema_n / self._bb_ema_s) if (self._bb_ema_n and self._bb_ema_s) else 1.0
            self.last_branch_balance = {"w_normal": w, "c_synth": cs, "L_N": float(loss_N), "L_S": float(loss)}
            tot = w * loss_N + (1.0 - w) * cs * loss
            self.last_terms = dict(self._tkacc, total=tot.detach(), n_syn=float(n_s), batch=float(B))
            return tot
        self.last_terms = dict(self._tkacc, total=loss.detach(),
                               n_syn=float(len(syn_idx)) if has_syn else 0.0, batch=float(student_features[0].shape[0]))
        return loss


# ── Extension Model (Mode 3) ───────────────────────────────────────────
class UniNetModelExtension(UniNetModel):
    """Mode 3/4: Extension with canvas-normal L_KD+L_SC.

    During training:
      1. Standard pipeline (teachers → bottleneck → student → DFS)
      2. Teacher forward on canvas normals (detached)
      3. ExtensionLoss with all 3 terms (normal align, canvas align, syn two-sided)
      4. Optionally: anomalib's native BCE on predictions (avgpool+fc on student_features[0])
         when native_bce=True. Gradient flows back through fc → avgpool → student backbone.
         This is the "Mode 2-style image-level BCE supervision" applied to Mode 4. Default OFF.

    During eval: standard UniNet inference.
    """

    def __init__(self, *args, native_bce: bool = False, native_bce_per_stage: bool = False,
                 bce_all_normal: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        # native_bce: when True, add anomalib's per-prediction BCE term to the loss (Mode 2 style).
        # Uses the inherited self.avgpool + self.fc + self.bce_loss from UniNetModel.
        # Single BCE applied to predictions from avgpool+fc on student_features[0] (deepest stage).
        self.native_bce = native_bce
        # native_bce_per_stage: when True, BCE is applied PER STAGE — 6 BCE terms total
        # (3 stages × source/target halves). Each stage uses a LazyLinear head so dimensions
        # are detected on first forward (backbone-agnostic). Mutually exclusive with native_bce.
        self.native_bce_per_stage = native_bce_per_stage
        # bce_all_normal: when True, override BCE labels to ZERO for all samples (treats every
        # input as if it were normal). Sanity ablation to test whether globally-suppressing-anomaly
        # pressure adds anything beyond Term 2's restoration. Only meaningful with native_bce.
        self.bce_all_normal = bce_all_normal
        if native_bce_per_stage:
            # 3 LazyLinear FCs, one per stage. Lazy because channel dims differ by backbone.
            self.stage_fcs = nn.ModuleList([nn.LazyLinear(1) for _ in range(3)])

    def _compute_loss(
        self,
        student_features,
        teacher_features,
        predictions=None,
        label=None,
        mask=None,
        stop_gradient=False,
    ):
        """Feature loss + optional native BCE (predictions-based, Mode-2 style)."""
        loss = self.loss(student_features, teacher_features, mask=mask)
        if self.native_bce and predictions is not None and label is not None:
            label_target = torch.zeros_like(label.float()) if self.bce_all_normal else label.float()
            loss = loss + self.bce_loss(predictions[0], label_target) + self.bce_loss(predictions[1], label_target)
        return loss

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        canvas_normals: torch.Tensor | None = None,
        has_canvas: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.training or canvas_normals is None or has_canvas is None:
            return super().forward(images, masks, labels)

        # 1. Standard forward pipeline
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)
        student_features = self.student(bottleneck_outputs)

        # Compute native-BCE predictions BEFORE chunking (anomalib pattern: avgpool+fc on
        # the raw [2B, C, H, W] student[0], then chunk to source/target halves).
        predictions = None
        if self.native_bce and labels is not None:
            preds_pre = self.avgpool(student_features[0])
            preds_pre = torch.flatten(preds_pre, 1)
            preds_pre = self.fc(preds_pre).squeeze()
            predictions = preds_pre.chunk(dim=0, chunks=2)

        # Per-stage BCE: 3 separate avgpool+fc heads, one per stage. Each produces predictions
        # of shape [2B, 1] → chunk(2) → ([B], [B]). Stored as list-of-tuples for BCE loop later.
        predictions_per_stage = None
        if self.native_bce_per_stage and labels is not None:
            predictions_per_stage = []
            for stage_i, sf in enumerate(student_features):
                p = self.avgpool(sf)
                p = torch.flatten(p, 1)
                p = self.stage_fcs[stage_i](p).squeeze()
                predictions_per_stage.append(p.chunk(dim=0, chunks=2))

        # Chunk and rearrange (same as base UniNetModel)
        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0], student_features[1][0], student_features[2][0],
            student_features[0][1], student_features[1][1], student_features[2][1],
        ]
        student_features = self._feature_selection(source_target_features, student_features)

        # 2. Teacher forward on canvas normals (detached)
        syn_idx = has_canvas.nonzero(as_tuple=True)[0]
        canvas_teacher_features = None
        canvas_student_features = None
        bottleneck_synth = None
        bottleneck_canvas = None
        if len(syn_idx) > 0:
            with torch.no_grad():
                canvas_teacher_features, canvas_bn_in = self.teachers(canvas_normals[syn_idx])
            # Compute canvas bottleneck once; shared by proj_consistency AND bottleneck geometry.
            _need_bn_geom = getattr(self.loss, "_need_bottleneck_features", False)
            need_canvas_bn = getattr(self.loss, "proj_consistency", False) or _need_bn_geom
            canvas_bn_out = self.bottleneck(canvas_bn_in) if need_canvas_bn else None
            # Group D: also run student on canvas (with grad) for projection-consistency loss.
            if getattr(self.loss, "proj_consistency", False) and canvas_bn_out is not None:
                cs_raw = self.student(canvas_bn_out)
                cs_chunks = [d.chunk(dim=0, chunks=2) for d in cs_raw]
                canvas_student_features = [
                    cs_chunks[0][0], cs_chunks[1][0], cs_chunks[2][0],
                    cs_chunks[0][1], cs_chunks[1][1], cs_chunks[2][1],
                ]
            # Bottleneck geometry: B_s = target half of synth, B_n = target half of canvas.
            if _need_bn_geom and canvas_bn_out is not None:
                n_batch = images.shape[0]
                bottleneck_synth = bottleneck_outputs[:n_batch][syn_idx]    # [n_syn, 1024, 8, 8]
                bottleneck_canvas = canvas_bn_out[:len(syn_idx)]             # [n_syn, 1024, 8, 8]

        # 3. Prepare mask
        mask_ = None
        if masks is not None:
            mask_ = masks.float()
            if mask_.dim() == 3:
                mask_ = mask_.unsqueeze(1)

        # 4. ExtensionLoss: all 3 terms + optional native BCE
        loss = self.loss(
            student_features, source_target_features,
            canvas_teacher_features=canvas_teacher_features,
            syn_idx=syn_idx,
            mask=mask_,
            canvas_student_features=canvas_student_features,
            bottleneck_synth=bottleneck_synth,
            bottleneck_canvas=bottleneck_canvas,
        )
        if self.native_bce and predictions is not None and labels is not None:
            label_target = torch.zeros_like(labels.float()) if self.bce_all_normal else labels.float()
            loss = loss + self.bce_loss(predictions[0], label_target) + self.bce_loss(predictions[1], label_target)
        if self.native_bce_per_stage and predictions_per_stage is not None and labels is not None:
            # 6 BCE terms: 3 stages × 2 source/target. Each contributes equally.
            label_target = torch.zeros_like(labels.float()) if self.bce_all_normal else labels.float()
            for pred_chunks in predictions_per_stage:
                loss = loss + self.bce_loss(pred_chunks[0], label_target) + self.bce_loss(pred_chunks[1], label_target)
        tbce = self._teacher_bce_term(source_target_features, labels)
        if tbce is not None:
            loss = loss + tbce
        return loss

    def _teacher_bce_term(self, source_target_features, labels):
        """Auxiliary image-level BCE on the ADAPTIVE TARGET teacher features only.

        source_target_features = [s0,s1,s2, t0,t1,t2]; indices 3-5 are the trainable
        target teacher (carry grad). The frozen source teacher (0-2, no_grad) is untouched.
        Grad from this term reaches: teacher_bce_head + target_teacher. NOT the student
        (student is a downstream sibling of target_feats, never an ancestor of this logit).
        Returns weight * loss, or None when disabled.
        """
        if not getattr(self, "use_teacher_bce", False):
            return None
        if labels is None or getattr(self, "teacher_bce_weight", 0.0) == 0.0:
            return None
        layer = getattr(self, "teacher_bce_feature_layer", 0)
        tf = source_target_features[3 + layer]           # [B, C, H, W] target teacher, grad
        if getattr(self, "teacher_bce_pool", "avg") == "max":
            pooled = F.adaptive_max_pool2d(tf, (1, 1))
        else:
            pooled = F.adaptive_avg_pool2d(tf, (1, 1))
        pooled = torch.flatten(pooled, 1)                # [B, C]
        logit = self.teacher_bce_head(pooled).squeeze(-1)  # [B]
        y = labels.float()
        if getattr(self, "teacher_bce_loss_type", "bce") == "focal":
            bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            p = torch.sigmoid(logit)
            p_t = p * y + (1 - p) * (1 - y)
            gamma = getattr(self, "teacher_bce_focal_gamma", 2.0)
            loss_t = (((1 - p_t) ** gamma) * bce).mean()
        else:
            loss_t = F.binary_cross_entropy_with_logits(logit, y)
        with torch.no_grad():
            p = torch.sigmoid(logit)
            self.last_teacher_bce_stats = {
                "teacher_bce_loss": float(loss_t.item()),
                "teacher_bce_weighted": float(self.teacher_bce_weight * loss_t.item()),
                "logit_mean": float(logit.mean().item()),
                "logit_std": float(logit.std(unbiased=False).item()),
                "pos_rate": float((p > 0.5).float().mean().item()),
            }
        return self.teacher_bce_weight * loss_t

    def _recon_concat_bce_term(self, source_target_features, student_features, labels):
        """Discriminative BCE on [teacher input ⊕ student reconstruction].

        Concatenates the frozen source-teacher feature T_L (the "input"/reference) with the
        student's reconstruction S_L along channel dim, pools, and feeds a Linear head → BCE
        on the image label. Unlike native-BCE (student-only, which fights CAAO Term 2), the
        head also sees T, so it can read the T–S discrepancy without forcing S itself to look
        anomalous. Grad: head + student (via S). Source teacher is frozen (no grad through T).
        Returns weight * loss, or None when disabled.
        """
        if not getattr(self, "recon_concat_bce", False) or labels is None:
            return None
        if getattr(self, "recon_bce_weight", 0.0) == 0.0:
            return None
        L = getattr(self, "teacher_bce_feature_layer", 0)
        # Both teacher pairings (the detector sums both discrepancies):
        #   source: (T_src frozen, S_src);  target: (T_tgt, S_tgt)
        # Both teacher pairings. Source teacher is frozen (no grad regardless); the ADAPTIVE
        # target teacher DOES receive gradient here (intended — it adapts), as does the student.
        T_src = source_target_features[L]          # frozen source teacher [B,C,H,W]
        S_src = student_features[L]                # student recon (source-paired)
        T_tgt = source_target_features[3 + L]      # adaptive target teacher (receives grad)
        S_tgt = student_features[3 + L]            # student recon (target-paired)
        if getattr(self, "recon_l2", False):
            # L2-normalize each map along channel dim → magnitude-free directional features
            # (aligns head input with the cosine geometry the detector actually uses).
            T_src = F.normalize(T_src, dim=1, eps=1e-6); S_src = F.normalize(S_src, dim=1, eps=1e-6)
            T_tgt = F.normalize(T_tgt, dim=1, eps=1e-6); S_tgt = F.normalize(S_tgt, dim=1, eps=1e-6)
        htype = getattr(self, "recon_head_type", "grouped")
        if htype == "cosonly":
            # ONLY the two explicit native-res cosine-distance maps (no raw features).
            d_src = (1.0 - F.cosine_similarity(T_src, S_src, dim=1, eps=1e-6)).unsqueeze(1)
            d_tgt = (1.0 - F.cosine_similarity(T_tgt, S_tgt, dim=1, eps=1e-6)).unsqueeze(1)
            x = torch.cat([d_src, d_tgt], dim=1)                      # [B, 2, H, W]
            score_map = self.recon_bce_head(x)                        # [B, 1, H, W]
            logit = F.adaptive_max_pool2d(score_map, (1, 1)).flatten(1).squeeze(-1)
            y = labels.float()
            loss_t = F.binary_cross_entropy_with_logits(logit, y)
            with torch.no_grad():
                p = torch.sigmoid(logit)
                self.last_teacher_bce_stats = {
                    "teacher_bce_loss": float(loss_t.item()),
                    "teacher_bce_weighted": float(self.recon_bce_weight * loss_t.item()),
                    "logit_mean": float(logit.mean().item()),
                    "logit_std": float(logit.std(unbiased=False).item()),
                    "pos_rate": float((p > 0.5).float().mean().item()),
                }
            return self.recon_bce_weight * loss_t
        def _bce_ret(logit):
            y = labels.float(); loss_t = F.binary_cross_entropy_with_logits(logit, y)
            with torch.no_grad():
                p = torch.sigmoid(logit)
                self.last_teacher_bce_stats = {"teacher_bce_loss": float(loss_t.item()),
                    "teacher_bce_weighted": float(self.recon_bce_weight*loss_t.item()),
                    "logit_mean": float(logit.mean().item()), "logit_std": float(logit.std(unbiased=False).item()),
                    "pos_rate": float((p>0.5).float().mean().item())}
            return self.recon_bce_weight * loss_t
        def _interleave(use_cos):
            B, C, H, W = T_src.shape
            maps = [T_src, S_src, T_tgt, S_tgt]
            if use_cos:
                maps += [(1.0 - F.cosine_similarity(T_src, S_src, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W),
                         (1.0 - F.cosine_similarity(T_tgt, S_tgt, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)]
            return torch.stack(maps, dim=2).reshape(B, len(maps) * C, H, W)
        if htype in ("f_mix_lin",):
            # F input, mix→1 map, learned spatial Linear (clean MAX↔LIN test vs grouped/F-MIX-MAX).
            smap = self.recon_bce_head(_interleave(False))      # [B,1,H,W]
            return _bce_ret(self.recon_lin(smap.flatten(1)).squeeze(-1))
        if htype in ("fc_per_gridmax",):
            # FC, per-channel grouped → LOCAL max-pool to 8×8 grid → flatten → Linear (regional, not 1 spike).
            feat = self.recon_bce_head(_interleave(True))       # [B,C,H,W]
            g = F.adaptive_max_pool2d(feat, (8, 8)).flatten(1)  # [B, C*64]
            return _bce_ret(self.recon_grid_lin(g).squeeze(-1))
        if htype in ("fc_mix_gridmax",):
            # FC, mix→1 map → LOCAL max-pool to 8×8 grid → flatten → Linear.
            smap = self.recon_bce_head(_interleave(True))       # [B,1,H,W]
            g = F.adaptive_max_pool2d(smap, (8, 8)).flatten(1)  # [B, 64]
            return _bce_ret(self.recon_grid_lin(g).squeeze(-1))
        if htype == "tsrd_dilated":
            # TSRD-dilated: per branch the 4 views [T, S, R=|T-S|, D=(T-S) signed]; concat both
            # pairings → 8C channels → dilated conv stack (multi-scale RF) → 1-ch score map →
            # spatial max. Port of the gate's l2_TSRD_dilated champion into the recon-BCE head.
            def _tsrd(T, S):
                return [T, S, (T - S).abs(), (T - S)]
            x = torch.cat(_tsrd(T_src, S_src) + _tsrd(T_tgt, S_tgt), dim=1)  # [B, 8C, H, W]
            score_map = self.recon_bce_head(x)                              # [B, 1, H, W]
            logit = F.adaptive_max_pool2d(score_map, (1, 1)).flatten(1).squeeze(-1)
            return _bce_ret(logit)
        if htype in ("grouped_pmax_cos", "grouped_pmax_cos_avg"):
            # Like grouped_pmax, but each group ALSO gets the 2 cosine-distance maps (broadcast
            # into every group): per channel c → [T_src_c, S_src_c, T_tgt_c, S_tgt_c, d_src, d_tgt].
            B, C, H, W = T_src.shape
            d_src = (1.0 - F.cosine_similarity(T_src, S_src, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)
            d_tgt = (1.0 - F.cosine_similarity(T_tgt, S_tgt, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)
            x = torch.stack([T_src, S_src, T_tgt, S_tgt, d_src, d_tgt], dim=2).reshape(B, 6 * C, H, W)
            feat = self.recon_bce_head(x)                              # grouped conv (6C→C, groups=C)
            pool = F.adaptive_avg_pool2d if htype.endswith("_avg") else F.adaptive_max_pool2d
            peaks = pool(feat, (1, 1)).flatten(1)                      # [B, C]
            logit = self.recon_pmax_linear(peaks).squeeze(-1)
            y = labels.float(); loss_t = F.binary_cross_entropy_with_logits(logit, y)
            with torch.no_grad():
                p = torch.sigmoid(logit)
                self.last_teacher_bce_stats = {"teacher_bce_loss": float(loss_t.item()),
                    "teacher_bce_weighted": float(self.recon_bce_weight*loss_t.item()),
                    "logit_mean": float(logit.mean().item()), "logit_std": float(logit.std(unbiased=False).item()),
                    "pos_rate": float((p>0.5).float().mean().item())}
            return self.recon_bce_weight * loss_t
        if htype == "fc_mix_lin":
            # FC-MIX-LIN: feats+cos per group → grouped conv → mix to 1 map → flatten →
            # learned spatial Linear(H*W→1) (no max; learns WHERE anomalies tend to be).
            B, C, H, W = T_src.shape
            d_src = (1.0 - F.cosine_similarity(T_src, S_src, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)
            d_tgt = (1.0 - F.cosine_similarity(T_tgt, S_tgt, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)
            x = torch.stack([T_src, S_src, T_tgt, S_tgt, d_src, d_tgt], dim=2).reshape(B, 6 * C, H, W)
            smap = self.recon_bce_head(x)                    # [B,1,H,W]
            logit = self.recon_lin(smap.flatten(1)).squeeze(-1)
            y = labels.float(); loss_t = F.binary_cross_entropy_with_logits(logit, y)
            with torch.no_grad():
                p = torch.sigmoid(logit)
                self.last_teacher_bce_stats = {"teacher_bce_loss": float(loss_t.item()),
                    "teacher_bce_weighted": float(self.recon_bce_weight*loss_t.item()),
                    "logit_mean": float(logit.mean().item()), "logit_std": float(logit.std(unbiased=False).item()),
                    "pos_rate": float((p>0.5).float().mean().item())}
            return self.recon_bce_weight * loss_t
        if htype == "grouped_cosrep":
            # 6 views per group (4 features + 2 broadcast cosine maps), grouped conv → mix → single max.
            B, C, H, W = T_src.shape
            d_src = (1.0 - F.cosine_similarity(T_src, S_src, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)
            d_tgt = (1.0 - F.cosine_similarity(T_tgt, S_tgt, dim=1, eps=1e-6)).unsqueeze(1).expand(B, C, H, W)
            x = torch.stack([T_src, S_src, T_tgt, S_tgt, d_src, d_tgt], dim=2).reshape(B, 6 * C, H, W)
        elif htype in ("grouped", "grouped_pmax", "grouped_cos"):
            # Interleave so channel c's 4 views are adjacent → grouped 1×1 conv (groups=C)
            # combines ONLY the 4 views per channel (learned per-channel affine over the 4 maps).
            B, C, H, W = T_src.shape
            x = torch.stack([T_src, S_src, T_tgt, S_tgt], dim=2).reshape(B, 4 * C, H, W)
        else:
            x = torch.cat([T_src, S_src, T_tgt, S_tgt], dim=1)   # [B, 4C, H, W]
        if htype == "grouped_pmax":
            # grouped conv → [B,C,H,W] per-channel maps; spatial-max EACH channel independently
            # (channels may peak at different locations), then Linear over the C peaks → logit.
            feat = self.recon_bce_head(x)                              # [B, C, H, W]
            peaks = F.adaptive_max_pool2d(feat, (1, 1)).flatten(1)     # [B, C]
            logit = self.recon_pmax_linear(peaks).squeeze(-1)          # [B]
        elif htype == "grouped_cos":
            # grouped per-channel features + EXPLICIT native-res cosine-distance maps per branch
            # (1-cos(T_src,S_src), 1-cos(T_tgt,S_tgt)) concatenated → final 1×1 mix → max.
            feat = self.recon_bce_head(x)                              # [B, C, H, W]
            d_src = (1.0 - F.cosine_similarity(T_src, S_src, dim=1, eps=1e-6)).unsqueeze(1)  # [B,1,H,W]
            d_tgt = (1.0 - F.cosine_similarity(T_tgt, S_tgt, dim=1, eps=1e-6)).unsqueeze(1)
            mixed = torch.cat([feat, d_src, d_tgt], dim=1)            # [B, C+2, H, W]
            score_map = self.recon_cos_mix(mixed)                     # [B, 1, H, W]
            logit = F.adaptive_max_pool2d(score_map, (1, 1)).flatten(1).squeeze(-1)
        else:
            # head → 1-ch score map → single spatial MAX (evidence must co-localize)
            score_map = self.recon_bce_head(x)     # [B, 1, H, W]
            logit = F.adaptive_max_pool2d(score_map, (1, 1)).flatten(1).squeeze(-1)  # [B]
        y = labels.float()
        loss_t = F.binary_cross_entropy_with_logits(logit, y)
        with torch.no_grad():
            p = torch.sigmoid(logit)
            self.last_teacher_bce_stats = {
                "teacher_bce_loss": float(loss_t.item()),
                "teacher_bce_weighted": float(self.recon_bce_weight * loss_t.item()),
                "logit_mean": float(logit.mean().item()),
                "logit_std": float(logit.std(unbiased=False).item()),
                "pos_rate": float((p > 0.5).float().mean().item()),
            }
        return self.recon_bce_weight * loss_t

    def _teacher_bce_real_only_term(self, source_target_features, labels, real_images, syn_idx):
        """Real-anomaly-counterpart teacher-BCE. BCE positives = REAL anomaly counterparts of
        the in-batch synthetic samples (run through the teacher here); negatives = normals in
        the batch. Synthetic samples are EXCLUDED from this BCE (they only do restoration).

        Real images NEVER touch the restoration/student pathway — they only feed the teacher
        (target-teacher features) + the BCE head. Grad: head + target_teacher. Frozen source
        teacher and student get nothing.
        Returns weight * loss, or None when disabled/unavailable.
        """
        if not getattr(self, "use_teacher_bce", False) or labels is None:
            return None
        if getattr(self, "teacher_bce_weight", 0.0) == 0.0:
            return None
        if real_images is None or syn_idx is None or len(syn_idx) == 0:
            return None
        layer = getattr(self, "teacher_bce_feature_layer", 0)
        y = labels.float()
        normal_idx = (y == 0).nonzero(as_tuple=True)[0]
        if len(normal_idx) == 0:
            return None

        def pool_logit(feat):
            if getattr(self, "teacher_bce_pool", "avg") == "max":
                p = F.adaptive_max_pool2d(feat, (1, 1))
            else:
                p = F.adaptive_avg_pool2d(feat, (1, 1))
            return self.teacher_bce_head(torch.flatten(p, 1)).squeeze(-1)

        # Negatives: normals' target-teacher features (already computed, grad to target teacher)
        neg_feat = source_target_features[3 + layer][normal_idx]
        neg_logit = pool_logit(neg_feat)
        # Positives: REAL counterparts of the in-batch synth samples -> teacher forward (grad)
        real_batch = real_images[syn_idx]
        real_stf, _ = self.teachers(real_batch)        # source(no_grad)+target(grad)
        pos_feat = real_stf[3 + layer]
        pos_logit = pool_logit(pos_feat)

        logit = torch.cat([neg_logit, pos_logit], dim=0)
        tgt = torch.cat([torch.zeros_like(neg_logit), torch.ones_like(pos_logit)], dim=0)
        if getattr(self, "teacher_bce_loss_type", "bce") == "focal":
            bce = F.binary_cross_entropy_with_logits(logit, tgt, reduction="none")
            p = torch.sigmoid(logit); p_t = p * tgt + (1 - p) * (1 - tgt)
            gamma = getattr(self, "teacher_bce_focal_gamma", 2.0)
            loss_t = (((1 - p_t) ** gamma) * bce).mean()
        else:
            loss_t = F.binary_cross_entropy_with_logits(logit, tgt)
        with torch.no_grad():
            p = torch.sigmoid(logit)
            self.last_teacher_bce_stats = {
                "teacher_bce_loss": float(loss_t.item()),
                "teacher_bce_weighted": float(self.teacher_bce_weight * loss_t.item()),
                "logit_mean": float(logit.mean().item()),
                "logit_std": float(logit.std(unbiased=False).item()),
                "pos_rate": float((p > 0.5).float().mean().item()),
            }
        return self.teacher_bce_weight * loss_t


class UniNetModelExtensionDFSAblate(UniNetModelExtension):
    """Variant of UniNetModelExtension exposing toggles for DFS at training/inference.

    Default flags (train_apply_dfs=True, eval_apply_dfs=False) reproduce
    UniNetModelExtension's behavior bit-identically (DFS at train, none at eval).

    Flags:
      train_apply_dfs: if False, skip the _feature_selection (DFS) call at training.
      eval_apply_dfs:  if True,  apply _feature_selection (DFS) at inference before cosine.
    """
    # 9 inference aggregations: 3 teacher-handling × 3 cross-stage methods.
    #
    # Teacher-handling axis (how to combine the 2 teachers):
    #   - "standard": no reduction → 6 maps (3 layers × 2 teachers)
    #   - "min":      element-wise min(T1, T2) per layer → 3 consensus maps (AND filter)
    #   - "max":      element-wise max(T1, T2) per layer → 3 OR-per-layer maps (loud teacher wins)
    #
    # Cross-stage axis (how to combine the N maps into 1 image score + 1 anomaly map):
    #   - "sum_max":         sum upsampled stages → σ=4 Gaussian blur → max of map
    #                        (matches anomalib's weighted_decision_mechanism for the score)
    #   - "softmax_max_mix": softmax over per-stage maxes → linear-combine upsampled stages → max of map
    #   - "paper":           sum upsampled → blur → adaptive top-k MEAN with α=0.01, β=3e-5
    #                        (paper-faithful, top-k varies with stage confidence ~2 to ~400 pixels)
    ALL_AGGS = (
        "standard_paper", "standard_sum_max", "standard_paper_inverted",
    )

    def __init__(self, *args, train_apply_dfs: bool = True, eval_apply_dfs: bool = False,
                 infer_aggs=None, lambda_idem: float = 0.0, idem_caao: bool = False,
                 no_detach_any: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_apply_dfs = train_apply_dfs
        self.eval_apply_dfs = eval_apply_dfs
        self.lambda_idem = float(lambda_idem)
        self.idem_caao = bool(idem_caao)
        self.no_detach_any = bool(no_detach_any)
        # infer_aggs: list of aggregation names to compute at inference (forward returns dict if >1).
        # See ALL_AGGS docstring above for the 9-grid (teacher × cross-stage).
        if infer_aggs is None:
            infer_aggs = list(self.ALL_AGGS)
        if isinstance(infer_aggs, str):
            infer_aggs = [infer_aggs]
        for a in infer_aggs:
            assert a in self.ALL_AGGS, f"Unknown agg: {a}. Valid: {self.ALL_AGGS}"
        assert len(infer_aggs) > 0, "Need at least one aggregation"
        self.infer_aggs = list(infer_aggs)

    def _compute_output_list(self, source_target_features, student_features):
        """Per-stage 1 - cos(student, teacher) distance maps. Shared between aggregation modes."""
        output_list = []
        for tf, sf in zip(source_target_features, student_features, strict=True):
            output_list.append(1 - F.cosine_similarity(tf, sf))
        return output_list

    @staticmethod
    def _reduce_teachers(output_list, mode):
        """Reduce 6 maps (3 layers × 2 teachers) → N maps based on teacher-handling mode.

        Pairing convention (set in forward(), line ~1414): output_list ordering is
        [T1_l1, T1_l2, T1_l3, T2_l1, T2_l2, T2_l3], so layer_i pair = (output_list[i], output_list[i+3]).

        Returns:
            "standard" → list of 6 maps (passes through unchanged)
            "min"      → list of 3 element-wise min(T1, T2) per layer (AND consensus)
            "max"      → list of 3 element-wise max(T1, T2) per layer (loud-teacher-wins)
        """
        if mode == "standard":
            return list(output_list)
        n = len(output_list) // 2
        op = torch.min if mode == "min" else torch.max
        return [op(output_list[i], output_list[i + n]) for i in range(n)]

    @staticmethod
    def _xstage_sum_max(maps, output_size, batch_size):
        """Sum upsampled stages → σ=4 Gaussian blur → max of map.

        Matches anomalib's weighted_decision_mechanism scoring (top_k_values[0] = max).
        Returns (anomaly_score, anomaly_map).
        """
        import einops
        from anomalib.models.components.filters import GaussianBlur2d
        device = maps[0].device
        ks = BLUR_KERNEL_SIZE
        gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(ks, ks), channels=1).to(device)

        anomaly_map = torch.zeros(batch_size, *output_size, device=device)
        for d in maps:
            up = F.interpolate(d.unsqueeze(1), size=output_size, mode="bilinear", align_corners=True).squeeze(1)
            anomaly_map = anomaly_map + up

        scores = torch.zeros(batch_size, device=device)
        for idx in range(batch_size):
            single = anomaly_map[idx]
            blurred = gaussian_blur(einops.rearrange(single, "h w -> 1 1 h w")).squeeze()
            scores[idx] = blurred.view(-1).max()
        return scores.unsqueeze(1), anomaly_map

    @staticmethod
    def _xstage_softmax_max_mix(maps, output_size, batch_size):
        """Softmax over per-stage maxes → linear-combine upsampled stages → max of combined map."""
        stage_maxes = torch.stack([d.flatten(1).max(dim=1).values for d in maps], dim=1)  # [B, N]
        weights = F.softmax(stage_maxes, dim=1)
        upsampled = [F.interpolate(d.unsqueeze(1), size=output_size, mode="bilinear", align_corners=True).squeeze(1)
                     for d in maps]
        stacked = torch.stack(upsampled, dim=1)
        w = weights.unsqueeze(-1).unsqueeze(-1)
        anomaly_map = (stacked * w).sum(dim=1)
        anomaly_score = anomaly_map.flatten(1).max(dim=1).values.unsqueeze(-1)
        return anomaly_score, anomaly_map

    @staticmethod
    def _xstage_paper(maps, output_size, batch_size):
        """Paper-faithful: sum upsampled → blur → adaptive top-k MEAN. α=0.01, β=3e-5."""
        import einops
        from anomalib.models.components.filters import GaussianBlur2d
        device = maps[0].device
        alpha, beta = 0.01, 3e-05
        ks = BLUR_KERNEL_SIZE
        gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(ks, ks), channels=1).to(device)

        total_weights = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            max_values = torch.stack([torch.max(d[i]) for d in maps])
            probs = F.softmax(max_values, dim=0)
            prob_mean = probs.mean()
            mask = probs > prob_mean
            if mask.any():
                w = torch.max(torch.stack([
                    max_values[mask].mean() * alpha,
                    torch.tensor(beta, device=device),
                ]))
            else:
                w = torch.tensor(beta, device=device)
            total_weights[i] = w

        anomaly_map = torch.zeros(batch_size, *output_size, device=device)
        for d in maps:
            up = F.interpolate(d.unsqueeze(1), size=output_size, mode="bilinear", align_corners=True).squeeze(1)
            anomaly_map = anomaly_map + up

        scores = torch.zeros(batch_size, device=device)
        for idx in range(batch_size):
            top_k = max(int(output_size[0] * output_size[1] * total_weights[idx]), 1)
            single = anomaly_map[idx]
            blurred = gaussian_blur(einops.rearrange(single, "h w -> 1 1 h w")).squeeze()
            scores[idx] = blurred.view(-1).topk(top_k).values.mean()
        return scores.unsqueeze(1), anomaly_map

    @staticmethod
    def _xstage_paper_inverted(maps, output_size, batch_size):
        """Inverted paper: HIGH stage_max → NARROW top_k. LOW stage_max → WIDE top_k.

        Tests the opposite hypothesis to paper: when model is confident the anomaly is
        a sharp peak, we should NOT broaden the pool (would dilute with background).
        Same summed-blurred map as paper, only top_k differs.
        """
        import einops
        from anomalib.models.components.filters import GaussianBlur2d
        device = maps[0].device
        alpha, beta = 0.01, 3e-05
        ks = BLUR_KERNEL_SIZE
        gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(ks, ks), channels=1).to(device)

        total_weights = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            max_values = torch.stack([torch.max(d[i]) for d in maps])
            probs = F.softmax(max_values, dim=0)
            prob_mean = probs.mean()
            mask = probs > prob_mean
            if mask.any():
                inverted = torch.clamp(1.0 - max_values[mask].mean(), min=0.0)
                w = torch.max(torch.stack([
                    inverted * alpha,
                    torch.tensor(beta, device=device),
                ]))
            else:
                w = torch.tensor(alpha, device=device)
            total_weights[i] = w

        anomaly_map = torch.zeros(batch_size, *output_size, device=device)
        for d in maps:
            up = F.interpolate(d.unsqueeze(1), size=output_size, mode="bilinear", align_corners=True).squeeze(1)
            anomaly_map = anomaly_map + up

        scores = torch.zeros(batch_size, device=device)
        for idx in range(batch_size):
            top_k = max(int(output_size[0] * output_size[1] * total_weights[idx]), 1)
            single = anomaly_map[idx]
            blurred = gaussian_blur(einops.rearrange(single, "h w -> 1 1 h w")).squeeze()
            scores[idx] = blurred.view(-1).topk(top_k).values.mean()
        return scores.unsqueeze(1), anomaly_map

    _XSTAGE_FNS = {
        "sum_max": "_xstage_sum_max",
        "softmax_max_mix": "_xstage_softmax_max_mix",
        "paper": "_xstage_paper",
        "paper_inverted": "_xstage_paper_inverted",
    }

    @classmethod
    def _parse_agg_name(cls, name):
        """Parse '<teacher>_<xstage>' → ('standard'|'min'|'max', xstage_name)."""
        for teacher_mode in ("standard", "min", "max"):
            prefix = teacher_mode + "_"
            if name.startswith(prefix):
                xstage = name[len(prefix):]
                if xstage in cls._XSTAGE_FNS:
                    return teacher_mode, xstage
        raise ValueError(f"Unknown aggregation: {name!r}. Expected <standard|min|max>_<sum_max|softmax_max_mix|paper>")

    def _dispatch_agg(self, agg_name, output_list, output_size, batch_size):
        """Apply teacher reduction + cross-stage aggregation. Returns (score, anomaly_map)."""
        teacher_mode, xstage = self._parse_agg_name(agg_name)
        reduced = self._reduce_teachers(output_list, teacher_mode)
        fn = getattr(self, self._XSTAGE_FNS[xstage])
        return fn(reduced, output_size, batch_size)

    def forward(self, images, masks=None, labels=None, canvas_normals=None, has_canvas=None,
                real_images=None):
        # ── INFERENCE path
        if not self.training:
            from anomalib.data import InferenceBatch
            with torch.no_grad():
                source_target_features, bottleneck_inputs = self.teachers(images)
                bottleneck_outputs = self.bottleneck(bottleneck_inputs)
                student_features = self.student(bottleneck_outputs)
                student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
                student_features = [
                    student_features[0][0], student_features[1][0], student_features[2][0],
                    student_features[0][1], student_features[1][1], student_features[2][1],
                ]
                if self.eval_apply_dfs:
                    student_features = self._feature_selection(source_target_features, student_features)
                output_list = self._compute_output_list(source_target_features, student_features)

                output_size = images.shape[-2:]
                B = images.shape[0]
                results = {}
                for agg in self.infer_aggs:
                    score, amap = self._dispatch_agg(agg, output_list, output_size, B)
                    results[agg] = InferenceBatch(pred_score=score, anomaly_map=amap)
            if len(self.infer_aggs) == 1:
                return results[self.infer_aggs[0]]
            return results

        # ── TRAINING path
        if canvas_normals is None or has_canvas is None:
            # Falls into base UniNetModel.forward training path
            return super().forward(images, masks, labels)

        if self.train_apply_dfs:
            # Default: delegate to parent (which applies DFS)
            return super().forward(images, masks, labels, canvas_normals, has_canvas)

        # Replicate parent's training path WITHOUT DFS
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)
        student_features = self.student(bottleneck_outputs)
        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0], student_features[1][0], student_features[2][0],
            student_features[0][1], student_features[1][1], student_features[2][1],
        ]
        # SKIP DFS — student_features = self._feature_selection(...) intentionally omitted

        syn_idx = has_canvas.nonzero(as_tuple=True)[0]
        canvas_teacher_features = None
        canvas_student_features = None
        if len(syn_idx) > 0:
            if self.no_detach_any:
                canvas_teacher_features, canvas_bn_in = self.teachers(canvas_normals[syn_idx])
            else:
                with torch.no_grad():
                    canvas_teacher_features, canvas_bn_in = self.teachers(canvas_normals[syn_idx])
            if getattr(self.loss, "proj_consistency", False) or self.lambda_idem > 0:
                canvas_bn_out = self.bottleneck(canvas_bn_in)
                cs_raw = self.student(canvas_bn_out)
                cs_chunks = [d.chunk(dim=0, chunks=2) for d in cs_raw]
                canvas_student_features = [
                    cs_chunks[0][0], cs_chunks[1][0], cs_chunks[2][0],
                    cs_chunks[0][1], cs_chunks[1][1], cs_chunks[2][1],
                ]
        mask_ = None
        if masks is not None:
            mask_ = masks.float()
            if mask_.dim() == 3:
                mask_ = mask_.unsqueeze(1)
        loss = self.loss(
            student_features, source_target_features,
            canvas_teacher_features=canvas_teacher_features,
            syn_idx=syn_idx,
            mask=mask_,
            canvas_student_features=canvas_student_features,
        )

        if self.lambda_idem > 0 and canvas_student_features is not None and len(syn_idx) > 0:
            # Idempotency: S(B(S(B(T(x^s))))) ≈ stopgrad(S(B(T(x^c)))))
            # Feed synth student output back through bottleneck+decoder (dual pass),
            # align with student-on-canvas at anomaly pixels only.
            syn_sf = [student_features[i][syn_idx] for i in range(6)]
            idem_bn_in = [
                torch.cat([syn_sf[i + 3], syn_sf[i]], dim=0)
                for i in range(3)
            ]
            idem_bn_out = self.bottleneck(idem_bn_in)
            idem_raw = self.student(idem_bn_out)
            idem_chunks = [d.chunk(dim=0, chunks=2) for d in idem_raw]
            idem_sf = [
                idem_chunks[0][0], idem_chunks[1][0], idem_chunks[2][0],
                idem_chunks[0][1], idem_chunks[1][1], idem_chunks[2][1],
            ]
            syn_mask = mask_[syn_idx] if mask_ is not None else None
            idem_loss = torch.tensor(0.0, device=loss.device)
            n_syn = len(syn_idx)
            for i in range(6):
                sf_dual = idem_sf[i]                          # [n_syn, c, h, w]
                sf_canvas = canvas_student_features[i]        # [n_syn, c, h, w]
                c, h, w = sf_dual.shape[1], sf_dual.shape[2], sf_dual.shape[3]
                sf1 = F.normalize(sf_dual.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)
                sf_canvas_t = sf_canvas if self.no_detach_any else sf_canvas.detach()
                sf2 = F.normalize(sf_canvas_t.view(n_syn, c, -1).transpose(1, 2), p=2, dim=2)
                cos_dist = 1 - F.cosine_similarity(sf1, sf2, dim=2)  # [n_syn, h*w]
                if self.idem_caao and syn_mask is not None:
                    m_down = F.max_pool2d(syn_mask, kernel_size=syn_mask.shape[-1] // h)
                    anom = (m_down.view(n_syn, -1) > 0.5)
                    if anom.any():
                        idem_loss = idem_loss + cos_dist[anom].mean()
                    else:
                        idem_loss = idem_loss + cos_dist.mean()
                else:
                    idem_loss = idem_loss + cos_dist.mean()
            loss = loss + self.lambda_idem * idem_loss

        if getattr(self, "recon_concat_bce", False):
            tbce = self._recon_concat_bce_term(source_target_features, student_features, labels)
        elif getattr(self, "teacher_bce_real_only", False):
            tbce = self._teacher_bce_real_only_term(source_target_features, labels, real_images, syn_idx)
        else:
            tbce = self._teacher_bce_term(source_target_features, labels)
        if tbce is not None:
            loss = loss + tbce

        return loss


# ── Learned anomaly score head ─────────────────────────────────────────
class AnomalyScoreHead(nn.Module):
    """Learned image-level anomaly score from per-layer distance maps.

    Replaces the handcrafted max(blurred_map) with a small CNN + attention pooling.

    Args:
        in_channels: 1 (summed maps) or 6 (per-layer maps).
        dropout: Dropout probability (default 0.5).
    """

    def __init__(self, in_channels: int = 6, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.drop = nn.Dropout2d(dropout)
        self.attn_conv = nn.Conv2d(32, 1, 1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, 64, 64] → [B, 1] score."""
        x = self.drop(F.relu(self.conv1(x)))   # [B, 16, 32, 32]
        x = self.drop(F.relu(self.conv2(x)))   # [B, 32, 16, 16]
        # Attention pooling
        attn = torch.sigmoid(self.attn_conv(x))  # [B, 1, 16, 16]
        weighted = x * attn                       # [B, 32, 16, 16]
        pooled = weighted.sum(dim=(2, 3)) / (attn.sum(dim=(2, 3)) + 1e-6)  # [B, 32]
        return self.fc(pooled)                    # [B, 1]


class StatisticsHead(nn.Module):
    """5-parameter statistics head for image-level anomaly scoring.

    Computes [max, mean, std, kurtosis, skewness] of the summed distance map,
    then a single Linear(5, 1) layer. Cannot overfit 50 samples with 5 weights.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 64, 64] (summed distance map) → [B, 1] score."""
        flat = x.view(x.shape[0], -1)  # [B, 4096]
        max_val = flat.max(dim=1).values
        mean_val = flat.mean(dim=1)
        std_val = flat.std(dim=1)
        centered = flat - mean_val.unsqueeze(1)
        var = (centered ** 2).mean(dim=1)
        safe_std = std_val.clamp(min=1e-6)
        skewness = (centered ** 3).mean(dim=1) / (safe_std ** 3)
        kurtosis = (centered ** 4).mean(dim=1) / (safe_std ** 4) - 3.0  # excess kurtosis
        stats = torch.stack([max_val, mean_val, std_val, kurtosis, skewness], dim=1)  # [B, 5]
        return self.fc(stats)  # [B, 1]


class TinySpatialHead(nn.Module):
    """Tiny spatial head: training-time BCE scaffold on a [B,1,64,64] distance map.

    Architecture (hidden=8 fixed):
        Conv2d(1,8,3) → GN(1,8) → SiLU → Conv2d(8,8,3, groups=8) → GN(1,8) → SiLU → Conv2d(8,1,1) → AdaptiveMaxPool(1,1)

    Rationale: small enough to avoid becoming a synthetic-artifact classifier;
    spatial enough to test whether BCE needs local map structure beyond global
    statistics. AdaptiveMaxPool reflects the AD prior that one strong region
    suffices for image-level classification. ~250 params.

    NOT used at inference — purely a training-time BCE gradient scaffold.
    Returns [B] logits (squeezed).
    """

    def __init__(self):
        super().__init__()
        hidden = 8
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return self.pool(z).flatten(1).squeeze(1)  # [B]


def _topk_mean_score(A_sum: torch.Tensor, frac: float = 0.01) -> torch.Tensor:
    """Differentiable scalar image score: mean of the top-`frac` highest pixels of A_sum.

    Used by B9 (direct_score_bce) and B10 (direct_score_rank) as a fallback for the
    inference-time paper-correct top-k aggregation (which uses an adaptive k derived
    from stage softmax + Gaussian blur on a 256x256 map). Here we operate on the raw
    64x64 A_sum without blur — k = max(1, int(frac * 4096)) ≈ 40 pixels at frac=0.01.

    A_sum: [B, 1, H, W] -> scores: [B]. Gradient flows through topk.
    """
    B = A_sum.shape[0]
    flat = A_sum.view(B, -1)  # [B, H*W]
    k = max(1, int(frac * flat.shape[-1]))
    topk_vals, _ = flat.topk(k, dim=-1)  # [B, k]
    return topk_vals.mean(dim=-1)  # [B]


# ── Learned anomaly map aggregation decoders ─────────────────────────────

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft dice loss. pred/target: [B, 1, H, W] in [0, 1]."""
    pred_flat = pred.view(pred.shape[0], -1)
    target_flat = target.view(target.shape[0], -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)).mean()


def combo_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0,
               bce_weight: float = 0.5) -> torch.Tensor:
    """Dice + BCE combo on logits. BCE with logits has better gradient flow than sigmoid + BCE."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    pred = torch.sigmoid(logits)
    d = dice_loss(pred, target, smooth)
    return bce_weight * bce + (1 - bce_weight) * d


class LearnedWeights(nn.Module):
    """Minimal: 6 learned scalar weights for per-layer aggregation. 6 parameters total."""

    def __init__(self, n_layers: int = 6):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_layers) / n_layers)

    def forward(self, distance_maps: list[torch.Tensor], target_size: int = 256) -> torch.Tensor:
        """distance_maps: list of 6 [H_i, W_i] tensors → [1, 1, target_size, target_size]."""
        w = torch.softmax(self.weights, dim=0)
        dev = distance_maps[0].device
        out = torch.zeros(1, 1, target_size, target_size, device=dev)
        for i, dm in enumerate(distance_maps):
            up = F.interpolate(dm[None, None], size=target_size,
                               mode="bilinear", align_corners=True)
            out = out + w[i] * up
        return out  # logits, no sigmoid


class ConcatDecoder(nn.Module):
    """U-Net-style: project each scale, merge bottom-up with concat + conv.

    16×16 (2 maps) → project+concat → upsample → merge with 32×32 (2 maps)
    → upsample → merge with 64×64 (2 maps) → upsample to 256×256 → 1ch output.
    """

    def __init__(self, ch: int = 16):
        super().__init__()
        # Per-scale projections: 1ch distance map → ch channels
        self.proj_16 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.proj_32 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.proj_64 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True))

        # Merge blocks: concat(upsampled_coarse, fine) → conv → ch
        self.merge_32 = nn.Sequential(
            nn.Conv2d(ch + ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.merge_64 = nn.Sequential(
            nn.Conv2d(ch + ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )

        # Final: 64→256 via bilinear + 1×1 conv
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, distance_maps: list[torch.Tensor], target_size: int = 256) -> torch.Tensor:
        """distance_maps: 6 tensors at [H_i, W_i]. Returns [1,1,256,256]."""
        # Group by resolution: source [0,1,2] + target [3,4,5]
        # [0],[3] = 64×64;  [1],[4] = 32×32;  [2],[5] = 16×16
        d64 = torch.stack([distance_maps[0], distance_maps[3]], dim=0).unsqueeze(0)  # [1, 2, 64, 64]
        d32 = torch.stack([distance_maps[1], distance_maps[4]], dim=0).unsqueeze(0)  # [1, 2, 32, 32]
        d16 = torch.stack([distance_maps[2], distance_maps[5]], dim=0).unsqueeze(0)  # [1, 2, 16, 16]

        # Bottom-up
        x16 = self.proj_16(d16)                                          # [1, ch, 16, 16]
        x16_up = F.interpolate(x16, size=32, mode="bilinear", align_corners=True)
        x32 = self.merge_32(torch.cat([x16_up, self.proj_32(d32)], dim=1))  # [1, ch, 32, 32]
        x32_up = F.interpolate(x32, size=64, mode="bilinear", align_corners=True)
        x64 = self.merge_64(torch.cat([x32_up, self.proj_64(d64)], dim=1))  # [1, ch, 64, 64]

        x_full = F.interpolate(x64, size=target_size, mode="bilinear", align_corners=True)
        return self.head(x_full)  # logits, no sigmoid


class ResidualDecoder(nn.Module):
    """Residual variant: add instead of concat at each merge point."""

    def __init__(self, ch: int = 16):
        super().__init__()
        self.proj_16 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.proj_32 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.proj_64 = nn.Sequential(nn.Conv2d(2, ch, 3, padding=1), nn.ReLU(inplace=True))

        self.refine_32 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.refine_64 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1),
        )

    def forward(self, distance_maps: list[torch.Tensor], target_size: int = 256) -> torch.Tensor:
        d64 = torch.stack([distance_maps[0], distance_maps[3]], dim=0).unsqueeze(0)
        d32 = torch.stack([distance_maps[1], distance_maps[4]], dim=0).unsqueeze(0)
        d16 = torch.stack([distance_maps[2], distance_maps[5]], dim=0).unsqueeze(0)

        x16 = self.proj_16(d16)
        x16_up = F.interpolate(x16, size=32, mode="bilinear", align_corners=True)
        x32 = self.proj_32(d32) + x16_up                              # residual merge
        x32 = x32 + self.refine_32(x32)                               # residual refine
        x32_up = F.interpolate(x32, size=64, mode="bilinear", align_corners=True)
        x64 = self.proj_64(d64) + x32_up                              # residual merge
        x64 = x64 + self.refine_64(x64)                               # residual refine

        x_full = F.interpolate(x64, size=target_size, mode="bilinear", align_corners=True)
        return self.head(x_full)  # logits, no sigmoid


def _compute_distance_maps(teacher_features, student_features, return_native_mask: bool = False):
    """Compute per-layer distance maps, upsampled to 64×64.

    Args:
        teacher_features: list of 6 tensors from UniNet source_target_features.
        student_features: list of 6 tensors (chunked student features).
        return_native_mask: if True, also return a list of bool flags indicating
            whether each input map was already native 64×64 (no upsampling applied).

    Returns:
        stacked: [B, 6, 64, 64] tensor of distance maps.
        summed:  [B, 1, 64, 64] tensor of summed distance maps.
        (optional) native_mask: list of 6 bools, True = map was native 64×64.
    """
    maps = []
    native_mask = []
    for tf, sf in zip(teacher_features, student_features):
        d = 1 - F.cosine_similarity(tf, sf)  # [B, H, W]
        d = d.unsqueeze(1)                    # [B, 1, H, W]
        is_native = (d.shape[-1] == 64)
        native_mask.append(is_native)
        if not is_native:
            d = F.interpolate(d, size=64, mode='bilinear', align_corners=True)
        maps.append(d)
    stacked = torch.cat(maps, dim=1)  # [B, 6, 64, 64]
    summed = stacked.sum(dim=1, keepdim=True)  # [B, 1, 64, 64]
    if return_native_mask:
        return stacked, summed, native_mask
    return stacked, summed


class UniNetModelWithHead(UniNetModelExtension):
    """Extension model with learned anomaly score head.

    Replaces the handcrafted weighted_decision_mechanism with a learned
    CNN + attention pooling head for image-level scoring.
    Training: ExtensionLoss + BCE on head output.
    Inference: head score + standard anomaly map (for pAUROC).
    """

    def __init__(self, student_backbone, teacher_backbone, loss, score_head_channels=6, head_dropout=0.5,
                 head_type="cnn", head_grad_to_student=False,
                 bce_variant: str = "summed", lambda_bce: float = 1.0, lambda_stage: float = 0.1):
        super().__init__(student_backbone=student_backbone, teacher_backbone=teacher_backbone, loss=loss)
        # bce_variant: which BCE family to use for the head's loss.
        #   'summed'           — B0: 1 StatisticsHead on the summed 64×64 map (sum of all 6).
        #   'native64'         — B1: 1 StatisticsHead on sum of only the natively-64×64 maps (no upsample).
        #   'per_stage'        — B2: 3 StatisticsHeads, each on the per-stage sum (source+target paths).
        #   'per_path'         — B4: 6 StatisticsHeads, one per individual map.
        #   'final_plus_stage' — B3: 1 final-summed head + 3 per-stage heads. Loss formula:
        #                        L_bce = lambda_bce * (BCE_final + lambda_stage * (1/3)*sum_l BCE_stage_l)
        # lambda_bce: outer weight on the BCE block.
        # lambda_stage: inner weight for the stage term in 'final_plus_stage' variant only.
        assert bce_variant in ("summed", "native64", "per_stage", "per_path", "final_plus_stage",
                                "tiny_spatial_sum", "tiny_spatial_native64", "tiny_spatial_perstage",
                                "tiny_spatial_final_plus_stage",
                                "direct_score_bce", "direct_score_rank", "direct_score_pairwise",
                                "asymmetric_normal"), \
            f"bce_variant must be one of summed/native64/per_stage/per_path/final_plus_stage/"\
            f"tiny_spatial_sum/tiny_spatial_native64/tiny_spatial_perstage/tiny_spatial_final_plus_stage/"\
            f"direct_score_bce/direct_score_rank/asymmetric_normal, got {bce_variant}"
        self.bce_variant = bce_variant
        self.lambda_bce = float(lambda_bce)
        self.lambda_stage = float(lambda_stage)

        # Backward-compat: legacy "score_head + head_grad_to_student" path uses head_type & one head.
        if head_type == "stats":
            self.score_head = StatisticsHead()
            self.score_head_channels = 1
        else:
            self.score_head = AnomalyScoreHead(in_channels=score_head_channels, dropout=head_dropout)
            self.score_head_channels = score_head_channels
        self.head_type = head_type
        self.head_bce = nn.BCEWithLogitsLoss()
        self.head_grad_to_student = head_grad_to_student

        # Allocate additional StatisticsHeads per variant. Each is ~6 params; cheap.
        if bce_variant == "per_stage" or bce_variant == "final_plus_stage":
            self.stage_heads = nn.ModuleList([StatisticsHead() for _ in range(3)])
        if bce_variant == "per_path":
            self.path_heads = nn.ModuleList([StatisticsHead() for _ in range(6)])
        # `summed`, `native64`, and final part of `final_plus_stage` reuse self.score_head.

        # B6/B7/B8 — TinySpatialHead variants (training-only BCE scaffold).
        # NOT used at inference; the standard discrepancy-map inference path is preserved.
        if bce_variant in ("tiny_spatial_sum", "tiny_spatial_native64", "tiny_spatial_final_plus_stage"):
            self.tiny_head = TinySpatialHead()
        if bce_variant in ("tiny_spatial_perstage", "tiny_spatial_final_plus_stage"):
            self.tiny_stage_heads = nn.ModuleList([TinySpatialHead() for _ in range(3)])

        # B9 — direct_score_bce: learnable scale + bias mapping topk_mean(A_sum) -> logit.
        # No learned map reader. inverse_softplus(1.0) ≈ 0.5413 -> initial scale ≈ 1.0.
        if bce_variant == "direct_score_bce":
            self.raw_scale = nn.Parameter(torch.tensor(0.5413))
            self.score_bias = nn.Parameter(torch.tensor(0.0))
        # B10 — direct_score_rank: no learned params (pairwise loss on the topk_mean score).
        # B11 — asymmetric_normal: reuses self.score_head (StatisticsHead allocated above).

        # Skip counter for B10 batches with no normals OR no anomalies (pairwise loss undefined).
        self._rank_skip_count = 0

        # Once-per-run native64 index cache (filled on first forward).
        self._native64_indices = None
        # Loss component logging — per-step magnitudes. Cleared per epoch by external code.
        self.last_loss_components = None

    def forward(self, images, masks=None, labels=None, canvas_normals=None, has_canvas=None):
        if not self.training:
            return self._inference(images)

        # Standard extension forward
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)
        student_features = self.student(bottleneck_outputs)

        # Chunk student features
        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0], student_features[1][0], student_features[2][0],
            student_features[0][1], student_features[1][1], student_features[2][1],
        ]
        student_features = self._feature_selection(source_target_features, student_features)

        # Compute distance maps + native-64 index mask. Gradient flows by default (head_grad_to_student=True)
        # so BCE can shape student features.
        stacked, summed, native_mask = _compute_distance_maps(
            source_target_features, student_features, return_native_mask=True)
        if self._native64_indices is None:
            self._native64_indices = [i for i, n in enumerate(native_mask) if n]
            print(f"[UniNetModelWithHead] native-64 map indices detected: {self._native64_indices} "
                  f"(of 6 total maps)")

        # Detach toggle applies uniformly to all variants below.
        def maybe_detach(t):
            return t if self.head_grad_to_student else t.detach()

        # Build the input(s) and compute BCE logits per variant.
        labels_f = labels.float()
        bce_loss_raw = torch.tensor(0.0, device=summed.device)

        def _safe_head_call(head, x, name=""):
            logit = head(x).squeeze(-1)
            assert logit.dim() == 1 and logit.shape[0] == labels.shape[0], \
                f"BCE {name}: logit shape {logit.shape} mismatch labels {labels.shape}"
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                raise AssertionError(f"BCE {name}: NaN/Inf in logits. input stats: "
                                     f"max={x.max():.3f} min={x.min():.3f} mean={x.mean():.3f}")
            return logit

        if self.bce_variant == "summed":
            inp = maybe_detach(summed)
            logit = _safe_head_call(self.score_head, inp, "B0_summed")
            bce_loss_raw = self.head_bce(logit, labels_f)

        elif self.bce_variant == "native64":
            assert len(self._native64_indices) > 0, "B1_native64: no native-64 maps detected!"
            native_maps = [stacked[:, i:i+1] for i in self._native64_indices]
            native_sum = torch.stack(native_maps, dim=0).sum(dim=0)  # [B, 1, 64, 64]
            inp = maybe_detach(native_sum)
            logit = _safe_head_call(self.score_head, inp, "B1_native64")
            bce_loss_raw = self.head_bce(logit, labels_f)

        elif self.bce_variant == "per_stage":
            # Stage l = paths {l, l+3}. Sum each pair, apply per-stage head, average BCE.
            stage_losses = []
            for l in range(3):
                stage_map = stacked[:, l:l+1] + stacked[:, l+3:l+4]   # [B, 1, 64, 64]
                inp = maybe_detach(stage_map)
                logit = _safe_head_call(self.stage_heads[l], inp, f"B2_stage{l}")
                stage_losses.append(self.head_bce(logit, labels_f))
            bce_loss_raw = torch.stack(stage_losses).mean()

        elif self.bce_variant == "per_path":
            path_losses = []
            for p in range(6):
                path_map = stacked[:, p:p+1]
                inp = maybe_detach(path_map)
                logit = _safe_head_call(self.path_heads[p], inp, f"B4_path{p}")
                path_losses.append(self.head_bce(logit, labels_f))
            bce_loss_raw = torch.stack(path_losses).mean()

        elif self.bce_variant == "final_plus_stage":
            # B3: lambda_bce * (BCE_final + lambda_stage * mean_l BCE_stage_l)
            inp_final = maybe_detach(summed)
            logit_final = _safe_head_call(self.score_head, inp_final, "B3_final")
            bce_final = self.head_bce(logit_final, labels_f)
            stage_losses = []
            for l in range(3):
                stage_map = stacked[:, l:l+1] + stacked[:, l+3:l+4]
                inp_s = maybe_detach(stage_map)
                logit_s = _safe_head_call(self.stage_heads[l], inp_s, f"B3_stage{l}")
                stage_losses.append(self.head_bce(logit_s, labels_f))
            stage_mean = torch.stack(stage_losses).mean()
            bce_loss_raw = bce_final + self.lambda_stage * stage_mean

        elif self.bce_variant == "tiny_spatial_sum":
            # B6: TinySpatialHead on summed map. Gradient flows into student.
            inp = maybe_detach(summed)
            logit = _safe_head_call(self.tiny_head, inp, "B6_tiny_spatial_sum")
            bce_loss_raw = self.head_bce(logit, labels_f)

        elif self.bce_variant == "tiny_spatial_native64":
            # B7: TinySpatialHead on sum of natively-64x64 maps only.
            assert len(self._native64_indices) > 0, "B7_tiny_spatial_native64: no native-64 maps detected!"
            native_maps = [stacked[:, i:i+1] for i in self._native64_indices]
            native_sum = torch.stack(native_maps, dim=0).sum(dim=0)
            inp = maybe_detach(native_sum)
            logit = _safe_head_call(self.tiny_head, inp, "B7_tiny_spatial_native64")
            bce_loss_raw = self.head_bce(logit, labels_f)

        elif self.bce_variant == "tiny_spatial_perstage":
            # B8: per-stage TinySpatialHeads, BCE averaged.
            stage_losses = []
            for l in range(3):
                stage_map = stacked[:, l:l+1] + stacked[:, l+3:l+4]
                inp = maybe_detach(stage_map)
                logit = _safe_head_call(self.tiny_stage_heads[l], inp, f"B8_tiny_spatial_stage{l}")
                stage_losses.append(self.head_bce(logit, labels_f))
            bce_loss_raw = torch.stack(stage_losses).mean()

        elif self.bce_variant == "tiny_spatial_final_plus_stage":
            # B12: B3 (final_plus_stage) but with TinySpatialHead replacing StatisticsHead.
            # 1 final TinySpatialHead on summed + 3 per-stage TinySpatialHeads.
            # L_bce = lambda_bce * (BCE_final + lambda_stage * mean_l BCE_stage_l)
            inp_final = maybe_detach(summed)
            logit_final = _safe_head_call(self.tiny_head, inp_final, "B12_tiny_final")
            bce_final = self.head_bce(logit_final, labels_f)
            stage_losses = []
            for l in range(3):
                stage_map = stacked[:, l:l+1] + stacked[:, l+3:l+4]
                inp_s = maybe_detach(stage_map)
                logit_s = _safe_head_call(self.tiny_stage_heads[l], inp_s, f"B12_tiny_stage{l}")
                stage_losses.append(self.head_bce(logit_s, labels_f))
            stage_mean = torch.stack(stage_losses).mean()
            bce_loss_raw = bce_final + self.lambda_stage * stage_mean

        elif self.bce_variant == "direct_score_bce":
            # B9: BCE on the actual inference-like image score.
            # No learned map reader. Score = mean of top-1% pixels of A_sum.
            # logit = softplus(raw_scale) * score + bias  (2 learnable scalars).
            inp = maybe_detach(summed)
            s = _topk_mean_score(inp, frac=0.01)  # [B]
            scale = F.softplus(self.raw_scale) + 1e-6
            logit = scale * s + self.score_bias  # [B]
            if torch.isnan(logit).any() or torch.isinf(logit).any():
                raise AssertionError(f"B9 direct_score_bce: NaN/Inf in logit. "
                                     f"score max={s.max():.3f} min={s.min():.3f} mean={s.mean():.3f}, "
                                     f"scale={scale.item():.4f} bias={self.score_bias.item():.4f}")
            bce_loss_raw = self.head_bce(logit, labels_f)

        elif self.bce_variant in ("direct_score_rank", "direct_score_pairwise"):
            # B10: pairwise softplus margin loss on the topk_mean score.
            # Directly optimizes image-level ranking — closer to iAUC than BCE.
            # If batch has no normals OR no anomalies, skip with zero loss.
            inp = maybe_detach(summed)
            s = _topk_mean_score(inp, frac=0.01)  # [B]
            s_pos = s[labels == 1]
            s_neg = s[labels == 0]
            if s_pos.numel() == 0 or s_neg.numel() == 0:
                self._rank_skip_count += 1
                bce_loss_raw = torch.zeros((), device=summed.device, dtype=s.dtype)
            else:
                margin = 0.1
                diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)  # [P, N]
                bce_loss_raw = F.softplus(margin - diff).mean()
                if torch.isnan(bce_loss_raw) or torch.isinf(bce_loss_raw):
                    raise AssertionError(f"B10 direct_score_rank: NaN/Inf in loss. "
                                         f"score max={s.max():.3f} min={s.min():.3f}")

        elif self.bce_variant == "asymmetric_normal":
            # B11: BCE ONLY on normal samples (push toward label=0).
            # No loss on anomaly samples — does NOT amplify anomaly direction.
            # Reuses self.score_head (StatisticsHead).
            inp = maybe_detach(summed)
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                bce_loss_raw = torch.zeros((), device=summed.device, dtype=inp.dtype)
            else:
                inp_norm = inp[normal_mask]
                labels_norm = labels_f[normal_mask]  # all zeros
                logit = _safe_head_call(self.score_head, inp_norm, "B11_asymmetric_normal")
                bce_loss_raw = self.head_bce(logit, labels_norm)

        head_loss = self.lambda_bce * bce_loss_raw
        # For logging only:
        self.last_loss_components = {
            "bce_raw": float(bce_loss_raw.detach()),
            "bce_weighted": float(head_loss.detach()),
        }

        # Canvas teacher features (same as UniNetModelExtension)
        syn_idx = has_canvas.nonzero(as_tuple=True)[0] if has_canvas is not None else torch.tensor([], dtype=torch.long)
        canvas_teacher_features = None
        if len(syn_idx) > 0 and canvas_normals is not None:
            with torch.no_grad():
                canvas_teacher_features, _ = self.teachers(canvas_normals[syn_idx])

        mask_ = None
        if masks is not None:
            mask_ = masks.float()
            if mask_.dim() == 3:
                mask_ = mask_.unsqueeze(1)

        feature_loss = self.loss(
            student_features, source_target_features,
            canvas_teacher_features=canvas_teacher_features,
            syn_idx=syn_idx,
            mask=mask_,
        )
        if self.last_loss_components is not None:
            self.last_loss_components["feature_loss"] = float(feature_loss.detach())
            self.last_loss_components["total"] = float((feature_loss + head_loss).detach())
        return feature_loss + head_loss

    def _inference(self, images):
        """Inference: use learned head for image score, standard map for pixel score."""
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)
        student_features = self.student(bottleneck_outputs)

        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0], student_features[1][0], student_features[2][0],
            student_features[0][1], student_features[1][1], student_features[2][1],
        ]

        # Distance maps (64x64) for score head
        stacked, summed = _compute_distance_maps(source_target_features, student_features)

        # Image score from learned head.
        # B6/B7/B8 (tiny_spatial_*) and B9/B10/B11 (direct_score_*/asymmetric_normal):
        # heads are training-only scaffolds; do NOT consult at inference.
        # pred_score falls back to summed-map max. Canonical iAUC comes from paper-correct
        # top-k on anomaly_map anyway.
        training_only_variants = ("tiny_spatial_sum", "tiny_spatial_native64", "tiny_spatial_perstage",
                                   "direct_score_bce", "direct_score_rank", "asymmetric_normal")
        if self.bce_variant in training_only_variants:
            pred_score = summed.view(summed.shape[0], -1).max(dim=1).values.unsqueeze(1)
        else:
            head_input = summed if self.score_head_channels == 1 else stacked
            head_logits = self.score_head(head_input)  # [B, 1]
            pred_score = torch.sigmoid(head_logits)

        # Anomaly map: upsample each layer to full res, then sum (matches standard UniNet)
        target_size = images.shape[-2:]
        anomaly_map = torch.zeros(images.shape[0], *target_size, device=images.device)
        for tf, sf in zip(source_target_features, student_features):
            d = 1 - F.cosine_similarity(tf, sf)  # [B, H, W]
            d = F.interpolate(
                d.unsqueeze(1), size=target_size,
                mode='bilinear', align_corners=True,
            ).squeeze(1)
            anomaly_map += d

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


# ── Paper-correct scoring ───────────────────────────────────────────────
def _apply_paper_score_mode():
    """Monkey-patch weighted_decision_mechanism to use mean of top-k pixels.

    The anomalib implementation uses top_k_values[0] (always the max), making
    the adaptive top-k computation pointless. The UniNet paper intends the
    image-level score to be the MEAN of the top-k highest-anomaly pixels,
    where k is adaptively determined by the dominant-scale anomaly signal.
    """
    import anomalib.models.image.uninet.components.anomaly_map as _amap_mod
    import anomalib.models.image.uninet.torch_model as _tm_mod

    import einops
    import torch
    from torch.nn import functional as F
    from anomalib.models.components.filters import GaussianBlur2d

    def _weighted_decision_mechanism_paper(batch_size, output_list, alpha, beta,
                                           output_size=(256, 256)):
        device = output_list[0].device
        num_outputs = len(output_list)

        total_weights = torch.zeros(batch_size, device=device)
        ks = BLUR_KERNEL_SIZE
        gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(ks, ks), channels=1).to(device)

        for i in range(batch_size):
            max_values = torch.zeros(num_outputs, device=device)
            for j, output_tensor in enumerate(output_list):
                max_values[j] = torch.max(output_tensor[i])
            probs = F.softmax(max_values, dim=0)
            prob_mean = torch.mean(probs)
            mask = probs > prob_mean
            if mask.any():
                weight_tensor = max_values[mask]
                weight = torch.max(torch.stack([
                    torch.mean(weight_tensor) * alpha,
                    torch.tensor(beta, device=device)
                ]))
            else:
                weight = torch.tensor(beta, device=device)
            total_weights[i] = weight

        processed_anomaly_maps = torch.zeros(batch_size, *output_size, device=device)
        for output_tensor in output_list:
            output_resized = F.interpolate(
                output_tensor.unsqueeze(1), output_size,
                mode="bilinear", align_corners=True,
            ).squeeze(1)
            processed_anomaly_maps += output_resized

        anomaly_scores = torch.zeros(batch_size, device=device)
        for idx in range(batch_size):
            top_k = max(int(output_size[0] * output_size[1] * total_weights[idx]), 1)
            single_map = processed_anomaly_maps[idx]
            single_map = gaussian_blur(
                einops.rearrange(single_map, "h w -> 1 1 h w")
            ).squeeze()
            top_k_values = single_map.view(-1).topk(top_k).values
            # Paper intent: mean of top-k pixels (not just the max / top_k_values[0])
            anomaly_scores[idx] = top_k_values.mean().detach()

        return anomaly_scores.unsqueeze(1), processed_anomaly_maps.detach()

    # Patch both the module and the already-imported reference in torch_model
    _amap_mod.weighted_decision_mechanism = _weighted_decision_mechanism_paper
    _tm_mod.weighted_decision_mechanism = _weighted_decision_mechanism_paper
    print("Score mode: 'paper' — using mean of adaptive top-k pixels (monkey-patched)")


# ── Model factory ───────────────────────────────────────────────────────
def make_model(mode: str, device: torch.device, w_canvas: float = 1.0, w_twosided: float = 1.0,
               lambda_push: float = 1.0,
               lambda_norm_cos: float = 1.0,
               loss_form: str = "cos",
               scale_kd_push: bool = False,
               scale_push_contrastive: bool = False,
               lw_align_only: bool = False,
               weight_push_only: bool = False,
               score_head: str = None, head_dropout: float = 0.5, head_grad_to_student: bool = False,
               native_bce: bool = False, native_bce_per_stage: bool = False, bce_all_normal: bool = False,
               bce_variant: str = "summed", lambda_bce: float = 1.0, lambda_stage: float = 0.1,
               train_seed: int = SEED,
               backbone: str = "wrn50", lkd_mode: str = "local",
               margin_mode: str = "off", lsc_pre_relu: bool = False,
               train_apply_dfs: bool = True, eval_apply_dfs: bool = False,
               dinomaly_p: float = 0.0, dinomaly_factor: float = 0.1,
               dinomaly_ramp_iters: int = 100,
               infer_aggs: list = None, temperature: float = 2.0,
               two_sided_lkd: bool = False, tau: float | None = None,
               canvas_align_mode: str = "uniform",
               proj_consistency: bool = False, lambda_proj: float = 0.1,
               residual_direction_alignment: bool = False, lambda_resdir: float = 0.05,
               pairrank: bool = False, lambda_pairrank: float = 0.05, pairrank_margin: float = 0.1,
               reltail: bool = False, lambda_reltail: float = 0.05, reltail_rho: float = 0.5,
               reltail_frac: float = 0.01,
               maskrank: bool = False, lambda_maskrank: float = 0.05, maskrank_margin: float = 0.1,
               adaptive_triplet: bool = False, triplet_alpha: float = 0.5,
               preserve_outside: bool = False, w_preserve: float = 1.0,
               gap_gated_outside_canvas: bool = False, lambda_gated_outcanvas: float = 0.25,
               ct_twosided: str = "off", w_ct_twosided: float = 1.0,
               tproto: str = "off", w_tproto: float = 1.0,
               real_tproto: str = "off", w_real_tproto: float = 1.0,
               real_student_sup: bool = True,
               target_teacher_lr: float = 1e-6,
               stop_gradient: bool = False, lambda_idem: float = 0.0,
               idem_caao: bool = False, no_detach_any: bool = False,
               no_twosided_push: bool = False,
               use_teacher_bce: bool = False, teacher_bce_weight: float = 0.0,
               teacher_bce_feature_layer: int = 0, teacher_bce_pool: str = "avg",
               teacher_bce_loss_type: str = "bce", teacher_bce_focal_gamma: float = 2.0,
               teacher_bce_real_only: bool = False, push_to_max: bool = False,
               recon_concat_bce: bool = False, recon_bce_weight: float = 0.0,
               recon_head: str = "grouped", recon_l2: bool = False, bce_pool_max: bool = False,
               # Geometry modules (CAAO extensions)
               teacher_repulsion: bool = False,
               teacher_repulsion_weight: float = 1.0,
               teacher_repulsion_margin: float = 0.5,
               teacher_repulsion_clean_sg: bool = False,
               bottleneck_pair: bool = False,
               bottleneck_pair_weight: float = 1.0,
               bottleneck_pair_metric: str = "cosine",
               bottleneck_pair_clean_sg: bool = False,
               bottleneck_protos: bool = False,
               bottleneck_protos_k: int = 8,
               bottleneck_protos_weight: float = 1.0,
               bottleneck_protos_assignment: str = "nearest",
               bottleneck_protos_temperature: float = 0.1,
               bottleneck_protos_ema: float = 0.99,
               bottleneck_protos_sinkhorn_iters: int = 3,
               bottleneck_protos_update_mode: str = "gradient",
               bottleneck_protos_lr: float = 5e-3,
               teacher_repulsion_stages: str = "trainable_only",
               geom_diagnostics: bool = True,
               # Geom2 modules
               spm: bool = False,
               spm_k: int = 500,
               spm_weight: float = 1.0,
               spm_tau: float = 0.05,
               spm_eps: float = 0.05,
               spm_sinkhorn_iters: int = 3,
               spm_ema: float = 0.99,
               spm_init_file: str | None = None,
               paired_proto: bool = False,
               paired_proto_weight: float = 0.1,
               paired_proto_sg_target: bool = True,
               soft_pair: bool = False,
               soft_pair_weight: float = 0.01,
               trep: bool = False,
               trep_weight: float = 0.05,
               trep_adaptive_margin: bool = False,
               trep_margin: float = 0.0):
    """Create model + optimizer for the given mode."""
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    random.seed(train_seed)        # patched 2026-05-01: was missing; controls aug hflip/vflip/rotation
    np.random.seed(train_seed)     # patched 2026-05-01: was missing

    if mode in ("baseline", "supervised"):
        # Modes 1 (Baseline), 2 (Supervised real), 3 (Supervised synthetic) — all use UniNetLoss.
        # Modes 2 & 3 share the same code-level "supervised" mode (different training datasets).
        # Controllable variant respects margin_mode flag.
        loss_fn = UniNetLossControllable(
            temperature=temperature,
            margin_enabled=(margin_mode == "on"),
            lkd_mode=lkd_mode,
            two_sided=two_sided_lkd,
            tau=tau,
            loss_form=loss_form,
            stop_gradient=stop_gradient,
            tproto=tproto, w_tproto=w_tproto,
        )
        loss_fn.scale_push_contrastive = scale_push_contrastive
        model = UniNetModel(
            student_backbone="wide_resnet50_2",
            teacher_backbone="wide_resnet50_2",
            loss=loss_fn,
        )
        if not train_apply_dfs:
            # Minimal-UniNet ablation: anomalib calls self._feature_selection(source_target_features,
            # student_features) at training → return the student features unmodified (no DFS).
            model._feature_selection = lambda target_features, source_features, maximize=True: source_features
        if globals().get("_NO_UNINET_BCE", False):
            model.bce_loss = _ZeroLoss()
    else:
        # Mode 4: Extension (ExtensionLoss with canvas alignment + two-sided syn term, no BCE)
        # WRN50 bottleneck outputs 2048 channels; ConvNeXt-B outputs 1024 channels.
        _bn_dim = 2048 if backbone == "wrn50" else 1024
        loss_fn = ExtensionLoss(temperature=temperature, w_canvas=w_canvas, w_twosided=w_twosided,
                                lambda_push=lambda_push,
                                lambda_norm_cos=lambda_norm_cos,
                                loss_form=loss_form,
                                lkd_mode=lkd_mode, margin_mode=margin_mode,
                                lsc_pre_relu=lsc_pre_relu,
                                dinomaly_p=dinomaly_p, dinomaly_factor=dinomaly_factor,
                                dinomaly_ramp_iters=dinomaly_ramp_iters,
                                tau=tau, canvas_align_mode=canvas_align_mode,
                                proj_consistency=proj_consistency, lambda_proj=lambda_proj,
                                residual_direction_alignment=residual_direction_alignment,
                                lambda_resdir=lambda_resdir,
                                pairrank=pairrank, lambda_pairrank=lambda_pairrank,
                                pairrank_margin=pairrank_margin,
                                reltail=reltail, lambda_reltail=lambda_reltail,
                                reltail_rho=reltail_rho, reltail_frac=reltail_frac,
                                maskrank=maskrank, lambda_maskrank=lambda_maskrank,
                                maskrank_margin=maskrank_margin,
                                adaptive_triplet=adaptive_triplet, triplet_alpha=triplet_alpha,
                                preserve_outside=preserve_outside, w_preserve=w_preserve,
                                gap_gated_outside_canvas=gap_gated_outside_canvas,
                                lambda_gated_outcanvas=lambda_gated_outcanvas,
                                ct_twosided=ct_twosided, w_ct_twosided=w_ct_twosided,
                                stop_gradient=stop_gradient,
                                teacher_repulsion=teacher_repulsion,
                                teacher_repulsion_weight=teacher_repulsion_weight,
                                teacher_repulsion_margin=teacher_repulsion_margin,
                                teacher_repulsion_clean_sg=teacher_repulsion_clean_sg,
                                teacher_repulsion_stages=teacher_repulsion_stages,
                                bottleneck_pair=bottleneck_pair,
                                bottleneck_pair_weight=bottleneck_pair_weight,
                                bottleneck_pair_metric=bottleneck_pair_metric,
                                bottleneck_pair_clean_sg=bottleneck_pair_clean_sg,
                                bottleneck_protos=bottleneck_protos,
                                bottleneck_protos_k=bottleneck_protos_k,
                                bottleneck_protos_weight=bottleneck_protos_weight,
                                bottleneck_protos_assignment=bottleneck_protos_assignment,
                                bottleneck_protos_temperature=bottleneck_protos_temperature,
                                bottleneck_protos_ema=bottleneck_protos_ema,
                                bottleneck_protos_sinkhorn_iters=bottleneck_protos_sinkhorn_iters,
                                bottleneck_protos_update_mode=bottleneck_protos_update_mode,
                                bottleneck_dim=_bn_dim,
                                geom_diagnostics=geom_diagnostics,
                                spm=spm,
                                spm_k=spm_k,
                                spm_weight=spm_weight,
                                spm_tau=spm_tau,
                                spm_eps=spm_eps,
                                spm_sinkhorn_iters=spm_sinkhorn_iters,
                                spm_ema=spm_ema,
                                spm_init_file=spm_init_file,
                                paired_proto=paired_proto,
                                paired_proto_weight=paired_proto_weight,
                                paired_proto_sg_target=paired_proto_sg_target,
                                soft_pair=soft_pair,
                                soft_pair_weight=soft_pair_weight,
                                trep=trep,
                                trep_weight=trep_weight,
                                trep_adaptive_margin=trep_adaptive_margin,
                                trep_margin=trep_margin)
        loss_fn.real_tproto = real_tproto          # set externally (Mode 5 real-anomaly teacher push)
        loss_fn.w_real_tproto = w_real_tproto
        loss_fn.tproto = tproto                     # M4 CAAO synth-anomaly teacher push
        loss_fn.w_tproto = w_tproto
        loss_fn.real_student_sup = bool(real_student_sup)
        if score_head is not None:
            if score_head == "stats":
                model = UniNetModelWithHead(
                    student_backbone="wide_resnet50_2",
                    teacher_backbone="wide_resnet50_2",
                    loss=loss_fn,
                    head_type="stats",
                    head_grad_to_student=head_grad_to_student,
                    bce_variant=bce_variant,
                    lambda_bce=lambda_bce,
                    lambda_stage=lambda_stage,
                )
            else:
                channels = 1 if score_head == "1ch" else 6
                model = UniNetModelWithHead(
                    student_backbone="wide_resnet50_2",
                    teacher_backbone="wide_resnet50_2",
                    loss=loss_fn,
                    score_head_channels=channels,
                    head_dropout=head_dropout,
                    head_grad_to_student=head_grad_to_student,
                    bce_variant=bce_variant,
                    lambda_bce=lambda_bce,
                    lambda_stage=lambda_stage,
                )
        else:
            # Always use the DFS-ablate variant. Default flags reproduce
            # UniNetModelExtension's behavior bit-identically (DFS at train, no DFS at eval).
            model = UniNetModelExtensionDFSAblate(
                student_backbone="wide_resnet50_2",
                teacher_backbone="wide_resnet50_2",
                loss=loss_fn,
                train_apply_dfs=train_apply_dfs,
                eval_apply_dfs=eval_apply_dfs,
                infer_aggs=infer_aggs,
                native_bce=native_bce,
                native_bce_per_stage=native_bce_per_stage,
                bce_all_normal=bce_all_normal,
                lambda_idem=lambda_idem,
                idem_caao=idem_caao,
                no_detach_any=no_detach_any,
            )

    loss_fn.scale_kd_push = scale_kd_push
    loss_fn.scale_push_contrastive = scale_push_contrastive
    loss_fn.lw_align_only = lw_align_only
    loss_fn.weight_push_only = weight_push_only
    loss_fn.no_detach_any = no_detach_any
    loss_fn.no_twosided_push = no_twosided_push
    loss_fn.push_to_max = push_to_max
    if globals().get("_LAMBDA_WEIGHT") is not None:
        loss_fn.lambda_weight = float(_LAMBDA_WEIGHT)
        print(f"  lambda_weight override: {loss_fn.lambda_weight} (L_SC weight {1 - loss_fn.lambda_weight:g})")

    if backbone == "convnext_b":
        _swap_to_convnext(model)
    if globals().get("_SINGLE_TEACHER", False):
        # No frozen teacher: the "source" branch IS the adaptive target teacher (shared weights).
        model.teachers.source_teacher = model.teachers.target_teacher
        print("  single adaptive teacher: source branch tied to the adaptive target teacher")

    model = model.to(device)
    model._skip_sup_bce = bool(globals().get("_SKIP_SUP_BCE", False))

    if bce_pool_max and hasattr(model, 'avgpool'):
        model.avgpool = nn.AdaptiveMaxPool2d((1, 1))  # swap image-BCE pooling avg->max

    # ── Teacher-BCE head (image-level BCE on adaptive target teacher features) ──
    # Materialized eagerly so its params are present when the optimizer is built.
    # Channel count read from a dummy target-teacher forward (backbone/layer-agnostic).
    model.use_teacher_bce = bool(use_teacher_bce)
    model.teacher_bce_weight = float(teacher_bce_weight)
    model.teacher_bce_feature_layer = int(teacher_bce_feature_layer)
    model.teacher_bce_pool = teacher_bce_pool
    model.teacher_bce_loss_type = teacher_bce_loss_type
    model.teacher_bce_focal_gamma = float(teacher_bce_focal_gamma)
    model.teacher_bce_real_only = bool(teacher_bce_real_only)
    model.recon_concat_bce = bool(recon_concat_bce)
    model.recon_bce_weight = float(recon_bce_weight)
    model.recon_l2 = bool(recon_l2)
    model.last_teacher_bce_stats = {}
    if use_teacher_bce:
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            _tfeats = model.teachers.target_teacher(_dummy)
        _C = _tfeats[teacher_bce_feature_layer].shape[1]
        model.teacher_bce_head = nn.Linear(_C, 1).to(device)
    model.recon_head_type = recon_head
    if recon_concat_bce:
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            _sfeats = model.teachers.source_teacher(_dummy)
        _C = _sfeats[teacher_bce_feature_layer].shape[1]    # per-stage channels (128 CNX / 256 WRN)
        if recon_head == "grouped":
            # Grouped 1×1: each of C groups sees only its 4 interleaved views → 1 per-channel
            # signal (~5·C params), then 1×1 mixes the C channels → 1 score map. Tiny.
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(4 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(),
                nn.Conv2d(_C, 1, 1),
            ).to(device)
        elif recon_head == "grouped_pmax":
            # Grouped conv → [B,C,H,W] per-channel maps (no channel mixing); the term then
            # spatial-maxes each channel independently → Linear over the C peaks → logit.
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(4 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(),
            ).to(device)
            model.recon_pmax_linear = nn.Linear(_C, 1).to(device)
        elif recon_head in ("grouped_pmax_cos","grouped_pmax_cos_avg"):
            # grouped_pmax but 6 views per group (4 features + 2 broadcast cosine maps).
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(6 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(),
            ).to(device)
            model.recon_pmax_linear = nn.Linear(_C, 1).to(device)
        elif recon_head == "f_mix_lin":
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(4 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(), nn.Conv2d(_C, 1, 1)).to(device)
            _HW = int(_sfeats[teacher_bce_feature_layer].shape[2] * _sfeats[teacher_bce_feature_layer].shape[3])
            model.recon_lin = nn.Linear(_HW, 1).to(device)
        elif recon_head == "fc_per_gridmax":
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(6 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU()).to(device)
            model.recon_grid_lin = nn.Linear(_C * 64, 1).to(device)
        elif recon_head == "fc_mix_gridmax":
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(6 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(), nn.Conv2d(_C, 1, 1)).to(device)
            model.recon_grid_lin = nn.Linear(64, 1).to(device)
        elif recon_head == "fc_mix_lin":
            # grouped(6C->C) -> GN/GELU -> mix(C->1) -> [B,1,H,W]; spatial Linear(H*W->1) learns the readout.
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(6 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(),
                nn.Conv2d(_C, 1, 1),
            ).to(device)
            _HW = int(_sfeats[teacher_bce_feature_layer].shape[2] * _sfeats[teacher_bce_feature_layer].shape[3])
            model.recon_lin = nn.Linear(_HW, 1).to(device)
        elif recon_head == "grouped_cosrep":
            # 6 views/group grouped conv (6C->C) -> GN/GELU -> 1x1 mix (C->1) -> single max.
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(6 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(),
                nn.Conv2d(_C, 1, 1),
            ).to(device)
        elif recon_head == "grouped_cos":
            # Grouped per-channel features + 2 explicit cosine-distance maps → mix (C+2 → 1).
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(4 * _C, _C, 1, groups=_C), nn.GroupNorm(8, _C), nn.GELU(),
            ).to(device)
            model.recon_cos_mix = nn.Conv2d(_C + 2, 1, 1).to(device)
        elif recon_head == "cosonly":
            # ONLY the 2 cosine-distance maps → small conv → score map.
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.GELU(),
                nn.Conv2d(8, 1, 1),
            ).to(device)
        elif recon_head == "tsrd_dilated":
            # TSRD-dilated: 8C in (T,S,|T-S|,T-S per pairing) → 1×1 compress to 32 →
            # dilated 3×3 stack (d=1,2,4 → multi-scale RF) → 1-ch score map.
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(8 * _C, 32, 1), nn.GroupNorm(4, 32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=1, dilation=1), nn.GroupNorm(4, 32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GroupNorm(4, 32), nn.GELU(),
                nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.GroupNorm(4, 32), nn.GELU(),
                nn.Conv2d(32, 1, 1),
            ).to(device)
        elif recon_head == "tiny":
            # Per-pixel linear over the 4C channels → 1 score map (~4C+1 params).
            model.recon_bce_head = nn.Conv2d(4 * _C, 1, 1).to(device)
        else:  # dense (overfit-prone, kept for comparison)
            _h = 64
            model.recon_bce_head = nn.Sequential(
                nn.Conv2d(4 * _C, _h, 3, padding=1), nn.GroupNorm(8, _h), nn.GELU(),
                nn.Conv2d(_h, _h, 3, padding=1), nn.GroupNorm(8, _h), nn.GELU(),
                nn.Conv2d(_h, 1, 1),
            ).to(device)

    param_groups = [
        {"params": model.student.parameters()},
        {"params": model.bottleneck.parameters()},
        {"params": model.dfs.parameters()},
        {"params": model.teachers.target_teacher.parameters(), "lr": target_teacher_lr},
    ]
    if score_head is not None and hasattr(model, 'score_head'):
        param_groups.append({"params": model.score_head.parameters(), "lr": 1e-3, "weight_decay": 0.01})
    if use_teacher_bce:
        param_groups.append({"params": model.teacher_bce_head.parameters(), "lr": 1e-3, "weight_decay": 0.0})
    if recon_concat_bce:
        param_groups.append({"params": model.recon_bce_head.parameters(), "lr": 1e-3, "weight_decay": 0.0})
        if hasattr(model, "recon_pmax_linear"):
            param_groups.append({"params": model.recon_pmax_linear.parameters(), "lr": 1e-3, "weight_decay": 0.0})
        if hasattr(model, "recon_cos_mix"):
            param_groups.append({"params": model.recon_cos_mix.parameters(), "lr": 1e-3, "weight_decay": 0.0})
        if hasattr(model, "recon_lin"):
            param_groups.append({"params": model.recon_lin.parameters(), "lr": 1e-3, "weight_decay": 0.0})
        if hasattr(model, "recon_grid_lin"):
            param_groups.append({"params": model.recon_grid_lin.parameters(), "lr": 1e-3, "weight_decay": 0.0})
    # Prototype parameters live in loss_fn — add separately so AdamW tracks them.
    # Only in gradient mode: ema mode uses a buffer (not a Parameter), so optimizer skips it.
    # weight_decay=0 because prototypes are constrained to the unit sphere; WD fights normalization.
    if (hasattr(loss_fn, "prototypes") and loss_fn.prototypes is not None
            and isinstance(loss_fn.prototypes, nn.Parameter)):
        param_groups.append({
            "params": [loss_fn.prototypes],
            "lr": bottleneck_protos_lr,
            "weight_decay": 0.0,
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=5e-3, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-10, amsgrad=True,
    )
    return model, optimizer


# ── Training ────────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, dataloader, device, mode):
    model.train()
    model.teachers.eval()
    total_loss, n_batches = 0.0, 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["gt_label"].to(device)
        masks = batch["gt_mask"].to(device)
        if masks.dim() == 4:
            masks = masks.squeeze(1)

        # Stash foreground mask on the loss for the on-object tproto reference (mask==0 & fg>0.5).
        # Loss reads self._batch_fg; only used when tproto/real_tproto is active. Overwritten/step.
        _ls = getattr(model, "loss", None)
        if _ls is not None:
            fgm = batch.get("fg_mask")
            _ls._batch_fg = fgm.to(device) if fgm is not None else None

        optimizer.zero_grad()

        if mode == "baseline":
            # Unsupervised: no masks, pass labels for BCE + scalar mask
            loss = model(images=images, masks=None, labels=labels)
        elif mode == "supervised":
            # Original UniNet supervised: spatial masks + labels (BCE included unless skipped)
            _lab = None if getattr(model, "_skip_sup_bce", False) else labels
            loss = model(images=images, masks=masks, labels=_lab)
        else:
            # Extension: spatial masks + canvas normals (no BCE, no margin)
            canvas_normals = batch["canvas_normal"].to(device)
            has_canvas = batch["has_canvas"].to(device)
            if getattr(model, "teacher_bce_real_only", False):
                # Only the real-only DFSAblate path accepts real_images; other model
                # classes' forwards don't take the kwarg.
                loss = model(
                    images=images, masks=masks, labels=labels,
                    canvas_normals=canvas_normals, has_canvas=has_canvas,
                    real_images=batch["real_image"].to(device),
                )
            else:
                loss = model(
                    images=images, masks=masks, labels=labels,
                    canvas_normals=canvas_normals, has_canvas=has_canvas,
                )

        if GRAD_BALANCE and n_batches % GB_EVERY == 0:
            _lm = getattr(model, 'loss', None) or getattr(model, '_ext_loss', None)
            _parts = getattr(_lm, '_gb_parts', None)
            if _parts is not None and _parts[0].requires_grad and _parts[1].requires_grad and 0.0 < _parts[2] < 1.0:
                _ps = [q for g in optimizer.param_groups for q in g['params'] if q.requires_grad]
                def _gn(L):
                    gs = torch.autograd.grad(L, _ps, retain_graph=True, allow_unused=True)
                    return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
                gN, gS = _gn(_parts[0]), _gn(_parts[1])
                if gN > 0 and gS > 0:
                    _lm._gn_ema_n = 0.9 * getattr(_lm, '_gn_ema_n', gN) + 0.1 * gN
                    _lm._gn_ema_s = 0.9 * getattr(_lm, '_gn_ema_s', gS) + 0.1 * gS
                    _lm._gn_ratio = _lm._gn_ema_n / _lm._gn_ema_s
            if _lm is not None:
                _lm._gb_parts = None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        _lt = getattr(getattr(model, 'loss', None) or getattr(model, '_ext_loss', None), 'last_terms', None)
        if _lt is not None:
            for _k, _v in _lt.items():
                TERM_LOG.setdefault(_k, []).append(float(_v))
            _lt.clear() if isinstance(_lt, dict) else None
        # Accumulate teacher-BCE diagnostics for per-epoch logging
        st = getattr(model, "last_teacher_bce_stats", None)
        if st:
            acc = getattr(model, "_tbce_epoch_acc", None)
            if acc is None:
                acc = {"tbce": 0.0, "tbce_w": 0.0, "lmean": 0.0, "lstd": 0.0, "pos": 0.0, "n": 0}
                model._tbce_epoch_acc = acc
            acc["tbce"] += st["teacher_bce_loss"]; acc["tbce_w"] += st["teacher_bce_weighted"]
            acc["lmean"] += st["logit_mean"]; acc["lstd"] += st["logit_std"]
            acc["pos"] += st["pos_rate"]; acc["n"] += 1
        # Accumulate geometry diagnostics for per-epoch logging
        gs = getattr(getattr(model, "loss", None), "last_geometry_stats", None)
        if gs:
            gacc = getattr(model, "_geom_epoch_acc", None)
            if gacc is None:
                gacc = {"_n": 0}
                model._geom_epoch_acc = gacc
            for k, v in gs.items():
                gacc[k] = gacc.get(k, 0.0) + v
            gacc["_n"] += 1

    acc = getattr(model, "_tbce_epoch_acc", None)
    if acc and acc["n"] > 0:
        nb = acc["n"]
        print(f"    [teacher-BCE] loss={acc['tbce']/nb:.4f} weighted={acc['tbce_w']/nb:.4f} "
              f"logit_mean={acc['lmean']/nb:.3f} logit_std={acc['lstd']/nb:.3f} "
              f"pos_rate={acc['pos']/nb:.3f}")
        model._tbce_epoch_acc = None  # reset for next epoch

    gacc = getattr(model, "_geom_epoch_acc", None)
    if gacc and gacc.get("_n", 0) > 0:
        nb = gacc["_n"]
        parts = [f"{k}={v/nb:.4f}" for k, v in sorted(gacc.items()) if k != "_n"]
        print(f"    [geometry] " + "  ".join(parts))
        # Track per-epoch V mean for baseline q50 saving (used by trep2 adaptive margin)
        v_key = "geom/anom_teacher_clean_synth_cosdist"
        if v_key in gacc:
            if not hasattr(model, "_geom_V_history"):
                model._geom_V_history = []
            model._geom_V_history.append(gacc[v_key] / nb)
        model._geom_epoch_acc = None  # reset for next epoch

    return total_loss / n_batches


def _confusion_matrix_at_optimal(labels, scores):
    """Compute confusion matrix at optimal threshold (Youden's J)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j = tpr - fpr
    best_idx = np.argmax(j)
    thresh = thresholds[best_idx]
    preds = (scores >= thresh).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "threshold": float(thresh)}


@torch.no_grad()
@torch.no_grad()
def _collect_test_shoot(model, dataloader, device, shift_label, shoot_log_list):
    """Per-image element-wise sf-vs-tf comparison restricted to GT anomaly pixels.

    Logged once per (image, stage) for all images in dataloader that have
    anomaly pixels in their ground-truth mask. Mirrors the training-side
    `_log_shoot` but at test time on real anomalies. Each entry tagged with
    `shift_label` so 0x/1x/2x/3x/4x can be split apart later.

    Stages 0-2 = student/source-target features for the source (anomaly→normal)
    pathway; 3-5 = same for the target pathway, after DFS.
    """
    model.eval()
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["gt_mask"].to(device)
        if masks.dim() == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [B, H, W]
        # Reproduce model.forward's feature extraction
        source_target_features, bottleneck_inputs = model.teachers(images)
        bottleneck_outputs = model.bottleneck(bottleneck_inputs)
        s_feats = model.student(bottleneck_outputs)
        s_feats = [d.chunk(dim=0, chunks=2) for d in s_feats]
        s_feats = [
            s_feats[0][0], s_feats[1][0], s_feats[2][0],
            s_feats[0][1], s_feats[1][1], s_feats[2][1],
        ]
        s_feats = model._feature_selection(source_target_features, s_feats)

        for stage_i, (tf, sf) in enumerate(zip(source_target_features, s_feats)):
            h, w = tf.shape[-2:]
            mask_down = _downsample_mask(masks.float().unsqueeze(1), h, w)  # [B, H, W]
            m_bool = (mask_down > 0.5).unsqueeze(1).expand_as(sf)
            if not m_bool.any():
                continue
            sf_anom = sf[m_bool]
            tf_anom = tf[m_bool]
            diff = sf_anom - tf_anom
            shoot_log_list.append({
                "phase": "test",
                "shift": shift_label,
                "stage": stage_i,
                "n_elements": int(sf_anom.numel()),
                "elem_under_frac": float((sf_anom < tf_anom).float().mean()),
                "signed_diff_mean": float(diff.mean()),
                "signed_diff_p10": float(diff.quantile(0.10)),
                "signed_diff_p50": float(diff.quantile(0.50)),
                "signed_diff_p90": float(diff.quantile(0.90)),
                "s_mean": float(sf_anom.mean()),
                "t_mean": float(tf_anom.mean()),
            })


@torch.no_grad()
def evaluate(model, dataloader, device, blur_pixel_map: bool = False):
    """Evaluate. Returns dict keyed by aggregation name → flat metrics dict.

    Always returns dict-of-dicts (even for single-agg models). Single forward pass per batch
    feeds all aggregations the model is configured to produce (model.infer_aggs).
    """
    from anomalib.models.components.filters import GaussianBlur2d
    model.eval()

    aggs = list(model.infer_aggs) if hasattr(model, "infer_aggs") else ["default"]

    # Per-aggregation data accumulators
    per_agg = {a: {
        "img_scores": [], "topk": {3: [], 5: []}, "top1pct": [],
        "rel": {"max_minus_median": [], "max_over_mean": []},
        "px_labels": [], "px_scores": [],
        "obj_px_labels": [], "obj_px_scores": [],
    } for a in aggs}
    all_labels = []
    pixel_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device) if blur_pixel_map else None
    # Top-1% image score (added 2026-09-19, same for every mode/agg): mean of the highest 1% of pixels of the
    # returned anomaly map after the anomalib σ=4 5×5 Gaussian blur. Stored alongside the primary score.
    top1_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device)

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["gt_mask"]
        labels = batch["gt_label"]
        fg_masks = batch.get("fg_mask")  # [B, 1, H, W] or None

        all_labels.append(labels.cpu().numpy())
        preds = model(images)
        # Normalize: model returns dict for multi-agg, single InferenceBatch for single-agg
        if not isinstance(preds, dict):
            preds = {aggs[0]: preds}

        for agg_name in aggs:
            p = preds[agg_name]
            d_acc = per_agg[agg_name]
            d_acc["img_scores"].append(p.pred_score.cpu().numpy().squeeze(-1))

            amap = p.anomaly_map
            if pixel_blur is not None:
                amap = pixel_blur(amap.unsqueeze(1)).squeeze(1)
            flat = amap.view(amap.shape[0], -1)
            for k in [3, 5]:
                d_acc["topk"][k].append(flat.topk(k, dim=1).values.mean(dim=1).cpu().numpy())
            _fb = top1_blur(amap.unsqueeze(1).float()).flatten(1)
            d_acc["top1pct"].append(_fb.topk(max(1, int(0.01 * _fb.shape[1])), dim=1).values.mean(dim=1).cpu().numpy())
            max_vals = flat.max(dim=1).values
            median_vals = flat.median(dim=1).values
            mean_vals = flat.mean(dim=1)
            d_acc["rel"]["max_minus_median"].append((max_vals - median_vals).cpu().numpy())
            d_acc["rel"]["max_over_mean"].append((max_vals / (mean_vals + 1e-8)).cpu().numpy())

            for i in range(images.shape[0]):
                m = masks[i].cpu().numpy().flatten()
                s = amap[i].cpu().numpy().flatten()
                if m.max() > 0:
                    d_acc["px_labels"].append(m)
                    d_acc["px_scores"].append(s)
                if fg_masks is not None:
                    fg = fg_masks[i].cpu().numpy().flatten() > 0.5
                    gt_bin = m > 0.5
                    roi = fg | gt_bin
                    if roi.sum() > 0:
                        d_acc["obj_px_labels"].append(m[roi])
                        d_acc["obj_px_scores"].append(s[roi])

    all_labels_concat = np.concatenate(all_labels)
    metrics_per_agg = {}
    for agg_name in aggs:
        d = per_agg[agg_name]
        img_scores = np.concatenate(d["img_scores"])
        img_auroc = roc_auc_score(all_labels_concat, img_scores)
        img_cm = _confusion_matrix_at_optimal(all_labels_concat, img_scores)
        topk_aurocs = {k: roc_auc_score(all_labels_concat, np.concatenate(d["topk"][k])) for k in [3, 5]}
        rel_aurocs = {key: roc_auc_score(all_labels_concat, np.concatenate(d["rel"][key])) for key in d["rel"]}

        px_auroc = 0.0
        px_cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "threshold": 0.0}
        if d["px_labels"]:
            px_labels = np.concatenate(d["px_labels"])
            px_scores = np.concatenate(d["px_scores"])
            px_auroc = roc_auc_score(px_labels, px_scores)
            px_cm = _confusion_matrix_at_optimal((px_labels > 0.5).astype(int), px_scores)

        obj_px_auroc = 0.0
        obj_px_cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "threshold": 0.0}
        obj_px_cm_per_image = []
        if d["obj_px_labels"]:
            obj_labels = np.concatenate(d["obj_px_labels"])
            obj_scores = np.concatenate(d["obj_px_scores"])
            if obj_labels.max() > 0 and obj_labels.min() < 1:
                obj_px_auroc = roc_auc_score(obj_labels, obj_scores)
                obj_px_cm = _confusion_matrix_at_optimal((obj_labels > 0.5).astype(int), obj_scores)
            for img_labels_pi, img_scores_pi in zip(d["obj_px_labels"], d["obj_px_scores"]):
                img_labels_bin = (img_labels_pi > 0.5).astype(int)
                if img_labels_bin.max() > 0 and img_labels_bin.min() < 1:
                    obj_px_cm_per_image.append(_confusion_matrix_at_optimal(img_labels_bin, img_scores_pi))
                else:
                    obj_px_cm_per_image.append(None)

        metrics_per_agg[agg_name] = {
            "image_AUROC": img_auroc, "pixel_AUROC": px_auroc, "obj_pixel_AUROC": obj_px_auroc,
            "top1pct_AUROC": roc_auc_score(all_labels_concat, np.concatenate(d["top1pct"])),
            "image_CM": img_cm, "pixel_CM": px_cm, "obj_pixel_CM": obj_px_cm,
            "obj_pixel_CM_per_image": obj_px_cm_per_image,
            "topk3_AUROC": topk_aurocs[3], "topk5_AUROC": topk_aurocs[5],
            "max_minus_median_AUROC": rel_aurocs["max_minus_median"],
            "max_over_mean_AUROC": rel_aurocs["max_over_mean"],
        }
    return metrics_per_agg


# ── Discrepancy map visualization ────────────────────────────────────────
def _prep_sample(img_t, amap_t, target_size=None):
    """Denormalize image and upsample anomaly map. Returns (img_np, amap_np)."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = (img_t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    sz = target_size or img_np.shape[:2]
    amap_np = F.interpolate(
        amap_t.unsqueeze(0).unsqueeze(0), size=sz,
        mode="bilinear", align_corners=True,
    ).squeeze().numpy()
    return img_np, amap_np


@torch.no_grad()
def save_discrepancy_maps(model, dataloader, device, output_dir: Path, n_samples: int = 10,
                          blur_pixel_map: bool = False, label_filter: int = 1):
    """Save per-sample discrepancy map overlays for anomaly test images.

    Foreground-normalized: colorscale spans the object's own distance range,
    so intra-object anomaly contrast is visible even at early training.
    Background pixels show the original image without overlay.

    Multi-agg aware: when `model(images)` returns a dict {agg_name: InferenceBatch},
    one subdirectory is written per agg under output_dir/<agg>/. Single-agg models
    keep the legacy flat layout (output_dir/sample_*.png).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    from scipy import ndimage
    from anomalib.models.components.filters import GaussianBlur2d

    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device) if blur_pixel_map else None

    # Collect anomaly samples — image/mask/fg shared across aggs, amap is per-agg.
    base_samples = []  # (image_tensor, mask_tensor, fg_mask_tensor)
    amaps_per_agg = {}  # {agg_name: [amap_tensor, ...]}
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["gt_mask"]
        labels = batch["gt_label"]
        fg_masks = batch.get("fg_mask")

        preds = model(images)
        # Normalize: model returns dict for multi-agg, single InferenceBatch for single-agg.
        if not isinstance(preds, dict):
            preds = {"_": preds}

        # Apply optional pixel blur per agg.
        amap_dict = {}
        for agg_name, p in preds.items():
            amap = p.anomaly_map  # [B, H, W]
            if pixel_blur is not None:
                amap = pixel_blur(amap.unsqueeze(1)).squeeze(1)
            amap_dict[agg_name] = amap

        for i in range(images.shape[0]):
            if labels[i].item() == label_filter and len(base_samples) < n_samples:
                fg = fg_masks[i].cpu().squeeze() if fg_masks is not None else None
                base_samples.append((
                    images[i].cpu(),
                    masks[i].cpu().squeeze(),
                    fg,
                ))
                for agg_name, amap in amap_dict.items():
                    amaps_per_agg.setdefault(agg_name, []).append(amap[i].cpu())
        if len(base_samples) >= n_samples:
            break

    if not base_samples:
        print("  No anomaly samples found for visualization")
        return

    # Per-agg plotting: each agg gets its own subdir + own vmin/vmax (raw distance
    # ranges differ across aggregations, so a shared colorscale would be misleading).
    is_multi_agg = len(amaps_per_agg) > 1 or list(amaps_per_agg.keys()) != ["_"]
    aggs_to_plot = list(amaps_per_agg.keys())

    for agg_name in aggs_to_plot:
        amaps = amaps_per_agg[agg_name]
        samples = [(img, mask, amap, fg) for (img, mask, fg), amap in zip(base_samples, amaps)]
        if is_multi_agg and agg_name != "_":
            agg_dir = output_dir / agg_name
            agg_dir.mkdir(parents=True, exist_ok=True)
        else:
            agg_dir = output_dir
        _plot_one_agg(samples, agg_dir, plt, Normalize, cm, ndimage)
    print(f"  Saved {len(base_samples)} discrepancy maps × {len(aggs_to_plot)} agg(s) → {output_dir}/")


def _plot_one_agg(samples, output_dir, plt, Normalize, cm, ndimage):
    """Plot per-sample figures + summary grid for a single aggregation's amaps.

    Pulled out of save_discrepancy_maps so the same plotting code is reused per agg.
    Computes its own vmin/vmax (foreground-pixel 2/99 percentiles) so the colorscale
    matches the agg's actual distance range.
    """
    fg_pixels_all = []
    for _, _, amap_t, fg_t in samples:
        amap_np = amap_t.numpy()
        if fg_t is not None:
            fg_np = F.interpolate(
                fg_t.unsqueeze(0).unsqueeze(0), size=amap_np.shape,
                mode="nearest",
            ).squeeze().numpy() > 0.5
        else:
            fg_np = np.ones_like(amap_np, dtype=bool)
        fg_pixels_all.append(amap_np[fg_np])
    fg_all = np.concatenate(fg_pixels_all)
    vmin_fg = np.percentile(fg_all, 2)
    vmax_fg = np.percentile(fg_all, 99)

    def _make_overlay(img_np, amap_np, fg_np, alpha=0.55):
        """Heatmap overlay: BiRefNet FG mask, jet colormap, background = original image."""
        norm = Normalize(vmin=vmin_fg, vmax=vmax_fg, clip=True)
        heatmap = cm.jet(norm(amap_np))[:, :, :3]

        # Blend only on foreground, keep original image on background
        fg_3d = fg_np[:, :, np.newaxis].astype(np.float32)
        overlay = fg_3d * (img_np * (1 - alpha) + heatmap * alpha) + (1 - fg_3d) * img_np
        return overlay.clip(0, 1)

    def _make_gt_overlay(img_np, mask_np):
        """GT mask: red contour on image."""
        if mask_np.max() < 0.5:
            return img_np.copy()
        binary = mask_np > 0.5
        edge = ndimage.binary_dilation(binary, iterations=2).astype(float) - binary.astype(float)
        out = img_np.copy()
        out[edge > 0.5] = [1.0, 0.0, 0.0]
        return out

    def _get_fg_np(fg_t, target_shape, dilate_frac=0.10):
        """Upsample FG mask to target spatial size, dilated by dilate_frac of image edge."""
        if fg_t is not None:
            fg = F.interpolate(
                fg_t.unsqueeze(0).unsqueeze(0), size=target_shape,
                mode="nearest",
            ).squeeze().numpy() > 0.5
            # Dilate by ~10% of image size to avoid clipping object edges
            dilate_px = max(1, int(dilate_frac * max(target_shape)))
            fg = ndimage.binary_dilation(fg, iterations=dilate_px)
            return fg
        return np.ones(target_shape, dtype=bool)

    # ── Individual per-sample figures (3 panels: Input | GT | Discrepancy) ──
    for idx, (img_t, mask_t, amap_t, fg_t) in enumerate(samples):
        img_np, amap_np = _prep_sample(img_t, amap_t)
        mask_np = mask_t.numpy()
        norm = Normalize(vmin=vmin_fg, vmax=vmax_fg, clip=True)
        heatmap = cm.jet(norm(amap_np))[:, :, :3]
        overlay = (img_np * 0.45 + heatmap * 0.55).clip(0, 1)
        gt_overlay = _make_gt_overlay(img_np, mask_np)

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(gt_overlay)
        axes[1].set_title("GT Mask", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Discrepancy Map", fontsize=11)
        axes[2].axis("off")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap="jet", norm=Normalize(vmin=vmin_fg, vmax=vmax_fg))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("Feature Distance", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_dir / f"sample_{idx:03d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Summary grid (3 columns) ──
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.8 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx, (img_t, mask_t, amap_t, fg_t) in enumerate(samples):
        img_np, amap_np = _prep_sample(img_t, amap_t)
        mask_np = mask_t.numpy()
        norm = Normalize(vmin=vmin_fg, vmax=vmax_fg, clip=True)
        heatmap = cm.jet(norm(amap_np))[:, :, :3]
        overlay = (img_np * 0.45 + heatmap * 0.55).clip(0, 1)
        gt_overlay = _make_gt_overlay(img_np, mask_np)

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].axis("off")
        axes[idx, 1].imshow(gt_overlay)
        axes[idx, 1].axis("off")
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].axis("off")
        if idx == 0:
            axes[idx, 0].set_title("Input", fontsize=11)
            axes[idx, 1].set_title("GT Mask", fontsize=11)
            axes[idx, 2].set_title("Discrepancy Map", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Decoder training (post-hoc, frozen UniNet) ──────────────────────────

@torch.no_grad()
def _extract_distance_maps(model, dataloader, device):
    """Extract per-layer distance maps + GT masks from a frozen UniNet model.

    Returns list of (distance_maps: list[6× [H_i, W_i]], gt_mask: [1, 256, 256], fg_mask).
    """
    model.eval()
    all_samples = []
    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["gt_mask"].to(device)  # [B, 1, 256, 256]
        fg_masks = batch.get("fg_mask")

        # Run teacher + student
        source_target_features, bottleneck_inputs = model.teachers(images)
        bottleneck_outputs = model.bottleneck(bottleneck_inputs)
        student_features = model.student(bottleneck_outputs)
        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0], student_features[1][0], student_features[2][0],
            student_features[0][1], student_features[1][1], student_features[2][1],
        ]
        student_features = model._feature_selection(source_target_features, student_features)

        # Per-layer distance maps (keep native resolution)
        for b in range(images.shape[0]):
            dmaps = []
            for tf, sf in zip(source_target_features, student_features):
                d = (1 - F.cosine_similarity(tf[b:b+1], sf[b:b+1])).squeeze(0)  # [H, W]
                dmaps.append(d.cpu())
            gt = masks[b].cpu()  # [1, 256, 256]
            fg = fg_masks[b].cpu() if fg_masks is not None else torch.ones_like(gt)
            all_samples.append((dmaps, gt, fg))
    return all_samples


def train_decoder(model, decoder, train_loader, test_loader, device, n_epochs, lr,
                  display_name, output_dir):
    """Train decoder on frozen UniNet distance maps with dice loss."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n  Training decoder ({decoder.__class__.__name__}, "
          f"{sum(p.numel() for p in decoder.parameters())} params, {n_epochs} epochs)")

    # Extract all distance maps once (frozen UniNet)
    train_data = _extract_distance_maps(model, train_loader, device)
    test_data = _extract_distance_maps(model, test_loader, device)
    print(f"  Extracted {len(train_data)} train / {len(test_data)} test distance maps")

    # Compute per-layer normalization stats (mean/std from training data)
    all_dmaps = [[] for _ in range(6)]
    for dmaps, _, _ in train_data:
        for j, dm in enumerate(dmaps):
            all_dmaps[j].append(dm.flatten())
    dmap_means = [torch.cat(vals).mean().item() for vals in all_dmaps]
    dmap_stds = [max(torch.cat(vals).std().item(), 1e-6) for vals in all_dmaps]
    print(f"  Distance map stats: means={[f'{m:.3f}' for m in dmap_means]}, stds={[f'{s:.3f}' for s in dmap_stds]}")

    def _normalize_dmaps(dmaps):
        """Z-normalize distance maps using training stats."""
        return [(dm - dmap_means[j]) / dmap_stds[j] for j, dm in enumerate(dmaps)]

    decoder = decoder.to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    best_dice = 1.0
    train_losses, test_dices, test_opxs = [], [], []

    for epoch in range(n_epochs):
        # Train
        decoder.train()
        random.shuffle(train_data)
        epoch_loss = 0.0
        n = 0
        for dmaps, gt, fg in train_data:
            gt_dev = gt.to(device).unsqueeze(0).float()  # [1, 1, 256, 256]
            if gt_dev.shape[-1] != 256:
                gt_dev = F.interpolate(gt_dev, size=256, mode="nearest")

            # Skip normals: dice on all-zero masks just teaches "predict zero"
            if gt_dev.sum() == 0:
                continue

            dmaps_dev = [d.to(device) for d in _normalize_dmaps(dmaps)]
            pred = decoder(dmaps_dev)  # [1, 1, 256, 256]
            loss = combo_loss(pred, gt_dev)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg_loss = epoch_loss / max(n, 1)
        train_losses.append(avg_loss)

        # Eval: compute pAUROC + opx using decoder output as anomaly map
        decoder.eval()
        all_px_labels, all_px_scores = [], []
        all_obj_labels, all_obj_scores = [], []
        with torch.no_grad():
            for dmaps, gt, fg in test_data:
                dmaps_dev = [d.to(device) for d in _normalize_dmaps(dmaps)]
                pred = torch.sigmoid(decoder(dmaps_dev)).squeeze().cpu().numpy()  # [256, 256]
                gt_np = gt.squeeze().numpy()  # [256, 256]
                fg_np = fg.squeeze().numpy() > 0.5

                if gt_np.max() > 0:
                    all_px_labels.append(gt_np.flatten())
                    all_px_scores.append(pred.flatten())

                roi = fg_np | (gt_np > 0.5)
                if roi.sum() > 0:
                    all_obj_labels.append(gt_np.flatten()[roi.flatten()])
                    all_obj_scores.append(pred.flatten()[roi.flatten()])

        test_dice = avg_loss  # approximate
        opx = 0.0
        if all_obj_labels:
            obj_l = np.concatenate(all_obj_labels)
            obj_s = np.concatenate(all_obj_scores)
            if obj_l.max() > 0 and obj_l.min() < 1:
                opx = roc_auc_score(obj_l, obj_s)
        test_opxs.append(opx)

        px_auroc = 0.0
        if all_px_labels:
            px_l = np.concatenate(all_px_labels)
            px_s = np.concatenate(all_px_scores)
            if px_l.max() > 0:
                px_auroc = roc_auc_score(px_l, px_s)
        test_dices.append(px_auroc)

        if avg_loss < best_dice:
            best_dice = avg_loss
            torch.save(decoder.state_dict(), output_dir / "decoder_best.pt")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Ep {epoch+1:3d} | dice_loss={avg_loss:.4f} pAUROC={px_auroc:.3f} opx={opx:.3f}")

    # Save final + plot
    torch.save(decoder.state_dict(), output_dir / "decoder_final.pt")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(train_losses)
    axes[0].set_title("Dice Loss (train)")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(test_dices)
    axes[1].set_title("pAUROC (test)")
    axes[1].set_xlabel("Epoch")
    axes[2].plot(test_opxs)
    axes[2].set_title("Object pAUROC (test)")
    axes[2].set_xlabel("Epoch")
    fig.suptitle(f"Decoder: {decoder.__class__.__name__} ({display_name})")
    fig.tight_layout()
    fig.savefig(output_dir / "decoder_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Decoder training done. Best dice={best_dice:.4f}, final opx={test_opxs[-1]:.3f}")
    return decoder


# ── Run ─────────────────────────────────────────────────────────────────
def save_results(results, json_path):
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)


def run_mode(mode, display_name, train_ds, test_ds, results, json_path, n_epochs, batch_size,
             w_canvas=1.0, w_twosided=1.0, lambda_push=1.0, lambda_norm_cos=1.0,
             loss_form="cos", scale_kd_push=False, scale_push_contrastive=False, lw_align_only=False, weight_push_only=False,
             score_head=None, head_dropout=0.5,
             head_grad_to_student=False, native_bce=False, native_bce_per_stage=False,
             bce_all_normal=False,
             bce_variant="summed", lambda_bce=1.0, lambda_stage=0.1, train_seed=SEED,
             extra_test_datasets=None, backbone="wrn50", save_viz=True, viz_samples=10,
             decoder_type=None, decoder_epochs=50, decoder_lr=1e-3,
             blur_pixel_map=False, lkd_mode="local", margin_mode="off", lsc_pre_relu=False,
             train_apply_dfs=True, eval_apply_dfs=False,
             dinomaly_p=0.0, dinomaly_factor=0.1, dinomaly_ramp_iters=100,
             infer_aggs=None, temperature: float = 2.0,
             two_sided_lkd: bool = False, tau: float | None = None,
             canvas_align_mode: str = "uniform",
             proj_consistency: bool = False, lambda_proj: float = 0.1,
             residual_direction_alignment: bool = False, lambda_resdir: float = 0.05,
             pairrank: bool = False, lambda_pairrank: float = 0.05, pairrank_margin: float = 0.1,
             reltail: bool = False, lambda_reltail: float = 0.05, reltail_rho: float = 0.5,
             reltail_frac: float = 0.01,
             maskrank: bool = False, lambda_maskrank: float = 0.05, maskrank_margin: float = 0.1,
             adaptive_triplet: bool = False, triplet_alpha: float = 0.5,
             preserve_outside: bool = False, w_preserve: float = 1.0,
             gap_gated_outside_canvas: bool = False, lambda_gated_outcanvas: float = 0.25,
             ct_twosided: str = "off", w_ct_twosided: float = 1.0, target_teacher_lr: float = 1e-6,
             tproto: str = "off", w_tproto: float = 1.0,
             real_tproto: str = "off", w_real_tproto: float = 1.0,
             real_student_sup: bool = True,
             balanced_sampler: bool = False, stop_gradient: bool = False,
             lambda_idem: float = 0.0, idem_caao: bool = False,
             no_detach_any: bool = False, no_twosided_push: bool = False,
             use_teacher_bce: bool = False, teacher_bce_weight: float = 0.0,
             teacher_bce_feature_layer: int = 0, teacher_bce_pool: str = "avg",
             teacher_bce_loss_type: str = "bce", teacher_bce_focal_gamma: float = 2.0,
             teacher_bce_real_only: bool = False, push_to_max: bool = False,
             recon_concat_bce: bool = False, recon_bce_weight: float = 0.0,
             recon_head: str = "grouped", recon_l2: bool = False, bce_pool_max: bool = False,
             # Geometry modules (CAAO extensions)
             teacher_repulsion: bool = False,
             teacher_repulsion_weight: float = 1.0,
             teacher_repulsion_margin: float = 0.5,
             teacher_repulsion_clean_sg: bool = False,
             bottleneck_pair: bool = False,
             bottleneck_pair_weight: float = 1.0,
             bottleneck_pair_metric: str = "cosine",
             bottleneck_pair_clean_sg: bool = False,
             bottleneck_protos: bool = False,
             bottleneck_protos_k: int = 8,
             bottleneck_protos_weight: float = 1.0,
             bottleneck_protos_assignment: str = "nearest",
             bottleneck_protos_temperature: float = 0.1,
             bottleneck_protos_ema: float = 0.99,
             bottleneck_protos_sinkhorn_iters: int = 3,
             bottleneck_protos_update_mode: str = "gradient",
             bottleneck_protos_lr: float = 5e-3,
             teacher_repulsion_stages: str = "trainable_only",
             geom_diagnostics: bool = True,
             # Geom2 modules
             spm: bool = False,
             spm_k: int = 500,
             spm_weight: float = 1.0,
             spm_tau: float = 0.05,
             spm_eps: float = 0.05,
             spm_sinkhorn_iters: int = 3,
             spm_ema: float = 0.99,
             spm_init_file: str | None = None,
             paired_proto: bool = False,
             paired_proto_weight: float = 0.1,
             paired_proto_sg_target: bool = True,
             soft_pair: bool = False,
             soft_pair_weight: float = 0.01,
             trep: bool = False,
             trep_weight: float = 0.05,
             trep_adaptive_margin: bool = False,
             trep_margin: float = 0.0,
             trep_margin_file: str | None = None):
    """Train a model and evaluate against one or more test datasets.

    Args:
        extra_test_datasets: Optional dict of {"label": test_dataset} for multi-shift evaluation.
            If provided, after training against the primary test_ds, evaluates the trained model
            on each extra test dataset and stores results under "{display_name} @ {label}".
        backbone: "wrn50" or "convnext_b".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer = make_model(mode, device, w_canvas=w_canvas, w_twosided=w_twosided,
                                  lambda_push=lambda_push,
                                  lambda_norm_cos=lambda_norm_cos,
                                  loss_form=loss_form,
                                  scale_kd_push=scale_kd_push,
                                  scale_push_contrastive=scale_push_contrastive,
                                  lw_align_only=lw_align_only,
                                  weight_push_only=weight_push_only,
                                  score_head=score_head if mode == "extension" else None,
                                  head_dropout=head_dropout,
                                  head_grad_to_student=head_grad_to_student,
                                  native_bce=native_bce,
                                  native_bce_per_stage=native_bce_per_stage,
                                  bce_all_normal=bce_all_normal,
                                  bce_variant=bce_variant,
                                  lambda_bce=lambda_bce,
                                  lambda_stage=lambda_stage,
                                  train_seed=train_seed,
                                  backbone=backbone, lkd_mode=lkd_mode, margin_mode=margin_mode,
                                  lsc_pre_relu=lsc_pre_relu,
                                  train_apply_dfs=train_apply_dfs, eval_apply_dfs=eval_apply_dfs,
                                  dinomaly_p=dinomaly_p, dinomaly_factor=dinomaly_factor,
                                  dinomaly_ramp_iters=dinomaly_ramp_iters,
                                  infer_aggs=infer_aggs, temperature=temperature,
                                  two_sided_lkd=two_sided_lkd, tau=tau,
                                  canvas_align_mode=canvas_align_mode,
                                  proj_consistency=proj_consistency, lambda_proj=lambda_proj,
                                  residual_direction_alignment=residual_direction_alignment,
                                  lambda_resdir=lambda_resdir,
                                  pairrank=pairrank, lambda_pairrank=lambda_pairrank,
                                  pairrank_margin=pairrank_margin,
                                  reltail=reltail, lambda_reltail=lambda_reltail,
                                  reltail_rho=reltail_rho, reltail_frac=reltail_frac,
                                  maskrank=maskrank, lambda_maskrank=lambda_maskrank,
                                  maskrank_margin=maskrank_margin,
                                  adaptive_triplet=adaptive_triplet, triplet_alpha=triplet_alpha,
                                  preserve_outside=preserve_outside, w_preserve=w_preserve,
                                gap_gated_outside_canvas=gap_gated_outside_canvas,
                                lambda_gated_outcanvas=lambda_gated_outcanvas,
                                ct_twosided=ct_twosided, w_ct_twosided=w_ct_twosided,
                                tproto=tproto, w_tproto=w_tproto,
                                real_tproto=real_tproto, w_real_tproto=w_real_tproto,
                                real_student_sup=real_student_sup,
                                target_teacher_lr=target_teacher_lr,
                                stop_gradient=stop_gradient,
                                lambda_idem=lambda_idem,
                                idem_caao=idem_caao,
                                no_detach_any=no_detach_any,
                                no_twosided_push=no_twosided_push,
                                use_teacher_bce=use_teacher_bce,
                                teacher_bce_weight=teacher_bce_weight,
                                teacher_bce_feature_layer=teacher_bce_feature_layer,
                                teacher_bce_pool=teacher_bce_pool,
                                teacher_bce_loss_type=teacher_bce_loss_type,
                                teacher_bce_focal_gamma=teacher_bce_focal_gamma,
                                teacher_bce_real_only=teacher_bce_real_only,
                                push_to_max=push_to_max,
                                recon_concat_bce=recon_concat_bce,
                                recon_bce_weight=recon_bce_weight,
                                recon_head=recon_head,
                                recon_l2=recon_l2,
                                bce_pool_max=bce_pool_max,
                                teacher_repulsion=teacher_repulsion,
                                teacher_repulsion_weight=teacher_repulsion_weight,
                                teacher_repulsion_margin=teacher_repulsion_margin,
                                teacher_repulsion_clean_sg=teacher_repulsion_clean_sg,
                                teacher_repulsion_stages=teacher_repulsion_stages,
                                bottleneck_pair=bottleneck_pair,
                                bottleneck_pair_weight=bottleneck_pair_weight,
                                bottleneck_pair_metric=bottleneck_pair_metric,
                                bottleneck_pair_clean_sg=bottleneck_pair_clean_sg,
                                bottleneck_protos=bottleneck_protos,
                                bottleneck_protos_k=bottleneck_protos_k,
                                bottleneck_protos_weight=bottleneck_protos_weight,
                                bottleneck_protos_assignment=bottleneck_protos_assignment,
                                bottleneck_protos_temperature=bottleneck_protos_temperature,
                                bottleneck_protos_ema=bottleneck_protos_ema,
                                bottleneck_protos_sinkhorn_iters=bottleneck_protos_sinkhorn_iters,
                                bottleneck_protos_update_mode=bottleneck_protos_update_mode,
                                bottleneck_protos_lr=bottleneck_protos_lr,
                                geom_diagnostics=geom_diagnostics,
                                spm=spm, spm_k=spm_k, spm_weight=spm_weight,
                                spm_tau=spm_tau, spm_eps=spm_eps,
                                spm_sinkhorn_iters=spm_sinkhorn_iters, spm_ema=spm_ema,
                                spm_init_file=spm_init_file,
                                paired_proto=paired_proto, paired_proto_weight=paired_proto_weight,
                                paired_proto_sg_target=paired_proto_sg_target,
                                soft_pair=soft_pair, soft_pair_weight=soft_pair_weight,
                                trep=trep, trep_weight=trep_weight,
                                trep_adaptive_margin=trep_adaptive_margin,
                                trep_margin=trep_margin)

    # Geom2: load adaptive trep margin from baseline V q50 file
    _loss = getattr(model, 'loss', None)
    if _loss is None:
        _loss = getattr(model, '_ext_loss', None)
    if _loss is not None and getattr(_loss, 'trep_adaptive_margin', False):
        if trep_margin_file is not None and os.path.isfile(trep_margin_file):
            with open(trep_margin_file) as _f:
                _v_q50 = float(_f.read().strip())
            _loss._trep_margin = _v_q50
            print(f"  [trep2] adaptive margin loaded from {trep_margin_file}: {_v_q50:.4f}")
        else:
            print(f"  [trep2] WARNING: --trep-adaptive-margin set but file not found: {trep_margin_file}")

    if balanced_sampler:
        # Build per-sample weights so E[#normal] = E[#synth] per epoch.
        # Sample is normal iff label==0 (samples[i][2] == 0).
        from torch.utils.data import WeightedRandomSampler
        sample_labels = [s[2] for s in train_ds.samples]   # 0=normal, 1=anomaly (synth or real)
        n_normal = sum(1 for l in sample_labels if l == 0)
        n_anom = sum(1 for l in sample_labels if l == 1)
        if n_normal == 0 or n_anom == 0:
            print(f"  WARNING: balanced_sampler enabled but counts are normal={n_normal} anom={n_anom} — falling back to shuffle")
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
                pin_memory=True, persistent_workers=True, worker_init_fn=_worker_init_fn,
                generator=torch.Generator().manual_seed(train_seed))
        else:
            # 50/50 weighting: w_normal = 1/n_normal, w_anom = 1/n_anom; total = 2 → P(normal sample) = 0.5
            weights = [1.0/n_normal if l == 0 else 1.0/n_anom for l in sample_labels]
            num_samples = len(train_ds)   # keep epoch length unchanged
            sampler = WeightedRandomSampler(
                weights, num_samples=num_samples, replacement=True,
                generator=torch.Generator().manual_seed(train_seed))
            print(f"  Balanced sampler: n_normal={n_normal}, n_anom={n_anom}; "
                  f"num_samples={num_samples} → E[#normal]={num_samples//2}, E[#anom]={num_samples//2} per epoch")
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler, num_workers=4,
                pin_memory=True, persistent_workers=True, worker_init_fn=_worker_init_fn,
                generator=torch.Generator().manual_seed(train_seed),
            )
    elif SYNTH_EPOCH_SLICES is not None and any(s[2] == 1 and s[3] is not None for s in train_ds.samples):
        # --synthetic-epochs: epoch e = all normals + synthetic pool e (EpochPoolSampler; set_epoch in the loop)
        normal_idx = [i for i, s in enumerate(train_ds.samples) if s[2] == 0]
        syn_idx = [i for i, s in enumerate(train_ds.samples) if s[2] == 1 and s[3] is not None]
        assert syn_idx == list(range(syn_idx[0], syn_idx[0] + SYNTH_EPOCH_SLICES[-1][1])), "synthetics not contiguous"
        sampler = EpochPoolSampler(normal_idx, syn_idx[0], SYNTH_EPOCH_SLICES, train_seed)
        print(f"  EpochPoolSampler: {len(normal_idx)} normals + {SYNTH_EPOCH_SLICES[0][1] - SYNTH_EPOCH_SLICES[0][0]} "
              f"synthetics per epoch ({len(SYNTH_EPOCH_SLICES)} pools)")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=4,
            pin_memory=True, persistent_workers=True, worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(train_seed),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
            pin_memory=True, persistent_workers=True,
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(train_seed),
        )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True,
                             worker_init_fn=_worker_init_fn)

    print(f"\n{'='*60}")
    print(f"  {display_name} ({n_epochs} epochs, seed={train_seed})")
    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")
    print(f"{'='*60}")
    print(f"{'Ep':>4} | {'Loss':>8} {'iAUC':>8} {'pAUC':>8} | {'Time':>5}")
    print("-" * 45)
    sys.stdout.flush()

    torch.manual_seed(train_seed + 1000)

    # Pre-build shift dataloaders (reused every epoch)
    shift_loaders = {}
    if extra_test_datasets:
        for shift_label, shift_ds in extra_test_datasets.items():
            shift_loaders[shift_label] = DataLoader(
                shift_ds, batch_size=batch_size, shuffle=False,
                num_workers=2, pin_memory=True, persistent_workers=True,
                worker_init_fn=_worker_init_fn,
            )
            shift_key = f"{display_name} @ {shift_label}"
            results[shift_key] = {"img_aurocs": [], "pix_aurocs": []}

    def _store_per_agg(target_dict, metrics_dict):
        """Append per-aggregation metrics into target_dict["per_agg"][agg_name][metric_list]."""
        per_agg = target_dict.setdefault("per_agg", {})
        for agg_name, m in metrics_dict.items():
            agg_block = per_agg.setdefault(agg_name, {})
            agg_block.setdefault("img_aurocs", []).append(m["image_AUROC"])
            agg_block.setdefault("pix_aurocs", []).append(m["pixel_AUROC"])
            agg_block.setdefault("obj_pix_aurocs", []).append(m["obj_pixel_AUROC"])
            agg_block.setdefault("img_cms", []).append(m["image_CM"])
            agg_block.setdefault("pix_cms", []).append(m["pixel_CM"])
            agg_block.setdefault("obj_pix_cms", []).append(m["obj_pixel_CM"])
            agg_block.setdefault("obj_pix_cms_per_image", []).append(m["obj_pixel_CM_per_image"])
            agg_block.setdefault("topk3_aurocs", []).append(m["topk3_AUROC"])
            agg_block.setdefault("top1pct_aurocs", []).append(m["top1pct_AUROC"])
            agg_block.setdefault("topk5_aurocs", []).append(m["topk5_AUROC"])
            agg_block.setdefault("mmm_aurocs", []).append(m["max_minus_median_AUROC"])
            agg_block.setdefault("mom_aurocs", []).append(m["max_over_mean_AUROC"])

    for epoch in range(n_epochs):
        # Tag training-side shoot_log entries with current epoch
        if hasattr(model, "loss") and hasattr(model.loss, "_current_epoch"):
            model.loss._current_epoch = epoch
        t0 = time.time()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        TERM_LOG.clear()
        loss = train_one_epoch(model, optimizer, train_loader, device, mode)
        if TERM_LOG:   # per-epoch mean of each loss term over batches (T1_normal, T2_canvas, T3_twosided, T4_proj, total, n_syn)
            results[display_name].setdefault("term_losses", []).append({k: float(np.mean(v)) for k, v in TERM_LOG.items()})
        metrics_per_agg = evaluate(model, test_loader, device, blur_pixel_map=blur_pixel_map)
        # Primary aggregation = first one in the model's list (used for top-level back-compat keys + console print)
        primary_agg = list(metrics_per_agg.keys())[0]
        metrics = metrics_per_agg[primary_agg]

        results[display_name]["losses"].append(loss)
        # Top-level back-compat: primary aggregation's metrics
        results[display_name]["img_aurocs"].append(metrics["image_AUROC"])
        results[display_name]["pix_aurocs"].append(metrics["pixel_AUROC"])
        results[display_name].setdefault("obj_pix_aurocs", []).append(metrics["obj_pixel_AUROC"])
        results[display_name]["img_cms"].append(metrics["image_CM"])
        results[display_name]["pix_cms"].append(metrics["pixel_CM"])
        results[display_name].setdefault("obj_pix_cms", []).append(metrics["obj_pixel_CM"])
        results[display_name].setdefault("obj_pix_cms_per_image", []).append(metrics["obj_pixel_CM_per_image"])
        results[display_name].setdefault("topk3_aurocs", []).append(metrics["topk3_AUROC"])
        results[display_name].setdefault("topk5_aurocs", []).append(metrics["topk5_AUROC"])
        results[display_name].setdefault("mmm_aurocs", []).append(metrics["max_minus_median_AUROC"])
        results[display_name].setdefault("mom_aurocs", []).append(metrics["max_over_mean_AUROC"])
        results[display_name].setdefault("primary_agg", primary_agg)
        # Per-agg breakdown
        _store_per_agg(results[display_name], metrics_per_agg)

        # Shift evaluation every epoch
        shift_parts = []
        for shift_label, shift_loader in shift_loaders.items():
            shift_key = f"{display_name} @ {shift_label}"
            shift_metrics_per_agg = evaluate(model, shift_loader, device, blur_pixel_map=blur_pixel_map)
            shift_metrics = shift_metrics_per_agg[primary_agg]
            results[shift_key]["img_aurocs"].append(shift_metrics["image_AUROC"])
            results[shift_key].setdefault("top1pct_aurocs", []).append(shift_metrics["top1pct_AUROC"])
            results[shift_key]["pix_aurocs"].append(shift_metrics["pixel_AUROC"])
            results[shift_key].setdefault("obj_pix_aurocs", []).append(shift_metrics["obj_pixel_AUROC"])
            results[shift_key].setdefault("img_cms", []).append(shift_metrics["image_CM"])
            results[shift_key].setdefault("pix_cms", []).append(shift_metrics["pixel_CM"])
            results[shift_key].setdefault("obj_pix_cms", []).append(shift_metrics["obj_pixel_CM"])
            results[shift_key].setdefault("obj_pix_cms_per_image", []).append(shift_metrics["obj_pixel_CM_per_image"])
            _store_per_agg(results[shift_key], shift_metrics_per_agg)
            shift_parts.append(f"{shift_label}={shift_metrics['image_AUROC']:.3f}/{shift_metrics['pixel_AUROC']:.3f}")

        elapsed = time.time() - t0
        opx = metrics["obj_pixel_AUROC"]
        t3 = metrics["topk3_AUROC"]
        mmm = metrics["max_minus_median_AUROC"]
        mom = metrics["max_over_mean_AUROC"]
        # --save-best: keep the checkpoint of the best composite epoch (mean image AUROC over the evaluated levels)
        if globals().get("_SAVE_BEST", False):
            _lv = [results[f"{display_name} @ {sl}"]["img_aurocs"][-1] for sl in shift_loaders] or [metrics["image_AUROC"]]
            _comp = sum(_lv) / len(_lv)
            if _comp > globals().get("_BEST_COMP", -1.0):
                globals()["_BEST_COMP"] = _comp
                _bdir = json_path.parent / "checkpoints"; _bdir.mkdir(parents=True, exist_ok=True)
                _bname = display_name.replace(" ", "_").replace("/", "_").replace("—", "-")
                torch.save(model.state_dict(), _bdir / f"{_bname}_best.pt")
                json.dump({"epoch": epoch + 1, "composite_img_auroc": _comp},
                          open(_bdir / f"{_bname}_best.json", "w"), indent=1)
        shift_str = f" | {' '.join(shift_parts)}" if shift_parts else ""
        print(f"{epoch+1:4d} | {loss:8.4f} {metrics['image_AUROC']:8.4f} {metrics['pixel_AUROC']:8.4f} opx={opx:.3f} t3={t3:.3f} mm={mmm:.3f} mo={mom:.3f} | {elapsed:5.1f}s{shift_str}")
        sys.stdout.flush()
        save_results(results, json_path)

    # Save checkpoint
    ckpt_dir = json_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    safe_name = display_name.replace(" ", "_").replace("/", "_").replace("—", "-")
    ckpt_path = ckpt_dir / f"{safe_name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint → {ckpt_path}")

    # Geom2: save baseline V q50 for adaptive trep margin (baseline A run only)
    v_hist = getattr(model, "_geom_V_history", None)
    if v_hist and trep_margin_file is not None and not trep_adaptive_margin:
        import statistics
        v_q50 = statistics.median(v_hist)
        os.makedirs(os.path.dirname(trep_margin_file), exist_ok=True)
        with open(trep_margin_file, "w") as _f:
            _f.write(f"{v_q50:.6f}\n")
        print(f"  [geom2] Baseline V q50={v_q50:.4f} saved to {trep_margin_file}")

    # Test-time student-vs-teacher undershoot stats on real anomalies (only for
    # extension-mode model where this hypothesis is being tested).
    if mode == "extension":
        test_shoot_log = []
        try:
            _collect_test_shoot(model, test_loader, device, "0x", test_shoot_log)
            for shift_label, shift_loader in shift_loaders.items():
                _collect_test_shoot(model, shift_loader, device, shift_label, test_shoot_log)
        except Exception as e:
            print(f"  WARN: test shoot collection failed: {e}")
        # Dump training (per-batch, epoch-tagged) and test (per-image, per-shift) logs.
        train_shoot_log = list(getattr(model.loss, "shoot_log", []))
        shoot_path = json_path.parent / f"{safe_name}_shoot_log.json"
        with open(shoot_path, "w") as f:
            json.dump({"training": train_shoot_log, "test": test_shoot_log}, f)
        print(f"  Shoot log → {shoot_path} (train={len(train_shoot_log)}, test={len(test_shoot_log)})")

    # Discrepancy map visualization — 0x test loader + each shift loader.
    # Lets us inspect predictions under shift later without saving full per-pixel
    # score arrays.
    if save_viz and viz_samples > 0:
        viz_root = json_path.parent / "discrepancy_maps" / safe_name
        save_discrepancy_maps(model, test_loader, device, viz_root / "0x",
                              n_samples=viz_samples, blur_pixel_map=blur_pixel_map)
        for shift_label, shift_loader in shift_loaders.items():
            try:
                save_discrepancy_maps(model, shift_loader, device, viz_root / shift_label,
                                      n_samples=viz_samples, blur_pixel_map=blur_pixel_map)
            except Exception as e:
                print(f"  WARN: discrepancy maps for shift {shift_label} failed: {e}")

    # Decoder training (post-hoc, frozen UniNet)
    if decoder_type is not None:
        decoder_dir = json_path.parent / "decoder" / f"{safe_name}_{decoder_type}"
        decoder_dir.mkdir(parents=True, exist_ok=True)
        if decoder_type == "weights":
            decoder = LearnedWeights(n_layers=6)
        elif decoder_type == "concat":
            decoder = ConcatDecoder(ch=16)
        elif decoder_type == "residual":
            decoder = ResidualDecoder(ch=16)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        train_decoder(model, decoder, train_loader, test_loader, device,
                      decoder_epochs, decoder_lr, display_name, decoder_dir)

    del model, optimizer
    torch.cuda.empty_cache()


@torch.no_grad()
def compute_spm_prototypes(backbone: str, train_normals, spm_k: int, n_samples: int,
                           output_path: str, seed: int = 42):
    """Collect bottleneck tokens from normal cashew images and K-means init SPM prototypes.

    Args:
        backbone: 'wrn50' or 'convnext_b'.
        train_normals: list of normal image paths.
        spm_k: number of SPM prototypes.
        n_samples: max number of spatial tokens to collect.
        output_path: where to save the .npy file.
        seed: random seed.
    """
    from sklearn.cluster import MiniBatchKMeans
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compute_spm_init] backbone={backbone}, K={spm_k}, n_samples={n_samples}")

    # Build a minimal model with no losses to extract bottleneck features
    dummy_loss = nn.Identity()
    model_tmp = UniNetModelExtensionDFSAblate(
        student_backbone="wide_resnet50_2",
        teacher_backbone="wide_resnet50_2",
        loss=dummy_loss,
        train_apply_dfs=True,
        eval_apply_dfs=False,
    )
    if backbone == "convnext_b":
        _swap_to_convnext(model_tmp)
    model_tmp = model_tmp.to(device).eval()

    # Freeze all — we only use it for feature extraction
    for p in model_tmp.parameters():
        p.requires_grad_(False)

    from torchvision import transforms
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rng = random.Random(seed)
    paths = list(train_normals)
    rng.shuffle(paths)

    all_tokens = []
    for img_path in paths:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        # Extract bottleneck features: forward through teachers → bottleneck
        _, bottleneck_inputs = model_tmp.teachers(x)
        bn_out = model_tmp.bottleneck(bottleneck_inputs)  # [1, C, H, W]
        C = bn_out.shape[1]
        tokens = bn_out.permute(0, 2, 3, 1).reshape(-1, C).cpu().float().numpy()
        # L2-normalize
        norms = np.linalg.norm(tokens, axis=1, keepdims=True).clip(min=1e-8)
        tokens = tokens / norms
        all_tokens.append(tokens)
        if sum(t.shape[0] for t in all_tokens) >= n_samples:
            break

    all_tokens = np.concatenate(all_tokens, axis=0)[:n_samples]
    print(f"[compute_spm_init] Collected {all_tokens.shape[0]} tokens of dim {all_tokens.shape[1]}")

    kmeans = MiniBatchKMeans(n_clusters=spm_k, random_state=seed, n_init=5, max_iter=300)
    kmeans.fit(all_tokens)
    centers = kmeans.cluster_centers_  # [K, C]
    # L2-normalize
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-8)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.save(output_path, centers)
    print(f"[compute_spm_init] Saved K={spm_k} prototypes → {output_path}")
    return centers


def main():
    args = parse_args()
    global FG_ANOMALY_DIR, BRANCH_BALANCE, GRAD_BALANCE
    FG_ANOMALY_DIR = args.fg_anomaly_dir
    BRANCH_BALANCE = bool(args.branch_balance)
    GRAD_BALANCE = bool(args.grad_balance)
    assert not (BRANCH_BALANCE and GRAD_BALANCE), '--branch-balance and --grad-balance are exclusive'
    if GRAD_BALANCE:
        print('  --grad-balance: L = w L_N + (1-w) c L_S  (w = normal share, c = EMA ||grad L_N|| / EMA ||grad L_S||)')
    if BRANCH_BALANCE:
        print('  --branch-balance: L = w L_N + (1-w) c L_S  (w = normal share of batch, c = EMA L_N / EMA L_S)')
    print(f"FG masks for real anomalies: {FG_ANOMALY_DIR if FG_ANOMALY_DIR else 'LEGACY stem lookup (falls back to birefnet_masks/Normal)'}")
    global _SKIP_SUP_BCE; _SKIP_SUP_BCE = bool(getattr(args, 'no_supervised_bce', False))
    global _STUDENT_SKIPS
    _STUDENT_SKIPS = args.student_skips
    if _STUDENT_SKIPS != "off":
        print(f"  student skips: {_STUDENT_SKIPS} (source-teacher features added into the student decoder)")
    global _SAVE_BEST, _BEST_COMP
    _SAVE_BEST = bool(args.save_best); _BEST_COMP = -1.0
    global _LAMBDA_WEIGHT, _NO_UNINET_BCE, _SINGLE_TEACHER
    _SINGLE_TEACHER = bool(args.single_teacher)
    _LAMBDA_WEIGHT = args.lambda_weight
    _NO_UNINET_BCE = bool(args.no_uninet_bce)
    if args.score_mode == "paper":
        _apply_paper_score_mode()

    # Set the global mask-downsample mode used by _downsample_mask() at loss sites
    global MASK_DOWNSAMPLE_MODE
    MASK_DOWNSAMPLE_MODE = args.mask_downsample
    print(f"Mask downsample mode (loss-time): {MASK_DOWNSAMPLE_MODE}")

    # Geom2: --compute-spm-init mode — compute K-means SPM prototypes and exit
    if args.compute_spm_init:
        backbone_name = args.backbone if hasattr(args, "backbone") else "wrn50"
        out_path = (args.spm_init_file if args.spm_init_file is not None
                    else f"important_results/geom2_ablations/spm_init_{backbone_name}.npy")
        print(f"[compute_spm_init] Loading training normals...")
        train_normals_init, _, _, _, _, _ = load_data(args)
        print(f"[compute_spm_init] {len(train_normals_init)} training normals found")
        compute_spm_prototypes(
            backbone=backbone_name,
            train_normals=[p for p in train_normals_init],
            spm_k=args.spm_k,
            n_samples=args.spm_init_n_samples,
            output_path=out_path,
            seed=args.spm_init_seed,
        )
        return

    batch_size = args.batch_size
    n_epochs = args.epochs
    train_seed = args.seed if args.seed is not None else SEED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {device} ({gpu_name})")
    print(f"Data split seed: {SEED} | Training seed: {train_seed}")

    # Scope the ConvNeXt-decoder upsample knob to the ConvNeXt path via module-level global.
    if args.backbone == "convnext_b":
        global _CNX_UPSAMPLE_MODE
        _CNX_UPSAMPLE_MODE = args.cnx_upsample
        print(f"  ConvNeXt decoder upsample mode: {args.cnx_upsample}")

    train_normals, test_normals, synthetic_triples, real_anomaly_train, real_anomaly_test, extra_baseline_normals = load_data(args)

    # Training augmentation: geometric (if --augment) + CJ + blur (sampled U[0.01, train_blur]) + noise + dust
    train_aug = CashewAugmentation(
        geometric=args.augment,
        cj=args.train_cj,
        blur_sigma=args.train_blur, blur_range=True,
        noise_std=args.train_noise,
        dust_n=args.train_dust,
    )
    print(f"  Train augmentation: geometric={args.augment}, CJ={args.train_cj}, "
          f"blur=U[0,{args.train_blur}], noise={args.train_noise}, dust={args.train_dust}")

    # Primary test (1x): photometric only, no geometric
    test_aug = CashewAugmentation(
        geometric=False,
        cj=args.test_cj,
        blur_sigma=args.test_blur,
        noise_std=args.test_noise,
        fixed_brightness=args.test_fixed_brightness,
        fixed_contrast=args.test_fixed_contrast,
        fixed_saturation=args.test_fixed_saturation,
        dust_n=args.test_dust,
    )
    print(f"  Test augmentation: CJ={args.test_cj}, blur={args.test_blur}, noise={args.test_noise}, dust={args.test_dust}, "
          f"fixed(br={args.test_fixed_brightness}, con={args.test_fixed_contrast}, sat={args.test_fixed_saturation})")

    # Multi-shift mode: 0x = in-distribution (matches train 1:1), 1x-4x = increasing OOD drift.
    # 0x uses blur_range=True so blur is sampled U[0, train_blur] just like training.
    # Default: only 0x (in-dist) and 4x (max shift). Use --all-shifts for all 5 levels,
    # or --shift-levels '<json>' for fully custom.
    SHIFT_LEVELS = {
        "0x": {"cj": args.train_cj, "blur": args.train_blur, "blur_range": True, "noise": args.train_noise},
        "4x": {"cj": 0.4, "blur": 4.0, "noise": 0.005},
    }

    if args.all_shifts:
        SHIFT_LEVELS = {
            "0x": {"cj": args.train_cj, "blur": args.train_blur, "blur_range": True, "noise": args.train_noise},
            "1x": {"cj": 0.1, "blur": 1.0, "noise": 0.005},
            "2x": {"cj": 0.2, "blur": 2.0, "noise": 0.005},
            "3x": {"cj": 0.3, "blur": 3.0, "noise": 0.005},
            "4x": {"cj": 0.4, "blur": 4.0, "noise": 0.005},
        }
    elif args.shift_levels:
        import json as _json
        SHIFT_LEVELS = _json.loads(args.shift_levels)

    # Old-protocol dust schedule: test_dust scales with shift severity.
    # If --train-dust > 0 and user hasn't set --test-dust explicitly, apply per-shift
    # schedule matching the historical `blur*_dust15` experiments:
    #     0x/1x → dust=15 (matches training)
    #     2x    → dust=40
    #     3x/4x → dust=80
    OLD_DUST_SCHEDULE = {"0x": 15, "1x": 15, "2x": 40, "3x": 80, "4x": 80}
    use_old_dust_schedule = (args.train_dust > 0 and args.test_dust == 0)
    if use_old_dust_schedule:
        print(f"  Auto-applying old-protocol per-shift dust schedule: {OLD_DUST_SCHEDULE}")
        for shift_label, dust_n in OLD_DUST_SCHEDULE.items():
            if shift_label in SHIFT_LEVELS:
                SHIFT_LEVELS[shift_label]["dust"] = dust_n

    extra_test_datasets = {}
    for shift_label, shift_cfg in SHIFT_LEVELS.items():
        # Per-shift dust: prefer SHIFT_LEVELS["dust"] if present (old protocol),
        # else fall back to the global --test-dust value.
        shift_dust = shift_cfg.get("dust", args.test_dust)
        shift_aug = CashewAugmentation(
            geometric=False,
            cj=shift_cfg["cj"],
            blur_sigma=shift_cfg["blur"],
            blur_range=shift_cfg.get("blur_range", False),
            noise_std=shift_cfg["noise"],
            dust_n=shift_dust,
        )
        extra_test_datasets[shift_label] = CashewDataset(
            test_normals, real_anomaly_pairs=real_anomaly_test, aug=shift_aug,
        )
    print(f"Multi-shift evaluation: {list(SHIFT_LEVELS.keys())}")
    print(f"  Per-shift dust: " + ", ".join(
        f"{k}={SHIFT_LEVELS[k].get('dust', args.test_dust)}" for k in SHIFT_LEVELS))

    # Test dataset: same for all modes (50 normals + 50 DISTINCT real anomalies)
    test_ds = CashewDataset(test_normals, real_anomaly_pairs=real_anomaly_test, aug=test_aug)
    if getattr(args, "fixed_test_shifts", False):
        test_ds.fixed_shift_seed = 99
        for _k, _ds in extra_test_datasets.items():
            _ds.fixed_shift_seed = int(_k.rstrip('x')) if _k.rstrip('x').isdigit() else 98
        print(f"  --fixed-test-shifts: test photometric draws fixed per (image filename, shift level)")

    n_norm_tag = f"_n{len(train_normals)}" if args.train_normals is not None else ""
    aug_tag = "_aug" if args.augment else ""
    head_tag = f"_head{args.score_head}d{args.head_dropout}" if args.score_head else ""
    perturb_tag = f"_trncj{args.train_cj}b{args.train_blur}n{args.train_noise}_multishift{head_tag}"
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / f"results.json"
    else:
        json_path = SCRIPT_DIR / f"cashew_results_{args.placement}_{args.cfg_variant}{n_norm_tag}{aug_tag}{perturb_tag}.json"

    # Resume support
    if json_path.exists():
        try:
            with open(json_path) as f:
                results = json.load(f)
            existing = [k for k, v in results.items() if v.get("losses")]
            print(f"Resuming: {len(existing)} modes already done")
        except (json.JSONDecodeError, ValueError):
            results = {}
    else:
        results = {}

    # ── Dynamic-texture-blend synth setup (used by Mode 4 only when --n-dynamic-synth > 0) ──
    dyn_canvas_pool = None
    dyn_dtd_pool = None
    if args.n_dynamic_synth > 0:
        # Canvas pool
        cashew_root = Path(args.cashew_base) if hasattr(args, 'cashew_base') and args.cashew_base else \
                      CASHEW_ROOT
        if not cashew_root.exists():
            cashew_root = CASHEW_ROOT
        normal_dir = cashew_root / "Data" / "Images" / "Normal"
        splits_path = SPLITS
        with open(splits_path) as f:
            splits = json.load(f)
        if args.dynamic_canvas_pool == "train_only":
            cp_names = splits["normals"]["train"]
        elif args.dynamic_canvas_pool == "train_plus_canvas":
            cp_names = splits["normals"]["train"] + splits["normals"]["canvas"]
        elif args.dynamic_canvas_pool == "train_plus_unseen100":
            # Use train (100) + first 100 unseen normals (NOT test, NOT in cashew_100 prep)
            all_normals = [p.name for p in sorted(normal_dir.glob("*.JPG"))]
            seen = set(splits["normals"]["train"] + splits["normals"]["canvas"] + splits["normals"]["test"])
            unseen = [n for n in all_normals if n not in seen]
            cp_names = splits["normals"]["train"] + unseen[:100]
        dyn_canvas_pool = [normal_dir / n for n in cp_names if (normal_dir / n).exists()]
        # DTD pool minus excluded (used in ood_v2_small)
        excluded = set()
        if args.exclude_dtd_from_json and args.exclude_dtd_from_json.exists():
            with open(args.exclude_dtd_from_json) as f:
                excluded = set(json.load(f))   # entries like "category/file.jpg"
        dtd_all = list(args.dtd_train_dir.rglob("*.jpg"))
        dyn_dtd_pool = [p for p in dtd_all if f"{p.parent.name}/{p.name}" not in excluded]
        print(f"  Dynamic synth: n_slots={args.n_dynamic_synth}, "
              f"canvas_pool={len(dyn_canvas_pool)} normals ({args.dynamic_canvas_pool}), "
              f"dtd_pool={len(dyn_dtd_pool)} textures (excluded={len(excluded)})")

    MODES = [
        {
            "mode": "baseline",
            "name": "Mode 1 — Baseline",
            # Mode 1 uses train_normals + extra_baseline_normals so it sees the same
            # total number of training samples as Modes 2/3/4 (which include 44 anomaly
            # samples each). Off via --no-baseline-extend.
            "train_ds": CashewDataset(train_normals + extra_baseline_normals, aug=train_aug),
            "color": "#888888", "marker": "o", "linestyle": "--",
        },
        {
            "mode": "supervised",
            "name": "Mode 2 — Supervised (real)",
            "train_ds": CashewDataset(train_normals, real_anomaly_pairs=real_anomaly_train, aug=train_aug),
            "color": "#2a9d8f", "marker": "D", "linestyle": "-",
        },
        {
            "mode": "supervised",
            "name": "Mode 3 — Supervised (synthetic)",
            # Mode 3 now accepts dynsynth too: 44 anomagic + N dynsynth slots.
            "train_ds": CashewDataset(
                train_normals, synthetic_triples=synthetic_triples, aug=train_aug,
                n_dynamic_synth=args.n_dynamic_synth,
                dynamic_canvas_pool=dyn_canvas_pool,
                dtd_pool=dyn_dtd_pool,
                paired_photometric=args.paired_photometric_aug,
            ),
            "color": "#e63946", "marker": "s", "linestyle": "-",
        },
        {
            "mode": "extension",
            "name": "Mode 4 — Extension",
            "train_ds": CashewDataset(
                train_normals, synthetic_triples=synthetic_triples, aug=train_aug,
                n_dynamic_synth=args.n_dynamic_synth,
                dynamic_canvas_pool=dyn_canvas_pool,
                dtd_pool=dyn_dtd_pool,
                paired_photometric=args.paired_photometric_aug,
                synth_real_pairs=([p[0] for p in real_anomaly_train]
                                  if args.teacher_bce_real_only else None),
            ),
            "color": "#457b9d", "marker": "^", "linestyle": "-",
        },
        {
            # Mode 5: Mode 4 architecture extended to include real anomalies in the same
            # training run. Real anomalies have no canvas, so they only contribute to
            # Term 3 (two-sided spatial pull/push) — Term 2 (canvas-align) is skipped for
            # them per-sample inside ExtensionLoss. Synthetic samples still get all 3 terms.
            "mode": "extension",
            "name": "Mode 5 — Extension + real",
            "train_ds": CashewDataset(train_normals, synthetic_triples=synthetic_triples,
                                       real_anomaly_pairs=real_anomaly_train, aug=train_aug,
                                       paired_photometric=args.paired_photometric_aug),
            "color": "#9b5de5", "marker": "P", "linestyle": "-",
        },
    ]

    if args.modes is not None:
        MODES = [m for i, m in enumerate(MODES, 1) if i in args.modes]

    for cfg in MODES:
        name = cfg["name"]
        if name in results and len(results[name].get("losses", [])) >= n_epochs:
            print(f"Skipping '{name}' (already done)")
            continue

        results[name] = {
            "losses": [], "img_aurocs": [], "pix_aurocs": [], "img_cms": [], "pix_cms": [],
            "color": cfg["color"], "marker": cfg["marker"], "linestyle": cfg["linestyle"],
        }
        run_mode(cfg["mode"], name, cfg["train_ds"], test_ds, results, json_path, n_epochs, batch_size,
                 # set module flag for supervised-BCE skip (read in make_model)
                 w_canvas=args.w_canvas, w_twosided=args.w_twosided, lambda_push=args.lambda_push,
                 lambda_norm_cos=args.lambda_norm_cos,
                 loss_form=args.loss_form,
                 scale_kd_push=args.scale_kd_push or args.scale_push_contrastive,
                 scale_push_contrastive=args.scale_push_contrastive,
                 lw_align_only=args.lw_align_only,
                 weight_push_only=args.weight_push_only,
                 score_head=args.score_head,
                 head_grad_to_student=args.head_grad_to_student,
                 native_bce=args.native_bce,
                 native_bce_per_stage=args.native_bce_per_stage,
                 bce_all_normal=args.bce_all_normal,
                 bce_variant=args.bce_variant,
                 lambda_bce=args.lambda_bce,
                 lambda_stage=args.lambda_stage,
                 head_dropout=args.head_dropout, train_seed=train_seed,
                 extra_test_datasets=extra_test_datasets, backbone=args.backbone,
                 save_viz=not args.no_viz, viz_samples=args.viz_samples,
                 decoder_type=args.decoder, decoder_epochs=args.decoder_epochs,
                 decoder_lr=args.decoder_lr, blur_pixel_map=args.blur_pixel_map,
                 lkd_mode=args.lkd_mode, margin_mode=args.margin_mode,
                 lsc_pre_relu=args.lsc_pre_relu,
                 train_apply_dfs=args.train_apply_dfs, eval_apply_dfs=args.eval_apply_dfs,
                 dinomaly_p=args.dinomaly_p, dinomaly_factor=args.dinomaly_factor,
                 dinomaly_ramp_iters=args.dinomaly_ramp_iters,
                 infer_aggs=[a.strip() for a in args.infer_aggs.split(",") if a.strip()],
                 temperature=args.temperature,
                 two_sided_lkd=args.two_sided,
                 tau=args.tau,
                 canvas_align_mode=("anomaly-only" if args.canvas_align_anomaly_only else args.canvas_align_mode),
                 proj_consistency=args.proj_consistency, lambda_proj=args.lambda_proj,
                 residual_direction_alignment=args.residual_direction_alignment,
                 lambda_resdir=args.lambda_resdir,
                 pairrank=args.pairrank, lambda_pairrank=args.lambda_pairrank,
                 pairrank_margin=args.pairrank_margin,
                 reltail=args.reltail, lambda_reltail=args.lambda_reltail,
                 reltail_rho=args.reltail_rho, reltail_frac=args.reltail_frac,
                 maskrank=args.mask_rank_loss, lambda_maskrank=args.lambda_maskrank,
                 maskrank_margin=args.maskrank_margin,
                 adaptive_triplet=args.adaptive_triplet, triplet_alpha=args.triplet_alpha,
                 preserve_outside=args.preserve_outside_mask, w_preserve=args.w_preserve,
                 gap_gated_outside_canvas=args.gap_gated_outside_canvas,
                 lambda_gated_outcanvas=args.lambda_gated_outcanvas,
                 ct_twosided=args.ct_twosided, w_ct_twosided=args.w_ct_twosided,
                 tproto=args.tproto, w_tproto=args.w_tproto,
                 real_tproto=args.real_tproto, w_real_tproto=args.w_real_tproto,
                 real_student_sup=(not args.no_real_student_sup),
                 target_teacher_lr=args.target_teacher_lr,
                 balanced_sampler=args.balanced_sampler,
                 stop_gradient=args.stop_gradient,
                 lambda_idem=args.lambda_idem,
                 idem_caao=args.idem_caao,
                 no_detach_any=args.no_detach_any,
                 no_twosided_push=args.no_twosided_push,
                 use_teacher_bce=args.use_teacher_bce,
                 teacher_bce_weight=args.teacher_bce_weight,
                 teacher_bce_feature_layer=args.teacher_bce_feature_layer,
                 teacher_bce_pool=args.teacher_bce_pool,
                 teacher_bce_loss_type=args.teacher_bce_loss_type,
                 teacher_bce_focal_gamma=args.teacher_bce_focal_gamma,
                 teacher_bce_real_only=args.teacher_bce_real_only,
                 push_to_max=args.push_to_max,
                 recon_concat_bce=args.recon_concat_bce,
                 recon_bce_weight=args.recon_bce_weight,
                 recon_head=args.recon_head,
                 recon_l2=args.recon_l2,
                 bce_pool_max=args.bce_pool_max,
                 teacher_repulsion=args.teacher_repulsion,
                 teacher_repulsion_weight=args.teacher_repulsion_weight,
                 teacher_repulsion_margin=args.teacher_repulsion_margin,
                 teacher_repulsion_clean_sg=args.teacher_repulsion_clean_sg,
                 bottleneck_pair=args.bottleneck_pair,
                 bottleneck_pair_weight=args.bottleneck_pair_weight,
                 bottleneck_pair_metric=args.bottleneck_pair_metric,
                 bottleneck_pair_clean_sg=args.bottleneck_pair_clean_sg,
                 bottleneck_protos=args.bottleneck_protos,
                 bottleneck_protos_k=args.bottleneck_protos_k,
                 bottleneck_protos_weight=args.bottleneck_protos_weight,
                 bottleneck_protos_assignment=args.bottleneck_protos_assignment,
                 bottleneck_protos_temperature=args.bottleneck_protos_temperature,
                 bottleneck_protos_ema=args.bottleneck_protos_ema,
                 bottleneck_protos_sinkhorn_iters=args.bottleneck_protos_sinkhorn_iters,
                 bottleneck_protos_update_mode=args.bottleneck_protos_update_mode,
                 bottleneck_protos_lr=args.bottleneck_protos_lr,
                 teacher_repulsion_stages=args.teacher_repulsion_stages,
                 geom_diagnostics=(not args.no_geom_diagnostics),
                 spm=args.spm,
                 spm_k=args.spm_k,
                 spm_weight=args.spm_weight,
                 spm_tau=args.spm_tau,
                 spm_eps=args.spm_eps,
                 spm_sinkhorn_iters=args.spm_sinkhorn_iters,
                 spm_ema=args.spm_ema,
                 spm_init_file=args.spm_init_file,
                 paired_proto=args.paired_proto,
                 paired_proto_weight=args.paired_proto_weight,
                 paired_proto_sg_target=(not args.paired_proto_no_sg_target),
                 soft_pair=args.soft_pair,
                 soft_pair_weight=args.soft_pair_weight,
                 trep=args.trep,
                 trep_weight=args.trep_weight,
                 trep_adaptive_margin=args.trep_adaptive_margin,
                 trep_margin=args.trep_margin,
                 trep_margin_file=args.trep_margin_file)

    # Final summary
    print(f"\n{'='*72}")
    print(f"COMPLETE — {args.placement}/{args.cfg_variant}, data_seed={SEED}, train_seed={train_seed}")
    print(f"{'='*72}")
    print(f"{'Mode':<30} {'Loss':>8} {'iAUROC':>8} {'pAUROC':>8}")
    print("-" * 58)
    for k, v in results.items():
        if v.get("losses"):
            print(f"{k:<30} {v['losses'][-1]:8.4f} {v['img_aurocs'][-1]:8.4f} {v['pix_aurocs'][-1]:8.4f}")
    print(f"\nResults: {json_path}")


if __name__ == "__main__":
    main()
