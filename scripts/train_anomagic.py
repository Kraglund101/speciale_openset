"""
Anomagic training script — SD inpainting + IP-Adapter + captions.

Architecture — 2 independent conditioning pathways:
1. Text (captions) → standard UNet cross-attention K/V [frozen]
2. Visual (CLIP ref) → IP-Adapter cross-attention K/V [trained]

Training loop:
1. Sample batch: (image, mask, caption, anomaly_type)
2. Prepare reference image (full downscale or anomaly crop)
3. Encode reference: ip_adapter.encode_image(ref) → [B, K, 768]
4. Encode caption: pipeline.encode_text(caption) → [B, 77, 768]
5. Create inpainting inputs: noisy_latents + mask_latents + masked_image_latents
6. UNet forward with 2 pathways via cross_attention_kwargs
7. Masked noise prediction loss
8. Backprop through IP-Adapter params only
"""
import gc
import csv
import logging
import math
import os
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset
from src.utils.mask_utils import (
    dilate_mask_batch, downsample_mask_maxpool, create_latent_band_mask,
    unet_roundtrip_masks,
)
from src.utils.optim_utils import (
    flatten_modules, build_norm_param_id_set, split_decay_no_decay,
    L2SPRegularizer,
)
from src.utils.crop_utils import clip_crop_multi
from src.inference.generate import generate_anomagic_single
from scripts.validation_suite import run_validation_suite
from src.data.split_by_type import build_reverse_mapping

_SEED = 43


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of all trainable parameters. After each optimizer
    step, call update() to blend current weights into the shadow. Use
    store()/apply()/restore() to temporarily swap EMA weights in for inference.
    """

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.params = list(parameters)
        self.shadow = [p.data.clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def store(self):
        """Save current params before swapping in EMA for inference."""
        self.backup = [p.data.clone() for p in self.params]

    def apply(self):
        """Swap EMA weights into the model."""
        for p, s in zip(self.params, self.shadow):
            p.data.copy_(s)

    def restore(self):
        """Restore original (training) params after inference."""
        for p, b in zip(self.params, self.backup):
            p.data.copy_(b)
        del self.backup

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state):
        self.shadow = [s.to(self.params[0].device) for s in state["shadow"]]
        self.decay = state["decay"]


def _worker_init_fn(worker_id):
    """Seed Python random + numpy in DataLoader workers (required on Windows/spawn)."""
    worker_seed = _SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_anomagic(
    splits_dir: Path,
    save_dir: Path,
    captions_file: Path,
    n_steps: int = 20000,
    batch_size: int = 12,
    lr: float = 1e-4,
    lr_pretrained: float = 5e-5,
    lambda_sp: float = 0.0,
    device: str = "cuda",
    save_every: int = 5000,
    no_early_snapshots: bool = False,
    keep_all_checkpoints: bool = False,
    anomaly_types: list = None,
    exclude_sources: list = None,
    data_root: Path = None,
    master_json: Path = None,
    ip_adapter_type: str = "plus",
    ip_adapter_k: int = 16,
    ip_adapter_scale: float = 1.0,
    reference_mode: str = "full",
    mask_visual: bool = True,
    # CLIP dilation (transition zone around anomaly for visual tokens)
    clip_dilation_min_r: int = 2,
    clip_dilation_max_r: int = 10,
    # Latent-space band dilation
    band_mode: int = 2,
    # Loss weighting
    loss_core_ratio: float = 0.8,
    # Disable masked loss: weight_map becomes uniform (standard MSE over all pixels)
    no_masked_loss: bool = False,
    # Uniform loss inside dilated mask (no core/band split), 0 outside.
    uniform_mask_loss: bool = True,
    # Cross-attention mask type
    binary_cross_attn_mask: bool = False,
    # Binary IP-CA mask = core only (band gets 0). Isolates "where IP fires" without
    # touching UNet inpainting mask / T2I band channel / loss weighting.
    binary_cross_attn_mask_core: bool = False,
    # Core-only training: replace all dilated spatial signals with core-only (ablation)
    core_only: bool = False,
    # T2I-Adapter mode
    t2i_adapter_mode: str = "cascade",
    t2i_pair_inject: str = "last",
    t2i_decoder_inject: bool = False,
    # LoRA on UNet
    lora_rank: int = 0,
    lora_alpha: int = 16,
    lora_lr: float = 5e-5,
    lora_mode: str = "all",  # "all" = self+cross, "cross" = cross-attention only
    # Unfreeze W_Q and W_O: "" = off, "mid_up" = mid+up cross-attn,
    # "all_cross" = all cross-attn, "all" = all cross + self attn
    unfreeze_qo: str = "",
    unfreeze_qo_lr: float = 1e-5,
    # Conditioning dropout for CFG (mutually exclusive per-sample)
    drop_image_prob: float = 0.15,
    drop_text_prob: float = 0.05,
    drop_both_prob: float = 0.0,
    # Data augmentation
    augment: bool = False,
    # Resume from checkpoint
    resume_dir: Path = None,
    # Visual token processing mode
    visual_mode: int = 3,
    # Gate mode for masked self-attention
    learnable_gates: bool = True,
    # Force gates to 1.0 with normal projection init (block active from step 0)
    force_gates: bool = False,
    # Number of self-attention transformer layers
    sa_num_layers: int = 3,
    # Number of attention heads
    sa_num_heads: int = 12,
    # Multi-crop (2-group CLIP cropping)
    multi_crop: bool = False,
    # CFG direction for sample generation
    cfg_mode: str = "visual",
    # Headless mode (no live viewer)
    no_live_viewer: bool = False,
    # Noise offset (helps with global brightness/darkness)
    noise_offset: float = 0.05,
    # Timestep sampling strategy
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    # Optimizer
    optimizer_type: str = "adamw",
    # LR scheduler
    lr_scheduler: str = "none",       # "none" or "cosine"
    lr_min: float = 0.0,              # min LR for cosine annealing
    # Context corruption: zero masked_image_latents with this probability
    corrupt_context: float = 0.0,
    # x0-prediction pseudo-diffusion ablation
    x0_objective: bool = False,
    x0_start_ratio: float = 1.0,
    x0_end_ratio: float = 1.0,
    x0_no_context: bool = False,
    x0_warmup_frac: float = 0.2,
    x0_hold_frac: float = 0.1,
    # Attention diagnostics interval (ip_entropy, norm ratio)
    diag_interval: int = 1,
    # CLIP-UNet alignment: roundtripped masks + role embeddings
    clip_align: bool = True,
    clip_core_only: bool = False,
    # Validation suite
    val_data_dir: Path = None,
    val_panels: list = None,
    no_val_suite: bool = False,
    # EMA (Exponential Moving Average)
    ema_decay: float = 0.9999,
    # Base model (SDXL port): "sd_1.5" or "sdxl"; resolution None = model default (512 / 1024)
    sd_version: str = "sd_1.5",
    resolution: int = None,
    # Gradient accumulation: batch_size stays the EFFECTIVE batch (same data sampling), each batch is split into
    # grad_accum equal micro-batches on the GPU (memory), gradients summed with 1/grad_accum weights.
    grad_accum: int = 1,
    gradient_checkpointing: bool = False,
):
    """Train Anomagic: IP-Adapter + captions (2 pathways)."""
    # --no-val-suite: skip every validation-suite call (step-0, snapshots, checkpoints, final)
    _run_val = (lambda *a, **kw: None) if no_val_suite else run_validation_suite
    from src.models.base import SDVersion, get_sd_config
    if resolution is None:
        resolution = get_sd_config(SDVersion(sd_version)).image_size
    if batch_size % grad_accum != 0:
        raise ValueError(f"batch_size {batch_size} must be divisible by grad_accum {grad_accum}")
    if grad_accum > 1 and lambda_sp > 0:
        raise ValueError("L2-SP is not supported together with gradient accumulation")
    if sd_version != "sd_1.5":
        # The SDXL port trains the IP-Adapter only; the validation suite / inference helpers are SD 1.5-only.
        if t2i_adapter_mode != "off" or lora_rank > 0 or unfreeze_qo:
            raise ValueError("SDXL port trains the IP-Adapter only: use --t2i-adapter-mode off, no LoRA / unfreeze-qo")
        if x0_objective or corrupt_context > 0:
            raise ValueError("x0-objective / context corruption ablations are not ported to SDXL")
        no_val_suite = True
        _run_val = lambda *a, **kw: None
    # Seed everything for reproducibility across ablation runs
    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANOMAGIC TRAINING — IP-Adapter + Captions")
    print("=" * 70)
    print(f"Base model: {sd_version} @ {resolution}x{resolution}")
    print(f"Learning rate (scratch): {lr}")
    print(f"Learning rate (pretrained): {lr_pretrained}")
    print(f"L2-SP lambda: {lambda_sp}")
    print(f"Batch size: {batch_size} (grad_accum {grad_accum} -> micro-batch {batch_size // grad_accum}); "
          f"gradient checkpointing: {gradient_checkpointing}")
    print(f"Training steps: {n_steps}")
    print(f"IP-Adapter type: {ip_adapter_type} (K={ip_adapter_k})")
    print(f"IP-Adapter scale: {ip_adapter_scale}")
    print(f"Reference mode: {reference_mode}")
    print(f"Captions file: {captions_file}")
    print(f"Masked visual cross-attention: {mask_visual}")
    print(f"CLIP dilation: min_r={clip_dilation_min_r}, max_r={clip_dilation_max_r}")
    print(f"Band mode: {band_mode} (Chebyshev latent-space dilation)")
    if no_masked_loss:
        print(f"Loss ratio: UNIFORM (masked loss DISABLED — standard MSE over all pixels)")
    else:
        print(f"Loss ratio: {loss_core_ratio:.0%} core / {1-loss_core_ratio:.0%} band")
    if binary_cross_attn_mask_core:
        _ca_mask_str = "binary core"
    elif binary_cross_attn_mask:
        _ca_mask_str = "binary dilated"
    else:
        _ca_mask_str = "soft alpha"
    print(f"Cross-attn mask: {_ca_mask_str}")
    print(f"T2I-Adapter: {t2i_adapter_mode}")
    print(f"LoRA: {'off' if lora_rank == 0 else f'rank={lora_rank}, alpha={lora_alpha}, mode={lora_mode}, lr={lora_lr}'}")
    print(f"Unfreeze W_Q/W_O: {unfreeze_qo + ', lr=' + str(unfreeze_qo_lr) if unfreeze_qo else 'off'}")
    print(f"Conditioning dropout: image={drop_image_prob:.0%}, text={drop_text_prob:.0%}, both={drop_both_prob:.0%}")
    print(f"Data augmentation: {augment}")
    print(f"Visual mode: {visual_mode}")
    print(f"Learnable gates: {learnable_gates} (SA layers: {sa_num_layers}, heads: {sa_num_heads})")
    print(f"Multi-crop: {multi_crop}")
    print(f"CLIP-UNet alignment: {clip_align} (core_only={clip_core_only})")
    print(f"Optimizer: {optimizer_type}")
    print(f"EMA: {'off' if ema_decay <= 0 else f'decay={ema_decay}'}")
    print(f"Noise offset: {noise_offset}")
    print(f"Timestep sampling: {timestep_sampling}" +
          (f" (mean={logit_normal_mean}, std={logit_normal_std})" if timestep_sampling == "logit_normal" else ""))
    if x0_objective:
        warmup_steps = int(n_steps * x0_warmup_frac)
        hold_steps = int(n_steps * x0_hold_frac)
        print(f"x0-objective: ratio {x0_start_ratio:.0%}->{x0_end_ratio:.0%}, "
              f"warmup={warmup_steps} ({x0_warmup_frac:.0%}), hold={hold_steps} ({x0_hold_frac:.0%}), "
              f"no_context={x0_no_context}")
    if corrupt_context > 0:
        if x0_start_ratio != x0_end_ratio and not x0_objective:
            warmup_steps = int(n_steps * x0_warmup_frac)
            hold_steps = int(n_steps * x0_hold_frac)
            print(f"Context dropout: annealed {x0_start_ratio:.0%}->{x0_end_ratio:.0%}, "
                  f"warmup={warmup_steps} ({x0_warmup_frac:.0%}), hold={hold_steps} ({x0_hold_frac:.0%}), per-batch")
        else:
            print(f"Context dropout: {corrupt_context:.0%} per-batch (fixed)")
    if resume_dir:
        print(f"Resume from: {resume_dir}")
    print()

    # =========================================
    # Load Data
    # =========================================
    dataset = AnomalyDataset(
        splits_dir,
        anomaly_types=anomaly_types,
        exclude_sources=exclude_sources,
        data_root=data_root,
        captions_file=captions_file,
        return_reference=True,
        image_size=resolution,
        reference_mode=reference_mode,
        reference_crop_size=224,
        augment=augment,
        multi_crop=multi_crop,
        band_mode=band_mode,
        clip_align=clip_align,
        clip_core_only=clip_core_only,
    )

    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return

    n_captions = sum(1 for s in dataset.samples if s.get("caption"))
    print(f"Samples with captions: {n_captions}/{len(dataset.samples)}")
    # DATA GUARD (added 2026-09-28): the training samples must be EXACTLY the master dataset - nothing more, nothing
    # less - and must contain no VisA / cashew image (the evaluation data). Aborts before any model is built.
    if master_json is not None:
        import json as _json
        with open(master_json, encoding="utf-8") as _f:
            _master = _json.load(_f)
        _want = {(e["image_path"].replace("\\", "/"), e["mask_path"].replace("\\", "/")) for e in _master}
        _root = Path(data_root) if data_root else None
        def _rel(p):
            p = Path(p)
            return (p.relative_to(_root) if _root and p.is_absolute() else p).as_posix()
        _have = {(_rel(s["image_path"]), _rel(s["mask_path"])) for s in dataset.samples}
        _extra, _missing = _have - _want, _want - _have
        _visa = [s["image_path"] for s in dataset.samples if "visa" in s["image_path"].lower() or "cashew" in s["image_path"].lower()]
        print(f"DATA GUARD vs {master_json}: {len(_have)} samples | master {len(_want)} | extra {len(_extra)} | "
              f"missing {len(_missing)} | VisA/cashew paths {len(_visa)} | duplicates {len(dataset.samples) - len(_have)}")
        if _extra or _missing or _visa or len(dataset.samples) != len(_have):
            raise SystemExit(f"DATA GUARD FAILED - extra e.g. {sorted(_extra)[:3]}, missing e.g. {sorted(_missing)[:3]}, "
                             f"VisA/cashew e.g. {_visa[:3]}")
        print("DATA GUARD PASSED: training data == master dataset, no VisA/cashew")
    if n_captions == 0:
        print("WARNING: No captions found! Check --captions-file path.")

    # =========================================
    # Initialize Pipeline
    # =========================================
    print("\nInitializing models...")

    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline(sd_version, device=device)
    pipeline.load_pipeline()

    # Freeze everything on the pipeline (UNet, VAE, text encoder + embeddings)
    pipeline.freeze_all()
    if gradient_checkpointing:
        # Frozen UNet, but gradients must still flow through it to the IP-Adapter K/V projections.
        pipeline.unet.enable_gradient_checkpointing()

    # =========================================
    # Initialize IP-Adapter
    # =========================================
    print(f"\nLoading IP-Adapter ({ip_adapter_type}, K={ip_adapter_k}, visual_mode={visual_mode})...")
    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=ip_adapter_type,
        num_tokens=ip_adapter_k,
        scale=ip_adapter_scale,
        load_pretrained=True,
        mask_visual=mask_visual,
        visual_mode=visual_mode,
        learnable_gates=learnable_gates,
        force_gates=force_gates,
        sa_num_layers=sa_num_layers,
        sa_num_heads=sa_num_heads,
    )
    ip_adapter.freeze_image_encoder()

    # =========================================
    # Initialize T2I-Adapter (optional spatial conditioning)
    # =========================================
    t2i_adapter = None
    if t2i_adapter_mode != "off":
        from src.models.t2i_adapter import T2IAdapter
        t2i_adapter = T2IAdapter(
            in_channels=2,
            injection_mode=t2i_adapter_mode,
            pair_injection=t2i_pair_inject,
            decoder_inject=t2i_decoder_inject,
        ).to(device)
        n_t2i = sum(p.numel() for p in t2i_adapter.parameters())
        print(f"\nT2I-Adapter ({t2i_adapter_mode}, pair_inject={t2i_pair_inject}, "
              f"decoder_inject={t2i_decoder_inject}): {n_t2i:,} params")
        if t2i_pair_inject == "all":
            t2i_adapter.register_dense_hooks(pipeline.unet)
            print("  Dense pair injection hooks registered (encoder).")
        if t2i_decoder_inject:
            t2i_adapter.register_decoder_hooks(pipeline.unet)
            print(f"  Decoder hooks registered (pair_inject={t2i_pair_inject}).")

    # =========================================
    # Initialize LoRA on UNet (optional)
    # =========================================
    if lora_rank > 0:
        from peft import LoraConfig
        if lora_mode == "cross":
            # Cross-attention only: attn2 layers (text/IP conditioning)
            target_modules = r".*\.attn2\.(to_k|to_q|to_v|to_out\.0)"
        elif lora_mode == "mid_up":
            # All attention (attn1+attn2) in mid_block and up_blocks only
            # Must be a single string (not list) for PEFT to use regex matching
            # re.fullmatch requires matching the ENTIRE key, so prefix with .*
            target_modules = r"(mid_block|up_blocks)\..*\.(attn1|attn2)\.(to_k|to_q|to_v|to_out\.0)"
        elif lora_mode == "mid_up_notext":
            # mid+up blocks: full self-attention + cross-attention Q/out only (no text K/V)
            # attn1: to_q, to_k, to_v, to_out.0 (self-attn, no text involved)
            # attn2: to_q, to_out.0 only (skip to_k/to_v which project text embeddings)
            target_modules = r"(mid_block|up_blocks)\..*\.(attn1\.(to_k|to_q|to_v|to_out\.0)|attn2\.(to_q|to_out\.0))"
        else:
            # All attention layers: self-attention (attn1) + cross-attention (attn2)
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.unet.add_adapter(lora_config)
        # Cast LoRA params to fp32 (they inherit fp16 from UNet base weights)
        for p in pipeline.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        # Exclude IP-Adapter attn_processor params (already in Group A)
        ip_proc_ids = {id(p) for proc in ip_adapter.attn_processors.values() for p in proc.parameters()}
        lora_params = [p for p in pipeline.unet.parameters() if p.requires_grad and id(p) not in ip_proc_ids]
        n_lora = sum(p.numel() for p in lora_params)
        print(f"\nLoRA (rank={lora_rank}, alpha={lora_alpha}, mode={lora_mode}, lr={lora_lr}): {n_lora:,} params")
    else:
        lora_params = []

    # =========================================
    # Unfreeze W_Q and W_O in UNet attention layers
    # Modes: mid_up = mid+up cross-attn, all_cross = all cross-attn,
    #        all = all cross + self attn
    # =========================================
    qo_params = []
    qo_params_named = []  # (name, param) pairs for checkpoint save/load
    if unfreeze_qo:
        for name, param in pipeline.unet.named_parameters():
            # Block filter: mid_up restricts to mid+up, others allow all blocks
            if unfreeze_qo == "mid_up":
                if not (name.startswith("mid_block") or name.startswith("up_blocks")):
                    continue

            # Attention type filter
            if unfreeze_qo in ("mid_up", "all_cross"):
                # Cross-attention only (attn2)
                if "attn2" not in name:
                    continue
            else:  # "all": cross + self attention
                if "attn1" not in name and "attn2" not in name:
                    continue

            # Only W_Q and W_O
            if not (name.endswith("to_q.weight") or name.endswith("to_q.bias")
                    or "to_out.0.weight" in name or "to_out.0.bias" in name):
                continue
            param.requires_grad_(True)
            param.data = param.data.float()
            qo_params.append(param)
            qo_params_named.append((name, param))
        n_qo = sum(p.numel() for p in qo_params)
        mode_desc = {"mid_up": "mid+up cross-attn", "all_cross": "all cross-attn",
                     "all": "all cross+self attn"}[unfreeze_qo]
        print(f"\nUnfrozen W_Q + W_O ({mode_desc}, lr={unfreeze_qo_lr}): {n_qo:,} params")

    # Ensure all trainable modules are fp32 (pretrained weights may load as fp16)
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # =========================================
    # Build Optimizer Parameter Groups
    # =========================================
    # Group A (pretrained): image_projection + attn_processors  → low LR, L2-SP
    # Group B (scratch):    masked_self_attn + t2i_adapter       → higher LR, weight decay
    # Group C (gates/scales): gate scalars + t2i scales          → higher LR, no weight decay

    group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
    group_b_modules = []
    if hasattr(ip_adapter, 'masked_self_attn'):
        group_b_modules.append(ip_adapter.masked_self_attn)
    if t2i_adapter is not None:
        group_b_modules.append(t2i_adapter)

    norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)

    # --- Group A: pretrained IP-Adapter ---
    a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

    # --- Group C: gate/scale scalars (extract BEFORE splitting B) ---
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

    # --- Group B: from-scratch modules (minus Group C params) ---
    b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
    b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
    b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

    # Build param groups
    param_groups = [
        {"params": a_decay,     "lr": lr_pretrained, "weight_decay": 1e-4, "label": "A_decay"},
        {"params": a_no_decay,  "lr": lr_pretrained, "weight_decay": 0.0, "label": "A_no_decay"},
        {"params": b_decay,     "lr": lr,            "weight_decay": 1e-4, "label": "B_decay"},
        {"params": b_no_decay,  "lr": lr,            "weight_decay": 0.0, "label": "B_no_decay"},
        {"params": group_c_params, "lr": lr,         "weight_decay": 0.0, "label": "C_gates"},
    ]

    # Optional Group D: LoRA (exclude qo_params to avoid duplicates)
    if lora_params:
        qo_ids = {id(p) for p in qo_params}
        lora_params_clean = [p for p in lora_params if id(p) not in qo_ids]
        param_groups.append(
            {"params": lora_params_clean, "lr": lora_lr, "weight_decay": 1e-4, "label": "D_lora"}
        )

    # Optional Group E: Unfrozen W_Q + W_O
    if qo_params:
        param_groups.append(
            {"params": qo_params, "lr": unfreeze_qo_lr, "weight_decay": 0.0, "label": "E_qo"}
        )

    # --- Verify: no duplicate params across groups ---
    all_group_ids = []
    for pg in param_groups:
        for p in pg["params"]:
            pid = id(p)
            assert pid not in all_group_ids, f"Duplicate param in group {pg['label']}"
            all_group_ids.append(pid)

    # --- Verify: total param count matches expected ---
    # Skip when LoRA is active — peft restructures UNet module hierarchy,
    # making independent param counting unreliable. Duplicate check above
    # is the primary safety net.
    if not lora_params:
        group_numel = sum(p.numel() for pg in param_groups for p in pg["params"])
        expected_params = ip_adapter.get_trainable_parameters()
        if t2i_adapter is not None:
            expected_params.extend(list(t2i_adapter.parameters()))
        expected_params.extend(qo_params)
        expected_numel = sum(p.numel() for p in expected_params)
        assert group_numel == expected_numel, (
            f"Param count mismatch: groups={group_numel:,} vs expected={expected_numel:,}"
        )

    trainable_params = [p for pg in param_groups for p in pg["params"]]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n  Trainable parameters: {n_trainable:,}")

    # Per-group diagnostic
    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"    {pg['label']:12s}: {n:>10,} params  (lr={pg['lr']:.1e}, wd={pg['weight_decay']:.1e})")

    if optimizer_type == "prodigy":
        from prodigyopt import Prodigy
        # Prodigy auto-tunes LR: set lr=1.0 for all groups.
        # Zero out all per-group weight_decay — Prodigy handles its own regularization.
        for pg in param_groups:
            pg["lr"] = 1.0
            pg["weight_decay"] = 0.0
        optimizer = Prodigy(
            param_groups,
            lr=1.0,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            safeguard_warmup=True,
            use_bias_correction=True,
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # =========================================
    # LR Scheduler (pretrained groups only)
    # =========================================
    scheduler = None
    if lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        # Cosine annealing on pretrained IP-Adapter groups (A) only.
        # A: lr_pretrained → lr_min. All other groups: constant LR.
        lr_min_ratio = lr_min / lr_pretrained
        def _cosine_factor(step):
            return lr_min_ratio + 0.5 * (1.0 - lr_min_ratio) * (1.0 + math.cos(math.pi * step / n_steps))
        lambdas = []
        for pg in param_groups:
            if pg["label"].startswith("A_"):
                lambdas.append(_cosine_factor)
            else:
                lambdas.append(lambda step: 1.0)
        scheduler = LambdaLR(optimizer, lr_lambda=lambdas)
        print(f"\n  LR scheduler: cosine annealing on pretrained groups (A) only")
        print(f"    {lr_pretrained:.1e} → {lr_min:.1e} over {n_steps} steps")
        print(f"    Scratch/gates/LoRA/QO groups: constant LR")

    # =========================================
    # L2-SP Regularizer (anchored to pretrained weights, before resume)
    # =========================================
    l2sp_reg = None
    if lambda_sp > 0 and len(a_decay) > 0:
        l2sp_reg = L2SPRegularizer(a_decay, lambda_sp=lambda_sp)
        print(f"\n  L2-SP: {l2sp_reg.total_elements:,} elements, lambda={lambda_sp:.1e}")

    # AMP autocast: bf16 forward pass, trainable params stay fp32.
    # AMP — bf16 has same exponent range as fp32, no GradScaler needed
    use_amp = True
    amp_dtype = torch.bfloat16

    pipeline.text_encoder.float()
    if getattr(pipeline, "text_encoder_2", None) is not None:
        pipeline.text_encoder_2.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # =========================================
    # Resume from checkpoint (if requested)
    # =========================================
    start_step = 0
    resumed_losses = []
    _resumed_ema_state = None

    if resume_dir is not None:
        resume_dir = Path(resume_dir)
        print(f"\nResuming from checkpoint: {resume_dir}")

        # Load IP-Adapter weights
        ip_ckpt = resume_dir / "ip_adapter.pt"
        if ip_ckpt.exists():
            ip_adapter.load_finetuned(resume_dir)
            # Re-cast to fp32 after loading (weights may have been saved as fp16)
            ip_adapter.masked_self_attn.float()
            ip_adapter.image_projection.float()
            for proc in ip_adapter.attn_processors.values():
                proc.float()
            print(f"  Loaded IP-Adapter weights from {ip_ckpt}")
        else:
            print(f"  WARNING: {ip_ckpt} not found, starting with pretrained weights")

        # Load training state (optimizer, step, T2I, LoRA)
        state_path = resume_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)

            # Optimizer
            try:
                optimizer.load_state_dict(state["optimizer"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  WARNING: Could not load optimizer state ({e}), using fresh optimizer")

            # Step counter
            start_step = state.get("step", 0)
            print(f"  Resuming from step {start_step}")

            # Previous losses (for continuity in loss plot)
            resumed_losses = state.get("losses", [])
            if resumed_losses:
                print(f"  Loaded {len(resumed_losses)} previous loss values")

            # T2I-Adapter
            if t2i_adapter is not None and "t2i_adapter" in state:
                t2i_adapter.load_state_dict(state["t2i_adapter"])
                print(f"  Loaded T2I-Adapter weights")

            # LoRA
            if lora_rank > 0 and "lora_weights" in state:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(pipeline.unet, state["lora_weights"])
                # Re-cast LoRA params to fp32
                for p in pipeline.unet.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
                print(f"  Loaded LoRA weights")

            # Unfrozen W_Q/W_O
            if unfreeze_qo and "qo_weights" in state:
                unet_params = dict(pipeline.unet.named_parameters())
                loaded = 0
                for name, saved_tensor in state["qo_weights"].items():
                    if name in unet_params:
                        unet_params[name].data = saved_tensor.float().to(device)
                        loaded += 1
                print(f"  Loaded {loaded} unfrozen W_Q/W_O tensors")

            # LR scheduler
            if scheduler is not None and "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
                print(f"  Loaded LR scheduler state")

            # EMA state (loaded after EMA is initialized below)
            _resumed_ema_state = state.get("ema", None)
            if _resumed_ema_state is not None:
                print(f"  Found EMA state in checkpoint (will load after init)")

            del state
            torch.cuda.empty_cache()
        else:
            print(f"  WARNING: {state_path} not found, starting fresh (IP-Adapter weights loaded)")

    # =========================================
    # EMA (Exponential Moving Average)
    # =========================================
    ema = None
    if ema_decay > 0:
        ema = EMA(trainable_params, decay=ema_decay)
        if _resumed_ema_state is not None:
            ema.load_state_dict(_resumed_ema_state)
            _resumed_ema_state = None
            print(f"\n  EMA loaded from checkpoint: decay={ema_decay}, {len(trainable_params)} params")
        else:
            print(f"\n  EMA initialized fresh: decay={ema_decay}, {len(trainable_params)} params")

    remaining_steps = n_steps - start_step
    if remaining_steps <= 0:
        print(f"\nAlready completed {start_step}/{n_steps} steps. Nothing to do.")
        return

    # =========================================
    # Step-0 baseline samples (pretrained, before any training)
    # =========================================
    if start_step == 0 and not no_val_suite:
        print("\nGenerating step-0 baseline samples (pretrained IP-Adapter, no training)...")
        torch.cuda.empty_cache()
        try:
            _run_val(
                pipeline, ip_adapter, step=0,
                output_dir=save_dir, device=device,
                band_mode=band_mode, t2i_adapter=t2i_adapter,
                clip_align=clip_align, data_root=data_root,
                val_data_dir=val_data_dir, captions_file=captions_file,
                panels=val_panels,
            )
        except RuntimeError as e:
            print(f"  Warning: baseline sample generation failed ({e})")
        torch.cuda.empty_cache()

    # =========================================
    # Training Loop
    # =========================================
    print(f"\nStarting training for {remaining_steps} remaining steps ({start_step}/{n_steps} done)...")

    # Save run config for live viewer title
    import json
    config_file = save_dir / "run_config.json"
    run_config = {
        "visual_mode": visual_mode,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_lr": lora_lr,
        "lora_mode": lora_mode,
        "unfreeze_qo": unfreeze_qo,
        "unfreeze_qo_lr": unfreeze_qo_lr,
        "batch_size": batch_size,
        "lr_pretrained": lr_pretrained,
        "lr": lr,
        "drop_image_prob": drop_image_prob,
        "drop_text_prob": drop_text_prob,
        "drop_both_prob": drop_both_prob,
        "band_mode": band_mode,
        "noise_offset": noise_offset,
        "timestep_sampling": timestep_sampling,
        "sa_num_layers": sa_num_layers,
        "sa_num_heads": sa_num_heads,
        "learnable_gates": learnable_gates,
        "force_gates": force_gates,
        "multi_crop": multi_crop,
        "ip_adapter_type": ip_adapter_type,
        "ip_adapter_k": ip_adapter_k,
        "optimizer": optimizer_type,
        "lr_scheduler": lr_scheduler,
        "lr_min": lr_min,
        "corrupt_context": corrupt_context,
        "x0_objective": x0_objective,
        "x0_start_ratio": x0_start_ratio,
        "x0_end_ratio": x0_end_ratio,
        "x0_no_context": x0_no_context,
        "x0_warmup_frac": x0_warmup_frac,
        "x0_hold_frac": x0_hold_frac,
        "clip_align": clip_align,
        "clip_core_only": clip_core_only,
        "mask_visual": mask_visual,
        "no_early_snapshots": no_early_snapshots,
        "uniform_mask_loss": uniform_mask_loss,
        "no_masked_loss": no_masked_loss,
        "binary_cross_attn_mask": binary_cross_attn_mask,
        "binary_cross_attn_mask_core": binary_cross_attn_mask_core,
        "ema_decay": ema_decay,
        "t2i_pair_inject": t2i_pair_inject,
        "t2i_decoder_inject": t2i_decoder_inject,
        "sd_version": sd_version,
        "resolution": resolution,
        "grad_accum": grad_accum,
        "gradient_checkpointing": gradient_checkpointing,
    }
    with open(config_file, "w") as cf:
        json.dump(run_config, cf, indent=2)

    # Launch live loss viewer
    loss_file = save_dir / "losses.txt"
    # Pre-populate loss file with resumed losses so live viewer shows full history
    if resumed_losses:
        with open(loss_file, "w") as lf:
            lf.writelines(f"{v}\n" for v in resumed_losses)
    if not no_live_viewer:
        _launch_live_loss(loss_file)

    losses = resumed_losses
    type_losses = defaultdict(list)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True,
                        drop_last=True, worker_init_fn=_worker_init_fn,
                        generator=g)
    data_iter = iter(infinite_loader(loader))

    # Cache empty-string text encoding for conditioning dropout (constant, no need to recompute)
    with torch.no_grad():
        uncond_text = pipeline.encode_text([""] * batch_size, enable_grad=False)

    # Per-meta-group loss tracking (semantic groups from split_by_type.py)
    _type_to_group = build_reverse_mapping()
    meta_group_losses = defaultdict(list)  # group_name → list of recent losses
    meta_group_file = save_dir / "meta_group_loss.csv"
    meta_group_fh = open(meta_group_file, "a")
    # Sorted group names for consistent CSV column order
    _all_groups = sorted(set(_type_to_group.values()))
    if meta_group_fh.tell() == 0:
        meta_group_fh.write("step," + ",".join(_all_groups) + "\n")

    # Keep loss file open for the entire run (avoids Windows file locking issues)
    loss_fh = open(loss_file, "a")

    # Stats file: gate values + LR per step (CSV)
    stats_file = save_dir / "stats.csv"
    stats_fh = open(stats_file, "a")
    if stats_fh.tell() == 0:
        stats_fh.write("step,loss,lr_pretrained,lr_scratch,attn_gate,ff_gate,l2sp,core_loss,band_loss,x0_ratio,x0_loss,eps_loss,grad_norm,ctx_drop,emb_global_norm,emb_anomaly_norm,emb_normal_norm,loss_keep,loss_drop_vis,loss_drop_txt,ip_entropy,ip_norm_ratio,ip_key_ratio\n")

    pbar = tqdm(range(start_step, n_steps), desc="Training", initial=start_step, total=n_steps)
    skipped_nan = 0
    prev_ckpt_dir = None

    for step in pbar:
        batch = next(data_iter)

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        references = batch["reference"].to(device)
        batch_types = batch["anomaly_type"]
        batch_captions = batch.get("caption", [""] * batch_size)
        # CLIP masks (from crop_utils with UNet roundtrip alignment)
        clip_masks = batch["clip_mask"].to(device) if "clip_mask" in batch else None
        clip_core_masks = batch["clip_core_mask"].to(device) if "clip_core_mask" in batch else None
        # Multi-crop data (None when multi_crop=False)
        references_2 = batch["reference_2"].to(device) if "reference_2" in batch else None
        clip_masks_2 = batch["clip_mask_2"].to(device) if "clip_mask_2" in batch else None
        clip_core_masks_2 = batch["clip_core_mask_2"].to(device) if "clip_core_mask_2" in batch else None
        group_valid = batch["group_valid"].to(device) if "group_valid" in batch else None

        optimizer.zero_grad()

        # Annealing schedule: warmup → linear anneal → hold
        # Used by x0-objective OR context dropout (whichever is active)
        warmup_steps = int(n_steps * x0_warmup_frac)
        hold_steps = int(n_steps * x0_hold_frac)
        anneal_end = n_steps - hold_steps
        if step < warmup_steps:
            annealed_ratio = x0_start_ratio
        elif step >= anneal_end:
            annealed_ratio = x0_end_ratio
        else:
            progress = (step - warmup_steps) / max(anneal_end - warmup_steps - 1, 1)
            annealed_ratio = x0_start_ratio + (x0_end_ratio - x0_start_ratio) * progress

        if x0_objective:
            x0_ratio = annealed_ratio
        else:
            x0_ratio = 0.0

        # Context dropout: fixed rate OR annealed (when x0 annealing params are set)
        if corrupt_context > 0 and not x0_objective:
            if x0_start_ratio != x0_end_ratio:
                ctx_drop = annealed_ratio  # annealed
            else:
                ctx_drop = corrupt_context  # fixed
        else:
            ctx_drop = 0.0

        # AMP autocast wraps forward pass only; loss math runs in fp32
        # to avoid fp16 overflow in sum-of-squares reductions.
        loss_diff, loss_extras = _anomagic_loss_maybe_accumulated(
            grad_accum,
            pipeline, ip_adapter, images, masks, references,
            batch_types, batch_captions,
            reference_mode=reference_mode,
            clip_dilation_min_r=clip_dilation_min_r,
            clip_dilation_max_r=clip_dilation_max_r,
            band_mode=band_mode,
            loss_core_ratio=loss_core_ratio,
            no_masked_loss=no_masked_loss,
            uniform_mask_loss=uniform_mask_loss,
            binary_cross_attn_mask=binary_cross_attn_mask,
            binary_cross_attn_mask_core=binary_cross_attn_mask_core,
            core_only=core_only,
            t2i_adapter=t2i_adapter,
            drop_image_prob=drop_image_prob,
            drop_text_prob=drop_text_prob,
            drop_both_prob=drop_both_prob,
            clip_masks=clip_masks,
            clip_core_masks=clip_core_masks,
            references_2=references_2,
            clip_masks_2=clip_masks_2,
            clip_core_masks_2=clip_core_masks_2,
            group_valid=group_valid,
            noise_offset=noise_offset,
            timestep_sampling=timestep_sampling,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            corrupt_context=ctx_drop,
            x0_ratio=x0_ratio,
            x0_no_context=x0_no_context,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            collect_diagnostics=(step % diag_interval == 0),
            uncond_text=uncond_text,
        )

        # L2-SP regularization (fp32, outside autocast)
        l2sp_val = 0.0
        if l2sp_reg is not None:
            l2sp_loss = l2sp_reg.compute()
            l2sp_val = l2sp_loss.item()
            loss = loss_diff + l2sp_loss
        else:
            loss = loss_diff

        if torch.isnan(loss) or torch.isinf(loss):
            skipped_nan += 1
            del loss_diff, loss, loss_extras
            gc.collect()
            torch.cuda.empty_cache()
            continue

        if grad_accum == 1:
            loss.backward()  # (accumulated micro-batches already ran their backward passes)
        torch.cuda.synchronize()  # TDR prevention
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()

        # Guard against NaN/inf gradients — prevents single bad backward pass
        # from killing ALL parameters via NaN propagation through optimizer.
        if not math.isfinite(grad_norm):
            optimizer.zero_grad()
            skipped_nan += 1
            if grad_norm != grad_norm:  # NaN check
                logging.warning(f"Step {step}: NaN gradient detected, skipping update")
            continue

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()

        loss_val = loss_diff.item()
        losses.append(loss_val)

        # Append to loss file for live viewer
        loss_fh.write(f"{loss_val}\n")
        loss_fh.flush()

        # Log stats every step
        attn_gate = ff_gate = 0.0
        if hasattr(ip_adapter, 'masked_self_attn'):
            sa = ip_adapter.masked_self_attn
            if sa.learnable_gates:
                for layer in sa.layers:
                    attn_gate = layer["attn_gate"].gate.item()
                    if "ff_gate" in layer:
                        ff_gate = layer["ff_gate"].gate.item()
                    break  # first layer only
            else:
                attn_gate = ff_gate = 1.0  # fixed gates
        lr_pre = optimizer.param_groups[0]["lr"]
        lr_scr = optimizer.param_groups[2]["lr"]  # B_decay group
        core_l = loss_extras.get("core_loss", 0.0)
        band_l = loss_extras.get("band_loss", 0.0)
        x0_l = loss_extras.get("x0_loss", 0.0)
        eps_l = loss_extras.get("eps_loss", 0.0)
        l_keep = loss_extras.get("loss_keep", 0.0)
        l_dv = loss_extras.get("loss_drop_vis", 0.0)
        l_dt = loss_extras.get("loss_drop_txt", 0.0)
        ip_ent = loss_extras.get("ip_entropy", 0.0)
        ip_nr = loss_extras.get("ip_norm_ratio", 0.0)
        ip_kr = loss_extras.get("ip_key_ratio", 0.0)
        # Role embedding norms
        emb_g = emb_a = emb_n = 0.0
        if hasattr(ip_adapter, 'masked_self_attn'):
            sa = ip_adapter.masked_self_attn
            emb_g = sa.emb_global.data.norm().item()
            emb_a = sa.emb_anomaly.data.norm().item()
            emb_n = sa.emb_normal.data.norm().item()
        stats_fh.write(f"{step},{loss_val},{lr_pre},{lr_scr},{attn_gate},{ff_gate},{l2sp_val},{core_l},{band_l},{x0_ratio},{x0_l},{eps_l},{grad_norm},{ctx_drop},{emb_g},{emb_a},{emb_n},{l_keep},{l_dv},{l_dt},{ip_ent},{ip_nr},{ip_kr}\n")
        stats_fh.flush()

        for t in batch_types:
            type_losses[t].append(loss_val)
            group = _type_to_group.get(t)
            if group:
                meta_group_losses[group].append(loss_val)

        # Log per-meta-group rolling average every 100 steps
        if (step + 1) % 100 == 0:
            vals = []
            for grp in _all_groups:
                recent = meta_group_losses[grp][-200:]
                vals.append(f"{sum(recent)/len(recent):.5f}" if recent else "")
            meta_group_fh.write(f"{step}," + ",".join(vals) + "\n")
            meta_group_fh.flush()

        if step % 50 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            pbar.set_postfix(loss=f"{avg:.4f}")

        # Free unreferenced CUDA tensors every step to prevent VRAM creep
        del images, masks, references, clip_masks, clip_core_masks, references_2, clip_masks_2, clip_core_masks_2, group_valid, loss, loss_diff, loss_extras
        #torch.cuda.empty_cache()

        # Early sample snapshots (no checkpoint, just samples + loss plot)
        if (not no_early_snapshots) and (step + 1) in (500, 1000, 2000) and (step + 1) % save_every != 0:
            torch.cuda.empty_cache()
            if ema is not None:
                ema.store()
                ema.apply()
            try:
                _run_val(
                    pipeline, ip_adapter, step=step + 1,
                    output_dir=save_dir, device=device,
                    band_mode=band_mode, t2i_adapter=t2i_adapter,
                    clip_align=clip_align, data_root=data_root,
                    val_data_dir=val_data_dir, captions_file=captions_file,
                    panels=val_panels,
                )
            except RuntimeError as e:
                print(f"  Warning: sample generation failed ({e}), continuing training...")
            if ema is not None:
                ema.restore()
            torch.cuda.empty_cache()
            save_loss_plot(losses, save_dir / "stats.csv",
                           save_dir / f"loss_{step + 1}.png")

        # Save checkpoints + samples
        if (step + 1) % save_every == 0:
            print(f"\n  Checkpoint at step {step + 1}...", flush=True)
            ckpt_dir = save_dir / f"checkpoint_{step + 1}"
            save_anomagic_checkpoint(
                ip_adapter, optimizer,
                ckpt_dir,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                unet=pipeline.unet if lora_rank > 0 else None,
                step=step + 1,
                losses=losses,
                qo_params_named=qo_params_named if qo_params_named else None,
                ema=ema,
                scheduler=scheduler,
            )
            # Delete previous checkpoint (keep only latest + final), unless --keep-all-checkpoints
            if not keep_all_checkpoints and prev_ckpt_dir is not None and prev_ckpt_dir.exists():
                import shutil
                shutil.rmtree(prev_ckpt_dir)
            prev_ckpt_dir = ckpt_dir
            torch.cuda.empty_cache()
            if ema is not None:
                ema.store()
                ema.apply()
            try:
                _run_val(
                    pipeline, ip_adapter, step=step + 1,
                    output_dir=save_dir, device=device,
                    band_mode=band_mode, t2i_adapter=t2i_adapter,
                    clip_align=clip_align, data_root=data_root,
                    val_data_dir=val_data_dir, captions_file=captions_file,
                    panels=val_panels,
                )
            except RuntimeError as e:
                print(f"  Warning: sample generation failed ({e}), continuing training...")
            if ema is not None:
                ema.restore()
            torch.cuda.empty_cache()
            save_loss_plot(losses, save_dir / "stats.csv",
                           save_dir / f"loss_{step + 1}.png")

    loss_fh.close()
    stats_fh.close()
    meta_group_fh.close()

    # =========================================
    # Final Outputs
    # =========================================
    print("\nTraining complete!")
    print(f"  Total steps: {n_steps} (resumed from {start_step})" if start_step > 0 else f"  Total steps: {n_steps}")
    print(f"  Skipped (NaN/Inf): {skipped_nan}")
    print(f"  Valid losses recorded: {len(losses)}")

    # Final loss plot (same 2x3 layout as checkpoints)
    save_loss_plot(losses, save_dir / "stats.csv", save_dir / "training_loss.png")

    # Final checkpoint (with EMA state)
    save_anomagic_checkpoint(
        ip_adapter, optimizer,
        save_dir / "checkpoint_final",
        band_mode=band_mode,
        t2i_adapter=t2i_adapter,
        unet=pipeline.unet if lora_rank > 0 else None,
        step=n_steps,
        losses=losses,
        qo_params_named=qo_params_named if qo_params_named else None,
        ema=ema,
        scheduler=scheduler,
    )

    # Only checkpoint_final is used for inference — delete the last intermediate, unless --keep-all-checkpoints.
    if not keep_all_checkpoints and prev_ckpt_dir is not None and prev_ckpt_dir.exists():
        import shutil
        shutil.rmtree(prev_ckpt_dir)
        print(f"Deleted redundant intermediate checkpoint: {prev_ckpt_dir.name}")

    # Final sample grid (use EMA weights)
    torch.cuda.empty_cache()
    if ema is not None:
        ema.store()
        ema.apply()
    try:
        _run_val(
            pipeline, ip_adapter, step=n_steps,
            output_dir=save_dir, device=device,
            band_mode=band_mode, t2i_adapter=t2i_adapter,
            clip_align=clip_align, data_root=data_root,
            val_data_dir=val_data_dir, captions_file=captions_file,
            panels=val_panels,
        )
    except RuntimeError as e:
        print(f"  Warning: final samples failed ({e})")
    if ema is not None:
        ema.restore()

    print(f"\nResults saved to: {save_dir}")


def _anomagic_loss_maybe_accumulated(grad_accum, pipeline, ip_adapter, images, masks, references,
                                     batch_types, captions, **kwargs):
    """compute_anomagic_loss on the whole batch (grad_accum=1: unchanged path) or on grad_accum equal
    micro-batches, running backward(loss_i / grad_accum) per micro-batch.

    With accumulation the returned loss is DETACHED (mean of the micro-batch losses = full-batch loss, since the
    loss is a mean over samples) and the caller must not call backward again. Diagnostics are averaged over
    micro-batches. A non-finite micro-batch loss returns NaN at once (the caller skips the step and
    optimizer.zero_grad() at the next step clears the partial gradients).
    """
    if grad_accum == 1:
        return compute_anomagic_loss(pipeline, ip_adapter, images, masks, references, batch_types, captions,
                                     **kwargs)
    bsz = images.shape[0]
    mb = bsz // grad_accum
    per_sample_keys = ("clip_masks", "clip_core_masks", "references_2", "clip_masks_2", "clip_core_masks_2",
                       "group_valid", "uncond_text")
    total, extras_sum = 0.0, defaultdict(float)
    for m in range(grad_accum):
        sl = slice(m * mb, (m + 1) * mb)
        kw = dict(kwargs)
        for k in per_sample_keys:
            if kw.get(k) is not None:
                kw[k] = kw[k][sl]
        loss_m, extras_m = compute_anomagic_loss(pipeline, ip_adapter, images[sl], masks[sl], references[sl],
                                                 list(batch_types)[sl], list(captions)[sl], **kw)
        if not torch.isfinite(loss_m):
            return torch.tensor(float("nan"), device=images.device), {}
        (loss_m / grad_accum).backward()
        total += loss_m.item() / grad_accum
        for k, v in extras_m.items():
            extras_sum[k] += (v.item() if torch.is_tensor(v) else v) / grad_accum
        del loss_m, extras_m
    return torch.tensor(total, device=images.device), dict(extras_sum)


def compute_anomagic_loss(
    pipeline, ip_adapter, images, masks, references,
    batch_types, captions,
    reference_mode: str = "full",
    clip_dilation_min_r: int = 2,
    clip_dilation_max_r: int = 10,
    band_mode: int = 1,
    loss_core_ratio: float = 0.8,
    no_masked_loss: bool = False,
    uniform_mask_loss: bool = True,
    binary_cross_attn_mask: bool = False,
    binary_cross_attn_mask_core: bool = False,
    core_only: bool = False,
    t2i_adapter=None,
    drop_image_prob: float = 0.15,
    drop_text_prob: float = 0.05,
    drop_both_prob: float = 0.0,
    clip_masks=None,
    clip_core_masks=None,
    # Multi-crop (Mode 4)
    references_2=None,
    clip_masks_2=None,
    clip_core_masks_2=None,
    group_valid=None,
    # Noise offset
    noise_offset: float = 0.0,
    # Timestep sampling
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    # Context corruption
    corrupt_context: float = 0.0,
    # x0-prediction pseudo-diffusion ablation
    x0_ratio: float = 0.0,
    x0_no_context: bool = False,
    # AMP
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    # Diagnostics
    collect_diagnostics: bool = False,
    # Cached uncond text embedding (precomputed, avoids per-step recomputation)
    uncond_text=None,
):
    """
    Compute masked diffusion loss with 2 conditioning pathways.

    Pathway 1 (text):    caption → frozen UNet cross-attn K/V
    Pathway 2 (visual):  reference image → CLIP → IP-Adapter K/V [trained]

    Multi-crop: when references_2 is provided, encode both crops independently
    through the SAME IP-Adapter, concatenate embeddings → [B, 2K, dim], and
    build null_token_mask to suppress invalid group tokens.

    Mask strategy:
    - CLIP mask: tight image-space dilation for anomaly-focused feature extraction
    - UNet masks: latent-space band dilation (Chebyshev, 1-2 pixels at 64×64)
    - Cross-attn: soft alpha_map (default) or binary dilated mask
    - Loss weight: flat alpha + ratio scaling (no scipy)
    """
    batch_size = images.shape[0]
    device = images.device

    images = images.float()
    masks_fp32 = masks.float()

    # --- Forward pass under AMP autocast (bf16) ---
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):

        # --- CLIP mask ---
        # When augmented: clip_masks come from crop_utils (already cropped to 224×224)
        # When non-augmented: dilate the full-res mask for masked self-attention
        if clip_masks is None:
            clip_dilated = dilate_mask_batch(masks_fp32, clip_dilation_min_r, clip_dilation_max_r)
        else:
            clip_dilated = None  # not needed — clip_masks used directly

        # Encode image to latent space
        latents = pipeline.encode_image(images)

        # Timestep sampling
        if timestep_sampling == "logit_normal":
            # SD3-style: sample from logit-normal distribution (bell over t)
            u = torch.sigmoid(logit_normal_mean + logit_normal_std * torch.randn(batch_size, device=device))
            timesteps = (u * 1000).long().clamp(0, 999)
        elif timestep_sampling == "logit_normal_sigma2":
            # Bell over noise power sigma^2 (variance-preserving analog of SD3's bell-in-t).
            # Sample u ~ logit-normal, treat u as target sigma^2, invert schedule to find t.
            u = torch.sigmoid(logit_normal_mean + logit_normal_std * torch.randn(batch_size, device=device))
            alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)  # [1000], decreasing in t
            sigma2_all = 1.0 - alphas_cumprod                              # [1000], increasing in t
            timesteps = torch.searchsorted(sigma2_all, u).clamp(0, 999).long()
        else:
            # Uniform (standard DDPM)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)

        noise = torch.randn_like(latents)
        if noise_offset > 0:
            # Per-channel constant offset: helps model handle global brightness shifts
            noise += noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=device, dtype=latents.dtype
            )
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

        # --- UNet masks: maxpool to 64×64, then latent-space band dilation ---
        latent_h = latents.shape[-1]
        kernel = masks_fp32.shape[-1] // latent_h
        core_mask_64 = F.max_pool2d(masks_fp32, kernel_size=kernel)
        core_mask_64 = (core_mask_64 > 0.5).float()

        dilated_binary_64, alpha_map_64, weight_map_64, band_mask_64 = create_latent_band_mask(
            core_mask_64, band_mode,
        )

        if no_masked_loss:
            weight_map_64 = torch.ones_like(weight_map_64)
        elif uniform_mask_loss:
            # Uniform inside dilated mask (core = band = 1.0), zero outside.
            # Per-sample normalization in the loss reduction already divides by sum(weight),
            # so this gives mean-MSE over masked pixels, no core/band bias.
            weight_map_64 = dilated_binary_64.float()

        # --- Pathway 1: Text conditioning (captions only, no TI tokens) ---
        prompts = []
        for i, atype in enumerate(batch_types):
            cap = captions[i] if i < len(captions) else ""
            if cap:
                prompts.append(cap)
            else:
                type_word = atype.replace("_", " ")
                prompts.append(f"a photo of a {type_word} defect")

        text_emb = pipeline.encode_text(prompts, enable_grad=False)

        # --- Pathway 2: Visual conditioning via IP-Adapter ---
        ref_01 = (references + 1.0) / 2.0
        if clip_masks is not None:
            # clip_masks are [B, 1, 224, 224] from crop_utils (UNet-roundtripped dilated)
            clip_mask = clip_masks
            clip_core = clip_core_masks  # [B, 1, 224, 224] core mask for role embeddings
        else:
            clip_mask = clip_dilated if reference_mode == "full" else None
            # Use raw (undilated) mask as core so global token pools anomaly-only
            clip_core = masks_fp32 if reference_mode == "full" else None
        ip_image_embeds = ip_adapter.encode_image(ref_01, mask=clip_mask, core_mask=clip_core)  # [B, K, cross_attn_dim]

        # --- Multi-crop: encode second crop and concatenate ---
        null_token_mask = None
        if references_2 is not None:
            ref_2_01 = (references_2 + 1.0) / 2.0
            clip_mask_2 = clip_masks_2 if clip_masks_2 is not None else None
            clip_core_2 = clip_core_masks_2 if clip_core_masks_2 is not None else None
            ip_image_embeds_2 = ip_adapter.encode_image(ref_2_01, mask=clip_mask_2, core_mask=clip_core_2)  # [B, K, dim]

            # Concatenate: [B, K, dim] + [B, K, dim] → [B, 2K, dim]
            K = ip_image_embeds.shape[1]
            ip_image_embeds = torch.cat([ip_image_embeds, ip_image_embeds_2], dim=1)  # [B, 2K, dim]

            # Build null_token_mask [B, 2K]: 1=valid, 0=null (from invalid groups)
            if group_valid is not None:
                # group_valid [B, 2]: validity per group
                # Expand: group 1 valid → first K tokens valid, group 2 valid → last K tokens valid
                mask_1 = group_valid[:, 0:1].expand(-1, K)  # [B, K]
                mask_2 = group_valid[:, 1:2].expand(-1, K)  # [B, K]
                null_token_mask = torch.cat([mask_1, mask_2], dim=1)  # [B, 2K]

        # --- Conditioning dropout for CFG (mutually exclusive, per-sample) ---
        # Matches IP-Adapter training: single rand(), cumulative thresholds.
        # Image drop = mask-multiply (avoids in-place on grad tensor).
        # Text drop = empty-string encoding (text_emb has no grad, safe in-place).
        # drop_mode: 0=keep_both, 1=drop_image, 2=drop_text, 3=drop_both
        drop_modes = [0] * batch_size
        if (drop_image_prob + drop_text_prob + drop_both_prob) > 0 and batch_size > 0:
            # Use cached uncond_text if provided, else compute (fallback)
            if uncond_text is None:
                uncond_text = pipeline.encode_text([""] * batch_size, enable_grad=False)
            drop_image_mask = torch.ones(batch_size, 1, 1, device=device)
            for i in range(batch_size):
                r = random.random()
                if r < drop_image_prob:
                    drop_image_mask[i] = 0.0
                    drop_modes[i] = 1
                elif r < drop_image_prob + drop_text_prob:
                    text_emb[i] = uncond_text[i]
                    drop_modes[i] = 2
                elif r < drop_image_prob + drop_text_prob + drop_both_prob:
                    drop_image_mask[i] = 0.0
                    text_emb[i] = uncond_text[i]
                    drop_modes[i] = 3
            ip_image_embeds = ip_image_embeds * drop_image_mask  # new tensor, no in-place

        # --- Inpainting inputs — latent-space dilated mask defines regeneration region ---
        # Core-only ablation: swap dilated → core for mask channel + masked_image_latents.
        _inpaint_mask_64 = core_mask_64 if core_only else dilated_binary_64
        mask_latents = _inpaint_mask_64  # binary [B, 1, 64, 64]
        unet_mask_512 = F.interpolate(_inpaint_mask_64, size=images.shape[-2:], mode='nearest')
        masked_image = images * (1 - unet_mask_512)
        masked_image_latents = pipeline.encode_image(masked_image)

        # --- x0-objective: replace noisy input for pseudo-path samples ---
        x0_samples = None
        if x0_ratio > 0:
            x0_samples = torch.rand(batch_size, device=device) < x0_ratio  # [B] bool
            if x0_samples.any():
                noisy_pseudo = pipeline.scheduler.add_noise(masked_image_latents, noise, timesteps)
                noisy_latents = torch.where(
                    x0_samples[:, None, None, None].expand_as(noisy_latents),
                    noisy_pseudo, noisy_latents,
                )
                # Optionally zero inpainting context for x0-path samples only
                if x0_no_context:
                    masked_image_latents = torch.where(
                        x0_samples[:, None, None, None].expand_as(masked_image_latents),
                        torch.zeros_like(masked_image_latents),
                        masked_image_latents,
                    )
            else:
                x0_samples = None  # no samples selected, skip x0 logic in loss

        # Context dropout: per-batch zeroing of masked_image_latents
        if corrupt_context > 0.0 and random.random() < corrupt_context:
            masked_image_latents = torch.zeros_like(masked_image_latents)

        model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

        # --- T2I-Adapter spatial features (AMP autocast handles dtype) ---
        t2i_kwargs = {}
        if t2i_adapter is not None:
            # Core-only: zero the band input channel and use core for the internal mask.
            _band_in = torch.zeros_like(band_mask_64) if core_only else band_mask_64
            _t2i_mask = core_mask_64 if core_only else dilated_binary_64
            t2i_input = torch.cat([core_mask_64, _band_in], dim=1)  # [B, 2, 64, 64]
            t2i_features = t2i_adapter(t2i_input, mask=_t2i_mask)
            t2i_kwargs = t2i_adapter.prepare_unet_kwargs(t2i_features)
            if t2i_adapter.pair_injection == "all" or t2i_adapter.decoder_inject:
                t2i_adapter.set_hook_features(t2i_features)

        # --- UNet forward with both pathways via cross_attention_kwargs ---
        cross_attn_kwargs = {"ip_adapter_image_embeds": ip_image_embeds}
        # Cross-attn mask: core-only (full ablation) > binary core (IP-CA only) > binary dilated > soft alpha (default)
        if core_only:
            cross_attn_kwargs["ip_adapter_mask"] = core_mask_64
        elif binary_cross_attn_mask_core:
            cross_attn_kwargs["ip_adapter_mask"] = core_mask_64
        elif binary_cross_attn_mask:
            cross_attn_kwargs["ip_adapter_mask"] = dilated_binary_64
        else:
            cross_attn_kwargs["ip_adapter_mask"] = alpha_map_64
        # Multi-crop null masking
        if null_token_mask is not None:
            cross_attn_kwargs["null_token_mask"] = null_token_mask
        # Attention diagnostics (ip_entropy, norm ratio)
        diagnostics = None
        if collect_diagnostics:
            diagnostics = {}
            cross_attn_kwargs["diagnostics"] = diagnostics

        noise_pred = pipeline.unet_forward(
            model_input,
            timesteps,
            text_emb,
            cross_attention_kwargs=cross_attn_kwargs,
            **t2i_kwargs,
        ).sample.float()
        if t2i_adapter is not None and (
            t2i_adapter.pair_injection == "all" or t2i_adapter.decoder_inject
        ):
            t2i_adapter.clear_hook_features()
        torch.cuda.synchronize()  # TDR prevention

    # --- Loss computed OUTSIDE autocast (fp32) to avoid fp16 overflow in reductions ---
    weight_expanded = weight_map_64.expand_as(noise_pred)

    if x0_samples is not None and x0_samples.any():
        # Mixed batch: x0-space loss for pseudo-path, ε-space for standard
        alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device=device)
        alpha_t = alphas_cumprod[timesteps]  # [B]
        sqrt_alpha = alpha_t.sqrt()[:, None, None, None]
        sqrt_one_minus_alpha = (1 - alpha_t).sqrt()[:, None, None, None]

        # x0 prediction from ε prediction
        x0_pred = (noisy_latents - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        x0_diff = (x0_pred - latents.float()) ** 2
        eps_diff = (noise_pred - noise.float()) ** 2

        # Select per sample
        x0_sel = x0_samples[:, None, None, None].expand_as(noise_pred)
        diff = torch.where(x0_sel, x0_diff, eps_diff)
    else:
        diff = (noise_pred - noise.float()) ** 2

    weighted_diff = diff * weight_expanded

    # Per-sample loss: each sample contributes equally regardless of anomaly size
    per_sample_w = weight_expanded.sum(dim=(1, 2, 3)).clamp(min=1e-8)  # [B]
    per_sample_loss = weighted_diff.sum(dim=(1, 2, 3)) / per_sample_w  # [B]
    if per_sample_loss.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {}
    loss = per_sample_loss.mean()

    # --- Diagnostics: core vs band loss (detached, no grad impact) ---
    # Per-sample averaging so 0.8*core + 0.2*band = loss_keep exactly.
    # Only from fully-conditioned samples (drop_mode == 0).
    with torch.no_grad():
        dm = torch.tensor(drop_modes, device=device)
        both_cond = (dm == 0)  # [B]

        # Per-sample core/band MSE (mean over pixels per sample, then avg over both-cond)
        # Band uses weight_map weights (not binary mask) to respect inner/outer scaling,
        # ensuring 0.8*core + 0.2*band = loss_keep exactly.
        core_exp = core_mask_64.expand_as(noise_pred)   # [B,C,H,W]
        band_w_exp = (weight_map_64 * (1.0 - core_mask_64)).expand_as(noise_pred)  # [B,C,H,W]
        core_per = (diff * core_exp).sum(dim=(1, 2, 3)) / core_exp.sum(dim=(1, 2, 3)).clamp(min=1e-8)  # [B]
        band_per = (diff * band_w_exp).sum(dim=(1, 2, 3)) / band_w_exp.sum(dim=(1, 2, 3)).clamp(min=1e-8)  # [B]
        if both_cond.any():
            core_loss_val = core_per[both_cond].mean()
            band_loss_val = band_per[both_cond].mean()
        else:
            core_loss_val = torch.tensor(0.0, device=device)
            band_loss_val = torch.tensor(0.0, device=device)

        # x0 vs eps per-objective loss (only when mixed batch)
        x0_loss_val = 0.0
        eps_loss_val = 0.0
        if x0_samples is not None and x0_samples.any():
            x0_sel_4d = x0_samples[:, None, None, None].expand_as(noise_pred)
            eps_sel_4d = ~x0_sel_4d
            x0_w = (weight_expanded * x0_sel_4d.float())
            eps_w = (weight_expanded * eps_sel_4d.float())
            x0_w_sum = x0_w.sum().clamp(min=1e-8)
            eps_w_sum = eps_w.sum().clamp(min=1e-8)
            x0_loss_val = (x0_diff * x0_w).sum() / x0_w_sum
            eps_loss_val = (eps_diff * eps_w).sum() / eps_w_sum
            x0_loss_val = x0_loss_val.item()
            eps_loss_val = eps_loss_val.item()

        # Per-conditioning-mode loss (reuses per_sample_loss from training loss above)
        # drop_modes: 0=keep_both, 1=drop_image, 2=drop_text, 3=drop_both
        loss_keep = loss_drop_vis = loss_drop_txt = 0.0
        for mode_val, key in [(0, "keep"), (1, "drop_vis"), (2, "drop_txt")]:
            sel = dm == mode_val
            if sel.any():
                mode_loss = per_sample_loss[sel].mean().item()
                if key == "keep":
                    loss_keep = mode_loss
                elif key == "drop_vis":
                    loss_drop_vis = mode_loss
                else:
                    loss_drop_txt = mode_loss

    # Aggregate attention diagnostics (GPU tensors → single .item() sync)
    ip_entropy_val = 0.0
    ip_norm_ratio_val = 0.0
    ip_key_ratio_val = 0.0
    if diagnostics:
        n_layers = 0
        ip_out_sum = None
        h_pre_sum = None
        entropy_sum = None
        ip_k_sum = None
        text_k_sum = None
        for dkey, dval in diagnostics.items():
            if dkey.endswith("/ip_out_norm"):
                ip_out_sum = dval if ip_out_sum is None else ip_out_sum + dval
                n_layers += 1
            elif dkey.endswith("/h_pre_norm"):
                h_pre_sum = dval if h_pre_sum is None else h_pre_sum + dval
            elif dkey.endswith("/ip_entropy"):
                entropy_sum = dval if entropy_sum is None else entropy_sum + dval
            elif dkey.endswith("/ip_k_norm"):
                ip_k_sum = dval if ip_k_sum is None else ip_k_sum + dval
            elif dkey.endswith("/text_k_norm"):
                text_k_sum = dval if text_k_sum is None else text_k_sum + dval
        if n_layers > 0:
            # Stack and transfer to CPU in one sync
            agg = torch.stack([ip_out_sum, h_pre_sum, entropy_sum, ip_k_sum, text_k_sum])
            agg_cpu = agg.float().cpu()  # single GPU→CPU sync
            ip_out_avg = agg_cpu[0].item() / n_layers
            h_pre_avg = agg_cpu[1].item() / n_layers
            ip_entropy_val = agg_cpu[2].item() / n_layers
            ip_k_avg = agg_cpu[3].item() / n_layers
            text_k_avg = agg_cpu[4].item() / n_layers
            if h_pre_avg > 1e-8:
                ip_norm_ratio_val = ip_out_avg / h_pre_avg
            if text_k_avg > 1e-8:
                ip_key_ratio_val = ip_k_avg / text_k_avg

    return loss, {
        "core_loss": core_loss_val.item(),
        "band_loss": band_loss_val.item(),
        "x0_loss": x0_loss_val,
        "eps_loss": eps_loss_val,
        "loss_keep": loss_keep,
        "loss_drop_vis": loss_drop_vis,
        "loss_drop_txt": loss_drop_txt,
        "ip_entropy": ip_entropy_val,
        "ip_norm_ratio": ip_norm_ratio_val,
        "ip_key_ratio": ip_key_ratio_val,
    }


def _ema_smooth(values, smoothing: float = 0.99):
    """Exponential moving average."""
    arr = np.array(values, dtype=np.float64)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def save_loss_plot(losses: list, stats_file: Path, save_path: Path):
    """Render 3x3 training diagnostics plot.

    Row 0: [Cond Modes] [Core/Band (both cond)] [Band/Core Ratio | x0 | ctx]
    Row 1: [Gates] [Role Embeddings] [Grad Norm]
    Row 2: [IP/Text Output Ratio] [IP Entropy] [IP/Text Key Ratio]

    Core/Band and Band/Core use s_core/s_band directly (both-cond filtered).
    """
    if len(losses) < 2:
        return

    # --- Parse stats.csv ---
    stats_steps, diff_losses = [], []
    attn_gates, ff_gates, l2sp_vals = [], [], []
    core_losses, band_losses = [], []
    x0_losses, eps_losses, x0_ratios, grad_norms, ctx_drops = [], [], [], [], []
    emb_global_norms, emb_anomaly_norms, emb_normal_norms = [], [], []
    loss_keeps, loss_drop_viss, loss_drop_txts = [], [], []
    ip_entropies, ip_norm_ratios, ip_key_ratios = [], [], []
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                stats_steps.append(int(parts[0]))
                diff_losses.append(float(parts[1]))
                attn_gates.append(float(parts[4]))
                ff_gates.append(float(parts[5]))
                l2sp_vals.append(float(parts[6]))
                core_losses.append(float(parts[7]) if len(parts) > 8 else 0.0)
                band_losses.append(float(parts[8]) if len(parts) > 8 else 0.0)
                x0_ratios.append(float(parts[9]) if len(parts) > 9 else 0.0)
                x0_losses.append(float(parts[10]) if len(parts) > 10 else 0.0)
                eps_losses.append(float(parts[11]) if len(parts) > 11 else 0.0)
                grad_norms.append(float(parts[12]) if len(parts) > 12 else 0.0)
                ctx_drops.append(float(parts[13]) if len(parts) > 13 else 0.0)
                emb_global_norms.append(float(parts[14]) if len(parts) > 14 else 0.0)
                emb_anomaly_norms.append(float(parts[15]) if len(parts) > 15 else 0.0)
                emb_normal_norms.append(float(parts[16]) if len(parts) > 16 else 0.0)
                loss_keeps.append(float(parts[17]) if len(parts) > 17 else 0.0)
                loss_drop_viss.append(float(parts[18]) if len(parts) > 18 else 0.0)
                loss_drop_txts.append(float(parts[19]) if len(parts) > 19 else 0.0)
                ip_entropies.append(float(parts[20]) if len(parts) > 20 else 0.0)
                ip_norm_ratios.append(float(parts[21]) if len(parts) > 21 else 0.0)
                ip_key_ratios.append(float(parts[22]) if len(parts) > 22 else 0.0)
    except FileNotFoundError:
        pass

    s_steps = np.array(stats_steps)
    s_attn = np.array(attn_gates)
    s_ff = np.array(ff_gates)
    s_core = np.array(core_losses)
    s_band = np.array(band_losses)
    s_x0 = np.array(x0_losses)
    s_eps = np.array(eps_losses)
    s_x0r = np.array(x0_ratios)
    s_gn = np.array(grad_norms)
    s_ctx = np.array(ctx_drops)
    s_emb_g = np.array(emb_global_norms)
    s_emb_a = np.array(emb_anomaly_norms)
    s_emb_n = np.array(emb_normal_norms)
    s_loss_keep = np.array(loss_keeps)
    s_loss_dv = np.array(loss_drop_viss)
    s_loss_dt = np.array(loss_drop_txts)
    s_ip_ent = np.array(ip_entropies)
    s_ip_nr = np.array(ip_norm_ratios)
    s_ip_kr = np.array(ip_key_ratios)

    SM = 0.99
    SF = 0.6

    steps = np.arange(len(losses))
    arr = np.array(losses)
    ema_slow = _ema_smooth(arr, SM)

    has_cb = len(s_steps) > 0 and len(s_core) > 0 and s_core.any()
    has_x0 = len(s_steps) > 0 and s_x0.any()
    has_gn = len(s_steps) > 0 and s_gn.any()
    has_ctx = (len(s_steps) > 0 and s_ctx.any()
               and (s_ctx.max() - s_ctx.min()) > 0.01)
    has_emb = len(s_steps) > 0 and (s_emb_g.any() or s_emb_a.any() or s_emb_n.any())
    has_cond_modes = len(s_steps) > 0 and (s_loss_keep.any() or s_loss_dv.any() or s_loss_dt.any())
    has_ip_diag = len(s_steps) > 0 and (s_ip_ent.any() or s_ip_nr.any())

    # =====================================================
    # 3x3 grid — row 0 varies, rows 1-2 fixed
    # =====================================================
    fig, axes = plt.subplots(3, 3, figsize=(21, 15))

    ax_trend, ax_cb = axes[0, 0], axes[0, 1]
    ax_x0eps = None
    ax_ctx = None
    if has_x0:
        ax_x0eps = axes[0, 2]
    elif has_ctx:
        ax_ctx = axes[0, 2]
    else:
        ax_ratio = axes[0, 2]

    # Row 1 fixed: Gates, Embeddings, Grad Norm
    ax_gates = axes[1, 0]
    ax_emb = axes[1, 1]
    ax_gn = axes[1, 2]
    # Row 2 fixed: IP Output Ratio, IP Key Ratio, IP Entropy
    ax_ip_ratio = axes[2, 0]
    ax_ip_key = axes[2, 1]
    ax_ip_ent = axes[2, 2]

    # === [0,0] Conditioning Mode Losses ===
    if has_cond_modes:
        def _mode_ema_filtered(vals, steps_arr, sm):
            nz = vals > 0
            if not nz.any():
                return np.array([]), np.array([])
            return steps_arr[nz], _ema_smooth(vals[nz], sm)
        all_ema = _ema_smooth(np.array(diff_losses), SM)
        ax_trend.plot(s_steps, all_ema, color="black", linewidth=2,
                      alpha=1.0, label=f"Overall ({all_ema[-1]:.4f})")
        k_st, k_em = _mode_ema_filtered(s_loss_keep, s_steps, SM)
        if len(k_em):
            ax_trend.plot(k_st, k_em, color="blue", linewidth=2,
                          alpha=1.0, label=f"Keep both ({k_em[-1]:.4f})")
        dv_st, dv_em = _mode_ema_filtered(s_loss_dv, s_steps, SM)
        if len(dv_em):
            ax_trend.plot(dv_st, dv_em, color="red", linewidth=1,
                          alpha=0.35, label=f"Keep text ({dv_em[-1]:.4f})")
        dt_st, dt_em = _mode_ema_filtered(s_loss_dt, s_steps, SM)
        if len(dt_em):
            ax_trend.plot(dt_st, dt_em, color="orange", linewidth=1,
                          alpha=0.35, label=f"Keep visual ({dt_em[-1]:.4f})")
        ax_trend.set_title(f"Loss by Conditioning Mode, EMA({SM}) -- step {len(losses)}")
    else:
        ax_trend.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax_trend.set_title(f"Smooth Trend -- step {len(losses)}, trend: {ema_slow[-1]:.4f}")
    ax_trend.set_xlabel("Step"); ax_trend.set_ylabel("Loss")
    ax_trend.legend(loc="upper left", fontsize=8)
    ax_trend.grid(True, alpha=0.3)

    # === [0,1] Core vs Band (both cond only — direct from stats) ===
    if has_cb:
        ce = _ema_smooth(s_core, SM)
        be = _ema_smooth(s_band, SM)
        combined = 0.8 * ce + 0.2 * be
        ax_cb.fill_between(s_steps, 0, ce, color="#2196F3", alpha=0.4)
        ax_cb.fill_between(s_steps, ce, ce + be, color="#FF9800", alpha=0.4)
        ax_cb.plot(s_steps, ce, color="#1565C0", linewidth=1.5, label=f"Core ({ce[-1]:.4f})")
        ax_cb.plot(s_steps, ce + be, color="#E65100", linewidth=1.5, label=f"Band stacked ({be[-1]:.4f})")
        ax_cb.plot(s_steps, combined, color="black", linewidth=2, linestyle="--",
                   label=f"0.8\u00d7core+0.2\u00d7band = {combined[-1]:.4f}")
        ax_cb.legend(loc="upper left", fontsize=8)
        ax_cb.set_title(f"Core vs Band (both cond only), EMA({SM})")
    else:
        ax_cb.set_title("Core vs Band (no data yet)")
    ax_cb.set_xlabel("Step"); ax_cb.set_ylabel("Per-pixel MSE")
    ax_cb.grid(True, alpha=0.3)

    # === [0,2] Band/Core Ratio (default) | x0 | ctx ===
    if ax_x0eps is not None and has_x0:
        # x0 vs eps + annealing
        x0_mask = s_x0 > 0
        x0_steps_f, x0_vals_f = s_steps[x0_mask], s_x0[x0_mask]
        eps_mask = s_eps > 0
        eps_steps_f, eps_vals_f = s_steps[eps_mask], s_eps[eps_mask]
        lines = []
        if len(x0_vals_f) > 1:
            ax_x0eps.plot(x0_steps_f, _ema_smooth(x0_vals_f, SF), color="#CE93D8", linewidth=0.8, alpha=0.3)
            x0_slow = _ema_smooth(x0_vals_f, SM)
            l1, = ax_x0eps.plot(x0_steps_f, x0_slow, color="#6A1B9A", linewidth=2.5,
                                label=f"x0 = {x0_slow[-1]:.4f}")
            lines.append(l1)
        if len(eps_vals_f) > 1:
            ax_x0eps.plot(eps_steps_f, _ema_smooth(eps_vals_f, SF), color="#80CBC4", linewidth=0.8, alpha=0.3)
            eps_slow = _ema_smooth(eps_vals_f, SM)
            l2, = ax_x0eps.plot(eps_steps_f, eps_slow, color="#00695C", linewidth=2.5,
                                label=f"\u03b5 = {eps_slow[-1]:.4f}")
            lines.append(l2)
        ax_r2 = ax_x0eps.twinx()
        l3, = ax_r2.plot(s_steps, s_x0r, color="gray", linewidth=1.5, linestyle="--",
                         alpha=0.6, label="x0 ratio")
        ax_r2.set_ylim(-0.05, 1.1)
        ax_r2.set_ylabel("x0 ratio", color="gray")
        ax_r2.tick_params(axis="y", labelcolor="gray")
        lines.append(l3)
        ax_x0eps.legend(handles=lines, loc="upper left", fontsize=7)
        title_parts = []
        if len(x0_vals_f) > 1:
            title_parts.append(f"x0={x0_slow[-1]:.4f}")
        if len(eps_vals_f) > 1:
            title_parts.append(f"\u03b5={eps_slow[-1]:.4f}")
        ax_x0eps.set_title(f"x0 vs \u03b5 Loss, EMA({SM}) -- {', '.join(title_parts)}")
        ax_x0eps.set_xlabel("Step"); ax_x0eps.set_ylabel("Loss")
        ax_x0eps.grid(True, alpha=0.3)
    elif ax_ctx is not None:
        ax_ctx.plot(s_steps, s_ctx, color="#D32F2F", linewidth=2.5, label="Ctx dropout rate")
        ax_ctx.fill_between(s_steps, 0, s_ctx, color="#EF9A9A", alpha=0.3)
        ax_ctx.set_ylim(-0.05, 1.1)
        ax_ctx.set_title(f"Context Dropout Schedule -- current: {s_ctx[-1]:.1%}")
        ax_ctx.set_xlabel("Step"); ax_ctx.set_ylabel("Dropout rate")
        ax_ctx.legend(loc="upper left", fontsize=8)
        ax_ctx.grid(True, alpha=0.3)
    else:
        # Default: Band/Core Ratio (both cond only)
        if has_cb:
            safe_core = np.maximum(s_core, 1e-10)
            cbr = s_band / safe_core
            cbre = _ema_smooth(cbr, SM)
            ax_ratio.plot(s_steps, cbr, color="gray", linewidth=0.5, alpha=0.3, label="Raw")
            ax_ratio.plot(s_steps, cbre, color="purple", linewidth=2, label=f"EMA ({SM})")
            ax_ratio.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax_ratio.set_title(f"Band/Core Ratio (both cond only), EMA({SM}) -- {cbre[-1]:.2f}x")
            ax_ratio.legend(loc="upper left", fontsize=8)
        else:
            ax_ratio.set_title("Band/Core Ratio (no data yet)")
        ax_ratio.set_xlabel("Step"); ax_ratio.set_ylabel("Band / Core")
        ax_ratio.grid(True, alpha=0.3)

    # === [1,0] Gates ===
    if len(s_steps) > 0:
        ax_gates.plot(s_steps, s_attn, color="blue", linewidth=1.5, label="Attn gate")
        ax_gates.plot(s_steps, s_ff, color="green", linewidth=1.5, label="FF gate")
        ax_gates.set_title(f"Gates -- attn={s_attn[-1]:.4f}, ff={s_ff[-1]:.4f}")
        ax_gates.legend(loc="upper left", fontsize=8)
    else:
        ax_gates.set_title("Gates (no data yet)")
    ax_gates.set_xlabel("Step"); ax_gates.set_ylabel("Gate value")
    ax_gates.grid(True, alpha=0.3)

    # === [1,1] Role Embedding Norms ===
    if has_emb:
        ax_emb.plot(s_steps, s_emb_g, color="green", linewidth=1.5, label=f"Global ({s_emb_g[-1]:.4f})")
        ax_emb.plot(s_steps, s_emb_a, color="red", linewidth=1.5, label=f"Anomaly ({s_emb_a[-1]:.4f})")
        ax_emb.plot(s_steps, s_emb_n, color="blue", linewidth=1.5, label=f"Band ({s_emb_n[-1]:.4f})")
        ax_emb.set_title("Role Embedding Norms")
        ax_emb.legend(loc="upper left", fontsize=8)
    else:
        ax_emb.set_title("Role Embedding Norms (no data)")
    ax_emb.set_xlabel("Step"); ax_emb.set_ylabel("L2 Norm")
    ax_emb.grid(True, alpha=0.3)

    # === [1,2] Grad Norm ===
    if has_gn:
        gn_ema_fast = _ema_smooth(s_gn, SF)
        gn_ema_slow = _ema_smooth(s_gn, SM)
        ax_gn.plot(s_steps, gn_ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SF})")
        ax_gn.plot(s_steps, gn_ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax_gn.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
        ax_gn.set_title(f"Grad Norm -- trend: {gn_ema_slow[-1]:.4f}")
        ax_gn.legend(loc="upper left", fontsize=8)
    else:
        ax_gn.set_title("Grad Norm (no data yet)")
    ax_gn.set_xlabel("Step"); ax_gn.set_ylabel("Norm")
    ax_gn.grid(True, alpha=0.3)

    # === [2,0] IP/Text Output Ratio (||ip_out|| / ||h_pre||) ===
    if has_ip_diag and s_ip_nr.any():
        nz = s_ip_nr > 0
        if nz.any():
            ema_nr = _ema_smooth(s_ip_nr[nz], SM)
            ax_ip_ratio.plot(s_steps[nz], ema_nr, color="purple", linewidth=2,
                             label=f"ratio ({ema_nr[-1]:.4f})")
            ax_ip_ratio.legend(loc="upper left", fontsize=8)
            ax_ip_ratio.set_title(f"IP Output Ratio ||ip_out||/||h_pre|| -- {ema_nr[-1]:.4f}")
        else:
            ax_ip_ratio.set_title("IP Output Ratio ||ip_out||/||h_pre|| (no nonzero data)")
    else:
        ax_ip_ratio.set_title("IP Output Ratio ||ip_out||/||h_pre|| (no data yet)")
    ax_ip_ratio.set_xlabel("Step"); ax_ip_ratio.set_ylabel("Ratio")
    ax_ip_ratio.grid(True, alpha=0.3)

    # === [2,1] IP Attention Entropy ===
    if has_ip_diag and s_ip_ent.any():
        nz = s_ip_ent > 0
        if nz.any():
            ema_ent = _ema_smooth(s_ip_ent[nz], SM)
            ax_ip_ent.plot(s_steps[nz], ema_ent, color="teal", linewidth=2,
                           label=f"entropy ({ema_ent[-1]:.2f})")
            ax_ip_ent.legend(loc="upper left", fontsize=8)
            ax_ip_ent.set_title(f"IP Attention Entropy -- {ema_ent[-1]:.2f}")
        else:
            ax_ip_ent.set_title("IP Attention Entropy (no nonzero data)")
    else:
        ax_ip_ent.set_title("IP Attention Entropy (no data yet)")
    ax_ip_ent.set_xlabel("Step"); ax_ip_ent.set_ylabel("Entropy")
    ax_ip_ent.grid(True, alpha=0.3)

    # === [2,2] IP/Text Key Ratio (||ip_k|| / ||text_k||) ===
    if has_ip_diag and s_ip_kr.any():
        nz = s_ip_kr > 0
        if nz.any():
            ema_kr = _ema_smooth(s_ip_kr[nz], SM)
            ax_ip_key.plot(s_steps[nz], ema_kr, color="#E65100", linewidth=2,
                           label=f"key ratio ({ema_kr[-1]:.4f})")
            ax_ip_key.legend(loc="upper left", fontsize=8)
            ax_ip_key.set_title(f"IP Key Ratio ||ip_k||/||text_k|| -- {ema_kr[-1]:.4f}")
        else:
            ax_ip_key.set_title("IP Key Ratio ||ip_k||/||text_k|| (no nonzero data)")
    else:
        ax_ip_key.set_title("IP Key Ratio ||ip_k||/||text_k|| (no data yet)")
    ax_ip_key.set_xlabel("Step"); ax_ip_key.set_ylabel("Ratio")
    ax_ip_key.grid(True, alpha=0.3)

    # --- Run title from config ---
    run_title = None
    config_path = stats_file.parent / "run_config.json"
    try:
        with open(config_path) as _cf:
            cfg = json.load(_cf)
        parts = [
            f"VM{cfg.get('visual_mode', '?')}",
            f"LoRA={cfg['lora_rank']}" if cfg.get('lora_rank', 0) > 0 else "no-LoRA",
            f"BS={cfg.get('batch_size', '?')}",
            f"SA={cfg.get('sa_num_layers', '?')}L/{cfg.get('sa_num_heads', '?')}H",
            f"drop={cfg.get('drop_image_prob', 0) + cfg.get('drop_text_prob', 0) + cfg.get('drop_both_prob', 0):.0%}",
            f"gates={'yes' if cfg.get('learnable_gates', True) else 'no'}",
            f"UNet={'QO-' + cfg['unfreeze_qo'] if cfg.get('unfreeze_qo') else 'frozen'}",
            cfg.get('optimizer', 'adamw'),
            f"ts={cfg.get('timestep_sampling', '?')}",
            f"corrupt={cfg['corrupt_context']}" if cfg.get('corrupt_context', 0) > 0 else None,
            f"x0={cfg['x0_start_ratio']}->{cfg['x0_end_ratio']} w={cfg.get('x0_warmup_frac', 0)} ctx={'no' if cfg.get('x0_no_context') else 'yes'}" if cfg.get('x0_objective') else None,
        ]
        run_title = " | ".join(p for p in parts if p is not None)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    if run_title:
        fig.suptitle(run_title, fontsize=11, fontweight="bold", y=1.02)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_anomagic_checkpoint(
    ip_adapter, optimizer, save_dir,
    band_mode: int = 1, t2i_adapter=None, unet=None,
    step: int = 0, losses: list = None,
    qo_params_named: list = None,
    ema=None,
    scheduler=None,
):
    """Save IP-Adapter + T2I-Adapter + LoRA + unfrozen QO + EMA checkpoint."""
    save_dir = Path(save_dir)
    ip_adapter.save_ip_adapter(save_dir)

    state = {
        "optimizer": optimizer.state_dict(),
        "band_mode": band_mode,
        "step": step,
        "losses": losses or [],
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if ema is not None:
        state["ema"] = ema.state_dict()
    if t2i_adapter is not None:
        state["t2i_adapter"] = t2i_adapter.state_dict()
    if unet is not None:
        from peft import get_peft_model_state_dict
        state["lora_weights"] = get_peft_model_state_dict(unet)
    if qo_params_named:
        state["qo_weights"] = {name: param.data for name, param in qo_params_named}
    torch.save(state, save_dir / "training_state.pt")


# NOTE: generate_anomagic_samples and _generate_ood_grid removed —
# replaced by run_validation_suite() from scripts/validation_suite.py


def _launch_live_loss(loss_file: Path):
    """Launch live loss viewer as a detached subprocess."""
    import subprocess
    script = Path(__file__).parent / "live_loss.py"
    loss_file.write_text("")
    try:
        subprocess.Popen(
            [sys.executable, str(script), str(loss_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  Live loss viewer launched (reading {loss_file})")
    except Exception as e:
        print(f"  Warning: could not launch live loss viewer: {e}")


def infinite_loader(loader):
    epoch = 0
    while True:
        epoch += 1
        # Set epoch on dataset so __getitem__ can seed augmentation deterministically
        # per (epoch, index). Works with persistent_workers (init_fn doesn't re-run).
        loader.dataset._epoch = epoch
        for batch in loader:
            yield batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Anomagic training: IP-Adapter + captions")
    parser.add_argument("--sd-version", type=str, default="sd_1.5", choices=["sd_1.5", "sdxl"],
                        help="Base inpainting model: sd_1.5 (runwayml inpainting, 512) or sdxl "
                             "(diffusers/stable-diffusion-xl-1.0-inpainting-0.1, 1024; IP-Adapter only).")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Training resolution (default: 512 for sd_1.5, 1024 for sdxl).")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Split each --batch-size batch into this many micro-batches (same effective batch).")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Gradient checkpointing in the frozen UNet (less VRAM, slower).")
    parser.add_argument("--master-json", type=str, default=None,
                        help="DATA GUARD: abort unless the loaded training samples equal this master JSON exactly "
                             "(same image/mask pairs, no extras, no VisA/cashew).")
    parser.add_argument("--splits-dir", type=str, default="data/concepts",
                        help="Directory with anomaly type JSONs")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for resolving relative image paths")
    parser.add_argument("--captions-file", type=str, required=True,
                        help="Path to captions.json")
    parser.add_argument("--save-dir", type=str, default="results/anomagic_training",
                        help="Where to save results")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from (e.g. results/anomagic_training/checkpoint_8000)")
    parser.add_argument("--steps", type=int, default=20000,
                        help="Training steps (default 20000). Overridden to 60000 by --full-run.")
    parser.add_argument("--full-run", action="store_true",
                        help="Full run: 60000 steps (overrides --steps).")
    parser.add_argument("--save-every", type=int, default=5000,
                        help="Save checkpoint + samples every N steps")
    parser.add_argument("--no-early-snapshots", action="store_true",
                        help="Disable early sample snapshots at steps 500/1000/2000 "
                             "(use for matrix runs where only end-of-training val matters).")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                        help="Disable auto-deletion of intermediate checkpoints during training. "
                             "Use when you need to evaluate at multiple step counts later "
                             "(e.g., training-length sweep within a single run).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for IP-Adapter")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size")
    parser.add_argument("--ip-adapter-type", type=str, default="plus",
                        choices=["standard", "plus"],
                        help="standard: ViT-L/14 CLS→linear→4 tokens. plus: ViT-H/14 patches→perceiver→K tokens")
    parser.add_argument("--ip-adapter-k", type=int, default=16, choices=[1, 4, 16],
                        help="Visual tokens after projection: 1=no upproject, 4=standard, 16=pretrained Plus default")
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0,
                        help="Visual pathway strength")
    parser.add_argument("--reference-mode", type=str, default="full", choices=["full", "crop"],
                        help="Reference image mode: 'full'=downscale entire image, 'crop'=anomaly bbox crop")
    parser.add_argument("--types", type=str, nargs="+", default=None,
                        help="Specific anomaly types to train on (default: all)")
    parser.add_argument("--exclude-sources", type=str, nargs="+",
                        default=None,
                        help="Dataset sources to exclude (default: none)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-mask-visual", action="store_true",
                        help="Disable masked cross-attention for visual pathway "
                             "(ablation — all positions receive visual conditioning. "
                             "Default: masked, only anomaly positions get visual signal)")
    # Visual token processing mode
    parser.add_argument("--visual-mode", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Visual token processing: 0=all 256 tokens (default), "
                             "1=selective residual, 2=anomaly-only+attn-only, "
                             "3=anomaly-only+full transformer (padding dead everywhere)")
    parser.add_argument("--no-learnable-gates", action="store_true",
                        help="Replace scalar gates with zero-init output projections. "
                             "Same identity-at-init, but per-dimension gradient signal.")
    parser.add_argument("--force-gates", action="store_true",
                        help="Fixed gate=1.0 with normal projection init. "
                             "Block contributes fully from step 0 — forces reliance.")
    parser.add_argument("--sa-num-layers", type=int, default=3,
                        help="Number of transformer layers in masked self-attention (default 1)")
    parser.add_argument("--sa-num-heads", type=int, default=12,
                        help="Number of attention heads in masked self-attention (default 12, matching Resampler)")
    parser.add_argument("--cfg-mode", type=str, default="visual", choices=["text", "visual", "both"],
                        help="CFG direction for sample generation: 'text' (default) amplifies text "
                             "on visual baseline, 'visual' amplifies IP-Adapter on text baseline, "
                             "'both' amplifies both pathways on unconditional baseline")
    # Multi-crop (orthogonal to visual-mode)
    parser.add_argument("--multi-crop", action="store_true",
                        help="Enable 2-group CLIP cropping (disabled by default, K=1 single-crop)")
    parser.add_argument("--no-clip-align", action="store_true",
                        help="Disable CLIP-UNet roundtrip mask alignment and role embeddings. "
                             "Reverts to raw cropped masks with no anomaly/normal token distinction.")
    parser.add_argument("--clip-core-only", action="store_true",
                        help="CLIP attention sees only core (roundtripped, no band dilation). "
                             "Default: CLIP sees core+band (dilated).")
    # CLIP dilation settings — currently unused (CLIP path uses cropping instead).
    # Kept for potential future use with reference_mode="full".
    # parser.add_argument("--clip-dilation-min-r", type=int, default=2,
    #                     help="CLIP mask dilation min radius (tight, for anomaly-focused features)")
    # parser.add_argument("--clip-dilation-max-r", type=int, default=10,
    #                     help="CLIP mask dilation max radius")
    # Latent-space band dilation
    parser.add_argument("--band-mode", type=int, default=2, choices=[1, 2],
                        help="Band mode: 1=single 1px band (alpha=0.5), 2=inner+outer (alpha=2/3, 1/3)")
    # Loss settings
    parser.add_argument("--loss-core-ratio", type=float, default=0.8,
                        help="Fraction of total gradient to core (e.g. 0.8 = 80%% core, 20%% band)")
    parser.add_argument("--no-masked-loss", action="store_true",
                        help="Disable masked loss: uniform MSE over all 64x64 latent pixels (standard diffusion loss)")
    parser.add_argument("--uniform-mask-loss", action=argparse.BooleanOptionalAction, default=True,
                        help="Uniform weight inside dilated mask (core=band=1.0), zero outside. "
                             "Per-sample normalized — equivalent to mean MSE over masked pixels, no 80/20 bias. "
                             "Default True; use --no-uniform-mask-loss to enable the 80/20 core/band ratio loss instead.")
    # Cross-attention mask type
    parser.add_argument("--binary-cross-attn-mask", action="store_true",
                        help="Use binary dilated mask for cross-attn instead of soft alpha_map")
    parser.add_argument("--binary-cross-attn-mask-core", action="store_true",
                        help="Use binary CORE mask for IP cross-attn (band=0). Isolates the IP-CA "
                             "spatial gating; UNet inpainting mask, T2I band channel, and loss "
                             "weighting are unchanged. Mutually exclusive with --binary-cross-attn-mask "
                             "and --core-only (which take precedence).")
    parser.add_argument("--core-only", action="store_true",
                        help="Core-only training ablation: replace all dilated spatial signals "
                             "(UNet inpainting mask, T2I band channel, T2I internal mask, "
                             "IP-Adapter cross-attn mask) with core-only, and force loss_core_ratio→0.999.")
    # T2I-Adapter
    parser.add_argument("--t2i-adapter-mode", type=str, default="cascade",
                        choices=["cascade", "skip_only", "off"],
                        help="T2I-Adapter injection mode (cascade=encoder+decoder, skip_only=decoder, off=disabled)")
    parser.add_argument("--t2i-pair-inject", type=str, default="last",
                        choices=["last", "all"],
                        help="T2I intra-block injection: last=after last pair only (default), "
                             "all=before first pair + after every pair")
    parser.add_argument("--t2i-decoder-inject", action="store_true",
                        help="Also inject T2I features into decoder up-blocks via forward hooks "
                             "(symmetric with encoder). Respects --t2i-pair-inject: 'last' hooks "
                             "the last pair per up-block, 'all' hooks every pair.")
    # Conditioning dropout for CFG (3-category, mutually exclusive per-sample)
    parser.add_argument("--drop-image-prob", type=float, default=0.15,
                        help="Probability of zeroing IP-Adapter image embeddings per sample for CFG")
    parser.add_argument("--drop-text-prob", type=float, default=0.05,
                        help="Probability of replacing text with empty-string encoding per sample for CFG")
    parser.add_argument("--drop-both-prob", type=float, default=0.0,
                        help="Probability of dropping both image and text per sample for CFG")
    # LoRA
    # Data augmentation
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable data augmentation (default: on)")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank for UNet attention layers (0=disabled, 16=matches IP-Adapter Plus K)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora-lr", type=float, default=5e-5,
                        help="Learning rate for LoRA params (default 5e-5)")
    parser.add_argument("--lora-mode", type=str, default="all", choices=["all", "cross", "mid_up", "mid_up_notext"],
                        help="LoRA target: 'all' = self+cross attention, 'cross' = cross-attention only, 'mid_up' = all attention in mid+up blocks")
    parser.add_argument("--unfreeze-qo", type=str, default="",
                        choices=["", "mid_up", "all_cross", "all"],
                        help="Unfreeze W_Q/W_O: mid_up=mid+up cross-attn, all_cross=all cross-attn, all=cross+self")
    parser.add_argument("--unfreeze-qo-lr", type=float, default=1e-5,
                        help="Learning rate for unfrozen W_Q/W_O params (default 1e-5)")
    parser.add_argument("--lr-pretrained", type=float, default=1e-4,
                        help="Learning rate for pretrained IP-Adapter params (Group A)")
    parser.add_argument("--lambda-sp", type=float, default=0.0,
                        help="L2-SP regularization strength (default 0=disabled)")
    parser.add_argument("--no-live-viewer", action="store_true",
                        help="Disable live loss viewer (for headless servers)")
    parser.add_argument("--noise-offset", type=float, default=0.05,
                        help="Noise offset for global brightness/darkness (default 0.05, 0=disabled)")
    parser.add_argument("--timestep-sampling", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal", "logit_normal_sigma2"],
                        help="Timestep sampling: 'logit_normal' (default, SD3-style bell over t), "
                             "'logit_normal_sigma2' (bell over noise power sigma^2, centered at 0.5), "
                             "or 'uniform' (standard DDPM)")
    parser.add_argument("--logit-normal-mean", type=float, default=0.0,
                        help="Mean for logit-normal timestep sampling (default 0.0, centers on t=500)")
    parser.add_argument("--logit-normal-std", type=float, default=1.0,
                        help="Std for logit-normal timestep sampling (default 1.0, lower=tighter peak)")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "prodigy"],
                        help="Optimizer: 'adamw' (default) or 'prodigy' (auto-tuned LR)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["none", "cosine"],
                        help="LR scheduler: 'cosine' (default, cosine annealing) or 'none' (constant)")
    parser.add_argument("--lr-min", type=float, default=5e-5,
                        help="Minimum LR for cosine annealing on Group A (default 5e-5)")
    parser.add_argument("--corrupt-context", type=float, default=0.0,
                        help="Probability of zeroing masked_image_latents per step (0.0-1.0). "
                             "Forces UNet to rely on cross-attention instead of inpainting prior.")
    parser.add_argument("--x0-objective", action="store_true",
                        help="Enable pseudo-diffusion x0-prediction: input=noised(masked_image), "
                             "loss=||x0_pred - x0||². Forces cross-attention usage for anomaly generation.")
    parser.add_argument("--x0-start-ratio", type=float, default=1.0,
                        help="Fraction of samples using x0 path at step 0 (default 1.0)")
    parser.add_argument("--x0-end-ratio", type=float, default=1.0,
                        help="Fraction of samples using x0 path at final step (default 1.0, set to 0.1 to anneal)")
    parser.add_argument("--x0-no-context", action="store_true",
                        help="Zero masked_image_latents inpainting channel for x0-path samples "
                             "(removes all shortcuts, forces pure cross-attention)")
    parser.add_argument("--x0-warmup-frac", type=float, default=0.2,
                        help="Fraction of total steps to hold x0_ratio at start value before annealing "
                             "(default 0.2 = first 1/5 of training)")
    parser.add_argument("--x0-hold-frac", type=float, default=0.1,
                        help="Fraction of total steps at the END to hold x0_ratio at end value "
                             "(default 0.1 = last 10%% of training holds at end value)")
    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay rate (0=disabled, 0.9999=standard diffusion default)")
    # Attention diagnostics
    parser.add_argument("--diag-interval", type=int, default=1,
                        help="Attention diagnostics (ip_entropy, norm ratio) frequency (default: 1)")
    # Validation suite
    parser.add_argument("--val-data-dir", type=str, default=None,
                        help="Path to prepped validation data (from prep_validation_data.py)")
    parser.add_argument("--val-panels", type=str, nargs="+", default=None,
                        help="Which validation panels to run (default: all). Choices: A B C D E")
    parser.add_argument("--no-val-suite", action="store_true",
                        help="Skip ALL validation-suite generation (step-0, early snapshots, per-checkpoint, final)")
    # Post-training evaluation pipeline (on by default)
    parser.add_argument("--eval", action="store_true", default=True,
                        help="Run full evaluation pipeline after training (default: on)")
    parser.add_argument("--no-eval", dest="eval", action="store_false",
                        help="Skip evaluation pipeline after training")
    parser.add_argument("--eval-skip-uninet", action="store_true",
                        help="Skip UniNet evaluation in eval pipeline")
    parser.add_argument("--eval-dust", type=int, default=0,
                        help="Dust particles for UniNet in eval pipeline (default: 0)")
    parser.add_argument("--eval-resnet-epochs", type=int, default=30,
                        help="ResNet epochs in eval pipeline (default: 30, matches run_eval_pipeline.py)")
    parser.add_argument("--eval-uninet-epochs", type=int, default=30,
                        help="UniNet epochs in eval pipeline (default: 30, matches run_eval_pipeline.py)")

    args = parser.parse_args()

    if args.full_run:
        args.steps = 60000

    project_root = Path(__file__).parent.parent

    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    captions_file = Path(args.captions_file)
    if not captions_file.is_absolute():
        captions_file = project_root / captions_file

    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir and not resume_dir.is_absolute():
        resume_dir = project_root / resume_dir

    val_data_dir = Path(args.val_data_dir) if args.val_data_dir else None
    if val_data_dir and not val_data_dir.is_absolute():
        val_data_dir = project_root / val_data_dir

    save_dir = project_root / args.save_dir

    train_anomagic(
        master_json=(project_root / args.master_json) if args.master_json else None,
        splits_dir=project_root / args.splits_dir,
        save_dir=save_dir,
        captions_file=captions_file,
        n_steps=args.steps,
        save_every=args.save_every,
        no_early_snapshots=args.no_early_snapshots,
        keep_all_checkpoints=args.keep_all_checkpoints,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_pretrained=args.lr_pretrained,
        lambda_sp=args.lambda_sp,
        anomaly_types=args.types,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        device=args.device,
        ip_adapter_type=args.ip_adapter_type,
        ip_adapter_k=args.ip_adapter_k,
        ip_adapter_scale=args.ip_adapter_scale,
        reference_mode=args.reference_mode,
        mask_visual=not args.no_mask_visual,
        clip_dilation_min_r=getattr(args, 'clip_dilation_min_r', 2),
        clip_dilation_max_r=getattr(args, 'clip_dilation_max_r', 10),
        band_mode=args.band_mode,
        # loss_core_ratio passed below (overridden by --core-only)
        no_masked_loss=args.no_masked_loss,
        binary_cross_attn_mask=args.binary_cross_attn_mask,
        binary_cross_attn_mask_core=args.binary_cross_attn_mask_core,
        uniform_mask_loss=args.uniform_mask_loss,
        core_only=args.core_only,
        loss_core_ratio=(0.999 if args.core_only else args.loss_core_ratio),
        t2i_adapter_mode=args.t2i_adapter_mode,
        t2i_pair_inject=args.t2i_pair_inject,
        t2i_decoder_inject=args.t2i_decoder_inject,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_lr=args.lora_lr,
        lora_mode=args.lora_mode,
        unfreeze_qo=args.unfreeze_qo,
        unfreeze_qo_lr=args.unfreeze_qo_lr,
        drop_image_prob=args.drop_image_prob,
        drop_text_prob=args.drop_text_prob,
        drop_both_prob=args.drop_both_prob,
        augment=args.augment,
        resume_dir=resume_dir,
        visual_mode=args.visual_mode,
        learnable_gates=not args.no_learnable_gates,
        force_gates=args.force_gates,
        sa_num_layers=args.sa_num_layers,
        sa_num_heads=args.sa_num_heads,
        multi_crop=args.multi_crop,
        cfg_mode=args.cfg_mode,
        no_live_viewer=args.no_live_viewer,
        noise_offset=args.noise_offset,
        timestep_sampling=args.timestep_sampling.replace("-", "_"),
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        optimizer_type=args.optimizer,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        corrupt_context=args.corrupt_context,
        x0_objective=args.x0_objective,
        x0_start_ratio=args.x0_start_ratio,
        x0_end_ratio=args.x0_end_ratio,
        x0_no_context=args.x0_no_context,
        x0_warmup_frac=args.x0_warmup_frac,
        x0_hold_frac=args.x0_hold_frac,
        diag_interval=args.diag_interval,
        clip_align=not args.no_clip_align,
        clip_core_only=args.clip_core_only,
        val_data_dir=val_data_dir,
        val_panels=args.val_panels,
        no_val_suite=args.no_val_suite,
        ema_decay=args.ema_decay,
        sd_version=args.sd_version,
        resolution=args.resolution,
        grad_accum=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # ── Post-training evaluation pipeline ──────────────────────────────────
    if args.eval and args.sd_version != "sd_1.5":
        print("NOTE: the post-training eval pipeline is SD 1.5-only; skipped for", args.sd_version)
    elif args.eval:
        import subprocess as _sp
        # Release parent-process CUDA memory so child eval subprocess has headroom.
        # Without this, PyTorch's cached allocator keeps ~8-12 GB alive in the parent
        # (SD UNet, VAE, CLIP, IP-Adapter, T2I-Adapter, optimizer state) which combined
        # with UniNet's ~12 GB in the child is enough to OOM a 24 GB card.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        checkpoint_dir = save_dir / "checkpoint_final"
        if not checkpoint_dir.exists():
            print(f"\nWARNING: --eval requested but {checkpoint_dir} not found. Skipping eval.")
        else:
            print(f"\n{'='*70}")
            print(f"  RUNNING EVAL PIPELINE")
            print(f"{'='*70}")
            eval_cmd = [
                sys.executable, str(project_root / "scripts" / "run_eval_pipeline.py"),
                "--checkpoint", str(checkpoint_dir),
                "--output-dir", str(save_dir),
            ]
            if args.eval_skip_uninet:
                eval_cmd.append("--skip-uninet")
            if args.eval_dust > 0:
                eval_cmd.extend(["--dust", str(args.eval_dust)])
            eval_cmd.extend(["--resnet-epochs", str(args.eval_resnet_epochs)])
            eval_cmd.extend(["--uninet-epochs", str(args.eval_uninet_epochs)])
            print(f"  CMD: {' '.join(eval_cmd)}")
            _sp.run(eval_cmd, cwd=str(project_root))
