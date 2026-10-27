"""Anomagic single-image inference.

Contains `generate_anomagic_single` — the core generation function used by
both the training script (for periodic samples) and standalone generation
scripts (e.g. generate_cashew.py).
"""

import torch
import torch.nn.functional as F

from src.utils.mask_utils import create_latent_band_mask


class EvenStepDDIM:
    """Deterministic DDIM (eta 0, epsilon prediction) on an explicit, evenly spaced timestep list.

    The stock scheduler can only start part-way into its fixed grid, so a noise strength s < 1 runs just s * num_steps
    steps. This runs `num_steps` steps from the SAME start timestep down to 1 (fewer only when the start timestep is
    smaller than num_steps: then every timestep). On the stock grid it reproduces DDIMScheduler.step (checked 2026-10-03).
    """

    def __init__(self, base, t0: int, num_steps: int, device) -> None:
        n = min(num_steps, t0)
        ts = [int(round(t0 + (1 - t0) * k / max(n - 1, 1))) for k in range(n)]
        assert len(set(ts)) == n and ts[0] == t0 and ts[-1] == 1, ts
        self.base, self.custom = base, ts
        self.timesteps = torch.tensor(ts, dtype=torch.long, device=device)

    def add_noise(self, *a, **k):
        return self.base.add_noise(*a, **k)

    def step(self, model_output, timestep, sample):
        i = self.custom.index(int(timestep)); a_t = float(self.base.alphas_cumprod[self.custom[i]])
        a_p = float(self.base.alphas_cumprod[self.custom[i + 1]]) if i + 1 < len(self.custom) else float(self.base.final_alpha_cumprod)
        x0 = (sample - (1 - a_t) ** 0.5 * model_output) / a_t ** 0.5
        return type("Out", (), {"prev_sample": a_p ** 0.5 * x0 + (1 - a_p) ** 0.5 * model_output})()


def _dilate_clip_mask_16(mask: torch.Tensor) -> torch.Tensor:
    """Dilate CLIP mask by +1px at 16x16 grid, then upscale back.

    Downsamples mask to 16x16 via maxpool (matching MaskedAnomalySelfAttention),
    applies +1 cell dilation, then upscales to original resolution via nearest.
    This gives the self-attention 1 extra ring of context tokens.

    Args:
        mask: [B, 1, H, W] binary mask (e.g. [1, 1, 224, 224])

    Returns:
        Dilated mask at original resolution [B, 1, H, W]
    """
    h, w = mask.shape[-2:]
    # Downscale to 16x16 via maxpool (same as model internals)
    kernel = h // 16
    mask_16 = F.max_pool2d(mask.float(), kernel_size=kernel)
    mask_16 = (mask_16 > 0.5).float()
    # +1px dilation at 16x16
    mask_16_dilated = F.max_pool2d(mask_16, kernel_size=3, stride=1, padding=1)
    # Upscale back to original resolution
    return F.interpolate(mask_16_dilated, size=(h, w), mode="nearest")


def _cat_text(a, b):
    """Concatenate text conditioning along the batch (tensor for SD 1.5, SDXLTextCond for SDXL)."""
    if torch.is_tensor(a):
        return torch.cat([a, b])
    return type(a)(torch.cat([a.embeds, b.embeds]), torch.cat([a.pooled, b.pooled]))


def generate_anomagic_single(
    pipeline, ip_adapter, normal_image, mask, reference,
    anomaly_type, caption,
    num_steps=50, guidance_scale=7.5,
    noise_strength: float = 0.7,
    reference_mode: str = "full",
    band_mode: int = 1,
    t2i_adapter=None,
    seed: int = None,
    clip_mask: torch.Tensor = None,
    clip_core_mask: torch.Tensor = None,
    reference_2: torch.Tensor = None,
    clip_mask_2: torch.Tensor = None,
    clip_core_mask_2: torch.Tensor = None,
    group_valid: torch.Tensor = None,
    dilate_clip_mask: bool = False,
    cfg_mode: str = "visual",
    guidance_schedule=None,
    inference_mode: str = "same",
    cross_attn_mask_mode: str = "alpha",  # "alpha" = soft (default), "core_binary" = core-only binary
    zero_t2i_band: bool = False,            # zero the T2I band input channel (ablation)
    inpaint_mask_mode: str = "dilated",     # "dilated" (default) or "core_binary" — UNet mask channel ablation
    raw_background: bool = False,            # blend the paint band into the RAW canvas instead of the decoded one
    context_latents: torch.Tensor = None,   # latent-carry multi-edit: previous pass's state latent
    return_latents: bool = False,           # also return this pass's state latent (for the next pass)
    even_steps: bool = False,               # run num_steps evenly spaced steps from the start timestep (EvenStepDDIM)
):
    """Generate anomaly using IP-Adapter + text captions.

    Starts from the normal image noised to ``noise_strength`` (default 70%)
    and denoises from that point. This preserves structure outside the mask
    and gives the model a realistic starting point.

    Args:
        noise_strength: Fraction of the diffusion schedule to noise through.
            0.0 = no noise (returns normal image), 1.0 = pure noise.
        clip_mask: Optional [B, 1, H, W] dilated mask for CLIP self-attention.
            When provided, overrides the default mask logic (useful when
            reference is a crop with its own mask separate from the UNet mask).
        clip_core_mask: Optional [B, 1, H, W] core mask for role embeddings.
        reference_2: Optional second CLIP crop [1, 3, 224, 224] in [-1, 1]
            for multi-crop mode (matches training with --multi-crop).
        clip_mask_2: Optional mask for second crop [1, 1, 224, 224].
        clip_core_mask_2: Optional core mask for second crop [1, 1, 224, 224].
        group_valid: Optional [1, 2] validity flags per crop group.
        dilate_clip_mask: If True, expand CLIP mask by +1px at 16x16 grid
            before self-attention. Use for hard/deformation anomalies.
        cfg_mode: CFG direction. "visual" (default, the trained/thesis setting) = amplify the reference image on a
            text baseline: uncond = (real text, zero IP). "text" = amplify the caption only: uncond = (empty text,
            real IP) -- the reference image is then NEVER amplified. "both" = uncond = (empty text, zero IP).
        guidance_schedule: Optional callable mapping t_normalized [B, 1] to
            s [B, 1]. When provided, overrides ``guidance_scale`` with a
            per-timestep learned scale (used by CFG configs E/F).
        inference_mode: "same" = reference is from the same image as target
            (training-aligned: dilated CLIP mask + core/band role embeddings).
            "different" = reference is from a different image (core-only CLIP
            mask, no dilation — avoids contamination from wrong surface).
            UNet inpainting (band_mode) is unaffected by this setting.
        context_latents: Latent-carry multi-edit. When given, ``normal_image`` must be the PREVIOUS pass's
            output pixels and ``context_latents`` the previous pass's state latent. The pass then starts from and
            blends against that latent (no re-encode) and pixel-blends against those pixels (no re-decode), so the
            background is decoded exactly once however many passes run. ``normal_image`` is only encoded for the
            inpainting conditioning channel (masked image), never for output pixels. None = unchanged behaviour.
        raw_background: Option R. Final pixel blend uses the RAW canvas (``normal_image``) outside/in the band instead
            of its VAE round-trip, so the background is the untouched photo (no global VAE fingerprint). The matching
            refiner reference is compute_refined_masks' default alpha-blend (decoded in the band, raw outside).
            Default False = Option D (decoded canvas, seamless, VAE fingerprint everywhere).
        return_latents: If True, return ``(output, state_latent)`` where state_latent = alpha-composite of this
            pass's latent over the context latent - the context for the next pass.
    """
    device = normal_image.device

    with torch.amp.autocast('cuda'):
        normal_image = normal_image.float()
        mask = mask.float()

        # Pathway 1: Text
        if caption:
            prompt = caption
        else:
            type_word = anomaly_type.replace("_", " ")
            prompt = f"a photo of a {type_word} defect"

        text_emb = pipeline.encode_text([prompt], enable_grad=False)

        # Pathway 2: Visual (IP-Adapter) with masked mean pooling
        ref_01 = (reference + 1.0) / 2.0
        if clip_mask is not None:
            # Explicit CLIP mask provided (e.g. cropped mask from clip_crop)
            _clip_mask = clip_mask
            _clip_core = clip_core_mask
        elif reference_mode == "full":
            # Full reference: reuse the UNet inpainting mask for CLIP.
            # Use raw mask as core (undilated anomaly region) — the dilated
            # version will be computed inside encode_image via active_mask.
            _clip_mask = mask
            _clip_core = mask  # core = raw mask (before band dilation)
        else:
            _clip_mask = None
            _clip_core = None

        # "different" mode: use core-only for CLIP self-attention (no dilation).
        # Avoids contaminating the encoding with surface tokens from wrong image.
        if inference_mode == "different":
            if _clip_core is not None:
                _clip_mask = _clip_core  # core as both active mask and core
            # Skip dilate_clip_mask entirely — no dilation in "different" mode
        else:
            # "same" mode: optionally expand CLIP mask by +1px at 16x16 grid
            if dilate_clip_mask and _clip_mask is not None:
                _clip_mask = _dilate_clip_mask_16(_clip_mask)

        ip_image_embeds = ip_adapter.encode_image(ref_01, mask=_clip_mask, core_mask=_clip_core)  # [1, K, dim]

        # --- Multi-crop: encode second crop and concatenate (matches training) ---
        null_token_mask = None
        if reference_2 is not None:
            ref_2_01 = (reference_2 + 1.0) / 2.0
            _clip_mask_2 = clip_mask_2 if clip_mask_2 is not None else None
            _clip_core_2 = clip_core_mask_2 if clip_core_mask_2 is not None else None

            if inference_mode == "different":
                if _clip_core_2 is not None:
                    _clip_mask_2 = _clip_core_2
            else:
                if dilate_clip_mask and _clip_mask_2 is not None:
                    _clip_mask_2 = _dilate_clip_mask_16(_clip_mask_2)

            ip_image_embeds_2 = ip_adapter.encode_image(ref_2_01, mask=_clip_mask_2, core_mask=_clip_core_2)  # [1, K, dim]

            # Concatenate: [1, K, dim] + [1, K, dim] -> [1, 2K, dim]
            K = ip_image_embeds.shape[1]
            ip_image_embeds = torch.cat([ip_image_embeds, ip_image_embeds_2], dim=1)  # [1, 2K, dim]

            # Build null_token_mask [1, 2K]: 1=valid, 0=null (from invalid groups)
            if group_valid is not None:
                mask_1 = group_valid[:, 0:1].expand(-1, K)  # [1, K]
                mask_2 = group_valid[:, 1:2].expand(-1, K)  # [1, K]
                null_token_mask = torch.cat([mask_1, mask_2], dim=1)  # [1, 2K]

        # CFG: choose which pathway to amplify
        if cfg_mode == "visual":
            # Amplify visual: uncond = (real_text, zero_IP)
            # pred = text_only + scale * (text+visual - text_only)
            uncond_text = text_emb
            uncond_ip = torch.zeros_like(ip_image_embeds)
        elif cfg_mode == "both":
            # Amplify both: uncond = (empty_text, zero_IP)
            # pred = uncond + scale * (text+visual - uncond)
            uncond_text = pipeline.encode_text([""], enable_grad=False)
            uncond_ip = torch.zeros_like(ip_image_embeds)
        else:
            # Amplify text: uncond = (empty_text, real_IP) -- reference image never amplified
            # pred = visual_only + scale * (text+visual - visual_only)
            uncond_text = pipeline.encode_text([""], enable_grad=False)
            uncond_ip = ip_image_embeds

        # Inpainting setup — latent-space band dilation
        normal_latents = pipeline.encode_image(normal_image) if context_latents is None else context_latents
        kernel = mask.shape[-1] // normal_latents.shape[-1]
        core_mask_64 = F.max_pool2d(mask, kernel_size=kernel)
        core_mask_64 = (core_mask_64 > 0.5).float()

        dilated_binary_64, alpha_map_64, _, band_mask_64 = create_latent_band_mask(
            core_mask_64, band_mode,
        )

        # Ablation: use core-only mask for UNet inpainting channel (mask + masked_image_latents).
        # Alpha blend (below) still uses soft alpha_map_64 — this isolates inpainting-channel effect.
        _inpaint_mask_64 = core_mask_64 if inpaint_mask_mode == "core_binary" else dilated_binary_64
        mask_latents = _inpaint_mask_64
        unet_mask_512 = F.interpolate(_inpaint_mask_64, size=mask.shape[-2:], mode='nearest')
        _cond_img = normal_image if context_latents is None else normal_image.clamp(-1, 1)
        masked_image = _cond_img * (1 - unet_mask_512)
        masked_image_latents = pipeline.encode_image(masked_image)

        # T2I-Adapter features (constant across timesteps, computed ONCE outside loop)
        t2i_kwargs = {}
        t2i_features_doubled = None
        if t2i_adapter is not None:
            _band_in = torch.zeros_like(band_mask_64) if zero_t2i_band else band_mask_64
            t2i_input = torch.cat([core_mask_64, _band_in], dim=1)  # [1, 2, 64, 64]
            t2i_features = t2i_adapter(t2i_input, mask=dilated_binary_64)
            t2i_kwargs_single = t2i_adapter.prepare_unet_kwargs(t2i_features)
            # CFG: both branches get real T2I features (visual always present)
            t2i_kwargs = {
                k: [torch.cat([f, f]) for f in vs]
                for k, vs in t2i_kwargs_single.items()
            }
            if t2i_adapter.pair_injection == "all":
                t2i_features_doubled = [torch.cat([f, f]) for f in t2i_features]

        # Start from normal image noised to noise_strength (not pure noise)
        pipeline.scheduler.set_timesteps(num_steps, device=device)
        all_timesteps = pipeline.scheduler.timesteps
        # Find the starting timestep: noise_strength=0.7 → start at 70% through schedule
        start_step = max(0, int(len(all_timesteps) * (1 - noise_strength)))
        start_timestep = all_timesteps[start_step]
        sched = pipeline.scheduler
        if even_steps:                                   # same start noise, num_steps steps instead of the grid's remainder
            sched = EvenStepDDIM(pipeline.scheduler, int(start_timestep), num_steps, device)
            all_timesteps, start_step = sched.timesteps, 0
            start_timestep = all_timesteps[0]

        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn(normal_latents.shape, generator=gen, device=device, dtype=normal_latents.dtype)
        else:
            noise = torch.randn_like(normal_latents)
        latents = sched.add_noise(normal_latents, noise, start_timestep.unsqueeze(0))

        for i, t in enumerate(all_timesteps[start_step:]):
            latent_input = torch.cat([latents] * 2)
            mask_input = torch.cat([mask_latents] * 2)
            masked_input = torch.cat([masked_image_latents] * 2)

            model_input = torch.cat([latent_input, mask_input, masked_input], dim=1)

            combined_text = _cat_text(uncond_text, text_emb)
            combined_ip = torch.cat([uncond_ip, ip_image_embeds])

            cross_attn_kwargs = {"ip_adapter_image_embeds": combined_ip}
            # Cross-attention spatial mask — default soft alpha, optionally core-only binary.
            _ca_mask = core_mask_64 if cross_attn_mask_mode == "core_binary" else alpha_map_64
            cross_attn_kwargs["ip_adapter_mask"] = torch.cat([_ca_mask, _ca_mask])
            # Multi-crop null masking (duplicated for CFG)
            if null_token_mask is not None:
                cross_attn_kwargs["null_token_mask"] = torch.cat([null_token_mask, null_token_mask])

            if t2i_features_doubled is not None:
                t2i_adapter.set_hook_features(t2i_features_doubled)
            noise_pred = pipeline.unet_forward(
                model_input, t,
                combined_text,
                cross_attention_kwargs=cross_attn_kwargs,
                **t2i_kwargs,
            ).sample.float()
            if t2i_features_doubled is not None:
                t2i_adapter.clear_hook_features()

            noise_uncond, noise_cond = noise_pred.chunk(2)
            if guidance_schedule is not None:
                # Learned per-timestep scale: s(t) from GuidanceSchedule MLP
                # Match training: t_norm = t / 1000.0 (num_train_timesteps)
                t_norm = (t.float() / 1000.0).reshape(1, 1)  # [1, 1]
                s_t = guidance_schedule(t_norm)  # [1, 1]
                noise_pred = noise_uncond + s_t.unsqueeze(-1).unsqueeze(-1) * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latents = sched.step(noise_pred, t, latents).prev_sample

            # Alpha-blended diffusion: blend generated x_{t-1} with correctly-noised normal
            # Normal side: noise from clean x_0 at the next timestep (exact forward process)
            # Generated side: already at x_{t-1} from scheduler.step (correct noise level)
            remaining = len(all_timesteps) - (start_step + i + 1)
            if remaining > 0:
                next_t = all_timesteps[start_step + i + 1]
                noisy_normal = sched.add_noise(normal_latents, noise, next_t.unsqueeze(0))
                latents = latents * alpha_map_64 + noisy_normal * (1 - alpha_map_64)

    output = pipeline.decode_latents(latents.float())
    # Roundtrip the canvas through the VAE so both sides of the blend share the
    # same encoder/decoder bias — eliminates the seam from asymmetric decoding.
    # Latent carry: the previous pass's output pixels are already that decode — reuse them, never decode twice.
    if raw_background:
        normal_decoded = normal_image.float()      # Option R: the untouched canvas pixels
    else:
        normal_decoded = pipeline.decode_latents(normal_latents.float()) if context_latents is None else normal_image.float()
    alpha_512 = F.interpolate(alpha_map_64, size=output.shape[-2:], mode='nearest')
    output = output * alpha_512 + normal_decoded * (1 - alpha_512)

    if return_latents:
        state = latents.float() * alpha_map_64 + normal_latents.float() * (1 - alpha_map_64)
        return output, state
    return output
