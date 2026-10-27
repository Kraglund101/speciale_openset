"""
Stable Diffusion XL inpainting pipeline for IP-Adapter training (SDXL port of the SD 1.5 setup).

Differences from SD 1.5 handled here:
- Dual text encoders (CLIP ViT-L/14 + OpenCLIP ViT-bigG): token embeddings = concat of both penultimate hidden
  states [B, 77, 2048]; pooled embedding from text_encoder_2 [B, 1280].
- UNet micro-conditioning: ``added_cond_kwargs`` = pooled text + time ids (orig size, crop (0, 0), target size).
- Scheduler: DDIMScheduler built from the model's scheduler config (same scaled-linear betas, leading spacing,
  steps_offset 1) — the scheduler class of runwayml/stable-diffusion-inpainting used by the SD 1.5 setup (training
  ``add_noise`` + inference ``step``). The repo ships an Euler scheduler whose ``add_noise`` is sigma-parameterised.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionXLInpaintPipeline

from src.models.base import BaseSDPipeline, SDConfig


class SDXLTextCond:
    """SDXL text conditioning that can be row-indexed/assigned like a tensor.

    ``text_emb[i] = uncond_text[i]`` (conditioning dropout in training) replaces both the token embeddings and
    the pooled embedding of sample i.
    """

    def __init__(self, embeds: torch.Tensor, pooled: torch.Tensor):
        self.embeds = embeds
        self.pooled = pooled

    def __len__(self) -> int:
        return self.embeds.shape[0]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SDXLTextCond(self.embeds[i], self.pooled[i])
        return self.embeds[i], self.pooled[i]

    def __setitem__(self, i, value: Tuple[torch.Tensor, torch.Tensor]) -> None:
        embeds, pooled = value
        self.embeds[i] = embeds
        self.pooled[i] = pooled


class SDXLPipeline(BaseSDPipeline):
    """SDXL inpainting (9-channel UNet) with the interface used by scripts/train_anomagic.py."""

    def __init__(self, config: SDConfig, device: str = "cuda", **kwargs):
        super().__init__(config, device)
        self.dtype = kwargs.get("dtype", torch.float16)
        self.text_encoder_2 = None
        self.tokenizer_2 = None

    def load_pipeline(self, pretrained_model: Optional[str] = None, **kwargs) -> None:
        """Load the SDXL inpainting pipeline (fp16 weights variant when available)."""
        model_id = pretrained_model or self.config.pretrained_model
        try:
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_id, torch_dtype=self.dtype, variant="fp16", **kwargs)
        except (OSError, ValueError):
            self.pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, torch_dtype=self.dtype, **kwargs)
        self.pipeline.to(self.device)
        self.text_encoder = self.pipeline.text_encoder
        self.text_encoder_2 = self.pipeline.text_encoder_2
        self.tokenizer = self.pipeline.tokenizer
        self.tokenizer_2 = self.pipeline.tokenizer_2
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

    def _encode_text(self, text: List[str]) -> SDXLTextCond:
        tok_1 = self.tokenizer(text, padding="max_length", max_length=self.tokenizer.model_max_length,
                               truncation=True, return_tensors="pt").input_ids.to(self.device)
        tok_2 = self.tokenizer_2(text, padding="max_length", max_length=self.tokenizer_2.model_max_length,
                                 truncation=True, return_tensors="pt").input_ids.to(self.device)
        out_1 = self.text_encoder(tok_1, output_hidden_states=True)
        out_2 = self.text_encoder_2(tok_2, output_hidden_states=True)
        embeds = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)   # [B, 77, 2048]
        pooled = out_2[0]                                                                 # [B, 1280]
        return SDXLTextCond(embeds, pooled)

    def encode_text(self, text: Union[str, List[str]], enable_grad: bool = False) -> SDXLTextCond:
        """Encode prompts with both SDXL text encoders (penultimate hidden states + pooled)."""
        if isinstance(text, str):
            text = [text]
        if enable_grad:
            return self._encode_text(text)
        with torch.no_grad():
            return self._encode_text(text)

    def time_ids(self, batch_size: int, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
        """SDXL micro-conditioning: (orig_h, orig_w, crop_top, crop_left, target_h, target_w)."""
        ids = torch.tensor([height, width, 0, 0, height, width], device=self.device, dtype=dtype)
        return ids.unsqueeze(0).repeat(batch_size, 1)

    def unet_forward(self, model_input, timesteps, text_emb: SDXLTextCond, cross_attention_kwargs=None, **kwargs):
        """UNet forward with pooled text + time ids (image size inferred from the latent size)."""
        b, _, h, w = model_input.shape
        scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
        added = {"text_embeds": text_emb.pooled,
                 "time_ids": self.time_ids(b, h * scale, w * scale, text_emb.pooled.dtype)}
        return self.unet(model_input, timesteps, encoder_hidden_states=text_emb.embeds,
                         added_cond_kwargs=added, cross_attention_kwargs=cross_attention_kwargs, **kwargs)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image ([-1, 1]) to scaled VAE latents."""
        image = image.to(self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.vae.config.scaling_factor
        with torch.no_grad():
            return self.vae.decode(latents.to(self.vae.dtype)).sample

    def get_text_embedding_dim(self) -> int:
        return self.text_encoder.config.hidden_size + self.text_encoder_2.config.hidden_size

    def add_concept_token(self, token_name: str, initializer_token: Optional[str] = None,
                          num_vectors: int = 1) -> torch.Tensor:
        raise NotImplementedError("Concept tokens are not used by the SDXL IP-Adapter port")

    def get_concept_embedding(self, token_name: str) -> torch.Tensor:
        raise NotImplementedError("Concept tokens are not used by the SDXL IP-Adapter port")

    def set_concept_embedding(self, token_name: str, embedding: torch.Tensor) -> None:
        raise NotImplementedError("Concept tokens are not used by the SDXL IP-Adapter port")

    def freeze_all(self) -> None:
        for module in (self.unet, self.vae, self.text_encoder, self.text_encoder_2):
            module.requires_grad_(False)

    def freeze_all_except_embeddings(self) -> None:
        raise NotImplementedError("Concept tokens are not used by the SDXL IP-Adapter port")
