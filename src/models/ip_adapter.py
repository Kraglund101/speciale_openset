"""
IP-Adapter integration module with masked self-attention (Anomagic Eq. 3).

Supports two CLIP encoder types:
- Standard: CLIP ViT-L/14 (768-dim) → masked self-attn → mean pool → linear → [B, K, 768]
- Plus:     CLIP ViT-H/14 (1280-dim) → masked self-attn → resampler → [B, K, 768]

Pipeline:
1. Extract patch tokens from penultimate CLIP layer (NO CLS token) — frozen
2. Masked self-attention: all 256 tokens become anomaly-aware via
   P_v = Softmax(QK^T - (1-M)*C) * V — trainable
3. Projection to K cross-attention tokens (linear or perceiver resampler) — trainable

For SD 1.5 and SDXL, with version-specific handling.
"""
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseSDPipeline, SDVersion


@dataclass
class IPAdapterConfig:
    """Configuration for IP-Adapter."""
    adapter_type: str = "plus"  # "standard" (ViT-L/14) or "plus" (ViT-H/14)
    num_tokens: int = 4  # Visual tokens after projection: 1, 4, or 16
    ip_adapter_ckpt: Optional[str] = None
    scale: float = 1.0  # IP-Adapter visual pathway strength
    # Selective residual: anomaly tokens get gated residual, background tokens stay pure
    anomaly_residual: bool = False
    # Masked cross-attention: visual pathway only affects anomaly positions.
    # Text pathway remains unmasked. Default True — visual tokens from masked CLIP
    # self-attn are pure anomaly, useless noise for normal positions.
    mask_visual: bool = True
    # Visual token processing mode (0-3). See MaskedAnomalySelfAttention docstring.
    # 0: all 256 patches, gated residual on all
    # 1: all 256 patches, selective residual (background = pure attn mix)
    # 2: anomaly-only + attn-only (no MLP), padding dead everywhere
    # 3: anomaly-only + full transformer, padding dead everywhere
    visual_mode: int = 0
    # Force gates to 1.0 with normal projection init (block active from step 0)
    force_gates: bool = False
    # When False: remove scalar gates, zero-init output projections instead.
    # Identity at init (same as gates=0), but per-dimension gradient signal.
    learnable_gates: bool = True
    # Number of transformer layers in masked self-attention (default 1).
    sa_num_layers: int = 1
    # Number of attention heads in masked self-attention (default 12, matching Resampler).
    sa_num_heads: int = 12


# Default checkpoints keyed by (sd_version, adapter_type)
IP_ADAPTER_CHECKPOINTS = {
    (SDVersion.SD_1_5, "standard"): "h94/IP-Adapter",
    (SDVersion.SD_1_5, "plus"): "h94/IP-Adapter",
    (SDVersion.SD_XL, "standard"): "h94/IP-Adapter",
    (SDVersion.SD_XL, "plus"): "h94/IP-Adapter",
}

IP_ADAPTER_FILENAMES = {
    (SDVersion.SD_1_5, "standard"): "models/ip-adapter_sd15.safetensors",
    (SDVersion.SD_1_5, "plus"): "models/ip-adapter-plus_sd15.safetensors",
    (SDVersion.SD_XL, "standard"): "sdxl_models/ip-adapter_sdxl.safetensors",
    (SDVersion.SD_XL, "plus"): "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
}


# ---------------------------------------------------------------------------
# Resampler (official IP-Adapter Plus architecture — matches pretrained weights)
# ---------------------------------------------------------------------------

def _reshape_for_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """Reshape [B, L, D] → [B, heads, L, dim_head] for multi-head attention."""
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    return x.transpose(1, 2)


def _ff_block(dim: int, mult: int = 4) -> nn.Sequential:
    """Feed-forward block: LayerNorm → Linear → GELU → Linear (all bias=False on linears)."""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    """Cross-attention layer for perceiver resampler.

    Queries from learnable latents attend over image features (+ latents for self-attn).
    """

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Image features [B, N, D]
            latents: Learnable latent queries [B, K, D]
            key_padding_mask: [B, N] where 1=valid, 0=padding. Only masks the
                image tokens (x); latent self-attention positions are always valid.
        Returns:
            Updated latents [B, K, D]
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape
        n = x.shape[1]

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = _reshape_for_heads(q, self.heads)  # [B, heads, l, dim_head]
        k = _reshape_for_heads(k, self.heads)  # [B, heads, N+l, dim_head]
        v = _reshape_for_heads(v, self.heads)  # [B, heads, N+l, dim_head]

        # Stable scaled attention (double sqrt scaling from official IP-Adapter)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)

        # Mask padding tokens in key positions (image tokens only, latents always valid)
        if key_padding_mask is not None:
            # [B, N] → [B, N+l]: pad with 1s for latent positions (always valid)
            latent_valid = torch.ones(b, l, device=key_padding_mask.device, dtype=key_padding_mask.dtype)
            full_mask = torch.cat([key_padding_mask, latent_valid], dim=1)  # [B, N+l]
            attn_bias = (1 - full_mask).unsqueeze(1).unsqueeze(1) * -1e6  # [B, 1, 1, N+l]
            weight = weight + attn_bias

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    """Perceiver resampler for IP-Adapter Plus — matches official pretrained weights.

    Projects N CLIP patch tokens → K cross-attention tokens via learned latent queries
    and multi-layer perceiver cross-attention.

    Architecture: proj_in → depth × (PerceiverAttention + FeedForward) → proj_out → norm_out

    For SD 1.5 Plus: dim=768, depth=4, heads=12, dim_head=64, embedding_dim=1280, K=16
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 4,
        dim_head: int = 64,
        heads: int = 12,
        num_queries: int = 16,
        embedding_dim: int = 1280,
        output_dim: int = 768,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    _ff_block(dim=dim, mult=ff_mult),
                ])
            )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: CLIP patch tokens [B, N, embedding_dim] (e.g. [B, 256, 1280])
            key_padding_mask: [B, N] where 1=valid, 0=padding. Suppresses
                padding tokens in perceiver cross-attention (modes 2-3).
        Returns:
            Resampled tokens [B, K, output_dim] (e.g. [B, 16, 768])
        """
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)

        # Mask lives in embedding_dim space after proj_in, same token positions
        for attn, ff in self.layers:
            latents = attn(x, latents, key_padding_mask=key_padding_mask) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


# ---------------------------------------------------------------------------
# Linear projection (standard IP-Adapter)
# ---------------------------------------------------------------------------

class ImageProjection(nn.Module):
    """Projects CLIP image embeddings to cross-attention format via linear + norm.

    CLS projection [B, D] → Linear → reshape → LayerNorm → [B, K, cross_attn_dim]
    Used by standard IP-Adapter. Matches official h94/IP-Adapter checkpoint.

    Note: Standard IP-Adapter uses ViT-H CLS projection (1024-dim), NOT ViT-L.
    """

    def __init__(
        self,
        clip_embed_dim: int = 1024,
        cross_attention_dim: int = 768,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.clip_embed_dim = clip_embed_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Linear(clip_embed_dim, cross_attention_dim * num_tokens)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds: [B, D] CLS projection vector
        Returns:
            [B, K, cross_attention_dim]
        """
        batch_size = image_embeds.shape[0]
        if image_embeds.dim() == 3:
            image_embeds = image_embeds.mean(dim=1)
        out = self.proj(image_embeds)
        out = out.view(batch_size, self.num_tokens, self.cross_attention_dim)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Masked anomaly self-attention (Anomagic Eq. 3)
# ---------------------------------------------------------------------------

class MaskedAnomalySelfAttention(nn.Module):
    """
    Pre-norm transformer encoder for anomaly-aware feature extraction.

    N layers of: LayerNorm -> Multi-Head Self-Attention -> Residual -> LayerNorm -> MLP -> Residual

    Self-attention uses anomaly masking (Eq. 3) to suppress non-anomaly tokens:
        P_v = Softmax(QK^T / sqrt(d_k) - (1-M)*C) * V   where C=1e6

    Both sub-layers use gated residuals (gates init to 0 -> pure identity at start).

    Visual modes:
        0: All 256 tokens, masked self-attn, gated residuals on all tokens (default)
        1: All 256 tokens, selective residual — anomaly tokens get gated residual,
           background tokens become pure attention-weighted anomaly mixes
        2: Anomaly-only extraction + attention-only (no MLP), padding tokens
           fully zeroed (dead tokens, never read by self-attn or resampler)
        3: Anomaly-only extraction + full transformer, padding tokens fully
           zeroed at every stage (dead tokens, never read by self-attn or resampler)

    Args:
        in_channels: Dimension of input patch tokens (1280 for ViT-H)
        attn_dim: Q/K/V projection dimension (default 768)
        num_heads: Number of attention heads (default 1). attn_dim must be divisible by num_heads.
        num_layers: Number of transformer layers (default 1)
        ff_mult: MLP expansion factor (default 2)
        visual_mode: Token processing strategy (0-3)
    """

    def __init__(
        self,
        in_channels: int,
        attn_dim: int = 768,
        num_heads: int = 1,
        num_layers: int = 1,
        ff_mult: int = 2,
        visual_mode: int = 0,
        learnable_gates: bool = True,
        force_gates: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.visual_mode = visual_mode
        self.learnable_gates = learnable_gates and not force_gates
        self.force_gates = force_gates
        assert attn_dim % num_heads == 0, f"attn_dim {attn_dim} not divisible by num_heads {num_heads}"
        self.dim_head = attn_dim // num_heads

        # Mode 2 skips MLP (attention-only); all other modes have MLP
        has_mlp = (visual_mode != 2)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_dict = {
                # Attention sub-layer
                "attn_norm": nn.LayerNorm(in_channels),
                "to_q": nn.Linear(in_channels, attn_dim, bias=False),
                "to_k": nn.Linear(in_channels, attn_dim, bias=False),
                "to_v": nn.Linear(in_channels, attn_dim, bias=False),
                "to_out": nn.Linear(attn_dim, in_channels, bias=False),
            }
            if has_mlp:
                layer_dict["ff_norm"] = nn.LayerNorm(in_channels)
                layer_dict["ff"] = nn.Sequential(
                    nn.Linear(in_channels, in_channels * ff_mult, bias=False),
                    nn.GELU(),
                    nn.Linear(in_channels * ff_mult, in_channels, bias=False),
                )
            self.layers.append(nn.ModuleDict(layer_dict))

            if force_gates:
                # Forced gates: gate=1.0, normal projection init.
                # Block contributes fully from step 0 — forces reliance.
                pass
            elif learnable_gates:
                # Scalar gates (init 0 → identity at start)
                self.layers[-1]["attn_gate"] = _GateParameter()
                if has_mlp:
                    self.layers[-1]["ff_gate"] = _GateParameter()
            else:
                # No gates: zero-init output projections instead.
                # x + to_out(attn) starts as x + 0 = identity, but each weight
                # gets its own gradient (no scalar bottleneck).
                nn.init.zeros_(layer_dict["to_out"].weight)
                if has_mlp:
                    nn.init.zeros_(layer_dict["ff"][2].weight)

        self.scale = self.dim_head ** -0.5

        # Role embeddings for CLIP-UNet alignment: distinguish core (anomaly),
        # band (normal context), and global tokens. Zero-init = no-op at start.
        # Old checkpoints load fine (missing keys default to zeros).
        self.emb_anomaly = nn.Parameter(torch.zeros(1, 1, in_channels))  # [1, 1, 1280]
        self.emb_normal = nn.Parameter(torch.zeros(1, 1, in_channels))   # [1, 1, 1280]
        self.emb_global = nn.Parameter(torch.zeros(1, 1, in_channels))   # [1, 1, 1280]

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Patch tokens [B, N, D] where N=256 (modes 0-1) or variable (modes 2-3)
            mask: Binary anomaly mask [B, 1, H, W] in {0, 1}. Used by modes 0-1.
            padding_mask: [B, N] where 1=real token, 0=padding. Used by modes 2-3.

        Returns:
            Anomaly-aware tokens [B, N, D]
        """
        B, N, D = x.shape
        mode = self.visual_mode

        # --- Precompute masks ---
        attn_mask_bias = None  # [B, 1, 1, N] additive bias for attention scores
        blend_mask = None      # [B, N, 1] for selective residual blending

        if mode in (0, 1) and mask is not None:
            grid_size = int(N ** 0.5)
            kernel = mask.shape[-1] // grid_size
            mask_grid = F.max_pool2d(mask.float(), kernel_size=kernel)
            mask_grid = (mask_grid > 0.5).to(x.dtype)
            mask_flat = mask_grid.view(B, 1, N)  # [B, 1, N]
            # Broadcast: [B, 1, 1, N] for multi-head compatibility
            attn_mask_bias = (1 - mask_flat).unsqueeze(1) * -1e6  # [B, 1, 1, N]
            if mode == 1:
                # [B, N, 1] for element-wise blending with x [B, N, D]
                blend_mask = mask_flat.permute(0, 2, 1)  # [B, N, 1]
        elif mode == 1 and mask is None:
            # No mask → all tokens treated as anomaly (blend_mask = 1 everywhere)
            blend_mask = torch.ones(B, N, 1, device=x.device, dtype=x.dtype)

        elif mode in (2, 3) and padding_mask is not None:
            # padding_mask [B, N]: 1=real, 0=pad → attn bias on columns
            pad_flat_1n = padding_mask.unsqueeze(1)  # [B, 1, N]
            attn_mask_bias = (1 - pad_flat_1n).unsqueeze(1) * -1e6  # [B, 1, 1, N]
            # [B, N, 1] for zeroing padding positions
            blend_mask = padding_mask.unsqueeze(-1)  # [B, N, 1]

        for layer in self.layers:
            # --- Multi-Head Self-Attention with anomaly masking ---
            h = layer["attn_norm"](x)

            q = layer["to_q"](h)  # [B, N, attn_dim]
            k = layer["to_k"](h)
            v = layer["to_v"](h)

            # Reshape for multi-head: [B, N, attn_dim] -> [B, heads, N, dim_head]
            q = q.view(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
            k = k.view(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)
            v = v.view(B, N, self.num_heads, self.dim_head).permute(0, 2, 1, 3)

            # Scaled dot-product attention: [B, heads, N, N]
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if attn_mask_bias is not None:
                scores = scores + attn_mask_bias  # broadcast over heads

            weights = torch.softmax(scores, dim=-1)

            # Weighted sum: [B, heads, N, dim_head]
            attn_out = torch.matmul(weights, v)

            # Merge heads: [B, N, attn_dim]
            attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.attn_dim)
            attn_out = layer["to_out"](attn_out)  # [B, N, D]

            # --- Attention residual ---
            attn_g = layer["attn_gate"].gate if self.learnable_gates else 1.0
            if mode == 0:
                # All tokens: standard gated residual
                x = x + attn_g * attn_out
            elif mode == 1:
                # Selective: anomaly tokens get gated residual, background = pure attn mix
                x = blend_mask * (x + attn_g * attn_out) + (1 - blend_mask) * attn_out
            elif mode in (2, 3):
                # Real tokens: gated residual; padding tokens: zeroed out
                x = blend_mask * (x + attn_g * attn_out)

            # --- Feed-Forward (skipped for mode 2) ---
            if "ff" in layer:
                ff_out = layer["ff"](layer["ff_norm"](x))
                ff_g = layer["ff_gate"].gate if self.learnable_gates else 1.0
                if mode == 3:
                    # Mode 3: MLP residual on real tokens only, padding stays zero
                    x = x + blend_mask * ff_g * ff_out
                elif mode == 1:
                    # Mode 1: selective MLP residual too
                    x = blend_mask * (x + ff_g * ff_out) + (1 - blend_mask) * ff_out
                else:
                    # Mode 0: standard gated residual
                    x = x + ff_g * ff_out

        return x


class _GateParameter(nn.Module):
    """Learnable scalar gate initialized to 0."""
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))


# ---------------------------------------------------------------------------
# IP-Adapter cross-attention processor (injected into UNet)
# ---------------------------------------------------------------------------

class IPAdapterAttnProcessor(nn.Module):
    """
    Attention processor that injects IP-Adapter visual conditioning.

    Replaces the standard cross-attention in UNet with IP-Adapter augmented version.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 4,
        scale: float = 1.0,
        mask_visual: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale
        self.mask_visual = mask_visual

        # IP-Adapter specific key/value projections
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def _get_spatial_mask(
        self, mask: torch.Tensor, num_spatial: int
    ) -> torch.Tensor:
        """Downsample mask to match UNet layer spatial resolution via maxpool.

        Args:
            mask: [B, 1, H, W] mask (binary or soft alpha) at original resolution
            num_spatial: number of spatial positions at this UNet layer (HW)

        Returns:
            [B, num_spatial, 1] mask for multiplying with attention output
        """
        spatial_size = int(num_spatial ** 0.5)
        kernel = mask.shape[-1] // spatial_size
        if kernel > 1:
            mask_down = F.max_pool2d(mask.float(), kernel_size=kernel)
        else:
            mask_down = mask.float()
        # No binarization: soft alpha values pass through for graded routing.
        # [B, 1, h, w] → [B, h*w, 1] for broadcasting with [B, HW, C]
        return mask_down.view(mask.shape[0], -1).unsqueeze(-1)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        ip_adapter_mask: Optional[torch.Tensor] = None,
        null_token_mask: Optional[torch.Tensor] = None,
        diagnostics: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Process attention with IP-Adapter injection.

        The IP-Adapter adds visual conditioning by:
        1. Computing standard text cross-attention (all positions)
        2. Computing separate IP cross-attention with image embeddings
        3. Masking IP output to anomaly positions only (if mask provided)
        4. Adding scaled IP attention output to text attention output

        Args:
            null_token_mask: [B, T] where 1=valid IP token, 0=null (from invalid
                multi-crop groups). When provided, -1e6 bias is added to null
                token columns in IP attention scores before softmax.
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # Standard text cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Prepare spatial mask for visual pathway
        spatial_mask = None
        if self.mask_visual and ip_adapter_mask is not None:
            num_spatial = hidden_states.shape[1]
            spatial_mask = self._get_spatial_mask(ip_adapter_mask, num_spatial)
            spatial_mask = spatial_mask.to(hidden_states.dtype).to(hidden_states.device)

        # IP-Adapter cross-attention (if image embeds provided)
        if ip_adapter_image_embeds is not None:
            ip_key = self.to_k_ip(ip_adapter_image_embeds)
            ip_value = self.to_v_ip(ip_adapter_image_embeds)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            # Build null token bias for multi-crop (suppress invalid group tokens)
            ip_attn_mask = None
            if null_token_mask is not None:
                # null_token_mask [B, T] → [B, 1, T] → repeat for heads → [B*heads, 1, T]
                null_bias = (1 - null_token_mask.float()).unsqueeze(1) * -1e6  # [B, 1, T]
                n_heads = ip_key.shape[0] // null_token_mask.shape[0]
                ip_attn_mask = null_bias.repeat_interleave(n_heads, dim=0)  # [B*heads, 1, T]

            ip_attention_probs = attn.get_attention_scores(query, ip_key, ip_attn_mask)
            ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

            # --- Diagnostic capture (GPU tensors, no .item() sync) ---
            if diagnostics is not None:
                _layer = getattr(self, "_diag_layer_name", "unknown")
                diagnostics[f"{_layer}/ip_k_norm"] = ip_key.detach().norm()
                diagnostics[f"{_layer}/ip_out_norm"] = ip_hidden_states.detach().norm()
                diagnostics[f"{_layer}/h_pre_norm"] = hidden_states.detach().norm()
                diagnostics[f"{_layer}/text_k_norm"] = key.detach().norm()
                probs = ip_attention_probs.detach()
                diagnostics[f"{_layer}/ip_entropy"] = -(probs * probs.clamp(min=1e-8).log()).sum(-1).mean()

            # Mask IP output to anomaly positions only
            if spatial_mask is not None:
                ip_hidden_states = ip_hidden_states * spatial_mask

            # Add scaled IP attention
            hidden_states = hidden_states + self.scale * ip_hidden_states

        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ---------------------------------------------------------------------------
# Main IP-Adapter module
# ---------------------------------------------------------------------------

class IPAdapter(nn.Module):
    """
    IP-Adapter module for visual conditioning in Stable Diffusion.

    Integrates with the pipeline to provide image-conditioned generation:
    1. Encodes reference image with CLIP vision encoder
    2. Projects embeddings to cross-attention compatible format
    3. Injects into UNet cross-attention layers
    """

    def __init__(
        self,
        pipeline: BaseSDPipeline,
        config: IPAdapterConfig,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.sd_version = pipeline.config.version

        # Will be set during load
        self.image_encoder = None
        self.image_projection = None  # Resampler or ImageProjection
        self.attn_processors: Dict[str, IPAdapterAttnProcessor] = {}
        # Ordered processor names matching UNet iteration order (for checkpoint loading)
        self._ordered_proc_names: List[str] = []

        # Determine cross-attention dim based on SD version
        if self.sd_version == SDVersion.SD_1_5:
            self.cross_attention_dim = 768
        elif self.sd_version == SDVersion.SD_XL:
            self.cross_attention_dim = 2048
        else:
            self.cross_attention_dim = 4096  # SD3 estimate

    def load_ip_adapter(
        self,
        pretrained_model: Optional[str] = None,
        subfolder: Optional[str] = None,
        weight_name: Optional[str] = None,
    ) -> None:
        """
        Load IP-Adapter weights and set up attention processors.

        Args:
            pretrained_model: HuggingFace repo or local path
            subfolder: Subfolder in repo
            weight_name: Specific weight file name
        """
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

        # Both standard and Plus use ViT-H/14 — they differ in which output they use:
        # Standard: CLS projection (image_embeds, 1024-dim)
        # Plus: penultimate hidden states (patch tokens, 1280-dim)
        encoder_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        if self.config.adapter_type == "plus":
            clip_embed_dim = 1280  # ViT-H hidden dim (patch tokens)
        else:
            clip_embed_dim = 1024  # ViT-H projection dim (CLS token)

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            encoder_name
        ).to(self.pipeline.device, dtype=self.pipeline.dtype)
        self.image_encoder.eval()

        self.image_processor = CLIPImageProcessor.from_pretrained(encoder_name)

        # Create image projection — architecture must match pretrained weights
        if self.config.adapter_type == "plus":
            # Masked self-attention over CLIP patch tokens (Anomagic Eq. 3)
            # Only for Plus — standard uses CLS token, no spatial patches
            self.masked_self_attn = MaskedAnomalySelfAttention(
                in_channels=clip_embed_dim,
                visual_mode=self.config.visual_mode,
                learnable_gates=self.config.learnable_gates,
                force_gates=self.config.force_gates,
                num_layers=self.config.sa_num_layers,
                num_heads=self.config.sa_num_heads,
            ).to(self.pipeline.device, dtype=self.pipeline.dtype)

            # Official IP-Adapter Plus: 4-layer perceiver resampler.
            # SD 1.5 plus: dim 768 / 12 heads; SDXL plus (vit-h): dim 1280 / 20 heads, output 2048.
            resampler_dim = 1280 if self.sd_version == SDVersion.SD_XL else self.cross_attention_dim
            self.image_projection = Resampler(
                dim=resampler_dim,  # 768 for SD 1.5
                depth=4,
                dim_head=64,
                heads=resampler_dim // 64,  # 12 for SD 1.5
                num_queries=self.config.num_tokens,
                embedding_dim=clip_embed_dim,  # 1280 for ViT-H
                output_dim=self.cross_attention_dim,
                ff_mult=4,
            ).to(self.pipeline.device, dtype=self.pipeline.dtype)
        else:
            # Standard IP-Adapter: linear projection
            self.image_projection = ImageProjection(
                clip_embed_dim=clip_embed_dim,
                cross_attention_dim=self.cross_attention_dim,
                num_tokens=self.config.num_tokens,
            ).to(self.pipeline.device, dtype=self.pipeline.dtype)

        # Load pretrained weights if provided
        if pretrained_model:
            self._load_pretrained_weights(pretrained_model, subfolder, weight_name)

        # Set up attention processors in UNet
        self._setup_attn_processors()

    def _load_pretrained_weights(
        self,
        pretrained_model: str,
        subfolder: Optional[str],
        weight_name: Optional[str],
    ) -> None:
        """Load pretrained IP-Adapter weights."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Determine weight file
        if weight_name is None:
            key = (self.sd_version, self.config.adapter_type)
            weight_name = IP_ADAPTER_FILENAMES.get(key)

        if weight_name is None:
            print(f"Warning: No default weights for {self.sd_version} type={self.config.adapter_type}")
            return

        # Download or use local path
        if Path(pretrained_model).exists():
            weight_path = Path(pretrained_model) / weight_name
        else:
            weight_path = hf_hub_download(
                repo_id=pretrained_model,
                filename=weight_name,
                subfolder=subfolder,
            )

        print(f"  Loading pretrained weights from {weight_path}")

        # Load state dict (safetensors or torch format)
        try:
            state_dict = load_file(str(weight_path))
        except Exception:
            state_dict = torch.load(str(weight_path), map_location="cpu")
            # Torch .bin files may have nested dicts — flatten them
            if "ip_adapter" in state_dict and isinstance(state_dict["ip_adapter"], dict):
                flat = {}
                for k, v in state_dict.get("image_proj", {}).items():
                    flat[f"image_proj.{k}"] = v
                for k, v in state_dict.get("ip_adapter", {}).items():
                    flat[f"ip_adapter.{k}"] = v
                state_dict = flat

        # Load image projection weights
        proj_state = {
            k.replace("image_proj.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_proj.")
        }
        if proj_state:
            missing, unexpected = self.image_projection.load_state_dict(proj_state, strict=False)
            loaded = len(proj_state) - len(unexpected)
            print(f"  Image projection: loaded {loaded} tensors"
                  f"{f', missing {len(missing)}' if missing else ''}"
                  f"{f', unexpected {len(unexpected)}' if unexpected else ''}")

        # Store IP-adapter attention weights for loading after processors are created
        self._ip_adapter_state = {
            k: v for k, v in state_dict.items()
            if k.startswith("ip_adapter.")
        }

    def _setup_attn_processors(self) -> None:
        """Set up IP-Adapter attention processors in UNet."""
        attn_procs = {}
        unet = self.pipeline.unet

        # Track cross-attn processor names in UNet iteration order (NOT alphabetical)
        # This order matches the checkpoint's integer indices
        ordered_cross_attn_names = []

        for name, attn_processor in unet.attn_processors.items():
            # Only modify cross-attention layers (not self-attention)
            if name.endswith("attn2.processor"):
                if "down_blocks" in name or "up_blocks" in name or "mid_block" in name:
                    hidden_size = self._get_hidden_size_for_layer(name)

                    proc = IPAdapterAttnProcessor(
                        hidden_size=hidden_size,
                        cross_attention_dim=self.cross_attention_dim,
                        num_tokens=self.config.num_tokens,
                        scale=self.config.scale,
                        mask_visual=self.config.mask_visual,
                    )
                    # Store layer name for diagnostic capture
                    proc._diag_layer_name = name

                    attn_procs[name] = proc.to(self.pipeline.device, dtype=self.pipeline.dtype)
                    ordered_cross_attn_names.append(name)
                else:
                    attn_procs[name] = attn_processor
            else:
                attn_procs[name] = attn_processor

        # Save reference to IP processors BEFORE set_attn_processor (which clears the dict)
        self.attn_processors = {
            k: v for k, v in attn_procs.items()
            if isinstance(v, IPAdapterAttnProcessor)
        }
        # Preserve iteration order for checkpoint index mapping
        self._ordered_proc_names = ordered_cross_attn_names

        unet.set_attn_processor(attn_procs)
        print(f"  IP-Adapter cross-attention processors: {len(self.attn_processors)}")

        # Load pretrained weights for attention processors
        if hasattr(self, '_ip_adapter_state') and self._ip_adapter_state:
            self._load_attn_processor_weights()

    def _get_hidden_size_for_layer(self, layer_name: str) -> int:
        """Determine hidden size for a given attention layer."""
        # SD 1.5 hidden sizes by block
        if self.sd_version == SDVersion.SD_1_5:
            if "down_blocks.0" in layer_name:
                return 320
            elif "down_blocks.1" in layer_name:
                return 640
            elif "down_blocks.2" in layer_name or "down_blocks.3" in layer_name:
                return 1280
            elif "mid_block" in layer_name:
                return 1280
            elif "up_blocks.0" in layer_name or "up_blocks.1" in layer_name:
                return 1280
            elif "up_blocks.2" in layer_name:
                return 640
            elif "up_blocks.3" in layer_name:
                return 320

        # SDXL: read the attention width from the UNet itself (down_blocks.1 / up_blocks.1 = 640,
        # down_blocks.2 / mid / up_blocks.0 = 1280; down_blocks.0 and up_blocks.2 have no attention)
        elif self.sd_version == SDVersion.SD_XL:
            attn = self.pipeline.unet.get_submodule(layer_name[: -len(".processor")])
            return attn.to_k.out_features

        return 1280  # Default

    def _load_attn_processor_weights(self) -> None:
        """Load pretrained IP-Adapter attention processor K/V weights.

        The official IP-Adapter checkpoint uses integer indices based on UNet
        attn_processors iteration order (odd indices for cross-attn: 1,3,5,...,31).
        Our finetuned checkpoints use: ip_adapter.{layer_name}.to_k_ip.weight
        This method handles both formats.
        """
        state_dict = self._ip_adapter_state

        # Strip ip_adapter. prefix to normalize keys
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("ip_adapter."):
                stripped[k[len("ip_adapter."):]] = v
            else:
                stripped[k] = v

        # Detect format: official checkpoint uses integer indices, ours uses layer names
        sample_key = next(iter(stripped.keys()), "")
        if sample_key and sample_key[0].isdigit():
            # Official format: {idx}.to_k_ip.weight
            # Use UNet iteration order (stored during _setup_attn_processors)
            layer_indices = sorted(set(
                int(k.split(".")[0]) for k in stripped.keys()
            ))
            proc_names = self._ordered_proc_names

            if len(layer_indices) != len(proc_names):
                print(f"  WARNING: checkpoint has {len(layer_indices)} layers, "
                      f"but UNet has {len(proc_names)} cross-attn processors")

            loaded = 0
            for proc_name, layer_idx in zip(proc_names, layer_indices):
                processor = self.attn_processors[proc_name]
                k_key = f"{layer_idx}.to_k_ip.weight"
                v_key = f"{layer_idx}.to_v_ip.weight"
                if k_key in stripped:
                    processor.to_k_ip.weight.data.copy_(stripped[k_key])
                    loaded += 1
                if v_key in stripped:
                    processor.to_v_ip.weight.data.copy_(stripped[v_key])
                    loaded += 1
            print(f"  Attention processors: loaded {loaded} weight tensors "
                  f"(official format, {len(layer_indices)} layers)")
        else:
            # Finetuned format: {layer_name}.to_k_ip.weight
            loaded = 0
            for name in self._ordered_proc_names:
                processor = self.attn_processors[name]
                k_key = f"{name}.to_k_ip.weight"
                v_key = f"{name}.to_v_ip.weight"
                if k_key in stripped:
                    processor.to_k_ip.weight.data.copy_(stripped[k_key])
                    loaded += 1
                if v_key in stripped:
                    processor.to_v_ip.weight.data.copy_(stripped[v_key])
                    loaded += 1
            print(f"  Attention processors: loaded {loaded} weight tensors (finetuned format)")

        del self._ip_adapter_state

    def encode_image(
        self,
        image: Union[torch.Tensor, "PIL.Image.Image", List],
        mask: Optional[torch.Tensor] = None,
        core_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode reference image(s) using masked self-attention over CLIP patch tokens.

        Pipeline:
        1. Extract patch tokens from penultimate CLIP layer (frozen, no grad)
        2. Apply role embeddings (core=anomaly, band=normal context)
        3. Apply masked self-attention (Eq. 3) — trainable, with grad
           All 256 tokens become anomaly-aware while preserving spatial structure
        4. Plus: feed all 256 tokens → resampler → [B, K, 768]
           Standard: mean pool → linear → [B, K, 768]

        Visual modes 2-3 extract only anomaly patch tokens before self-attention:
        - Downsample mask to 16×16 grid
        - Gather tokens where mask=1, pad to batch max
        - Pass variable-length input through self-attention with padding_mask

        Args:
            image: Input image(s) - tensor [B, C, H, W] in [0,1], PIL image, or list
            mask: Dilated anomaly mask [B, 1, H, W] in {0, 1} (core+band region).
                  Used for attention routing (background suppressed).
            core_mask: Core anomaly mask [B, 1, H, W] in {0, 1}. Used for role
                  embeddings (core vs band distinction). If None, all active
                  patches get emb_anomaly (no band distinction).

        Returns:
            Image embeddings [B, K, cross_attention_dim]
        """
        from PIL import Image as PILImage

        # Process input
        if isinstance(image, torch.Tensor):
            # Tensor input assumed [B, C, H, W] in [0, 1] — apply CLIP normalization
            clip_mean = torch.tensor(self.image_processor.image_mean,
                                     device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            clip_std = torch.tensor(self.image_processor.image_std,
                                    device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
            pixel_values = (image - clip_mean) / clip_std
        else:
            if not isinstance(image, list):
                image = [image]
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
            ).pixel_values

        pixel_values = pixel_values.to(self.pipeline.device, dtype=self.pipeline.dtype)

        with torch.no_grad():
            outputs = self.image_encoder(pixel_values, output_hidden_states=True)

        if isinstance(self.image_projection, Resampler):
            # Plus: patch tokens from penultimate layer
            patch_tokens = outputs.hidden_states[-2][:, 1:, :]  # Remove CLS → [B, 256, 1280]
            B, N, D = patch_tokens.shape

            # --- Global token + active mask computation ---
            if mask is not None:
                grid_size = int(N ** 0.5)  # 16
                kernel = mask.shape[-1] // grid_size
                dil_16 = F.max_pool2d(mask.float(), kernel_size=kernel)  # [B, 1, 16, 16]
                active_mask = (dil_16 > 0.5).float().view(B, N)  # [B, 256]
            else:
                active_mask = torch.ones(B, N, device=patch_tokens.device, dtype=patch_tokens.dtype)

            if core_mask is not None:
                # Core mask available: pool over pure anomaly patches only
                if mask is None:
                    grid_size = int(N ** 0.5)  # 16
                    kernel = core_mask.shape[-1] // grid_size
                core_16 = F.max_pool2d(core_mask.float(), kernel_size=kernel)  # [B, 1, 16, 16]
                is_core = (core_16 > 0.5).float().view(B, N)  # [B, 256]
                core_3d = is_core.unsqueeze(-1)  # [B, 256, 1]
                core_sum = (patch_tokens * core_3d).sum(dim=1, keepdim=True)  # [B, 1, 1280]
                core_count = is_core.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
                global_anomaly_token = core_sum / core_count + self.masked_self_attn.emb_global  # [B, 1, 1280]

                # Role embeddings: core=anomaly, band=normal context
                is_band = (active_mask - is_core).clamp(min=0)  # [B, 256]
                role = (is_core.unsqueeze(-1) * self.masked_self_attn.emb_anomaly
                        + is_band.unsqueeze(-1) * self.masked_self_attn.emb_normal)
                patch_tokens = patch_tokens + role
            else:
                # No core mask: fallback to pooling over active (dilated) patches.
                # NOTE: this includes band tokens — callers should provide core_mask
                # for proper anomaly-only pooling. All training scripts do.
                active_3d = active_mask.unsqueeze(-1)  # [B, 256, 1]
                token_sum = (patch_tokens * active_3d).sum(dim=1, keepdim=True)  # [B, 1, 1280]
                count = active_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)  # [B, 1, 1]
                global_anomaly_token = token_sum / count + self.masked_self_attn.emb_global  # [B, 1, 1280]

            visual_mode = self.config.visual_mode

            if visual_mode in (0, 1):
                # Modes 0-1: all 256 tokens → masked self-attention (no padding)
                aware_tokens = self.masked_self_attn(patch_tokens, mask=mask)  # [B, 256, 1280]

                # Concat global summary + aware tokens → Resampler
                resampler_input = torch.cat([global_anomaly_token, aware_tokens], dim=1)  # [B, 257, 1280]

            elif visual_mode in (2, 3):
                # Modes 2-3: extract only anomaly patch tokens, pad to batch max
                # mask_grid = active_mask (already computed above from dilated mask)
                mask_grid = active_mask

                # Count anomaly tokens per sample
                counts = mask_grid.sum(dim=1).long()  # [B]
                max_count = max(counts.max().item(), 1)  # at least 1

                # Always pad to 256 to eliminate CUDA allocator fragmentation
                # (variable shapes cause non-reusable memory blocks at high VRAM usage)
                max_count = 256

                # Gather anomaly tokens with padding
                anomaly_tokens = torch.zeros(B, max_count, D, device=patch_tokens.device, dtype=patch_tokens.dtype)
                padding_mask = torch.zeros(B, max_count, device=patch_tokens.device, dtype=patch_tokens.dtype)

                for b in range(B):
                    indices = mask_grid[b].nonzero(as_tuple=True)[0]
                    n = len(indices)
                    if n > 0:
                        anomaly_tokens[b, :n] = patch_tokens[b, indices]
                        padding_mask[b, :n] = 1.0
                    else:
                        # No anomaly tokens — fallback: use mean-pooled token
                        anomaly_tokens[b, 0] = patch_tokens[b].mean(dim=0)
                        padding_mask[b, 0] = 1.0

                # Cast to float32 to match masked_self_attn weights
                # (patch_tokens are fp16 under AMP from CLIP encoder)
                anomaly_tokens = anomaly_tokens.float()
                padding_mask = padding_mask.float()

                # Self-attention over anomaly tokens only
                aware_tokens = self.masked_self_attn(
                    anomaly_tokens, padding_mask=padding_mask,
                )  # [B, max_count, 1280]

                # Concat global summary + anomaly tokens → Resampler
                resampler_input = torch.cat([global_anomaly_token, aware_tokens], dim=1)  # [B, max_count+1, 1280]

                # Build resampler mask: global token always valid + padding_mask
                global_valid = torch.ones(B, 1, device=padding_mask.device, dtype=padding_mask.dtype)
                resampler_mask = torch.cat([global_valid, padding_mask], dim=1)  # [B, 1+max_count]

            else:
                raise ValueError(f"Unknown visual_mode: {visual_mode}")

            # Pass padding mask to Resampler for modes 2-3 (modes 0-1: no padding)
            resampler_kwargs = {}
            if visual_mode in (2, 3):
                resampler_kwargs["key_padding_mask"] = resampler_mask
            image_embeds = self.image_projection(resampler_input, **resampler_kwargs)  # [B, K, 768]
        else:
            # Standard: CLS projection → linear + norm (no masked self-attn)
            image_embeds = self.image_projection(outputs.image_embeds)  # [B, K, 768]

        return image_embeds

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get parameters that should be trained (masked self-attn + image projection + attn processors)."""
        params = []
        if hasattr(self, 'masked_self_attn'):
            params.extend(self.masked_self_attn.parameters())
        params.extend(self.image_projection.parameters())

        for processor in self.attn_processors.values():
            params.extend(processor.parameters())

        return params

    def freeze_image_encoder(self) -> None:
        """Freeze the CLIP image encoder."""
        if self.image_encoder is not None:
            self.image_encoder.requires_grad_(False)

    def save_ip_adapter(self, save_path: Path) -> None:
        """Save IP-Adapter weights."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        state_dict = {}

        # Save masked self-attention (Plus only)
        if hasattr(self, 'masked_self_attn'):
            for k, v in self.masked_self_attn.state_dict().items():
                state_dict[f"masked_self_attn.{k}"] = v

        # Save image projection
        for k, v in self.image_projection.state_dict().items():
            state_dict[f"image_proj.{k}"] = v

        # Save attention processors
        for name, processor in self.attn_processors.items():
            for k, v in processor.state_dict().items():
                state_dict[f"ip_adapter.{name}.{k}"] = v

        torch.save(state_dict, save_path / "ip_adapter.pt")

        # Save config
        import json
        with open(save_path / "config.json", "w") as f:
            json.dump({
                "adapter_type": self.config.adapter_type,
                "num_tokens": self.config.num_tokens,
                "scale": self.config.scale,
                "sd_version": self.sd_version.value,
                "anomaly_residual": self.config.anomaly_residual,
                "mask_visual": self.config.mask_visual,
                "visual_mode": self.config.visual_mode,
                "learnable_gates": self.config.learnable_gates,
                "force_gates": self.config.force_gates,
            }, f, indent=2)

    def load_finetuned(self, load_path: Path) -> None:
        """Load finetuned IP-Adapter weights."""
        load_path = Path(load_path)

        state_dict = torch.load(load_path / "ip_adapter.pt", map_location="cpu")

        # Load masked self-attention
        self_attn_state = {
            k.replace("masked_self_attn.", ""): v
            for k, v in state_dict.items()
            if k.startswith("masked_self_attn.")
        }
        if self_attn_state:
            self.masked_self_attn.load_state_dict(self_attn_state, strict=False)

        # Load image projection
        proj_state = {
            k.replace("image_proj.", ""): v
            for k, v in state_dict.items()
            if k.startswith("image_proj.")
        }
        self.image_projection.load_state_dict(proj_state)

        # Load attention processors
        for name, processor in self.attn_processors.items():
            proc_state = {
                k.replace(f"ip_adapter.{name}.", ""): v
                for k, v in state_dict.items()
                if k.startswith(f"ip_adapter.{name}.")
            }
            if proc_state:
                processor.load_state_dict(proc_state)


def create_ip_adapter(
    pipeline: BaseSDPipeline,
    adapter_type: str = "plus",
    num_tokens: int = 4,
    scale: float = 1.0,
    load_pretrained: bool = True,
    anomaly_residual: bool = False,
    mask_visual: bool = True,
    visual_mode: int = 0,
    learnable_gates: bool = True,
    force_gates: bool = False,
    sa_num_layers: int = 1,
    sa_num_heads: int = 12,
) -> IPAdapter:
    """
    Factory function to create IP-Adapter for a pipeline.

    Args:
        pipeline: The SD pipeline to attach IP-Adapter to
        adapter_type: "standard" (ViT-L/14, 768-dim) or "plus" (ViT-H/14, 1280-dim).
                      Both use masked self-attention. Plus uses resampler, Standard uses linear.
        num_tokens: Visual tokens after projection (1, 4, or 16)
        scale: IP-Adapter visual pathway strength
        load_pretrained: Whether to load pretrained weights
        anomaly_residual: Add gated residual for anomaly-masked tokens only in self-attn.
                          Gate starts at 0 (pure attention), learns to blend original identity.
        mask_visual: Mask visual cross-attention to anomaly positions only.
                     Text pathway stays unmasked. Default True (masked).
        visual_mode: Token processing mode (0-3). See MaskedAnomalySelfAttention docstring.
        learnable_gates: If True, use scalar gates (init=0). If False, zero-init output
                         projections instead (per-dimension gradient, no scalar bottleneck).
        sa_num_layers: Number of transformer layers in masked self-attention.
        sa_num_heads: Number of attention heads in masked self-attention.

    Returns:
        Configured IPAdapter instance
    """
    config = IPAdapterConfig(
        adapter_type=adapter_type,
        num_tokens=num_tokens,
        scale=scale,
        anomaly_residual=anomaly_residual,
        mask_visual=mask_visual,
        visual_mode=visual_mode,
        learnable_gates=learnable_gates,
        force_gates=force_gates,
        sa_num_layers=sa_num_layers,
        sa_num_heads=sa_num_heads,
    )

    ip_adapter = IPAdapter(pipeline, config)

    if load_pretrained:
        # Get default checkpoint for this SD version + adapter type
        key = (pipeline.config.version, adapter_type)
        repo = IP_ADAPTER_CHECKPOINTS.get(key)

        if repo:
            ip_adapter.load_ip_adapter(pretrained_model=repo)
        else:
            # Just initialize without pretrained weights
            ip_adapter.load_ip_adapter()
    else:
        ip_adapter.load_ip_adapter()

    return ip_adapter
