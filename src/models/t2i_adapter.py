"""
T2I-Adapter for injecting anomaly mask spatial conditioning into UNet.

Input: 2 channels @ 64×64 (latent resolution)
  - Channel 0: Binary anomaly mask (maxpooled to 64×64)
  - Channel 1: Dilated ring around anomaly (maxpooled to 64×64)

Architecture:
  Block 0: Conv1×1(2→320) + GN+SiLU + Conv3×3(320→320) + GN+SiLU + Conv3×3(320→320, zero-init)
  Block 1: Conv3×3(320→640, s=2) + GN+SiLU + Conv3×3(640→640, zero-init)
  Block 2: Conv3×3(640→1280, s=2) + GN+SiLU + Conv3×3(1280→1280, zero-init)
  Block 3: Conv3×3(1280→1280, s=2) + GN+SiLU + Conv3×3(1280→1280, zero-init)

  Channels match down_block OUTPUTS: [320, 640, 1280, 1280] for SD 1.5.

Injection modes (controlled via `injection_mode` flag):

  "cascade": Features added to the block output BEFORE it continues to the
      next block. The modified output flows into the next encoder block AND
      gets saved as a skip connection. Both encoder and decoder are biased.

      block[0] → output + feat[0] → saved as skip → also flows to block[1]
      block[1] → output + feat[1] → saved as skip → also flows to block[2]
      ...

  "skip_only": Features added to the saved skip connections AFTER the entire
      encoder finishes. The flowing sample between blocks is never modified.
      Encoder runs clean, only decoder sees the adapter signal.

      block[0] → output → flows clean to block[1]
      block[1] → output → flows clean to block[2]
      ...
      then: skip[0] += feat[0], skip[1] += feat[1], ...
      decoder receives modified skips

Pair injection modes (controlled via `pair_injection` flag):

  "last" (default): Feature injected once per CrossAttnDownBlock2D, after the
      last ResBlock-Transformer pair (diffusers' native mechanism).

      ResBlock[0] → Transformer[0] → ResBlock[1] → Transformer[1] → (+feat) → Downsample

  "all": Feature injected at every point within each CrossAttnDownBlock2D:
      before the first pair (via pre-hook), after each intermediate pair
      (via post-hooks), and after the last pair (diffusers' native mechanism).

      (+feat) → ResBlock[0] → Transformer[0] → (+feat) → ResBlock[1] → Transformer[1] → (+feat) → Downsample

      Same feature tensor reused at each injection point. The learned scale
      compensates for the higher effective injection count (starts at zero
      due to zero-init convs, then converges to ~1/N of the "last" value).

Each feature is masked to anomaly regions and scaled by a learned parameter.
Zero-init on last conv per block → adapter contributes nothing at step 0.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBlock(nn.Module):
    """Block 0: per-pixel embedding (1×1) then spatial mixing (3×3s)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.embed(x)))
        x = self.act2(self.norm2(self.conv1(x)))
        x = self.conv_out(x)
        return x


class DownsampleBlock(nn.Module):
    """Blocks 1-3: stride-2 downsample + channel expansion, then zero-init conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.act1 = nn.SiLU()

        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.conv_out(x)
        return x


class T2IAdapter(nn.Module):
    """Lightweight spatial adapter injecting mask conditioning into UNet.

    Args:
        in_channels: Input channels (default: 2 for mask + dilated ring).
        block_out_channels: Channel dims matching UNet down_block outputs.
            Default: [320, 640, 1280, 1280] for SD 1.5.
        injection_mode: Where to inject features into UNet.
            "cascade"   — encoder + decoder (features flow through encoder)
            "skip_only" — decoder only (encoder untouched)
        pair_injection: When within each CrossAttnDownBlock2D to inject.
            "last" — after last ResBlock-Transformer pair only (default)
            "all"  — before first pair + after every pair (denser injection)
    """

    def __init__(
        self,
        in_channels: int = 2,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        injection_mode: str = "cascade",
        pair_injection: str = "last",
        decoder_inject: bool = False,
    ):
        super().__init__()
        assert injection_mode in ("cascade", "skip_only")
        assert pair_injection in ("last", "all")
        self.block_out_channels = block_out_channels
        self.injection_mode = injection_mode
        self.pair_injection = pair_injection
        self.decoder_inject = decoder_inject
        self._hook_features: List[torch.Tensor] = None  # set via set_hook_features()

        # Block 0: embedding + spatial mixing
        self.block0 = EmbeddingBlock(in_channels, block_out_channels[0])

        # Blocks 1-3: stride-2 downsample
        self.down_blocks = nn.ModuleList()
        for i in range(1, len(block_out_channels)):
            self.down_blocks.append(
                DownsampleBlock(block_out_channels[i - 1], block_out_channels[i])
            )

        # Learned scale per block, init to 1.0
        self.scales = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in block_out_channels
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Process adapter input and produce masked, scaled features.

        Args:
            x: Adapter input [B, in_channels, 64, 64].
            mask: Binary anomaly mask [B, 1, 64, 64] at latent resolution.

        Returns:
            List of 4 feature tensors (masked + scaled), matching down_block
            output dims: [B,320,64,64], [B,640,32,32], [B,1280,16,16], [B,1280,8,8]
        """
        # Precompute masks at each resolution via maxpool
        masks = [mask]  # 64×64
        m = mask
        for _ in range(len(self.block_out_channels) - 1):
            m = F.max_pool2d(m, kernel_size=2)
            masks.append(m)

        features = []

        # Block 0 — internal flow is unmasked
        h = self.block0(x)
        features.append(h * masks[0] * self.scales[0])

        # Blocks 1-3 — h flows unmasked, outputs get masked
        for i, block in enumerate(self.down_blocks):
            h = block(h)
            features.append(h * masks[i + 1] * self.scales[i + 1])

        return features

    def prepare_unet_kwargs(
        self,
        features: List[torch.Tensor],
    ) -> dict:
        """Build the kwargs to pass to UNet forward based on injection_mode.

        Args:
            features: Output of self.forward().

        Returns:
            Dict of kwargs to unpack into unet(...).

        Usage:
            features = adapter(x, mask)
            kwargs = adapter.prepare_unet_kwargs(features)
            noise_pred = unet(sample, timestep, encoder_hidden_states, **kwargs)
        """
        if self.injection_mode == "cascade":
            # Added inside each block → flows to next block + saved as skip
            return {"down_intrablock_additional_residuals": list(features)}
        else:
            # Added to saved skips after encoder finishes → decoder only
            return {"down_block_additional_residuals": list(features)}

    def set_hook_features(self, features: List[torch.Tensor]) -> None:
        """Activate dense injection hooks with the given features.

        Must be called before each UNet forward pass when pair_injection='all'.
        Call clear_hook_features() after the UNet call.
        """
        self._hook_features = features

    def clear_hook_features(self) -> None:
        """Deactivate dense injection hooks (hooks become no-ops)."""
        self._hook_features = None

    def register_dense_hooks(self, unet) -> None:
        """Register forward hooks for pair_injection='all' mode.

        Injects T2I feature at every pair endpoint within each
        CrossAttnDownBlock2D:
          - After pairs 0..N-2: post-hook on attentions[i] (non-last pairs)
          - After pair N-1: handled by diffusers' native mechanism
            (down_intrablock_additional_residuals, no hook needed)

        Total injections per block = N (one per pair). Pre-block injection is
        not possible because down-block input channels differ from output
        channels at block boundaries (e.g. 320→640 at block 1).

        Hooks are permanently registered but only active when _hook_features
        is set (i.e. between set_hook_features() and clear_hook_features()).
        Call once after model and UNet are fully initialised.

        Args:
            unet: The SD 1.5 UNet2DConditionModel instance.
        """
        from diffusers.models.unets.unet_2d_blocks import CrossAttnDownBlock2D

        block_idx = 0
        for down_block in unet.down_blocks:
            if not isinstance(down_block, CrossAttnDownBlock2D):
                continue

            bi = block_idx

            # Post-hook on each non-last Transformer: inject after pairs 0..N-2
            # (final pair handled by diffusers' native mechanism)
            for attn in down_block.attentions[:-1]:
                def _make_post_hook(b: int):
                    def hook(module, input, output):
                        if self._hook_features is None:
                            return output
                        feat = self._hook_features[b]
                        if isinstance(output, tuple):
                            return (output[0] + feat,) + output[1:]
                        return output + feat
                    return hook
                attn.register_forward_hook(_make_post_hook(bi))

            block_idx += 1

    def register_decoder_hooks(self, unet) -> None:
        """Register forward hooks to inject T2I features into decoder up-blocks.

        Diffusers UNet has no native `up_block_additional_residuals` argument, so
        decoder injection always requires hooks. Feature channel-dim mapping
        (symmetric to encoder): features[3-i] is injected at unet.up_blocks[i], so
        an up-block operating at resolution R receives the encoder feature at the
        matching R. Channel counts align for SD 1.5:
          up_blocks[0] (UpBlock2D,           1280 ch, 8×8)   ← features[3]
          up_blocks[1] (CrossAttnUpBlock2D,  1280 ch, 16×16) ← features[2]
          up_blocks[2] (CrossAttnUpBlock2D,   640 ch, 32×32) ← features[1]
          up_blocks[3] (CrossAttnUpBlock2D,   320 ch, 64×64) ← features[0]

        `pair_injection="last"` → post-hook on the last pair only.
        `pair_injection="all"`  → post-hook on every pair inside each up-block.

        For CrossAttnUpBlock2D the "pair" endpoint is the attention output; for
        UpBlock2D it is the resnet output (no attention module in that block).
        """
        from diffusers.models.unets.unet_2d_blocks import (
            CrossAttnUpBlock2D,
            UpBlock2D,
        )

        n_up = len(unet.up_blocks)
        for ui, up_block in enumerate(unet.up_blocks):
            feat_idx = n_up - 1 - ui  # features[3-i] for SD 1.5

            if isinstance(up_block, CrossAttnUpBlock2D):
                endpoints = list(up_block.attentions)
            elif isinstance(up_block, UpBlock2D):
                endpoints = list(up_block.resnets)
            else:
                continue  # unknown block type — skip

            if self.pair_injection == "last":
                endpoints = endpoints[-1:]
            # else "all" — keep every pair endpoint

            for ep in endpoints:
                def _make_hook(fi: int):
                    def hook(module, input, output):
                        if self._hook_features is None:
                            return output
                        feat = self._hook_features[fi]
                        if isinstance(output, tuple):
                            return (output[0] + feat,) + output[1:]
                        return output + feat
                    return hook
                ep.register_forward_hook(_make_hook(feat_idx))
