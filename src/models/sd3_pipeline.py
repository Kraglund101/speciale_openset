"""
Stable Diffusion 3 pipeline implementation.

SD3 uses:
- MMDiT (Multimodal Diffusion Transformer) instead of UNet
- Three text encoders: CLIP ViT-L, OpenCLIP ViT-bigG, T5-XXL
- Flow matching instead of diffusion
"""
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.models.base import BaseSDPipeline, SDConfig


class SD3Pipeline(BaseSDPipeline):
    """
    Stable Diffusion 3 pipeline for inpainting with textual inversion support.

    Key differences from SD 1.5/SDXL:
    - Uses MMDiT (transformer) instead of UNet
    - Three text encoders: CLIP ViT-L, OpenCLIP ViT-bigG, T5-XXL
    - Rectified flow matching
    - 16 latent channels (vs 4 in SD 1.5/SDXL)

    NOTE: SD3 inpainting may require custom implementation as official
    inpainting pipeline may not be available. This is a placeholder
    that will be implemented when SD3 inpainting becomes available.
    """

    def __init__(self, config: SDConfig, device: str = "cuda", **kwargs):
        super().__init__(config, device)
        self.concept_token_ids: Dict[str, List[int]] = {}
        self.concept_token_ids_2: Dict[str, List[int]] = {}
        self.concept_token_ids_3: Dict[str, List[int]] = {}  # For T5
        self.dtype = kwargs.get("dtype", torch.float16)

        # SD3 has three text encoders
        self.text_encoder_2 = None
        self.text_encoder_3 = None  # T5
        self.tokenizer_2 = None
        self.tokenizer_3 = None

    def load_pipeline(
        self,
        pretrained_model: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Load the SD3 pipeline.

        NOTE: As of implementation, SD3 inpainting may require custom setup.
        This uses the base SD3 pipeline and may need adaptation.
        """
        try:
            from diffusers import StableDiffusion3Pipeline

            model_id = pretrained_model or self.config.pretrained_model

            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                **kwargs
            ).to(self.device)

            # Extract components
            self.text_encoder = self.pipeline.text_encoder
            self.text_encoder_2 = self.pipeline.text_encoder_2
            self.text_encoder_3 = self.pipeline.text_encoder_3  # T5
            self.tokenizer = self.pipeline.tokenizer
            self.tokenizer_2 = self.pipeline.tokenizer_2
            self.tokenizer_3 = self.pipeline.tokenizer_3
            self.vae = self.pipeline.vae
            self.transformer = self.pipeline.transformer  # MMDiT, not UNet
            self.scheduler = self.pipeline.scheduler

        except ImportError:
            raise ImportError(
                "SD3 requires diffusers >= 0.29.0. "
                "Install with: pip install diffusers>=0.29.0"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SD3 pipeline. SD3 inpainting may not be "
                f"officially supported yet. Error: {e}"
            )

    def encode_text(
        self,
        text: Union[str, List[str]],
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """
        Encode text using all three SD3 text encoders.

        SD3 combines:
        - CLIP ViT-L pooled + sequence
        - OpenCLIP ViT-bigG pooled + sequence
        - T5-XXL sequence (no pooled)

        Args:
            text: Text prompt(s) to encode
            enable_grad: If True, enable gradient computation (for training)

        Returns:
            Combined prompt embeddings
        """
        if isinstance(text, str):
            text = [text]

        if enable_grad:
            # This is a simplified version - actual SD3 encoding is more complex
            prompt_embeds, pooled_prompt_embeds = self.pipeline.encode_prompt(
                prompt=text,
                device=self.device,
                do_classifier_free_guidance=False,
            )
        else:
            with torch.no_grad():
                # This is a simplified version - actual SD3 encoding is more complex
                prompt_embeds, pooled_prompt_embeds = self.pipeline.encode_prompt(
                    prompt=text,
                    device=self.device,
                    do_classifier_free_guidance=False,
                )

        return prompt_embeds

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space (16 channels for SD3)."""
        image = image.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to image space."""
        latents = latents / self.vae.config.scaling_factor

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return image

    def get_text_embedding_dim(self) -> int:
        """Get combined text embedding dimension for SD3."""
        # SD3 joint embedding dimension
        return 4096  # Approximate, actual may vary

    def add_concept_token(
        self,
        token_name: str,
        initializer_token: Optional[str] = None,
        num_vectors: int = 1,
    ) -> torch.Tensor:
        """
        Add concept token to all three text encoders.

        Note: T5 uses different tokenization (SentencePiece), so handling
        may differ from CLIP tokenizers.
        """
        placeholder_tokens = [token_name]
        if num_vectors > 1:
            placeholder_tokens = [f"{token_name}_{i}" for i in range(num_vectors)]

        # Add to CLIP tokenizers
        self.tokenizer.add_tokens(placeholder_tokens)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        token_ids_1 = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        self.concept_token_ids[token_name] = token_ids_1

        self.tokenizer_2.add_tokens(placeholder_tokens)
        self.text_encoder_2.resize_token_embeddings(len(self.tokenizer_2))
        token_ids_2 = self.tokenizer_2.convert_tokens_to_ids(placeholder_tokens)
        self.concept_token_ids_2[token_name] = token_ids_2

        # T5 tokenizer handling
        if self.tokenizer_3 is not None:
            self.tokenizer_3.add_tokens(placeholder_tokens)
            self.text_encoder_3.resize_token_embeddings(len(self.tokenizer_3))
            token_ids_3 = self.tokenizer_3.convert_tokens_to_ids(placeholder_tokens)
            self.concept_token_ids_3[token_name] = token_ids_3

        # Initialize embeddings
        embeds_1 = self.text_encoder.get_input_embeddings().weight.data

        if initializer_token:
            init_id = self.tokenizer.encode(initializer_token, add_special_tokens=False)
            if init_id:
                init_embed = embeds_1[init_id[0]].clone()
                for tid in token_ids_1:
                    embeds_1[tid] = init_embed + torch.randn_like(init_embed) * 0.01

        return embeds_1[token_ids_1]

    def get_concept_embedding(self, token_name: str) -> torch.Tensor:
        """Get embeddings from first CLIP encoder."""
        if token_name not in self.concept_token_ids:
            raise ValueError(f"Unknown concept token: {token_name}")

        token_ids = self.concept_token_ids[token_name]
        embeddings = self.text_encoder.get_input_embeddings().weight.data[token_ids]
        return embeddings.clone()

    def set_concept_embedding(self, token_name: str, embedding: torch.Tensor) -> None:
        """Set embeddings in all encoders."""
        if token_name not in self.concept_token_ids:
            raise ValueError(f"Unknown concept token: {token_name}")

        # Set in all CLIP encoders
        for encoder, token_ids_dict in [
            (self.text_encoder, self.concept_token_ids),
            (self.text_encoder_2, self.concept_token_ids_2),
        ]:
            if token_name in token_ids_dict:
                token_ids = token_ids_dict[token_name]
                embeds = encoder.get_input_embeddings().weight.data
                for i, tid in enumerate(token_ids):
                    if i < embedding.shape[0]:
                        if embedding.shape[-1] != embeds.shape[-1]:
                            # Pad or project
                            embeds[tid] = torch.nn.functional.pad(
                                embedding[i],
                                (0, max(0, embeds.shape[-1] - embedding.shape[-1]))
                            )[:embeds.shape[-1]]
                        else:
                            embeds[tid] = embedding[i]

        # T5 encoder may need different handling
        if self.text_encoder_3 is not None and token_name in self.concept_token_ids_3:
            # T5 has different embedding dimension - would need projection
            pass

    def freeze_all(self) -> None:
        """Freeze all model parameters including embeddings."""
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        if self.text_encoder_3:
            self.text_encoder_3.requires_grad_(False)

    def freeze_all_except_embeddings(self) -> None:
        """Freeze all except concept embeddings."""
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        if self.text_encoder_3:
            self.text_encoder_3.requires_grad_(False)

        # Unfreeze embeddings
        self.text_encoder.get_input_embeddings().requires_grad_(True)
        self.text_encoder_2.get_input_embeddings().requires_grad_(True)
        if self.text_encoder_3:
            self.text_encoder_3.get_input_embeddings().requires_grad_(True)

    def __call__(
        self,
        prompt: str,
        image: torch.Tensor,
        mask: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Run SD3 inpainting generation.

        NOTE: This is a placeholder. SD3 inpainting may need custom implementation.
        """
        raise NotImplementedError(
            "SD3 inpainting is not yet officially supported. "
            "Use SD 1.5 or SDXL for now, or implement custom inpainting logic."
        )
