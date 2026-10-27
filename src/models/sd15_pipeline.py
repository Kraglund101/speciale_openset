"""
Stable Diffusion 1.5 pipeline implementation.
"""
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from src.models.base import BaseSDPipeline, SDConfig


class SD15Pipeline(BaseSDPipeline):
    """
    Stable Diffusion 1.5 pipeline for inpainting with textual inversion support.
    """

    def __init__(self, config: SDConfig, device: str = "cuda", **kwargs):
        super().__init__(config, device)
        self.concept_token_ids: Dict[str, List[int]] = {}
        self.dtype = kwargs.get("dtype", torch.float16)

    def load_pipeline(
        self,
        pretrained_model: Optional[str] = None,
        **kwargs
    ) -> None:
        """Load the SD 1.5 inpainting pipeline."""
        model_id = pretrained_model or self.config.pretrained_model

        # Load full pipeline
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            **kwargs
        ).to(self.device)

        # Extract components for direct access
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler

    def encode_text(
        self,
        text: Union[str, List[str]],
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """
        Encode text prompts to CLIP embeddings.

        Args:
            text: Text prompt(s) to encode
            enable_grad: If True, enable gradient computation (for training)

        Returns:
            Text embeddings tensor
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Encode (with or without gradients)
        # Note: Embeddings may be fp32 for training stability while rest of model is fp16
        # We need to handle this mixed dtype situation carefully
        if enable_grad:
            embeddings = self.text_encoder(tokens)[0]
        else:
            with torch.no_grad():
                # Check if embeddings are fp32 (from training stability fix)
                embed_layer = self.text_encoder.get_input_embeddings()
                if embed_layer.weight.dtype == torch.float32:
                    # Run entire text encoder in fp32 to avoid dtype mismatch
                    # Store original dtypes to restore later
                    original_dtypes = {}
                    for name, param in self.text_encoder.named_parameters():
                        original_dtypes[name] = param.dtype
                        param.data = param.data.float()

                    embeddings = self.text_encoder(tokens)[0]

                    # Restore original dtypes (except embeddings which stay fp32)
                    for name, param in self.text_encoder.named_parameters():
                        if 'embeddings' not in name:
                            param.data = param.data.to(original_dtypes[name])
                else:
                    embeddings = self.text_encoder(tokens)[0]

        return embeddings

    def unet_forward(self, model_input, timesteps, text_emb, cross_attention_kwargs=None, **kwargs):
        """UNet forward used by training (SDXL overrides this to add pooled text + time ids)."""
        return self.unet(model_input, timesteps, encoder_hidden_states=text_emb,
                         cross_attention_kwargs=cross_attention_kwargs, **kwargs)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space."""
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
        """Get CLIP text embedding dimension (768 for SD 1.5)."""
        return self.text_encoder.config.hidden_size

    def add_concept_token(
        self,
        token_name: str,
        initializer_token: Optional[str] = None,
        num_vectors: int = 1,
    ) -> torch.Tensor:
        """
        Add a new concept token to the tokenizer and text encoder.

        Args:
            token_name: The placeholder token (e.g., "<ANOMALY_SCRATCH>")
            initializer_token: Token to initialize embedding from (e.g., "scratch")
            num_vectors: Number of embedding vectors for this concept

        Returns:
            The initialized embedding tensor
        """
        # For multi-vector tokens, create multiple placeholder tokens
        placeholder_tokens = [token_name]
        if num_vectors > 1:
            placeholder_tokens = [f"{token_name}_{i}" for i in range(num_vectors)]

        # Add tokens to tokenizer
        num_added = self.tokenizer.add_tokens(placeholder_tokens)
        if num_added == 0:
            # Tokens already exist
            pass

        # Resize token embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Get token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(placeholder_tokens)
        self.concept_token_ids[token_name] = token_ids

        # Initialize embeddings
        token_embeds = self.text_encoder.get_input_embeddings().weight.data

        if initializer_token:
            # Initialize from existing token
            init_token_id = self.tokenizer.encode(initializer_token, add_special_tokens=False)
            if init_token_id:
                init_embed = token_embeds[init_token_id[0]].clone()
                for tid in token_ids:
                    # Add small noise for diversity if multi-vector
                    noise = torch.randn_like(init_embed) * 0.01
                    token_embeds[tid] = init_embed + noise
        else:
            # Random initialization
            for tid in token_ids:
                token_embeds[tid] = torch.randn_like(token_embeds[0]) * 0.01

        return token_embeds[token_ids]

    def get_concept_embedding(self, token_name: str) -> torch.Tensor:
        """Get the current embedding vectors for a concept token."""
        if token_name not in self.concept_token_ids:
            raise ValueError(f"Unknown concept token: {token_name}")

        token_ids = self.concept_token_ids[token_name]
        embeddings = self.text_encoder.get_input_embeddings().weight.data[token_ids]
        return embeddings.clone()

    def set_concept_embedding(self, token_name: str, embedding: torch.Tensor) -> None:
        """Set the embedding vectors for a concept token."""
        if token_name not in self.concept_token_ids:
            raise ValueError(f"Unknown concept token: {token_name}")

        token_ids = self.concept_token_ids[token_name]
        token_embeds = self.text_encoder.get_input_embeddings().weight.data

        for i, tid in enumerate(token_ids):
            token_embeds[tid] = embedding[i]

    def get_trainable_embedding_params(self) -> List[torch.Tensor]:
        """Get the embedding parameters that should be trained."""
        params = []
        token_embeds = self.text_encoder.get_input_embeddings().weight

        for token_ids in self.concept_token_ids.values():
            for tid in token_ids:
                params.append(token_embeds[tid])

        return params

    def freeze_all(self) -> None:
        """Freeze all model parameters including embeddings."""
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def freeze_all_except_embeddings(self) -> None:
        """Freeze all model parameters except concept token embeddings."""
        # Freeze VAE
        self.vae.requires_grad_(False)

        # Freeze UNet
        self.unet.requires_grad_(False)

        # Freeze text encoder except embeddings
        self.text_encoder.requires_grad_(False)

        # Unfreeze only concept token embeddings
        token_embeds = self.text_encoder.get_input_embeddings()
        token_embeds.requires_grad_(True)

    def prepare_prompt_with_concept(
        self,
        concept_token: str,
        template: str = "a photo of a {concept} defect"
    ) -> str:
        """
        Prepare a prompt string with the concept token.

        Args:
            concept_token: The concept placeholder token
            template: Prompt template with {concept} placeholder

        Returns:
            Formatted prompt string
        """
        # For multi-vector tokens, join them
        if concept_token in self.concept_token_ids:
            token_ids = self.concept_token_ids[concept_token]
            if len(token_ids) > 1:
                tokens = [f"{concept_token}_{i}" for i in range(len(token_ids))]
                concept_str = " ".join(tokens)
            else:
                concept_str = concept_token
        else:
            concept_str = concept_token

        return template.format(concept=concept_str)

    def __call__(
        self,
        prompt: str,
        image: Union[torch.Tensor, Image.Image],
        mask: Union[torch.Tensor, Image.Image],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs
    ) -> Image.Image:
        """
        Run inpainting generation.

        Args:
            prompt: Text prompt (should include concept token)
            image: Input image (tensor [B, 3, H, W] in [-1, 1] or PIL Image)
            mask: Mask (tensor [B, 1, H, W] in [0, 1] or PIL Image), 1 = inpaint region
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale

        Returns:
            Generated PIL image
        """
        from torchvision.transforms.functional import to_pil_image
        from PIL import Image as PILImage

        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            # Denormalize from [-1, 1] to [0, 1]
            if image.dim() == 4:
                image = image[0]
            image_pil = to_pil_image((image + 1) / 2)
        else:
            image_pil = image

        if isinstance(mask, torch.Tensor):
            if mask.dim() == 4:
                mask = mask[0]
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask[0]
            mask_pil = to_pil_image(mask)
        else:
            mask_pil = mask

        result = self.pipeline(
            prompt=prompt,
            image=image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )

        return result.images[0]
