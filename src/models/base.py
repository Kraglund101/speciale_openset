"""
Base classes and model registry for multi-SD support.

Supports SD 1.5, SDXL, and SD 3.0 through a unified interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class SDVersion(Enum):
    """Supported Stable Diffusion versions."""
    SD_1_5 = "sd_1.5"
    SD_XL = "sdxl"
    SD_3 = "sd_3.0"


@dataclass
class SDConfig:
    """Configuration for a specific SD version."""
    version: SDVersion
    pretrained_model: str
    image_size: int
    clip_model: str
    clip_model_secondary: Optional[str] = None  # For SDXL/SD3
    t5_model: Optional[str] = None  # For SD3
    ip_adapter_ckpt: Optional[str] = None
    vae_model: Optional[str] = None

    # Architecture specifics
    num_text_encoders: int = 1
    uses_t5: bool = False
    latent_channels: int = 4


# Default configurations for each SD version
SD_CONFIGS: Dict[SDVersion, SDConfig] = {
    SDVersion.SD_1_5: SDConfig(
        version=SDVersion.SD_1_5,
        pretrained_model="runwayml/stable-diffusion-inpainting",
        image_size=512,
        clip_model="openai/clip-vit-large-patch14",
        num_text_encoders=1,
        latent_channels=4,
    ),
    SDVersion.SD_XL: SDConfig(
        version=SDVersion.SD_XL,
        pretrained_model="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        image_size=1024,
        clip_model="openai/clip-vit-large-patch14",
        clip_model_secondary="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        num_text_encoders=2,
        latent_channels=4,
    ),
    SDVersion.SD_3: SDConfig(
        version=SDVersion.SD_3,
        pretrained_model="stabilityai/stable-diffusion-3-medium",
        image_size=1024,
        clip_model="openai/clip-vit-large-patch14",
        clip_model_secondary="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        t5_model="google/t5-v1_1-xxl",
        num_text_encoders=3,
        uses_t5=True,
        latent_channels=16,
    ),
}


class BaseSDPipeline(ABC):
    """
    Abstract base class for Stable Diffusion pipelines.

    All SD versions implement this interface for unified usage.
    """

    def __init__(self, config: SDConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.pipeline = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.unet = None

    @abstractmethod
    def load_pipeline(self, **kwargs) -> None:
        """Load the diffusion pipeline and components."""
        pass

    @abstractmethod
    def encode_text(
        self,
        text: Union[str, List[str]],
        enable_grad: bool = False,
    ) -> torch.Tensor:
        """
        Encode text prompts to embeddings.

        Args:
            text: Text prompt(s) to encode
            enable_grad: If True, enable gradient computation (for training)

        Returns:
            Text embeddings tensor
        """
        pass

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space."""
        pass

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to image space."""
        pass

    @abstractmethod
    def get_text_embedding_dim(self) -> int:
        """Get dimension of text embeddings."""
        pass

    @abstractmethod
    def add_concept_token(
        self,
        token_name: str,
        initializer_token: Optional[str] = None,
        num_vectors: int = 1,
    ) -> torch.Tensor:
        """
        Add a new concept token to the text encoder.

        Returns the initialized embedding vectors.
        """
        pass

    @abstractmethod
    def get_concept_embedding(self, token_name: str) -> torch.Tensor:
        """Get the embedding for a concept token."""
        pass

    @abstractmethod
    def set_concept_embedding(self, token_name: str, embedding: torch.Tensor) -> None:
        """Set the embedding for a concept token."""
        pass

    @abstractmethod
    def freeze_all(self) -> None:
        """Freeze all model parameters including embeddings."""
        pass

    @abstractmethod
    def freeze_all_except_embeddings(self) -> None:
        """Freeze all model parameters except concept token embeddings."""
        pass


class ConceptTokenManager:
    """
    Manages concept tokens across different anomaly types.

    Includes special NULL token for uncertain/unmatched cases.
    """

    NULL_TOKEN = "<NULL_ANOMALY>"
    CONCEPT_PREFIX = "<ANOMALY_"
    CONCEPT_SUFFIX = ">"

    def __init__(
        self,
        num_vectors_per_concept: int = 4,
        initializer_token: str = "anomaly",
    ):
        self.num_vectors_per_concept = num_vectors_per_concept
        self.initializer_token = initializer_token
        self.concept_tokens: Dict[str, str] = {}  # concept_name -> token_string
        self.concept_embeddings: Dict[str, torch.Tensor] = {}

        # Always include NULL token
        self.concept_tokens["null"] = self.NULL_TOKEN

    def get_token_name(self, concept_name: str) -> str:
        """Get the token string for a concept."""
        if concept_name == "null":
            return self.NULL_TOKEN
        return f"{self.CONCEPT_PREFIX}{concept_name.upper()}{self.CONCEPT_SUFFIX}"

    def register_concept(self, concept_name: str) -> str:
        """Register a new concept and return its token name."""
        token_name = self.get_token_name(concept_name)
        self.concept_tokens[concept_name] = token_name
        return token_name

    def get_all_tokens(self) -> List[str]:
        """Get all registered token strings."""
        return list(self.concept_tokens.values())

    def save_embeddings(self, save_dir: Path) -> None:
        """Save all concept embeddings to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for concept_name, embedding in self.concept_embeddings.items():
            torch.save(
                {
                    "concept_name": concept_name,
                    "token_name": self.concept_tokens[concept_name],
                    "embedding": embedding,
                    "num_vectors": self.num_vectors_per_concept,
                },
                save_dir / f"{concept_name}.pt"
            )

    def load_embeddings(self, load_dir: Path) -> None:
        """Load concept embeddings from disk."""
        load_dir = Path(load_dir)

        for pt_file in load_dir.glob("*.pt"):
            data = torch.load(pt_file, map_location="cpu")
            concept_name = data["concept_name"]
            self.concept_tokens[concept_name] = data["token_name"]
            self.concept_embeddings[concept_name] = data["embedding"]

    def get_embedding(self, concept_name: str) -> Optional[torch.Tensor]:
        """Get embedding for a concept, or NULL embedding if not found."""
        if concept_name in self.concept_embeddings:
            return self.concept_embeddings[concept_name]
        # Return NULL token embedding for unknown concepts
        return self.concept_embeddings.get("null")


def get_sd_config(version: Union[str, SDVersion]) -> SDConfig:
    """Get configuration for an SD version."""
    if isinstance(version, str):
        version = SDVersion(version)
    return SD_CONFIGS[version]


def create_pipeline(
    version: Union[str, SDVersion],
    device: str = "cuda",
    **kwargs
) -> BaseSDPipeline:
    """
    Factory function to create the appropriate SD pipeline.

    Args:
        version: SD version string or enum
        device: Device to load model on
        **kwargs: Additional arguments passed to pipeline

    Returns:
        Initialized pipeline for the specified SD version
    """
    if isinstance(version, str):
        version = SDVersion(version)

    config = get_sd_config(version)

    # Import and instantiate the correct pipeline class
    if version == SDVersion.SD_1_5:
        from src.models.sd15_pipeline import SD15Pipeline
        return SD15Pipeline(config, device, **kwargs)
    elif version == SDVersion.SD_XL:
        from src.models.sdxl_pipeline import SDXLPipeline
        return SDXLPipeline(config, device, **kwargs)
    elif version == SDVersion.SD_3:
        from src.models.sd3_pipeline import SD3Pipeline
        return SD3Pipeline(config, device, **kwargs)
    else:
        raise ValueError(f"Unknown SD version: {version}")
