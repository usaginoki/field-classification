"""RemoteCLIP feature extraction using open_clip."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .dataset import load_image
from .logging_config import get_logger

logger = get_logger(__name__)

# RemoteCLIP model configurations
REMOTECLIP_VARIANTS = {
    "RN50": {
        "arch": "RN50",
        "checkpoint": "RemoteCLIP-RN50.pt",
    },
    "ViT-B-32": {
        "arch": "ViT-B-32",
        "checkpoint": "RemoteCLIP-ViT-B-32.pt",
    },
    "ViT-L-14": {
        "arch": "ViT-L-14",
        "checkpoint": "RemoteCLIP-ViT-L-14.pt",
    },
}


class RemoteCLIPFeatureExtractor:
    """Extract image features using RemoteCLIP for remote sensing imagery."""

    def __init__(
        self,
        variant: str = "ViT-L-14",
        device: str = "cuda",
        dtype: str = "float16",
        pooling: str = "none",
    ):
        """
        Initialize the RemoteCLIP feature extractor.

        Args:
            variant: RemoteCLIP variant (RN50, ViT-B-32, or ViT-L-14).
            device: Device to run the model on (cuda or cpu).
            dtype: Data type for model weights (float16, bfloat16, or float32).
            pooling: Ignored - RemoteCLIP outputs are already pooled.
        """
        import open_clip
        from huggingface_hub import hf_hub_download

        self.variant = variant
        self.device = device

        if variant not in REMOTECLIP_VARIANTS:
            raise ValueError(
                f"Unknown RemoteCLIP variant: {variant}. "
                f"Supported: {list(REMOTECLIP_VARIANTS.keys())}"
            )

        # Parse dtype
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        variant_config = REMOTECLIP_VARIANTS[variant]
        arch = variant_config["arch"]
        checkpoint_name = variant_config["checkpoint"]

        # Download checkpoint from HuggingFace Hub
        logger.info(f"Loading RemoteCLIP {variant} model...")
        load_start = time.time()

        checkpoint_path = hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename=checkpoint_name,
        )

        # Create model and load weights
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device).to(self.dtype)
        self.model.eval()

        logger.info(f"RemoteCLIP {variant} loaded in {time.time() - load_start:.2f}s")

        # Get feature dimension from model
        self.feature_dim = self._get_feature_dim()
        logger.info(f"RemoteCLIP feature dimension: {self.feature_dim}")

    def _get_feature_dim(self) -> int:
        """Get feature dimension from model or infer via test forward pass."""
        # Try to get from model attributes
        if hasattr(self.model, "visual"):
            visual = self.model.visual
            if hasattr(visual, "output_dim"):
                return visual.output_dim
            elif hasattr(visual, "proj") and visual.proj is not None:
                return visual.proj.shape[1]

        # Fallback: infer from a test forward pass
        test_image = torch.zeros(1, 3, 224, 224, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            features = self.model.encode_image(test_image)
        return features.shape[-1]

    def extract_single(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Feature vector as numpy array of shape (feature_dim,).
        """
        # Preprocess image using open_clip transforms
        image_tensor = self.preprocess(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device, self.dtype)

        # Extract features
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)

        return image_features.cpu().float().numpy().squeeze()

    def extract_batch(
        self, image_paths: list[Path], batch_size: int = 1
    ) -> np.ndarray:
        """
        Extract features from multiple images.

        Args:
            image_paths: List of paths to images.
            batch_size: Batch size (1 recommended for consistency).

        Returns:
            Feature matrix of shape (n_samples, feature_dim).
        """
        logger.info(
            f"Starting RemoteCLIP ({self.variant}) feature extraction "
            f"for {len(image_paths)} images"
        )
        start_time = time.time()
        features = []

        for i, path in enumerate(tqdm(image_paths, desc=f"Extracting RemoteCLIP features")):
            image = load_image(path)
            feat = self.extract_single(image)
            features.append(feat)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(image_paths) - i - 1) / rate
                logger.debug(
                    f"Processed {i + 1}/{len(image_paths)} images, "
                    f"{rate:.2f} img/s, ~{remaining:.0f}s remaining"
                )

        elapsed = time.time() - start_time
        logger.info(
            f"RemoteCLIP feature extraction completed in {elapsed:.2f}s "
            f"({len(image_paths) / elapsed:.2f} img/s)"
        )
        return np.stack(features)
