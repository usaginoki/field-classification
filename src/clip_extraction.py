"""CLIP feature extraction using transformers."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .dataset import load_image
from .logging_config import get_logger

logger = get_logger(__name__)


class CLIPFeatureExtractor:
    """Extract image features using CLIP vision encoder."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        dtype: str = "float16",
        pooling: str = "none",
    ):
        """
        Initialize the CLIP feature extractor.

        Args:
            model_name: HuggingFace CLIP model name.
            device: Device to run the model on (cuda or cpu).
            dtype: Data type for model weights (float16, bfloat16, or float32).
            pooling: Ignored - CLIP outputs are already pooled [CLS] embeddings.
        """
        self.model_name = model_name
        self.device = device

        # Parse dtype
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Load model and processor
        logger.info(f"Loading CLIP model: {model_name}")
        load_start = time.time()

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        ).to(device)
        self.model.eval()

        logger.info(f"CLIP model loaded in {time.time() - load_start:.2f}s")

        # Feature dimension from model config
        self.feature_dim = self.model.config.projection_dim
        logger.info(f"CLIP feature dimension: {self.feature_dim}")

    def extract_single(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from a single image.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Feature vector as numpy array of shape (feature_dim,).
        """
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

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
        logger.info(f"Starting CLIP feature extraction for {len(image_paths)} images")
        start_time = time.time()
        features = []

        for i, path in enumerate(tqdm(image_paths, desc="Extracting CLIP features")):
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
            f"CLIP feature extraction completed in {elapsed:.2f}s "
            f"({len(image_paths) / elapsed:.2f} img/s)"
        )
        return np.stack(features)
