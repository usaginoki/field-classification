"""Direct vision encoder feature extraction (bypasses LLM)."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .dataset import load_image
from .logging_config import get_logger

logger = get_logger(__name__)


class VisionEncoderFeatureExtractor:
    """Extract features directly from Qwen3-VL vision encoder (bypasses LLM)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        pooling: str = "mean",
    ):
        """
        Initialize the vision encoder feature extractor.

        This extractor uses only the vision encoder component, bypassing the LLM.
        Results in faster extraction but features are not contextualized by the LLM.

        Args:
            model_name: HuggingFace model name or path.
            device: Device to run the model on (cuda or cpu).
            dtype: Data type for model weights (float16 or bfloat16).
            pooling: Pooling strategy (mean, max, or cls).
        """
        self.model_name = model_name
        self.device = device
        self.pooling = pooling

        # Parse dtype
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Load full model first, then extract vision encoder
        logger.info(f"Loading vision encoder from: {model_name}")
        load_start = time.time()
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if device == "cuda" else None,
        )
        logger.info(f"Model loaded in {time.time() - load_start:.2f}s")

        # Extract vision encoder and move to device
        self.visual = full_model.visual
        self.visual.eval()

        # Keep processor for image preprocessing
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Feature dimension from vision config
        self.feature_dim = full_model.config.vision_config.out_hidden_size
        logger.info(f"Vision encoder output dimension: {self.feature_dim}")

        # Free LLM memory (keep only visual encoder)
        del full_model.model  # Delete LLM
        del full_model.lm_head  # Delete LM head
        torch.cuda.empty_cache()
        logger.info("Freed LLM memory, keeping only vision encoder")

    def extract_single(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from a single image using vision encoder directly.

        Args:
            image: PIL Image in RGB format.

        Returns:
            Feature vector as numpy array of shape (feature_dim,).
        """
        # Prepare input using processor with chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # Get pixel values and grid info
        pixel_values = inputs["pixel_values"].to(self.visual.device, self.dtype)
        image_grid_thw = inputs["image_grid_thw"]

        # Forward pass through vision encoder only
        with torch.no_grad():
            # visual() returns (visual_features, window_indices)
            visual_output = self.visual(pixel_values, grid_thw=image_grid_thw)
            visual_features = visual_output[0]  # (num_tokens, hidden_dim)

        # Pool features
        features = self._pool_features(visual_features)

        return features.cpu().float().numpy()

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling strategy to visual features.

        Args:
            features: Tensor of shape (num_tokens, hidden_dim).

        Returns:
            Pooled features of shape (hidden_dim,).
        """
        if self.pooling == "mean":
            return features.mean(dim=0)
        elif self.pooling == "max":
            return features.max(dim=0).values
        elif self.pooling == "cls":
            # Use first token
            return features[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    def extract_batch(
        self, image_paths: list[Path], batch_size: int = 1
    ) -> np.ndarray:
        """
        Extract features from multiple images.

        Args:
            image_paths: List of paths to images.
            batch_size: Batch size (1 recommended due to varying image sizes).

        Returns:
            Feature matrix of shape (n_samples, feature_dim).
        """
        logger.info(f"Starting feature extraction for {len(image_paths)} images")
        start_time = time.time()
        features = []
        for i, path in enumerate(tqdm(image_paths, desc="Extracting features (vision encoder)")):
            image = load_image(path)
            feat = self.extract_single(image)
            features.append(feat)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(image_paths) - i - 1) / rate
                logger.debug(f"Processed {i + 1}/{len(image_paths)} images, {rate:.2f} img/s, ~{remaining:.0f}s remaining")

        elapsed = time.time() - start_time
        logger.info(f"Feature extraction completed in {elapsed:.2f}s ({len(image_paths) / elapsed:.2f} img/s)")
        return np.stack(features)
