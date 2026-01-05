"""Feature extraction using Qwen3VL visual encoder with HDF5 caching."""

import hashlib
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .dataset import load_image
from .logging_config import get_logger

logger = get_logger(__name__)


class Qwen3VLFeatureExtractor:
    """Extract visual features from images using Qwen3-VL vision encoder."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype: str = "float16",
        pooling: str = "mean",
    ):
        """
        Initialize the feature extractor.

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

        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        load_start = time.time()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if device == "cuda" else None,
            attn_implementation="sdpa",
        )
        self.model.eval()
        logger.info(f"Model loaded in {time.time() - load_start:.2f}s")

        # Get token IDs for identifying visual tokens
        self.image_token_id = self.model.config.image_token_id

        # Feature dimension from config
        self.feature_dim = self.model.config.text_config.hidden_size
        logger.info(f"Feature dimension: {self.feature_dim}")

    def extract_single(self, image: Image.Image) -> np.ndarray:
        """
        Extract features from a single image.

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

        # Remove token_type_ids if present (not used by Qwen3VL)
        inputs.pop("token_type_ids", None)

        # Move to device
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)

        # Create mask for visual tokens
        input_ids = inputs["input_ids"]
        visual_mask = input_ids == self.image_token_id

        # Pool visual features
        features = self._pool_features(hidden_states, visual_mask)

        return features.cpu().float().numpy().squeeze()

    def _pool_features(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply pooling strategy to hidden states.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim).
            mask: Boolean mask of shape (batch, seq_len) for visual tokens.

        Returns:
            Pooled features of shape (batch, hidden_dim).
        """
        if self.pooling == "mean":
            # Mean pooling over visual tokens
            mask_float = mask.unsqueeze(-1).float()
            masked = hidden_states * mask_float
            return masked.sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            # Max pooling over visual tokens
            masked = hidden_states.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return masked.max(dim=1).values
        elif self.pooling == "cls":
            # Use first token (CLS-like)
            return hidden_states[:, 0, :]
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
        for i, path in enumerate(tqdm(image_paths, desc="Extracting features")):
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


class FeatureCache:
    """HDF5-based feature caching with metadata."""

    def __init__(self, cache_dir: Path | str):
        """
        Initialize the feature cache.

        Args:
            cache_dir: Directory to store cached features.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_key(
        self, model_name: str, pooling: str, dataset_name: str
    ) -> str:
        """
        Generate unique cache key based on model and dataset configuration.

        Args:
            model_name: Name of the model used for extraction.
            pooling: Pooling strategy used.
            dataset_name: Name of the dataset (e.g., "sample" or "full").

        Returns:
            Cache key string.
        """
        key_str = f"{model_name}_{pooling}_{dataset_name}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"features_{cache_key}.h5"

    def exists(self, cache_key: str) -> bool:
        """Check if cached features exist for the given key."""
        return self.get_cache_path(cache_key).exists()

    def save(
        self,
        cache_key: str,
        features: np.ndarray,
        labels: np.ndarray,
        filenames: list[str],
        metadata: dict,
    ) -> None:
        """
        Save features with metadata to HDF5.

        Args:
            cache_key: Unique cache identifier.
            features: Feature matrix of shape (n_samples, feature_dim).
            labels: Label array of shape (n_samples,).
            filenames: List of source filenames.
            metadata: Dictionary of metadata to store.
        """
        path = self.get_cache_path(cache_key)
        logger.info(f"Saving features to cache: {path}")
        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=features, compression="gzip")
            f.create_dataset("labels", data=labels)

            # Store filenames as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("filenames", data=filenames, dtype=dt)

            # Metadata
            for key, value in metadata.items():
                f.attrs[key] = value
            f.attrs["created"] = datetime.now().isoformat()
        logger.info(f"Cache saved: {features.shape[0]} samples, {features.shape[1]} features")

    def load(
        self, cache_key: str
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
        """
        Load cached features.

        Args:
            cache_key: Unique cache identifier.

        Returns:
            Tuple of (features, labels, filenames, metadata).
        """
        path = self.get_cache_path(cache_key)
        logger.info(f"Loading features from cache: {path}")
        with h5py.File(path, "r") as f:
            features = f["features"][:]
            labels = f["labels"][:]
            filenames = [
                fn.decode() if isinstance(fn, bytes) else fn
                for fn in f["filenames"][:]
            ]
            metadata = dict(f.attrs)
        logger.info(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
        return features, labels, filenames, metadata
