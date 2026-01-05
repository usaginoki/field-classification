"""Dataset loading utilities for TIFF images."""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def load_tiff_image(path: Path) -> Image.Image:
    """
    Load a TIFF image and convert to RGB PIL Image.

    Args:
        path: Path to the TIFF image file.

    Returns:
        RGB PIL Image.
    """
    img_array = tifffile.imread(path)

    # Handle different channel configurations
    if img_array.ndim == 2:
        # Grayscale - convert to RGB
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.ndim == 3:
        if img_array.shape[-1] == 4:
            # RGBA - drop alpha channel
            img_array = img_array[..., :3]
        elif img_array.shape[0] in (3, 4):
            # Channel-first format - transpose to channel-last
            img_array = np.transpose(img_array, (1, 2, 0))
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]

    # Normalize to uint8 if needed
    if img_array.dtype != np.uint8:
        if img_array.max() > 255:
            # Likely 16-bit, normalize
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    return Image.fromarray(img_array, mode="RGB")


def load_image(path: Path) -> Image.Image:
    """
    Load an image from path, handling both TIFF and other formats.

    Args:
        path: Path to the image file.

    Returns:
        RGB PIL Image.
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return load_tiff_image(path)
    return Image.open(path).convert("RGB")


class ImageDataset:
    """Dataset class for loading images from class directories."""

    def __init__(self, root: Path):
        """
        Initialize the dataset.

        Args:
            root: Root directory containing class_0 and class_1 subdirectories.
        """
        self.root = Path(root)
        self.class_dirs = {
            0: self.root / "class_0",
            1: self.root / "class_1",
        }

        # Verify directories exist
        for label, class_dir in self.class_dirs.items():
            if not class_dir.exists():
                raise ValueError(f"Class directory not found: {class_dir}")

    def get_image_paths(self, label: int) -> list[Path]:
        """Get all image paths for a given class label."""
        class_dir = self.class_dirs[label]
        extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        paths = []
        for ext in extensions:
            paths.extend(class_dir.glob(f"*{ext}"))
            paths.extend(class_dir.glob(f"*{ext.upper()}"))
        return sorted(paths)

    def get_all(self) -> tuple[list[Path], np.ndarray]:
        """
        Get all image paths and their labels.

        Returns:
            Tuple of (image_paths, labels) where labels is a numpy array.
        """
        all_paths = []
        all_labels = []

        for label in [0, 1]:
            paths = self.get_image_paths(label)
            all_paths.extend(paths)
            all_labels.extend([label] * len(paths))

        return all_paths, np.array(all_labels)

    def __len__(self) -> int:
        """Return total number of images."""
        return sum(len(self.get_image_paths(label)) for label in [0, 1])

    def get_stats(self) -> dict:
        """Get dataset statistics."""
        stats = {}
        for label in [0, 1]:
            paths = self.get_image_paths(label)
            stats[f"class_{label}"] = len(paths)
        stats["total"] = sum(stats.values())
        return stats
