"""CNN model trainer with frozen/unfrozen backbone support."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .dataset import load_image


class ImageClassificationDataset(Dataset):
    """Simple image classification dataset."""

    def __init__(self, image_paths: list, labels: np.ndarray, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Use the same load_image function as CLIP/SigLIP (handles TIFF properly)
        image = load_image(Path(img_path))

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def get_cnn_model(model_name: str, num_classes: int = 2, frozen: bool = True):
    """
    Get a pretrained CNN model with modified classifier.

    Args:
        model_name: One of efficientnet_b1, resnext50, vgg16, resnet50, densenet121
        num_classes: Number of output classes
        frozen: If True, freeze backbone weights

    Returns:
        model, input_size
    """
    model_configs = {
        "efficientnet_b1": {
            "fn": models.efficientnet_b1,
            "weights": models.EfficientNet_B1_Weights.IMAGENET1K_V2,
            "input_size": 240,
            "classifier_attr": "classifier",
            "in_features_idx": -1,
        },
        "resnext50": {
            "fn": models.resnext50_32x4d,
            "weights": models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
            "input_size": 224,
            "classifier_attr": "fc",
            "in_features_idx": None,
        },
        "vgg11": {
            "fn": models.vgg11,
            "weights": models.VGG11_Weights.IMAGENET1K_V1,
            "input_size": 224,
            "classifier_attr": "classifier",
            "in_features_idx": -1,
        },
        "resnet50": {
            "fn": models.resnet50,
            "weights": models.ResNet50_Weights.IMAGENET1K_V2,
            "input_size": 224,
            "classifier_attr": "fc",
            "in_features_idx": None,
        },
        "densenet121": {
            "fn": models.densenet121,
            "weights": models.DenseNet121_Weights.IMAGENET1K_V1,
            "input_size": 224,
            "classifier_attr": "classifier",
            "in_features_idx": None,
        },
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_configs.keys())}")

    cfg = model_configs[model_name]
    model = cfg["fn"](weights=cfg["weights"])

    # Freeze backbone if requested
    if frozen:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    classifier = getattr(model, cfg["classifier_attr"])

    if model_name.startswith("vgg"):
        # VGG classifier: Sequential(..., Linear(4096, 1000))
        # Replace just the last layer
        classifier[-1] = nn.Linear(4096, num_classes)
    elif cfg["in_features_idx"] is not None:
        # Sequential classifier (EfficientNet)
        in_features = classifier[cfg["in_features_idx"]].in_features
        new_classifier = list(classifier.children())[:-1]
        new_classifier.append(nn.Linear(in_features, num_classes))
        setattr(model, cfg["classifier_attr"], nn.Sequential(*new_classifier))
    else:
        # Single FC layer (ResNet, ResNeXt, DenseNet)
        in_features = classifier.in_features
        setattr(model, cfg["classifier_attr"], nn.Linear(in_features, num_classes))

    # Ensure classifier is trainable
    for param in getattr(model, cfg["classifier_attr"]).parameters():
        param.requires_grad = True

    return model, cfg["input_size"]


def get_transforms(input_size: int, is_train: bool = True):
    """Get image transforms for training/validation."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])


class CNNTrainer:
    """Trainer for CNN models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        frozen: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        # For unfrozen models, use lower LR for backbone to preserve pretrained features
        if frozen:
            # Only classifier is trainable, use standard LR
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            # Differential learning rates: lower for backbone, higher for classifier
            backbone_params = []
            classifier_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if "classifier" in name or "fc" in name:
                        classifier_params.append(param)
                    else:
                        backbone_params.append(param)

            self.optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": learning_rate * 0.01},  # 1e-5 for backbone
                {"params": classifier_params, "lr": learning_rate},       # 1e-3 for classifier
            ], weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        metrics = {
            "train_loss": total_loss / len(dataloader),
            "train_accuracy": accuracy_score(all_labels, all_preds),
            "train_f1": f1_score(all_labels, all_preds, zero_division=0),
        }
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on validation/test set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        metrics = {
            "val_loss": total_loss / len(dataloader),
            "val_accuracy": accuracy_score(all_labels, all_preds),
            "val_precision": precision_score(all_labels, all_preds, zero_division=0),
            "val_recall": recall_score(all_labels, all_preds, zero_division=0),
            "val_f1": f1_score(all_labels, all_preds, zero_division=0),
        }
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        wandb_run=None,
        early_stopping_patience: int = 7,
    ) -> dict:
        """
        Full training loop with early stopping.

        Returns:
            Best validation metrics
        """
        best_val_f1 = 0
        best_metrics = None
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Acc: {train_metrics['train_accuracy']:.4f}, "
                  f"F1: {train_metrics['train_f1']:.4f}")

            # Validate
            val_metrics = self.evaluate(val_loader)
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Acc: {val_metrics['val_accuracy']:.4f}, "
                  f"F1: {val_metrics['val_f1']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics["val_f1"])

            # Log to wandb
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    **train_metrics,
                    **val_metrics,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                })

            # Early stopping
            if val_metrics["val_f1"] > best_val_f1:
                best_val_f1 = val_metrics["val_f1"]
                best_metrics = {**train_metrics, **val_metrics, "best_epoch": epoch + 1}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

        return best_metrics
