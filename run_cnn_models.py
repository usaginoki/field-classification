#!/usr/bin/env python3
"""Run CNN classification experiments with frozen and unfrozen backbones."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.cnn_trainer import (
    CNNTrainer,
    ImageClassificationDataset,
    get_cnn_model,
    get_transforms,
)
from src.config import load_config
from src.dataset import ImageDataset


# CNN architectures to compare
CNN_MODELS = [
    "efficientnet_b1",
    "resnext50",
    "vgg11",
    "resnet50",
    "densenet121",
]


def run_experiment(
    model_name: str,
    frozen: bool,
    train_paths: list,
    train_labels: np.ndarray,
    val_paths: list,
    val_labels: np.ndarray,
    config: dict,
    fold: int,
    wandb_run,
) -> dict:
    """Run a single fold experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, input_size = get_cnn_model(model_name, num_classes=2, frozen=frozen)

    # Create data loaders
    train_transform = get_transforms(input_size, is_train=True)
    val_transform = get_transforms(input_size, is_train=False)

    train_dataset = ImageClassificationDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageClassificationDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )

    # Train
    trainer = CNNTrainer(
        model, device=device, learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"], frozen=frozen,
    )

    best_val_metrics = trainer.fit(
        train_loader, val_loader, epochs=config["epochs"],
        wandb_run=None, early_stopping_patience=config["early_stopping_patience"],
    )

    # Log fold results to wandb
    if wandb_run:
        wandb_run.log({
            f"fold_{fold}_val_accuracy": best_val_metrics["val_accuracy"],
            f"fold_{fold}_val_f1": best_val_metrics["val_f1"],
        })

    return best_val_metrics


def run_cv_experiment(
    model_name: str,
    frozen: bool,
    image_paths: list,
    labels: np.ndarray,
    config: dict,
    output_dir: Path,
    wandb_project: str,
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """Run 5-fold cross-validation for a model."""
    mode = "frozen" if frozen else "unfrozen"
    run_name = f"{model_name}_{mode}"
    print(f"\n{'='*60}")
    print(f"Running {n_folds}-fold CV: {run_name}")
    print(f"{'='*60}")

    # Initialize wandb
    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "model": model_name,
            "frozen": frozen,
            "n_folds": n_folds,
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
        },
        reinit=True,
    )

    # Get model info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, _ = get_cnn_model(model_name, num_classes=2, frozen=frozen)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    del model

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, labels), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        fold_metrics = run_experiment(
            model_name, frozen, train_paths, train_labels, val_paths, val_labels,
            config, fold, run,
        )
        fold_results.append(fold_metrics)
        print(f"  Fold {fold}: Acc={fold_metrics['val_accuracy']:.4f}, F1={fold_metrics['val_f1']:.4f}")

    # Compute mean and std
    accuracies = [r["val_accuracy"] for r in fold_results]
    f1_scores = [r["val_f1"] for r in fold_results]

    results = {
        "model": model_name,
        "frozen": frozen,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "fold_results": fold_results,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{run_name} Results:")
    print(f"  Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  F1:       {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")

    # Log to wandb
    wandb.log({
        "accuracy_mean": results["accuracy_mean"],
        "accuracy_std": results["accuracy_std"],
        "f1_mean": results["f1_mean"],
        "f1_std": results["f1_std"],
        "trainable_params": trainable_params,
    })

    # Save results
    exp_output_dir = output_dir / run_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    wandb.finish()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CNN classification experiments")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="field-classification-cnn",
                        help="Wandb project name")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cnn"),
                        help="Output directory")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help=f"Models to run (default: all). Options: {CNN_MODELS}")
    parser.add_argument("--frozen-only", action="store_true", help="Only run frozen experiments")
    parser.add_argument("--unfrozen-only", action="store_true", help="Only run unfrozen experiments")
    args = parser.parse_args()

    print("=" * 60)
    print("CNN Classification Experiments")
    print("=" * 60)

    # Load dataset config
    config = load_config(args.config)
    data_root = config.data.get_data_root()

    # Training config
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "early_stopping_patience": 10,
    }

    # Load dataset
    print(f"\nLoading dataset from {data_root}")
    dataset = ImageDataset(data_root)
    image_paths, labels = dataset.get_all()
    print(f"Total samples: {len(labels)} (class_0={sum(labels==0)}, class_1={sum(labels==1)})")

    random_seed = config.training.random_seed

    # Determine which models and modes to run
    models_to_run = args.models if args.models else CNN_MODELS
    modes = []
    if not args.unfrozen_only:
        modes.append(True)  # frozen
    if not args.frozen_only:
        modes.append(False)  # unfrozen

    print(f"\nModels: {models_to_run}")
    print(f"Modes: {'frozen' if True in modes else ''} {'unfrozen' if False in modes else ''}")
    print(f"Cross-validation: 5 folds")
    print(f"Wandb project: {args.wandb_project}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    for model_name in models_to_run:
        for frozen in modes:
            try:
                result = run_cv_experiment(
                    model_name=model_name,
                    frozen=frozen,
                    image_paths=image_paths,
                    labels=labels,
                    config=train_config,
                    output_dir=args.output_dir,
                    wandb_project=args.wandb_project,
                    n_folds=5,
                    random_seed=random_seed,
                )
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in {model_name} ({'frozen' if frozen else 'unfrozen'}): {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model_name,
                    "frozen": frozen,
                    "error": str(e),
                })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (5-fold CV)")
    print("=" * 60)
    print(f"{'Model':<20} {'Mode':<10} {'Accuracy':<20} {'F1':<20}")
    print("-" * 70)
    for r in all_results:
        mode = "frozen" if r.get("frozen") else "unfrozen"
        if "error" in r:
            print(f"{r['model']:<20} {mode:<10} ERROR: {r['error'][:30]}")
        else:
            acc = f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
            f1 = f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}"
            print(f"{r['model']:<20} {mode:<10} {acc:<20} {f1:<20}")

    # Save summary
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
