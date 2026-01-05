"""Qwen3VL Image Classification Pipeline - Main Entry Point."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.classification import get_enabled_classifiers
from src.config import load_config
from src.dataset import ImageDataset
from src.evaluation import CrossValidator, HyperparameterSearcher, format_params
from src.feature_extraction import FeatureCache, Qwen3VLFeatureExtractor
from src.logging_config import setup_logging, get_logger
from src.vision_encoder_extraction import VisionEncoderFeatureExtractor


console = Console()
logger = get_logger("main")


def main():
    """Main entry point for the classification pipeline."""
    parser = argparse.ArgumentParser(
        description="Qwen3VL Image Classification Pipeline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample dataset for testing",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache, re-extract features",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Hold out a test set for unbiased final evaluation",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (optional, logs to console by default)",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_file=args.log_file)
    logger.info("Starting Qwen3VL Image Classification Pipeline")
    pipeline_start = time.time()

    # Load configuration
    config = load_config(args.config)
    config.apply_overrides(use_sample=args.use_sample if args.use_sample else None)

    # Log configuration
    logger.info(f"Configuration loaded: model={config.model.name}, mode={config.model.extraction_mode}")
    logger.debug(f"Pooling: {config.model.pooling}, device: {config.model.device}")
    logger.debug(f"Scoring metric: {config.training.scoring_metric}")
    logger.debug(f"Use sample: {config.data.use_sample}, holdout: {args.holdout}")

    # Display configuration
    console.print("\n[bold blue]Configuration:[/bold blue]")
    console.print(f"  Model: {config.model.name}")
    console.print(f"  Extraction mode: {config.model.extraction_mode}")
    console.print(f"  Pooling: {config.model.pooling}")
    console.print(f"  Device: {config.model.device}")
    console.print(f"  Scoring metric: {config.training.scoring_metric}")
    console.print(f"  Use sample: {config.data.use_sample}")
    console.print(f"  Holdout test set: {args.holdout}" + (f" ({config.training.holdout_fraction:.0%})" if args.holdout else ""))

    # Setup paths
    data_root = config.data.get_data_root()
    dataset_name = "sample" if config.data.use_sample else "full"

    # Initialize cache (include extraction_mode in cache key)
    cache = FeatureCache(config.data.cache_dir)
    cache_key = cache.get_cache_key(
        config.model.name,
        config.model.pooling,
        f"{dataset_name}_{config.model.extraction_mode}",
    )

    # Load or extract features
    if cache.exists(cache_key) and not args.no_cache:
        console.print(f"\n[green]Loading cached features ({cache_key})...[/green]")
        features, labels, filenames, metadata = cache.load(cache_key)
        console.print(f"  Loaded {len(labels)} samples with {features.shape[1]} features")
    else:
        mode_desc = "vision encoder" if config.model.extraction_mode == "vision_encoder" else "full model"
        console.print(f"\n[yellow]Extracting features with {config.model.name} ({mode_desc})...[/yellow]")

        # Load dataset
        dataset = ImageDataset(data_root)
        stats = dataset.get_stats()
        console.print(f"  Dataset: {stats}")

        image_paths, labels = dataset.get_all()

        # Extract features using selected mode
        if config.model.extraction_mode == "vision_encoder":
            extractor = VisionEncoderFeatureExtractor(
                model_name=config.model.name,
                device=config.model.device,
                dtype=config.model.dtype,
                pooling=config.model.pooling,
            )
        else:
            extractor = Qwen3VLFeatureExtractor(
                model_name=config.model.name,
                device=config.model.device,
                dtype=config.model.dtype,
                pooling=config.model.pooling,
            )
        features = extractor.extract_batch(image_paths)

        # Cache features
        metadata = {
            "model": config.model.name,
            "extraction_mode": config.model.extraction_mode,
            "pooling": config.model.pooling,
            "dataset": dataset_name,
            "feature_dim": features.shape[1],
            "n_samples": len(labels),
        }
        filenames = [str(p) for p in image_paths]
        cache.save(cache_key, features, labels, filenames, metadata)
        console.print(f"[green]Features cached to {cache_key}[/green]")

    # Display feature info
    console.print(f"\n[bold]Features shape:[/bold] {features.shape}")
    console.print(
        f"[bold]Labels distribution:[/bold] class_0={sum(labels == 0)}, class_1={sum(labels == 1)}"
    )

    # Split into train/test if holdout is enabled
    if args.holdout:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=config.training.holdout_fraction,
            random_state=config.training.random_seed,
            stratify=labels,
        )
        console.print(
            f"\n[bold yellow]Holdout split:[/bold yellow] {len(y_train)} train, {len(y_test)} test"
        )
    else:
        X_train, y_train = features, labels
        X_test, y_test = None, None

    # Get enabled classifiers
    classifiers = get_enabled_classifiers(
        config.classifiers.mlp,
        config.classifiers.svm,
        config.classifiers.xgboost,
        config.classifiers.logistic_regression,
    )

    if not classifiers:
        console.print("[red]No classifiers enabled in configuration![/red]")
        return

    # Setup output directory
    output_root = Path(config.data.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Results table
    results_table = Table(title="Classification Results")
    results_table.add_column("Classifier", style="cyan")
    results_table.add_column("Best Params", style="white")
    results_table.add_column("HP Search Score", style="yellow")
    results_table.add_column(
        f"CV Accuracy ({config.training.final_cv_folds}-fold)", style="green"
    )
    results_table.add_column(
        f"CV F1 ({config.training.final_cv_folds}-fold)", style="green"
    )
    if args.holdout:
        results_table.add_column("Test Accuracy", style="magenta")
        results_table.add_column("Test F1", style="magenta")

    # Train each classifier
    for clf_idx, clf_wrapper in enumerate(classifiers, 1):
        clf_name = clf_wrapper.get_name()
        logger.info(f"Training classifier {clf_idx}/{len(classifiers)}: {clf_name}")
        clf_start = time.time()
        console.print(f"\n[bold blue]Training {clf_name}...[/bold blue]")

        # Create output directory for this classifier
        clf_output_dir = output_root / clf_name
        clf_output_dir.mkdir(parents=True, exist_ok=True)

        # Hyperparameter search (on train set only)
        searcher = HyperparameterSearcher(
            clf_wrapper,
            cv_folds=config.training.hp_search_cv_folds,
            scoring=config.training.scoring_metric,
            random_seed=config.training.random_seed,
        )
        search_results = searcher.search(X_train, y_train)

        console.print(f"  Best params: {search_results['best_params']}")
        console.print(f"  Best CV score: {search_results['best_score']:.4f}")

        # Save gridsearch results to CSV
        cv_results_df = pd.DataFrame(search_results["cv_results"])
        # Select relevant columns and sort by rank
        gridsearch_df = cv_results_df[
            [col for col in cv_results_df.columns if col.startswith(("param_", "mean_", "std_", "rank_"))]
        ].sort_values("rank_test_score")
        gridsearch_path = clf_output_dir / "gridsearch_results.csv"
        gridsearch_df.to_csv(gridsearch_path, index=False)
        console.print(f"  [green]Saved gridsearch results to {gridsearch_path}[/green]")

        # K-fold cross-validation on best model (on train set only)
        cv = CrossValidator(
            k_folds=config.training.final_cv_folds,
            random_seed=config.training.random_seed,
        )
        cv_results = cv.evaluate(search_results["best_estimator"], X_train, y_train)

        console.print(
            f"  Final CV Accuracy: {cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}"
        )
        console.print(
            f"  Final CV F1: {cv_results['f1_mean']:.4f} +/- {cv_results['f1_std']:.4f}"
        )

        # Evaluate on held-out test set if enabled
        test_results = None
        if args.holdout:
            # Retrain on full training set
            final_model = clone(search_results["best_estimator"])
            final_model.fit(X_train, y_train)

            # Predict on test set
            y_pred = final_model.predict(X_test)

            # Calculate test metrics
            test_results = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }

            # ROC-AUC if model supports predict_proba
            if hasattr(final_model, "predict_proba"):
                y_proba = final_model.predict_proba(X_test)[:, 1]
                test_results["roc_auc"] = float(roc_auc_score(y_test, y_proba))

            console.print(
                f"  [magenta]Test Accuracy: {test_results['accuracy']:.4f}[/magenta]"
            )
            console.print(
                f"  [magenta]Test F1: {test_results['f1']:.4f}[/magenta]"
            )

            # Save final model (trained on full train set)
            model_path = clf_output_dir / "best_model.joblib"
            joblib.dump(final_model, model_path)
            console.print(f"  [green]Saved final model to {model_path}[/green]")
        else:
            # Save best model from grid search
            model_path = clf_output_dir / "best_model.joblib"
            joblib.dump(search_results["best_estimator"], model_path)
            console.print(f"  [green]Saved best model to {model_path}[/green]")

        # Save results to JSON
        results_output = {
            "classifier": clf_name,
            "best_params": search_results["best_params"],
            "hp_search_score": search_results["best_score"],
            "cv_folds": config.training.final_cv_folds,
            "scoring_metric": config.training.scoring_metric,
            "cv_metrics": cv_results,
            "test_metrics": test_results,
            "holdout_enabled": args.holdout,
            "holdout_fraction": config.training.holdout_fraction if args.holdout else None,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model": config.model.name,
                "extraction_mode": config.model.extraction_mode,
                "pooling": config.model.pooling,
                "dataset": dataset_name,
                "n_samples": len(labels),
                "n_train": len(y_train),
                "n_test": len(y_test) if args.holdout else None,
                "feature_dim": features.shape[1],
            },
        }
        results_path = clf_output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results_output, f, indent=2)
        console.print(f"  [green]Saved results to {results_path}[/green]")

        # Log classifier completion
        clf_elapsed = time.time() - clf_start
        logger.info(f"{clf_name} training completed in {clf_elapsed:.2f}s")
        if test_results:
            logger.info(f"{clf_name} test results: accuracy={test_results['accuracy']:.4f}, f1={test_results['f1']:.4f}")

        # Add to results table
        row_data = [
            clf_name,
            format_params(search_results["best_params"]),
            f"{search_results['best_score']:.4f}",
            f"{cv_results['accuracy_mean']:.4f} +/- {cv_results['accuracy_std']:.4f}",
            f"{cv_results['f1_mean']:.4f} +/- {cv_results['f1_std']:.4f}",
        ]
        if args.holdout:
            row_data.extend([
                f"{test_results['accuracy']:.4f}",
                f"{test_results['f1']:.4f}",
            ])
        results_table.add_row(*row_data)

    # Display final results
    console.print("\n")
    console.print(results_table)
    console.print(f"\n[bold green]All results saved to {output_root}/[/bold green]")

    # Log pipeline completion
    pipeline_elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline completed in {pipeline_elapsed:.2f}s")
    logger.info(f"Results saved to {output_root}/")


if __name__ == "__main__":
    main()
