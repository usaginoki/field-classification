"""Hyperparameter search and cross-validation evaluation."""

import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .classification import BaseClassifierWrapper
from .logging_config import get_logger

logger = get_logger(__name__)


class HyperparameterSearcher:
    """Perform hyperparameter search with cross-validation."""

    def __init__(
        self,
        classifier_wrapper: BaseClassifierWrapper,
        cv_folds: int = 3,
        scoring: str = "f1",
        n_jobs: int = -1,
        random_seed: int = 42,
    ):
        """
        Initialize the hyperparameter searcher.

        Args:
            classifier_wrapper: Classifier wrapper with param grid.
            cv_folds: Number of CV folds for hyperparameter search.
            scoring: Scoring metric (f1, accuracy, roc_auc).
            n_jobs: Number of parallel jobs (-1 for all cores).
            random_seed: Random seed for reproducibility.
        """
        self.wrapper = classifier_wrapper
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_seed = random_seed

    def search(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Perform grid search to find best hyperparameters.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).

        Returns:
            Dictionary with best_params, best_score, best_estimator, cv_results.
        """
        estimator = self.wrapper.create_estimator(self.random_seed)
        param_grid = self.wrapper.get_param_grid()

        # Calculate total combinations
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        logger.info(f"Starting hyperparameter search: {n_combinations} combinations, {self.cv_folds}-fold CV")
        logger.debug(f"Param grid: {param_grid}")

        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            ),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1,
            return_train_score=True,
        )

        grid_search.fit(X, y)
        elapsed = time.time() - start_time

        logger.info(f"Hyperparameter search completed in {elapsed:.2f}s")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")

        return {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_estimator": grid_search.best_estimator_,
            "cv_results": grid_search.cv_results_,
        }


class CrossValidator:
    """Perform k-fold cross-validation on a model."""

    def __init__(self, k_folds: int = 5, random_seed: int = 42):
        """
        Initialize the cross-validator.

        Args:
            k_folds: Number of CV folds.
            random_seed: Random seed for reproducibility.
        """
        self.k_folds = k_folds
        self.random_seed = random_seed

    def evaluate(
        self, estimator: BaseEstimator, X: np.ndarray, y: np.ndarray
    ) -> dict[str, float]:
        """
        Perform k-fold CV and return detailed metrics.

        Args:
            estimator: Fitted or unfitted sklearn estimator.
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).

        Returns:
            Dictionary with mean and std of metrics across folds.
        """
        logger.info(f"Starting {self.k_folds}-fold cross-validation")
        start_time = time.time()

        cv = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=self.random_seed
        )

        metrics: dict[str, list[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Clone and fit
            est = clone(estimator)
            est.fit(X_train, y_train)

            # Predict
            y_pred = est.predict(X_val)

            # Calculate metrics
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            metrics["accuracy"].append(acc)
            metrics["precision"].append(
                precision_score(y_val, y_pred, zero_division=0)
            )
            metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            metrics["f1"].append(f1)

            # ROC-AUC requires probability predictions
            if hasattr(est, "predict_proba"):
                y_proba = est.predict_proba(X_val)[:, 1]
                metrics["roc_auc"].append(roc_auc_score(y_val, y_proba))
            else:
                metrics["roc_auc"].append(float("nan"))

            logger.debug(f"Fold {fold_idx}/{self.k_folds}: accuracy={acc:.4f}, f1={f1:.4f}")

        elapsed = time.time() - start_time

        # Calculate mean and std
        results: dict[str, float] = {}
        for metric, values in metrics.items():
            results[f"{metric}_mean"] = float(np.mean(values))
            results[f"{metric}_std"] = float(np.std(values))

        logger.info(f"Cross-validation completed in {elapsed:.2f}s")
        logger.info(f"CV Results: accuracy={results['accuracy_mean']:.4f}±{results['accuracy_std']:.4f}, f1={results['f1_mean']:.4f}±{results['f1_std']:.4f}")

        return results


def format_params(params: dict[str, Any]) -> str:
    """
    Format parameter dictionary for display.

    Args:
        params: Dictionary of parameters.

    Returns:
        Formatted string representation.
    """
    lines = []
    for k, v in params.items():
        # Remove pipeline prefix
        key = k.split("__")[-1]
        lines.append(f"{key}: {v}")
    return "\n".join(lines)
