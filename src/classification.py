"""Classification models with hyperparameter configurations."""

from abc import ABC, abstractmethod
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import LogisticRegressionConfig, MLPConfig, SVMConfig, XGBoostConfig


class BaseClassifierWrapper(ABC):
    """Base interface for classifiers with hyperparameter search."""

    @abstractmethod
    def get_param_grid(self) -> dict[str, list[Any]]:
        """Return hyperparameter grid for search."""
        pass

    @abstractmethod
    def create_estimator(self, random_seed: int = 42) -> BaseEstimator:
        """Create a new estimator instance."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return classifier name."""
        pass


class MLPClassifierWrapper(BaseClassifierWrapper):
    """MLP classifier with configurable hyperparameter search."""

    def __init__(self, config: MLPConfig | None = None):
        """
        Initialize the MLP classifier wrapper.

        Args:
            config: MLP configuration. If None, uses defaults.
        """
        self.config = config or MLPConfig()

    def get_name(self) -> str:
        return "MLP"

    def get_param_grid(self) -> dict[str, list[Any]]:
        """
        Return hyperparameter grid for GridSearchCV.

        The grid is built from the configuration, converting nested lists
        to tuples for hidden_layer_sizes.
        """
        # Convert nested lists to tuples for hidden_layer_sizes
        hidden_sizes = [tuple(hs) for hs in self.config.hidden_layer_sizes]

        return {
            "classifier__hidden_layer_sizes": hidden_sizes,
            "classifier__alpha": self.config.alpha,
            "classifier__learning_rate_init": self.config.learning_rate_init,
            "classifier__activation": self.config.activation,
        }

    def create_estimator(self, random_seed: int = 42) -> Pipeline:
        """
        Create MLP pipeline with StandardScaler.

        Args:
            random_seed: Random seed for reproducibility.

        Returns:
            sklearn Pipeline with scaler and MLP classifier.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        solver="adam",
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=20,
                        random_state=random_seed,
                    ),
                ),
            ]
        )


class SVMClassifierWrapper(BaseClassifierWrapper):
    """SVM classifier with configurable hyperparameter search."""

    def __init__(self, config: SVMConfig | None = None):
        """
        Initialize the SVM classifier wrapper.

        Args:
            config: SVM configuration. If None, uses defaults.
        """
        self.config = config or SVMConfig()

    def get_name(self) -> str:
        return "SVM"

    def get_param_grid(self) -> dict[str, list[Any]]:
        """Return hyperparameter grid for GridSearchCV."""
        return {
            "classifier__C": self.config.C,
            "classifier__gamma": self.config.gamma,
            "classifier__kernel": self.config.kernel,
        }

    def create_estimator(self, random_seed: int = 42) -> Pipeline:
        """
        Create SVM pipeline with StandardScaler.

        Args:
            random_seed: Random seed for reproducibility.

        Returns:
            sklearn Pipeline with scaler and SVM classifier.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        random_state=random_seed,
                        probability=True,  # Enable predict_proba
                    ),
                ),
            ]
        )


class XGBoostClassifierWrapper(BaseClassifierWrapper):
    """XGBoost classifier with configurable hyperparameter search."""

    def __init__(self, config: XGBoostConfig | None = None):
        """
        Initialize the XGBoost classifier wrapper.

        Args:
            config: XGBoost configuration. If None, uses defaults.
        """
        self.config = config or XGBoostConfig()

    def get_name(self) -> str:
        return "XGBoost"

    def get_param_grid(self) -> dict[str, list[Any]]:
        """Return hyperparameter grid for GridSearchCV."""
        return {
            "classifier__n_estimators": self.config.n_estimators,
            "classifier__max_depth": self.config.max_depth,
            "classifier__learning_rate": self.config.learning_rate,
            "classifier__subsample": self.config.subsample,
        }

    def create_estimator(self, random_seed: int = 42) -> Pipeline:
        """
        Create XGBoost pipeline with StandardScaler.

        Args:
            random_seed: Random seed for reproducibility.

        Returns:
            sklearn Pipeline with scaler and XGBoost classifier.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    XGBClassifier(
                        random_state=random_seed,
                        eval_metric="logloss",
                        use_label_encoder=False,
                        verbosity=0,
                    ),
                ),
            ]
        )


class LogisticRegressionClassifierWrapper(BaseClassifierWrapper):
    """Logistic Regression classifier with configurable hyperparameter search."""

    def __init__(self, config: LogisticRegressionConfig | None = None):
        """
        Initialize the Logistic Regression classifier wrapper.

        Args:
            config: Logistic Regression configuration. If None, uses defaults.
        """
        self.config = config or LogisticRegressionConfig()

    def get_name(self) -> str:
        return "LogisticRegression"

    def get_param_grid(self) -> dict[str, list[Any]]:
        """Return hyperparameter grid for GridSearchCV."""
        return {
            "classifier__C": self.config.C,
            "classifier__solver": self.config.solver,
        }

    def create_estimator(self, random_seed: int = 42) -> Pipeline:
        """
        Create Logistic Regression pipeline with StandardScaler.

        Args:
            random_seed: Random seed for reproducibility.

        Returns:
            sklearn Pipeline with scaler and Logistic Regression classifier.
        """
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_seed,
                        max_iter=1000,
                    ),
                ),
            ]
        )


def get_enabled_classifiers(
    mlp_config: MLPConfig,
    svm_config: SVMConfig,
    xgboost_config: XGBoostConfig,
    logistic_regression_config: LogisticRegressionConfig,
) -> list[BaseClassifierWrapper]:
    """
    Get list of enabled classifier wrappers based on configuration.

    Args:
        mlp_config: MLP configuration.
        svm_config: SVM configuration.
        xgboost_config: XGBoost configuration.
        logistic_regression_config: Logistic Regression configuration.

    Returns:
        List of enabled classifier wrappers.
    """
    classifiers = []
    if mlp_config.enabled:
        classifiers.append(MLPClassifierWrapper(mlp_config))
    if svm_config.enabled:
        classifiers.append(SVMClassifierWrapper(svm_config))
    if xgboost_config.enabled:
        classifiers.append(XGBoostClassifierWrapper(xgboost_config))
    if logistic_regression_config.enabled:
        classifiers.append(LogisticRegressionClassifierWrapper(logistic_regression_config))
    return classifiers
