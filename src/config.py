"""Configuration loading and validation for the classification pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "Qwen/Qwen3-VL-8B-Instruct"
    pooling: str = "mean"
    dtype: str = "float16"
    device: str = "cuda"
    extraction_mode: str = "vision_encoder"  # "full" or "vision_encoder"


@dataclass
class DataConfig:
    """Data configuration."""

    root: str = "data"
    use_sample: bool = False
    cache_dir: str = "cache/features"
    output_dir: str = "output"

    def get_data_root(self) -> Path:
        """Get the actual data root path based on use_sample flag."""
        root = Path(self.root)
        if self.use_sample:
            return root / "sample"
        return root


@dataclass
class TrainingConfig:
    """Training configuration."""

    scoring_metric: str = "f1"
    hp_search_cv_folds: int = 3
    final_cv_folds: int = 5
    random_seed: int = 42
    holdout_fraction: float = 0.2  # Fraction of data to hold out for testing


@dataclass
class MLPConfig:
    """MLP classifier configuration."""

    enabled: bool = True
    hidden_layer_sizes: list[list[int]] = field(
        default_factory=lambda: [[128], [256], [128, 64], [256, 128]]
    )
    alpha: list[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01])
    learning_rate_init: list[float] = field(default_factory=lambda: [0.001, 0.0001])
    activation: list[str] = field(default_factory=lambda: ["relu", "tanh"])


@dataclass
class SVMConfig:
    """SVM classifier configuration."""

    enabled: bool = True
    C: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0])
    gamma: list[Any] = field(
        default_factory=lambda: ["scale", "auto", 0.001, 0.01]
    )
    kernel: list[str] = field(default_factory=lambda: ["rbf", "linear"])


@dataclass
class XGBoostConfig:
    """XGBoost classifier configuration."""

    enabled: bool = True
    n_estimators: list[int] = field(default_factory=lambda: [100, 200, 300])
    max_depth: list[int] = field(default_factory=lambda: [3, 5, 7])
    learning_rate: list[float] = field(default_factory=lambda: [0.01, 0.1, 0.3])
    subsample: list[float] = field(default_factory=lambda: [0.8, 1.0])


@dataclass
class LogisticRegressionConfig:
    """Logistic Regression classifier configuration."""

    enabled: bool = True
    C: list[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    solver: list[str] = field(default_factory=lambda: ["lbfgs"])


@dataclass
class ClassifiersConfig:
    """Classifiers configuration."""

    mlp: MLPConfig = field(default_factory=MLPConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    logistic_regression: LogisticRegressionConfig = field(
        default_factory=LogisticRegressionConfig
    )


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    classifiers: ClassifiersConfig = field(default_factory=ClassifiersConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create configuration from a dictionary."""
        config = cls()

        if "model" in data:
            config.model = ModelConfig(**data["model"])

        if "data" in data:
            config.data = DataConfig(**data["data"])

        if "training" in data:
            config.training = TrainingConfig(**data["training"])

        if "classifiers" in data:
            clf_data = data["classifiers"]
            mlp_config = MLPConfig(**clf_data.get("mlp", {})) if "mlp" in clf_data else MLPConfig()
            svm_config = SVMConfig(**clf_data.get("svm", {})) if "svm" in clf_data else SVMConfig()
            xgb_config = XGBoostConfig(**clf_data.get("xgboost", {})) if "xgboost" in clf_data else XGBoostConfig()
            lr_config = LogisticRegressionConfig(**clf_data.get("logistic_regression", {})) if "logistic_regression" in clf_data else LogisticRegressionConfig()
            config.classifiers = ClassifiersConfig(
                mlp=mlp_config,
                svm=svm_config,
                xgboost=xgb_config,
                logistic_regression=lr_config,
            )

        return config

    def apply_overrides(
        self,
        use_sample: bool | None = None,
        no_cache: bool = False,
    ) -> None:
        """Apply CLI argument overrides to the configuration."""
        if use_sample is not None:
            self.data.use_sample = use_sample


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path is None:
        # Look for config.yaml in current directory
        default_path = Path("config.yaml")
        if default_path.exists():
            return Config.from_yaml(default_path)
        return Config()

    return Config.from_yaml(config_path)
