"""Pipeline configuration classes.

This module defines configuration classes for all pipeline phases:
injection, training, evaluation, and optimization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from CESTA.schema.fault import MarkovConfig
from CESTA.schema.window import DataConfig


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file as raw mapping data."""
    resolved = Path(path)
    if not resolved.exists():
        msg = f"Config file not found: {resolved}"
        raise FileNotFoundError(msg)

    suffix = resolved.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(resolved.read_text()) or {}
    elif suffix == ".json":
        raw = json.loads(resolved.read_text())
    else:
        msg = f"Unsupported config extension: {resolved.suffix}"
        raise ValueError(msg)

    if not isinstance(raw, dict):
        msg = f"Config file must contain a mapping: {resolved}"
        raise ValueError(msg)
    return raw


class InjectionConfig(BaseModel):
    """Complete configuration for fault injection pipeline.

    This is the main config object that gets serialized as metadata.

    Attributes:
        markov: Markov chain configuration.
        resample_freq: Resampling frequency string (e.g., "30s").
        target_features: Features to inject faults into.
        all_features: All features to include in the output.
        interpolation_method: Method for interpolating missing values.
        group_column: Column to group by (e.g., "moteid").
        seed: Global random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    markov: MarkovConfig = Field(default_factory=MarkovConfig)
    resample_freq: str = "5min"
    target_features: list[str] = Field(default_factory=lambda: ["temp"])
    all_features: list[str] = Field(default_factory=lambda: ["temp", "humid", "light", "volt"])
    interpolation_method: str = "linear"
    group_column: str = "moteid"
    seed: int | None = None

    @model_validator(mode="after")
    def _propagate_seed(self) -> "InjectionConfig":
        """Propagate seed to markov config if not set."""
        if self.seed is not None and self.markov.seed is None:
            object.__setattr__(
                self,
                "markov",
                self.markov.model_copy(update={"seed": self.seed}),
            )
        return self


class TrainConfig(BaseModel):
    """Configuration for model training.

    Attributes:
        model: Model architecture name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        use_focal_loss: Whether to use focal loss instead of cross-entropy.
        focal_gamma: Focusing parameter for focal loss (higher = more focus on hard examples).
        focal_alpha: Per-class balancing weights for focal loss. None means uniform.
        oversample: Whether to oversample minority (non-NORMAL) classes.
        oversample_ratio: Target ratio of minority to majority samples (1.0 = balanced).
        communication_penalty_weight: Weight for communication auxiliary loss (0 = disabled).
        communication_penalty_mode: ``"linear"`` for L1 penalty or ``"budget_hinge"`` for
            ``relu(ratio - target)^2``.
        target_request_ratio: Target active request ratio for budget_hinge mode.
        gate_entropy_weight: Weight for gate entropy regularization (0 = disabled).
            Positive weight encourages higher gate entropy to prevent collapse to all-zero.
        boundary_loss_weight: Weight for boundary/change-point auxiliary loss (0 = disabled).
        boundary_focal_gamma: Focusing parameter for focal BCE boundary loss.
        boundary_positive_weight: Optional positive-class weight for sparse boundary labels.
        boundary_dilation: Temporal dilation radius around boundary targets.
        crf_loss_weight: Weight for optional linear-chain CRF sequence loss (0 = disabled).
        gumbel_tau_start: Initial Gumbel-Softmax temperature.
        gumbel_tau_end: Final Gumbel-Softmax temperature after annealing.
        gumbel_tau_anneal_epochs: Number of epochs over which to linearly anneal
            temperature from ``gumbel_tau_start`` to ``gumbel_tau_end``.
        checkpoint_monitor: Metric to monitor for model checkpointing
            (``val_loss``, ``val_macro_f1``, or ``val_acc``).
        early_stopping_monitor: Metric to monitor for early stopping
            (``val_loss``, ``val_macro_f1``, or ``val_acc``).
        features: Subset of feature names to train on. None means all features.
        data: Train-time data windowing and split configuration.
        seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt=0.0)
    use_focal_loss: bool = False
    focal_gamma: float = Field(default=2.0, ge=0.0)
    focal_alpha: list[float] | None = None
    oversample: bool = False
    oversample_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
    communication_penalty_weight: float = Field(default=0.0, ge=0.0)
    communication_penalty_mode: str = Field(default="linear", pattern=r"^(linear|budget_hinge)$")
    target_request_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    gate_entropy_weight: float = Field(default=0.0, ge=0.0)
    boundary_loss_weight: float = Field(default=0.0, ge=0.0)
    boundary_focal_gamma: float = Field(default=2.0, ge=0.0)
    boundary_positive_weight: float | None = Field(default=None, gt=0.0)
    boundary_dilation: int = Field(default=0, ge=0)
    crf_loss_weight: float = Field(default=0.0, ge=0.0)
    gumbel_tau_start: float = Field(default=1.0, gt=0.0)
    gumbel_tau_end: float = Field(default=1.0, gt=0.0)
    gumbel_tau_anneal_epochs: int = Field(default=0, ge=0)
    checkpoint_monitor: str = Field(default="val_loss", pattern=r"^(val_loss|val_macro_f1|val_acc)$")
    early_stopping_monitor: str = Field(default="val_loss", pattern=r"^(val_loss|val_macro_f1|val_acc)$")
    features: list[str] | None = None
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = 42
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _flatten_config_file_sections(cls, data: Any) -> Any:
        if isinstance(data, dict) and "train" in data:
            train_section = data.get("train") or {}
            if not isinstance(train_section, dict):
                msg = "Train config 'train' section must be a mapping"
                raise ValueError(msg)
            merged = dict(train_section)
            if "model_kwargs" in data:
                merged["model_kwargs"] = data["model_kwargs"]
            if "data" in data:
                merged["data"] = data["data"]
            return merged
        return data

class EvaluateConfig(BaseModel):
    """Configuration for model evaluation.

    Attributes:
        batch_size: Evaluation batch size.
    """

    model_config = ConfigDict(frozen=True)

    batch_size: int = Field(default=64, ge=1)

class OptimizeConfig(BaseModel):
    """Configuration for hyperparameter optimization with Optuna.

    Attributes:
        model: Model architecture to optimize.
        n_trials: Number of Optuna trials.
        timeout: Optimization timeout in seconds (None = unlimited).
        seed: Random seed for sampler reproducibility.
        storage: Optuna storage URL (e.g. ``sqlite:///optuna.db``).
        study_name: Optuna study name. Defaults to ``cesta-<model>``.
        direction: ``minimize`` or ``maximize``.
        metric: Validation metric to optimize. One of ``val_loss``,
            ``val_macro_f1``, ``val_acc``.
        epochs: Number of training epochs per trial.
        sampler: Optuna sampler to use. One of ``tpe``, ``random``.
        pruner: Optuna pruner. One of ``median``, ``none``.
        startup_trials: Number of random trials before TPE/MedianPruner kicks in.
        load_if_exists: Resume an existing study with the same name.
        features: Subset of feature names to train on. None = all features.
        data: Train-time data windowing and split configuration.
    """

    model_config = ConfigDict(frozen=True)

    model: str = "lstm"
    n_trials: int = Field(default=20, ge=1)
    timeout: int | None = None
    seed: int = 42
    storage: str = "sqlite:///optuna.db"
    study_name: str | None = None
    direction: str = Field(default="minimize", pattern=r"^(minimize|maximize)$")
    metric: str = Field(
        default="val_loss",
        pattern=r"^(val_loss|val_macro_f1|val_acc)$",
    )
    epochs: int = Field(default=20, ge=1)
    sampler: str = Field(default="tpe", pattern=r"^(tpe|random)$")
    pruner: str = Field(default="median", pattern=r"^(median|none)$")
    startup_trials: int = Field(default=5, ge=0)
    load_if_exists: bool = True
    features: list[str] | None = None
    data: DataConfig = Field(default_factory=DataConfig)

    @model_validator(mode="after")
    def _align_direction_with_metric(self) -> "OptimizeConfig":
        direction = "minimize" if self.metric == "val_loss" else "maximize"
        if self.direction != direction:
            object.__setattr__(self, "direction", direction)
        return self

    def resolved_study_name(self) -> str:
        """Return the study name, defaulting to ``cesta-<model>``."""
        return self.study_name if self.study_name is not None else f"cesta-{self.model}"

    def resolved_direction(self) -> str:
        """Return the direction inferred from the metric if not overridden."""
        if self.metric == "val_loss":
            return "minimize"
        return "maximize"

