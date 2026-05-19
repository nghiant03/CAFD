"""Base class for deep learning models.

All model implementations should inherit from BaseModel and implement
the required abstract methods.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for fault diagnosis models.

    All models must implement:
        - name: Property returning the model's registered name
        - forward: Standard PyTorch forward pass
        - get_config: Return architecture config dict for serialization

    Subclasses that need extra data (e.g. graph topology) should list
    required metadata keys in ``required_metadata``.  The model registry
    validates these before construction.

    The base class provides common utilities for model management.
    """

    required_metadata: ClassVar[set[str]] = set()
    optional_metadata: ClassVar[set[str]] = set()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the registered name of this model."""
        ...

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, seq_len, features).

        Returns:
            Output tensor of shape (batch, seq_len, num_classes) for
            many-to-many classification.
        """
        ...

    @abstractmethod
    def get_config(self) -> dict[str, object]:
        """Return model architecture configuration for serialization."""
        ...

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(
        self,
        path: str | Path,
        config_dict: dict[str, object] | None = None,
    ) -> None:
        """Save model to a directory with weight.pt and config.json.

        Args:
            path: Directory path to save the model into.
            config_dict: Optional training config to embed alongside model config.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), directory / "weight.pt")

        meta: dict[str, object] = {
            "model_name": self.name,
            "model_config": self.get_config(),
        }
        if config_dict is not None:
            meta["train_config"] = config_dict
        (directory / "config.json").write_text(json.dumps(meta, indent=2))

    @staticmethod
    def load_config(path: str | Path) -> dict[str, object] | None:
        """Load training config from a saved model directory.

        Args:
            path: Path to the model directory.

        Returns:
            Training config dictionary if present, None otherwise.
        """
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            return None
        meta = json.loads(config_path.read_text())
        return meta.get("train_config")  # type: ignore[no-any-return]

    @staticmethod
    def load_metadata(path: str | Path) -> dict[str, object]:
        """Load full metadata (model_name, model_config, train_config) from directory.

        Args:
            path: Path to the model directory.

        Returns:
            Full metadata dictionary.
        """
        config_path = Path(path) / "config.json"
        return json.loads(config_path.read_text())  # type: ignore[no-any-return]
