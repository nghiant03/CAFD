"""GRU model for many-to-many fault classification.

This module implements a GRU-based architecture for sequence-to-sequence
fault diagnosis, predicting a fault label at each timestep.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
import torch.nn as nn

from CESTA.batch import TemporalWindowBatch
from CESTA.models.base import BaseModel


class GRUClassifier(BaseModel):
    """GRU model for many-to-many sequence classification.

    Architecture:
        Input -> GRU (bidirectional optional) -> Dropout -> Linear -> Output

    For each timestep in the input sequence, the model outputs class
    probabilities (logits) for fault classification.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Number of GRU hidden units.
        num_layers: Number of stacked GRU layers.
        num_classes: Number of output classes (fault types).
        dropout: Dropout probability between GRU layers.
        bidirectional: Whether to use bidirectional GRU.
    """

    optional_metadata: ClassVar[set[str]] = {"node_identity"}

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_nodes: int | None = None,
        node_embedding_dim: int = 0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.bidirectional = bidirectional
        self.num_nodes = num_nodes
        self.node_embedding_dim = node_embedding_dim

        if node_embedding_dim > 0 and num_nodes is None:
            raise ValueError("num_nodes is required when node_embedding_dim > 0")
        self.node_embedding = nn.Embedding(num_nodes, node_embedding_dim) if node_embedding_dim > 0 and num_nodes is not None else None
        gru_input_size = input_size + node_embedding_dim

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_output_size = hidden_size * 2 if bidirectional else hidden_size

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_size, num_classes)

    @property
    def name(self) -> str:
        return "gru"

    def forward(self, x: torch.Tensor | TemporalWindowBatch) -> torch.Tensor:
        """Forward pass for many-to-many classification.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch, seq_len, num_classes).
        """
        if isinstance(x, TemporalWindowBatch):
            node_ids = x.node_ids
            x = x.x
        else:
            node_ids = None
        if self.node_embedding is not None:
            if node_ids is None:
                raise ValueError("node_ids are required when node_embedding_dim > 0")
            node_embedding = self.node_embedding(node_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, node_embedding], dim=-1)
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out)
        return logits

    def get_config(self) -> dict[str, object]:
        """Return model configuration for serialization."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "bidirectional": self.bidirectional,
            "num_nodes": self.num_nodes,
            "node_embedding_dim": self.node_embedding_dim,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "GRUClassifier":
        """Load model from a saved directory.

        Args:
            path: Path to the model directory.

        Returns:
            Loaded GRUClassifier instance.
        """
        directory = Path(path)
        meta = BaseModel.load_metadata(directory)
        config = meta["model_config"]
        assert isinstance(config, dict)
        model = cls(
            input_size=int(config["input_size"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            num_classes=int(config["num_classes"]),
            dropout=float(config["dropout"]),
            bidirectional=bool(config["bidirectional"]),
            num_nodes=int(config["num_nodes"]) if config.get("num_nodes") is not None else None,
            node_embedding_dim=int(config.get("node_embedding_dim", 0)),
        )
        model.load_state_dict(
            torch.load(directory / "weight.pt", weights_only=True)
        )
        return model
