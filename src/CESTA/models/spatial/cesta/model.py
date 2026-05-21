"""CESTA spatial-temporal model for communication-aware fault diagnosis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, Literal

import torch
import torch.nn as nn

from CESTA.batch import GraphWindowBatch
from CESTA.models.base import BaseModel
from CESTA.models.spatial.cesta.communication import CESTACommunicationMixin, CommunicationStats
from CESTA.models.spatial.cesta.sequence import CESTASequenceMixin

CommunicationMode = Literal["none", "dense", "gumbel_request"]


class CESTAClassifier(CESTASequenceMixin, CESTACommunicationMixin, BaseModel):
    """Communication-Efficient Spatial-Temporal Aggregation classifier.

    Uses GAT-inspired single-head attention for neighbor aggregation.
    When zero neighbors are requested, produces a zero context vector.
    """

    required_metadata: ClassVar[set[str]] = {"graph"}

    def __init__(
        self,
        input_size: int,
        num_nodes: int,
        adjacency: list[list[float]] | None = None,
        edge_prob: list[list[float]] | None = None,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 4,
        dropout: float = 0.2,
        communication_mode: CommunicationMode = "none",
        fusion_hidden_size: int | None = None,
        precision_bits: int = 32,
        gumbel_temperature: float = 1.0,
        gate_hidden_size: int = 32,
        num_attention_heads: int = 1,
        graph_residual_init: float = 1.0,
        bidirectional: bool = False,
        use_logit_correction: bool = False,
        correction_hidden_size: int | None = None,
        correction_init: float = 0.1,
        use_neighbor_belief: bool = False,
        use_boundary_head: bool = False,
        boundary_hidden_size: int | None = None,
        use_boundary_gated_correction: bool = False,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        if input_size % num_nodes != 0:
            raise ValueError("input_size must be divisible by num_nodes")
        if communication_mode not in {"none", "dense", "gumbel_request"}:
            raise ValueError(
                "communication_mode must be one of: none, dense, gumbel_request"
            )
        if gumbel_temperature <= 0.0:
            raise ValueError("gumbel_temperature must be positive")
        if gate_hidden_size < 1:
            raise ValueError("gate_hidden_size must be positive")
        if num_attention_heads < 1:
            raise ValueError("num_attention_heads must be positive")
        if not 0.0 <= graph_residual_init <= 1.0:
            raise ValueError("graph_residual_init must be in [0, 1]")
        if correction_hidden_size is not None and correction_hidden_size < 1:
            raise ValueError("correction_hidden_size must be positive")
        if not 0.0 <= correction_init <= 1.0:
            raise ValueError("correction_init must be in [0, 1]")
        if boundary_hidden_size is not None and boundary_hidden_size < 1:
            raise ValueError("boundary_hidden_size must be positive")
        if use_boundary_gated_correction and not use_boundary_head:
            raise ValueError("use_boundary_gated_correction requires use_boundary_head")

        self.input_size = input_size
        self.num_nodes = num_nodes
        self.features_per_node = input_size // num_nodes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.communication_mode: CommunicationMode = communication_mode
        self.fusion_hidden_size = fusion_hidden_size
        self.precision_bits = precision_bits
        self.gumbel_temperature = gumbel_temperature
        self.gate_hidden_size = gate_hidden_size
        self.num_attention_heads = num_attention_heads
        self.graph_residual_init = graph_residual_init
        self.bidirectional = bidirectional
        self.use_logit_correction = use_logit_correction
        self.correction_hidden_size = correction_hidden_size
        self.correction_init = correction_init
        self.use_neighbor_belief = use_neighbor_belief
        self.use_boundary_head = use_boundary_head
        self.boundary_hidden_size = boundary_hidden_size
        self.use_boundary_gated_correction = use_boundary_gated_correction
        self.use_crf = use_crf
        self.encoder_output_size = hidden_size * (2 if bidirectional else 1)
        self.neighbor_belief_size = num_classes + 2

        self._gate_entropy: torch.Tensor | None = None
        self._last_boundary_logits: torch.Tensor | None = None

        if num_attention_heads != 1:
            raise NotImplementedError(
                "Multi-head attention (>1) is not yet implemented"
            )
        if self.encoder_output_size % num_attention_heads != 0:
            raise ValueError("encoder output size must be divisible by num_attention_heads")

        if adjacency is not None:
            adj_tensor = torch.tensor(adjacency, dtype=torch.float32)
        else:
            adj_tensor = torch.eye(num_nodes, dtype=torch.float32)
        if adj_tensor.shape != (num_nodes, num_nodes):
            raise ValueError("adjacency must have shape (num_nodes, num_nodes)")
        if edge_prob is not None:
            edge_prob_tensor = torch.tensor(edge_prob, dtype=torch.float32)
        else:
            edge_prob_tensor = adj_tensor.clone()
            edge_prob_tensor.fill_diagonal_(0.0)
        if edge_prob_tensor.shape != (num_nodes, num_nodes):
            raise ValueError("edge_prob must have shape (num_nodes, num_nodes)")
        self.register_buffer("adjacency", adj_tensor)
        self.register_buffer("edge_prob", edge_prob_tensor)
        self._adjacency_list: list[list[float]] = adj_tensor.tolist()
        self._edge_prob_list: list[list[float]] = edge_prob_tensor.tolist()

        self.temporal_encoder = nn.GRU(
            input_size=self.features_per_node,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_scale = self.encoder_output_size ** 0.5
        self.W_q = nn.Linear(self.encoder_output_size, self.encoder_output_size, bias=False)
        self.W_k = nn.Linear(self.encoder_output_size, self.encoder_output_size, bias=False)
        self.W_v = nn.Linear(self.encoder_output_size, self.encoder_output_size, bias=False)
        fusion_output_size = fusion_hidden_size or self.encoder_output_size
        fusion_input_size = self.encoder_output_size * 2 + (self.neighbor_belief_size if use_neighbor_belief else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_size, self.encoder_output_size),
        )
        residual_eps = 1e-4
        residual_init = min(max(graph_residual_init, residual_eps), 1.0 - residual_eps)
        self.graph_residual_logit = nn.Parameter(
            torch.tensor(math.log(residual_init / (1.0 - residual_init)), dtype=torch.float32)
        )
        self.request_gate = nn.Sequential(
            nn.Linear(self.encoder_output_size + 3, gate_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_size, 2),
        )
        self.classifier = nn.Linear(self.encoder_output_size, num_classes)
        boundary_layer_size = boundary_hidden_size or self.encoder_output_size
        self.boundary_head = nn.Sequential(
            nn.Linear(self.encoder_output_size, boundary_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(boundary_layer_size, 1),
        )
        correction_input_size = self.encoder_output_size * 3 + self.features_per_node * 3 + num_classes + 2
        if use_neighbor_belief:
            correction_input_size += self.neighbor_belief_size * 3
        correction_layer_size = correction_hidden_size or self.encoder_output_size
        self.logit_correction = nn.Sequential(
            nn.Linear(correction_input_size, correction_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(correction_layer_size, num_classes),
        )
        correction_eps = 1e-4
        correction_scale_init = min(max(correction_init, correction_eps), 1.0 - correction_eps)
        self.correction_logit = nn.Parameter(
            torch.tensor(math.log(correction_scale_init / (1.0 - correction_scale_init)), dtype=torch.float32)
        )
        self.crf_transitions = nn.Parameter(torch.zeros(num_classes, num_classes))
        self._last_communication_stats: CommunicationStats = (
            self._zero_communication_stats()
        )
        self._communication_loss: torch.Tensor | None = None
        self._gate_entropy: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "cesta"

    @property
    def last_communication_stats(self) -> CommunicationStats:
        return self._last_communication_stats.copy()

    @property
    def auxiliary_loss(self) -> torch.Tensor | None:
        """Communication loss for backward compatibility."""
        return self._communication_loss

    @property
    def communication_loss(self) -> torch.Tensor | None:
        return self._communication_loss

    @property
    def graph_residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.graph_residual_logit)

    @property
    def correction_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.correction_logit)

    @property
    def gate_entropy(self) -> torch.Tensor | None:
        return self._gate_entropy

    @property
    def last_boundary_logits(self) -> torch.Tensor | None:
        return self._last_boundary_logits

    def set_gumbel_temperature(self, tau: float) -> None:
        """Update Gumbel-Softmax temperature for annealing."""
        if tau <= 0.0:
            raise ValueError("gumbel_temperature must be positive")
        self.gumbel_temperature = tau

    def forward(self, x: torch.Tensor | GraphWindowBatch) -> torch.Tensor:
        edge_index: torch.Tensor | None = None
        edge_mask: torch.Tensor | None = None
        if isinstance(x, GraphWindowBatch):
            edge_index = x.edge_index
            edge_mask = x.edge_mask
            x = x.x

        if x.ndim == 4:
            batch, seq_len, _, _ = x.shape
            node_features = x
        else:
            batch, seq_len, _ = x.shape
            node_features = x.view(batch, seq_len, self.num_nodes, self.features_per_node)
        local_input = node_features.permute(0, 2, 1, 3).reshape(
            batch * self.num_nodes, seq_len, self.features_per_node
        )

        local_hidden, _ = self.temporal_encoder(local_input)
        local_hidden = local_hidden.view(
            batch, self.num_nodes, seq_len, self.encoder_output_size
        )
        local_hidden = local_hidden.permute(0, 2, 1, 3)

        correction_context: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None
        if self.communication_mode == "dense":
            neighbor_context, possible_mask = self._dense_neighbor_context(
                local_hidden, edge_index=edge_index, edge_mask=edge_mask
            )
            neighbor_belief_context = self._neighbor_belief_context(local_hidden, possible_mask) if self.use_neighbor_belief else None
            fusion_input = [local_hidden, neighbor_context]
            if neighbor_belief_context is not None:
                fusion_input.append(neighbor_belief_context)
            fused = self.fusion(torch.cat(fusion_input, dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
            correction_context = (neighbor_context, possible_mask, neighbor_belief_context)
            self._last_communication_stats = self._dense_communication_stats(
                possible_mask=possible_mask,
                batch=batch,
                seq_len=seq_len,
                device=x.device,
            )
            self._communication_loss = self._dense_communication_loss(possible_mask)
            self._gate_entropy = None
        elif self.communication_mode == "gumbel_request":
            neighbor_context, request_mask, possible_mask, soft_gate_probs = (
                self._gumbel_neighbor_context(
                    local_hidden, edge_index=edge_index, edge_mask=edge_mask
                )
            )
            neighbor_belief_context = self._neighbor_belief_context(local_hidden, request_mask) if self.use_neighbor_belief else None
            fusion_input = [local_hidden, neighbor_context]
            if neighbor_belief_context is not None:
                fusion_input.append(neighbor_belief_context)
            fused = self.fusion(torch.cat(fusion_input, dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
            correction_context = (neighbor_context, request_mask, neighbor_belief_context)
            self._last_communication_stats = self._request_communication_stats(
                request_mask=request_mask,
                possible_mask=possible_mask,
            )
            self._communication_loss = self._request_communication_loss(
                request_mask=request_mask,
                possible_mask=possible_mask,
            )
            self._gate_entropy = self._compute_gate_entropy(soft_gate_probs)
        else:
            hidden = self.dropout(local_hidden)
            self._last_communication_stats = self._zero_communication_stats()
            self._communication_loss = torch.zeros((), dtype=local_hidden.dtype, device=x.device)
            self._gate_entropy = None

        self._last_boundary_logits = self.boundary_head(hidden).squeeze(-1) if self.use_boundary_head else None
        logits = self.classifier(hidden)
        if self.use_logit_correction and correction_context is not None:
            neighbor_context, correction_mask, neighbor_belief_context = correction_context
            correction_delta = self._logit_correction(
                local_hidden=local_hidden,
                neighbor_context=neighbor_context,
                node_features=node_features,
                mask=correction_mask,
                local_logits=logits,
                neighbor_belief_context=neighbor_belief_context,
            )
            if self.use_boundary_gated_correction and self._last_boundary_logits is not None:
                boundary_gate = 1.0 + torch.sigmoid(self._last_boundary_logits).unsqueeze(-1)
                correction_delta = boundary_gate * correction_delta
            logits = logits + self.correction_scale * correction_delta
        return logits

    def get_config(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "num_nodes": self.num_nodes,
            "adjacency": self._adjacency_list,
            "edge_prob": self._edge_prob_list,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout_prob,
            "communication_mode": self.communication_mode,
            "fusion_hidden_size": self.fusion_hidden_size,
            "precision_bits": self.precision_bits,
            "gumbel_temperature": self.gumbel_temperature,
            "gate_hidden_size": self.gate_hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "graph_residual_init": self.graph_residual_init,
            "bidirectional": self.bidirectional,
            "use_logit_correction": self.use_logit_correction,
            "correction_hidden_size": self.correction_hidden_size,
            "correction_init": self.correction_init,
            "use_neighbor_belief": self.use_neighbor_belief,
            "use_boundary_head": self.use_boundary_head,
            "boundary_hidden_size": self.boundary_hidden_size,
            "use_boundary_gated_correction": self.use_boundary_gated_correction,
            "use_crf": self.use_crf,
        }

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> CESTAClassifier:
        directory = Path(path)
        meta = BaseModel.load_metadata(directory)
        config = meta["model_config"]
        assert isinstance(config, dict)
        model = cls(
            input_size=int(config["input_size"]),
            num_nodes=int(config["num_nodes"]),
            adjacency=config.get("adjacency"),  # type: ignore[arg-type]
            edge_prob=config.get("edge_prob"),  # type: ignore[arg-type]
            hidden_size=int(config.get("hidden_size", 64)),
            num_layers=int(config.get("num_layers", 1)),
            num_classes=int(config["num_classes"]),
            dropout=float(config.get("dropout", 0.2)),
            communication_mode=config.get("communication_mode", "none"),  # type: ignore[arg-type]
            fusion_hidden_size=(
                int(config["fusion_hidden_size"])
                if config.get("fusion_hidden_size") is not None
                else None
            ),
            precision_bits=int(config.get("precision_bits", 32)),
            gumbel_temperature=float(config.get("gumbel_temperature", 1.0)),
            gate_hidden_size=int(config.get("gate_hidden_size", 32)),
            num_attention_heads=int(config.get("num_attention_heads", 1)),
            graph_residual_init=float(config.get("graph_residual_init", 1.0)),
            bidirectional=bool(config.get("bidirectional", False)),
            use_logit_correction=bool(config.get("use_logit_correction", False)),
            correction_hidden_size=(
                int(config["correction_hidden_size"])
                if config.get("correction_hidden_size") is not None
                else None
            ),
            correction_init=float(config.get("correction_init", 0.1)),
            use_neighbor_belief=bool(config.get("use_neighbor_belief", False)),
            use_boundary_head=bool(config.get("use_boundary_head", False)),
            boundary_hidden_size=(
                int(config["boundary_hidden_size"])
                if config.get("boundary_hidden_size") is not None
                else None
            ),
            use_boundary_gated_correction=bool(config.get("use_boundary_gated_correction", False)),
            use_crf=bool(config.get("use_crf", False)),
        )
        model.load_state_dict(torch.load(directory / "weight.pt", weights_only=True))
        return model
