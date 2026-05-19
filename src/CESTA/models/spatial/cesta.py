"""CESTA spatial-temporal model for communication-aware fault diagnosis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, Literal, TypedDict, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from CESTA.batch import GraphWindowBatch
from CESTA.models.base import BaseModel

CommunicationMode = Literal["none", "dense", "gumbel_request"]


class CommunicationStats(TypedDict):
    """Communication statistics captured from the most recent forward pass."""

    active_request_ratio: float
    requested_edge_count: float
    possible_edge_count: float
    transmitted_bits_estimate: float
    full_embedding_message_count: float
    compressed_message_count: float
    average_compression_ratio: float


class CESTAClassifier(BaseModel):
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
        self.encoder_output_size = hidden_size * (2 if bidirectional else 1)

        self._gate_entropy: torch.Tensor | None = None

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
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder_output_size * 2, fusion_output_size),
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
    def gate_entropy(self) -> torch.Tensor | None:
        return self._gate_entropy

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
            local_input = x
        else:
            batch, seq_len, _ = x.shape
            local_input = x.view(batch, seq_len, self.num_nodes, self.features_per_node)
        local_input = local_input.permute(0, 2, 1, 3).reshape(
            batch * self.num_nodes, seq_len, self.features_per_node
        )

        local_hidden, _ = self.temporal_encoder(local_input)
        local_hidden = local_hidden.view(
            batch, self.num_nodes, seq_len, self.encoder_output_size
        )
        local_hidden = local_hidden.permute(0, 2, 1, 3)

        if self.communication_mode == "dense":
            neighbor_context, possible_mask = self._dense_neighbor_context(
                local_hidden, edge_index=edge_index, edge_mask=edge_mask
            )
            fused = self.fusion(torch.cat([local_hidden, neighbor_context], dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
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
            fused = self.fusion(torch.cat([local_hidden, neighbor_context], dim=-1))
            hidden = self.dropout(local_hidden + self.graph_residual_scale * fused)
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

        return self.classifier(hidden)

    def _dense_neighbor_context(
        self,
        local_hidden: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        possible_mask = self._possible_message_mask(
            local_hidden, edge_index=edge_index, edge_mask=edge_mask
        )
        return self._gat_aggregate(local_hidden, possible_mask), possible_mask

    def _gumbel_neighbor_context(
        self,
        local_hidden: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        possible_mask = self._possible_message_mask(
            local_hidden, edge_index=edge_index, edge_mask=edge_mask
        )
        edge_features = self._edge_gate_features(
            local_hidden, possible_mask, edge_index=edge_index
        )
        gate_logits = self.request_gate(edge_features)

        soft_gate_probs = F.softmax(gate_logits, dim=-1)

        if self.training:
            gate_probs = F.gumbel_softmax(
                gate_logits,
                tau=self.gumbel_temperature,
                hard=True,
                dim=-1,
            )
        else:
            gate_probs = F.one_hot(gate_logits.argmax(dim=-1), num_classes=2).to(
                local_hidden.dtype
            )

        request_mask = gate_probs[..., 1] * possible_mask
        neighbor_context = self._gat_aggregate(local_hidden, request_mask)
        return neighbor_context, request_mask, possible_mask, soft_gate_probs

    def _gat_aggregate(
        self,
        local_hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """GAT-inspired single-head attention aggregation over received neighbors.

        Args:
            local_hidden: ``(batch, window, num_nodes, hidden_size)``.
            mask: ``(num_nodes, num_nodes)`` for dense static neighbors or
                ``(batch, window, num_nodes, num_nodes)`` for Gumbel dynamic.
                1 where neighbor j is requested by receiver i.

        Returns:
            ``(batch, window, num_nodes, hidden_size)`` attention-weighted
            neighbor context. Zero vector when no neighbors requested.
        """
        B, T, N, H = local_hidden.shape

        Q = self.W_q(local_hidden)   # (B, T, N, H)
        K = self.W_k(local_hidden)   # (B, T, N, H)
        V = self.W_v(local_hidden)   # (B, T, N, H)

        scores = torch.einsum("btih,btjh->btij", Q, K) / self.attention_scale

        if mask.dim() == 2:
            mask_expanded = mask.view(1, 1, N, N).expand(B, T, N, N)
        else:
            mask_expanded = mask

        scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        has_neighbors = mask_expanded.sum(dim=-1, keepdim=True) > 0  # (B, T, N, 1)
        alpha = F.softmax(scores, dim=-1)                             # (B, T, N, N)
        alpha = torch.where(has_neighbors, alpha, torch.zeros_like(alpha))

        return torch.einsum("btij,btjh->btih", alpha, V)

    def _edge_gate_features(
        self,
        local_hidden: torch.Tensor,
        possible_mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, H = local_hidden.shape
        local_logits = self.classifier(local_hidden).detach()
        local_probs = F.softmax(local_logits, dim=-1)
        entropy = -(local_probs * torch.log(local_probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        if self.num_classes > 1:
            top2 = local_probs.topk(k=2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
        else:
            margin = torch.ones(B, T, N, 1, dtype=local_hidden.dtype, device=local_hidden.device)
        receiver_state = local_hidden.unsqueeze(3).expand(B, T, N, N, H)
        receiver_entropy = entropy.unsqueeze(3).expand(B, T, N, N, 1)
        receiver_margin = margin.unsqueeze(3).expand(B, T, N, N, 1)
        edge_prob = cast(torch.Tensor, self.edge_prob).to(device=local_hidden.device, dtype=local_hidden.dtype)
        edge_prob_features = edge_prob.view(1, 1, N, N, 1).expand(B, T, N, N, 1)
        return torch.cat([receiver_state, receiver_entropy, receiver_margin, edge_prob_features], dim=-1) * possible_mask.unsqueeze(-1)

    def _possible_message_mask(
        self,
        local_hidden: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, _ = local_hidden.shape
        device = local_hidden.device
        if edge_index is None or edge_mask is None:
            adjacency = cast(torch.Tensor, self.adjacency).to(device)
            message_mask = adjacency.clone()
            message_mask.fill_diagonal_(0.0)
            return message_mask.view(1, 1, N, N).expand(B, T, N, N)

        message_mask = torch.zeros(B, T, N, N, dtype=local_hidden.dtype, device=device)
        sender = edge_index[0].to(device)
        receiver = edge_index[1].to(device)
        active = edge_mask.to(device=device, dtype=local_hidden.dtype)
        message_mask[:, :, receiver, sender] = active
        return message_mask

    def _zero_communication_stats(self) -> CommunicationStats:
        return {
            "active_request_ratio": 0.0,
            "requested_edge_count": 0.0,
            "possible_edge_count": self._possible_edge_count(),
            "transmitted_bits_estimate": 0.0,
            "full_embedding_message_count": 0.0,
            "compressed_message_count": 0.0,
            "average_compression_ratio": 0.0,
        }

    def _dense_communication_stats(
        self,
        possible_mask: torch.Tensor,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> CommunicationStats:
        if possible_mask.dim() == 2:
            possible_edges = torch.tensor(
                float(possible_mask.sum().item() * batch * seq_len),
                dtype=torch.float32,
                device=device,
            )
        else:
            possible_edges = possible_mask.sum()
        requested_edges = possible_edges
        transmitted_bits = requested_edges * self.encoder_output_size * self.precision_bits
        return {
            "active_request_ratio": 1.0 if possible_edges.detach().cpu().item() > 0 else 0.0,
            "requested_edge_count": float(requested_edges.detach().cpu().item()),
            "possible_edge_count": float(possible_edges.detach().cpu().item()),
            "transmitted_bits_estimate": float(transmitted_bits.detach().cpu().item()),
            "full_embedding_message_count": float(requested_edges.detach().cpu().item()),
            "compressed_message_count": 0.0,
            "average_compression_ratio": 1.0 if possible_edges.detach().cpu().item() > 0 else 0.0,
        }

    def _request_communication_stats(
        self,
        request_mask: torch.Tensor,
        possible_mask: torch.Tensor,
    ) -> CommunicationStats:
        possible_edges = possible_mask.sum()
        requested_edges = request_mask.sum()
        active_ratio = requested_edges / possible_edges.clamp_min(1.0)
        transmitted_bits = requested_edges * self.encoder_output_size * self.precision_bits
        return {
            "active_request_ratio": float(active_ratio.detach().cpu().item()),
            "requested_edge_count": float(requested_edges.detach().cpu().item()),
            "possible_edge_count": float(possible_edges.detach().cpu().item()),
            "transmitted_bits_estimate": float(transmitted_bits.detach().cpu().item()),
            "full_embedding_message_count": float(
                requested_edges.detach().cpu().item()
            ),
            "compressed_message_count": 0.0,
            "average_compression_ratio": 1.0
            if requested_edges.detach().cpu().item() > 0
            else 0.0,
        }

    @staticmethod
    def _request_communication_loss(
        request_mask: torch.Tensor,
        possible_mask: torch.Tensor,
    ) -> torch.Tensor:
        possible_edges = possible_mask.sum().clamp_min(1.0)
        return request_mask.sum() / possible_edges

    @staticmethod
    def _dense_communication_loss(possible_mask: torch.Tensor) -> torch.Tensor:
        possible_edges = possible_mask.sum()
        return possible_edges / possible_edges.clamp_min(1.0)

    def _compute_gate_entropy(
        self,
        soft_gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy of gate probabilities averaged over all nodes and timesteps."""
        log_probs = torch.log(soft_gate_probs.clamp_min(1e-8))
        entropy_per_element = -(soft_gate_probs * log_probs).sum(dim=-1)
        return entropy_per_element.mean()

    def _possible_edge_count(self, device: torch.device | None = None) -> float:
        adjacency = cast(torch.Tensor, self.adjacency)
        if device is not None:
            adjacency = adjacency.to(device)
        message_mask = adjacency.clone()
        message_mask.fill_diagonal_(0.0)
        return float(message_mask.sum().item())

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
        )
        model.load_state_dict(torch.load(directory / "weight.pt", weights_only=True))
        return model
