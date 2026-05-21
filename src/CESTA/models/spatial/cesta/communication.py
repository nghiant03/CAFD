from __future__ import annotations

from typing import Any, TypedDict, cast

import torch
import torch.nn.functional as F


class CommunicationStats(TypedDict):
    active_request_ratio: float
    requested_edge_count: float
    possible_edge_count: float
    transmitted_bits_estimate: float
    full_embedding_message_count: float
    compressed_message_count: float
    average_compression_ratio: float


class CESTACommunicationMixin:
    request_gate: Any
    training: bool
    gumbel_temperature: float
    W_q: Any
    W_k: Any
    W_v: Any
    attention_scale: float
    num_classes: int
    classifier: Any
    logit_correction: Any
    edge_prob: torch.Tensor
    adjacency: torch.Tensor
    precision_bits: int
    encoder_output_size: int
    neighbor_belief_size: int
    use_neighbor_belief: bool

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

    def _masked_neighbor_mean(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, _ = values.shape
        if mask.dim() == 2:
            mask_expanded = mask.view(1, 1, N, N).expand(B, T, N, N)
        else:
            mask_expanded = mask
        weights = mask_expanded.to(dtype=values.dtype)
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.einsum("btij,btjf->btif", weights, values) / count

    def _belief_features(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.detach(), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1, keepdim=True)
        if self.num_classes > 1:
            top2 = probs.topk(k=2, dim=-1).values
            margin = (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
        else:
            margin = torch.ones_like(entropy)
        return torch.cat([probs, entropy, margin], dim=-1)

    def _neighbor_belief_context(
        self,
        local_hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._masked_neighbor_mean(self._belief_features(self.classifier(local_hidden)), mask)

    def _logit_correction(
        self,
        local_hidden: torch.Tensor,
        neighbor_context: torch.Tensor,
        node_features: torch.Tensor,
        mask: torch.Tensor,
        local_logits: torch.Tensor,
        neighbor_belief_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        neighbor_features = self._masked_neighbor_mean(node_features, mask)
        local_belief = self._belief_features(local_logits)
        correction_input_parts = [
            local_hidden,
            neighbor_context,
            local_hidden - neighbor_context,
            node_features,
            neighbor_features,
            node_features - neighbor_features,
            local_belief,
        ]
        if neighbor_belief_context is not None:
            correction_input_parts.extend(
                [
                    neighbor_belief_context,
                    local_belief - neighbor_belief_context,
                    local_belief * neighbor_belief_context,
                ]
            )
        correction_input = torch.cat(
            correction_input_parts,
            dim=-1,
        )
        return self.logit_correction(correction_input)

    def _edge_gate_features(
        self,
        local_hidden: torch.Tensor,
        possible_mask: torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, H = local_hidden.shape
        local_belief = self._belief_features(self.classifier(local_hidden))
        entropy = local_belief[..., self.num_classes : self.num_classes + 1]
        margin = local_belief[..., self.num_classes + 1 : self.num_classes + 2]
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
        transmitted_bits = requested_edges * self._message_size() * self.precision_bits
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
        transmitted_bits = requested_edges * self._message_size() * self.precision_bits
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

    def _message_size(self) -> int:
        return self.encoder_output_size + (self.neighbor_belief_size if self.use_neighbor_belief else 0)

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
