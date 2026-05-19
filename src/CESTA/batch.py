"""Runtime batch contracts shared by trainers, evaluators, and models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphWindowBatch:
    x: torch.Tensor
    y: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_mask: torch.Tensor


@dataclass
class TemporalWindowBatch:
    x: torch.Tensor
    y: torch.Tensor
    node_ids: torch.Tensor | None = None
