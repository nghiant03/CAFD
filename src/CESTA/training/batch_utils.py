"""Shared DataLoader and batch helpers for training and evaluation."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from CESTA.batch import GraphWindowBatch, TemporalWindowBatch
from CESTA.datasets.injected.graph import GraphMetadata
from CESTA.models.base import BaseModel
from CESTA.training.graph_batch import GraphWindowDataset, TemporalWindowDataset, collate_graph_batch, collate_temporal_batch


def make_window_loader(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    batch_size: int,
    *,
    shuffle: bool = False,
    metadata: dict[str, object] | None = None,
    node_mask: NDArray[np.bool_] | None = None,
    edge_mask: NDArray[np.bool_] | None = None,
    seed: int | None = None,
    node_identity_split: str = "test",
) -> DataLoader[object]:
    graph_meta = (metadata or {}).get("graph")
    if isinstance(graph_meta, GraphMetadata) and node_mask is not None and edge_mask is not None:
        dataset = GraphWindowDataset(X, y, node_mask, edge_mask, graph_meta.edge_index)
        collate_fn = collate_graph_batch
    elif isinstance(node_identity := (metadata or {}).get("node_identity"), dict):
        node_ids = node_identity.get(f"{node_identity_split}_node_ids")
        dataset = TemporalWindowDataset(X, y, node_ids)
        collate_fn = collate_temporal_batch
    else:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        collate_fn = None

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate_fn,
    )


def infer_num_classes(
    model: BaseModel,
    X: NDArray[np.float32],
    metadata: dict[str, object] | None,
    device: torch.device,
) -> int:
    graph_meta = (metadata or {}).get("graph")
    if isinstance(graph_meta, GraphMetadata) and X.ndim == 4:
        sample = GraphWindowBatch(
            x=torch.zeros(1, X.shape[1], X.shape[2], X.shape[3], device=device),
            y=torch.zeros(1, X.shape[1], X.shape[2], dtype=torch.long, device=device),
            node_mask=torch.ones(1, X.shape[1], X.shape[2], dtype=torch.bool, device=device),
            edge_index=torch.tensor(graph_meta.edge_index, dtype=torch.long, device=device),
            edge_mask=torch.ones(1, X.shape[1], graph_meta.edge_index.shape[1], dtype=torch.bool, device=device),
        )
        return int(model(sample).size(-1))
    if isinstance((metadata or {}).get("node_identity"), dict):
        sample = TemporalWindowBatch(
            x=torch.zeros(1, X.shape[1], X.shape[2], device=device),
            y=torch.zeros(1, X.shape[1], dtype=torch.long, device=device),
            node_ids=torch.zeros(1, dtype=torch.long, device=device),
        )
        return int(model(sample).size(-1))
    return int(model(torch.zeros(1, X.shape[1], X.shape[2], device=device)).size(-1))


def prepare_batch(
    batch: object,
    device: torch.device,
) -> tuple[torch.Tensor | GraphWindowBatch | TemporalWindowBatch, torch.Tensor, torch.Tensor | None, int]:
    if isinstance(batch, GraphWindowBatch):
        graph_batch = GraphWindowBatch(
            x=batch.x.to(device),
            y=batch.y.to(device),
            node_mask=batch.node_mask.to(device),
            edge_index=batch.edge_index.to(device),
            edge_mask=batch.edge_mask.to(device),
        )
        return graph_batch, graph_batch.y, graph_batch.node_mask, graph_batch.x.size(0)
    if isinstance(batch, TemporalWindowBatch):
        temporal_batch = TemporalWindowBatch(
            x=batch.x.to(device),
            y=batch.y.to(device),
            node_ids=batch.node_ids.to(device) if batch.node_ids is not None else None,
        )
        return temporal_batch, temporal_batch.y, None, temporal_batch.x.size(0)
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError("Expected a tensor batch, TemporalWindowBatch, or GraphWindowBatch")
    X_batch = batch[0].to(device)
    y_batch = batch[1].to(device)
    return X_batch, y_batch, None, X_batch.size(0)
