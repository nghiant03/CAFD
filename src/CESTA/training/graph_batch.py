"""Graph-window dataset and collate utilities."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from CESTA.batch import GraphWindowBatch, TemporalWindowBatch


class GraphWindowDataset(Dataset[GraphWindowBatch]):
    def __init__(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        node_mask: NDArray[np.bool_],
        edge_mask: NDArray[np.bool_],
        edge_index: NDArray[np.int64],
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.node_mask = torch.tensor(node_mask, dtype=torch.bool)
        self.edge_mask = torch.tensor(edge_mask, dtype=torch.bool)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> GraphWindowBatch:
        return GraphWindowBatch(
            x=self.X[index],
            y=self.y[index],
            node_mask=self.node_mask[index],
            edge_index=self.edge_index,
            edge_mask=self.edge_mask[index],
        )


def collate_graph_batch(items: list[GraphWindowBatch]) -> GraphWindowBatch:
    return GraphWindowBatch(
        x=torch.stack([item.x for item in items]),
        y=torch.stack([item.y for item in items]),
        node_mask=torch.stack([item.node_mask for item in items]),
        edge_index=items[0].edge_index,
        edge_mask=torch.stack([item.edge_mask for item in items]),
    )


class TemporalWindowDataset(Dataset[TemporalWindowBatch]):
    def __init__(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        node_ids: NDArray[np.int64] | None = None,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.node_ids = torch.tensor(node_ids, dtype=torch.long) if node_ids is not None else None

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> TemporalWindowBatch:
        return TemporalWindowBatch(
            x=self.X[index],
            y=self.y[index],
            node_ids=self.node_ids[index] if self.node_ids is not None else None,
        )


def collate_temporal_batch(items: list[TemporalWindowBatch]) -> TemporalWindowBatch:
    node_ids = None
    if items[0].node_ids is not None:
        node_tensors = [item.node_ids for item in items if item.node_ids is not None]
        node_ids = torch.stack(node_tensors)
    return TemporalWindowBatch(
        x=torch.stack([item.x for item in items]),
        y=torch.stack([item.y for item in items]),
        node_ids=node_ids,
    )
