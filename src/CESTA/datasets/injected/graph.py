"""Dynamic directed graph dataset construction for sensor networks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from CESTA.datasets.injected.tabular import InjectedDataset
from CESTA.datasets.injected.windowed import (
    WindowedSplits,
    create_windows_with_starts,
    split_boundaries,
    validate_features,
)
from CESTA.logging import logger
from CESTA.schema.window import DataSplitConfig, WindowConfig


@dataclass
class GraphMetadata:
    edge_index: NDArray[np.int64]
    edge_prob: NDArray[np.float32]
    node_ids: list[int]
    num_nodes: int
    threshold: float
    edge_convention: str = "sender_to_receiver"
    dynamic_link_seed: int | None = None
    burst_params: dict[str, float] = field(default_factory=dict)
    timestamps: list[Any] = field(default_factory=list)
    link_mask_shape: tuple[int, int] | None = None
    adjacency: NDArray[np.float32] | None = None

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def load_directed_edges(
    connectivity_path: str | Path,
    node_ids: list[int],
    threshold: float = 0.5,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    connectivity_path = Path(connectivity_path)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    edges: list[tuple[int, int]] = []
    probs: list[float] = []

    with connectivity_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            src, dst = int(parts[0]), int(parts[1])
            prob = float(parts[2])
            if src == dst or prob < threshold:
                continue
            if src in id_to_idx and dst in id_to_idx:
                edges.append((id_to_idx[src], id_to_idx[dst]))
                probs.append(prob)

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_prob = np.asarray(probs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_prob = np.empty((0,), dtype=np.float32)

    logger.info(
        "Graph: {} nodes, {} directed edges (threshold={:.2f})",
        len(node_ids),
        edge_index.shape[1],
        threshold,
    )
    return edge_index, edge_prob


def load_adjacency_matrix(
    connectivity_path: str | Path,
    node_ids: list[int],
    threshold: float = 0.5,
) -> NDArray[np.float32]:
    edge_index, _ = load_directed_edges(connectivity_path, node_ids, threshold)
    adj = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
        adj[edge_index[1], edge_index[0]] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def simulate_bursty_link_mask(
    num_timestamps: int,
    edge_index: NDArray[np.int64],
    edge_prob: NDArray[np.float32],
    *,
    seed: int,
    rho: float,
    q_bad_base: float,
    q_recover_base: float,
    bad_success_floor: float,
) -> NDArray[np.bool_]:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if not 0.0 <= bad_success_floor <= 1.0:
        raise ValueError("bad_success_floor must be in [0, 1]")

    num_edges = edge_index.shape[1]
    rng = np.random.default_rng(seed)
    link_mask = np.zeros((num_timestamps, num_edges), dtype=np.bool_)
    if num_timestamps == 0 or num_edges == 0:
        return link_mask

    q_bad_env = q_bad_base * rho * (1.0 - edge_prob)
    q_recover_env = q_recover_base * rho * edge_prob
    q_bad_dir = q_bad_base * (1.0 - rho) * (1.0 - edge_prob)
    q_recover_dir = q_recover_base * (1.0 - rho) * edge_prob

    pair_keys: list[tuple[int, int]] = [
        (min(int(s), int(r)), max(int(s), int(r))) for s, r in edge_index.T
    ]
    unique_pairs = sorted(set(pair_keys))
    pair_to_idx = {pair: idx for idx, pair in enumerate(unique_pairs)}
    pair_edge_indices: dict[tuple[int, int], list[int]] = {pair: [] for pair in unique_pairs}
    for edge_idx, pair in enumerate(pair_keys):
        pair_edge_indices[pair].append(edge_idx)

    env_state = np.zeros((len(unique_pairs),), dtype=np.bool_)
    for pair, pair_idx in pair_to_idx.items():
        edge_indices = pair_edge_indices[pair]
        qb = float(np.mean(q_bad_env[edge_indices]))
        qr = float(np.mean(q_recover_env[edge_indices]))
        bad_prob = qb / max(qb + qr, 1e-12)
        env_state[pair_idx] = rng.random() < bad_prob

    dir_bad_prob = q_bad_dir / np.maximum(q_bad_dir + q_recover_dir, 1e-12)
    dir_state = rng.random(num_edges) < dir_bad_prob

    for t in range(num_timestamps):
        if t > 0:
            for pair, pair_idx in pair_to_idx.items():
                edge_indices = pair_edge_indices[pair]
                qb = float(np.mean(q_bad_env[edge_indices]))
                qr = float(np.mean(q_recover_env[edge_indices]))
                if env_state[pair_idx]:
                    env_state[pair_idx] = not (rng.random() < qr)
                else:
                    env_state[pair_idx] = rng.random() < qb

            recover = rng.random(num_edges) < q_recover_dir
            fail = rng.random(num_edges) < q_bad_dir
            dir_state = np.where(dir_state, ~recover, fail)

        env_bad = np.asarray([env_state[pair_to_idx[pair]] for pair in pair_keys], dtype=np.bool_)
        effective_bad = env_bad | dir_state
        success_prob = np.where(effective_bad, bad_success_floor, edge_prob)
        link_mask[t] = rng.random(num_edges) < success_prob

    return link_mask


def pack_link_mask(mask: NDArray[np.bool_]) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    return np.packbits(mask.reshape(-1)), np.asarray(mask.shape, dtype=np.int64)


def unpack_link_mask(packed: NDArray[np.uint8], shape: tuple[int, int] | NDArray[np.integer[Any]]) -> NDArray[np.bool_]:
    shape_tuple = tuple(int(x) for x in shape)
    total = int(np.prod(shape_tuple))
    return np.unpackbits(packed)[:total].reshape(shape_tuple).astype(np.bool_)


@dataclass
class GraphDataset(InjectedDataset):
    edge_index: NDArray[np.int64] = field(default_factory=lambda: np.empty((2, 0), dtype=np.int64))
    edge_prob: NDArray[np.float32] = field(default_factory=lambda: np.empty((0,), dtype=np.float32))
    node_ids: list[int] = field(default_factory=list)
    threshold: float = 0.5
    link_mask: NDArray[np.bool_] = field(default_factory=lambda: np.empty((0, 0), dtype=np.bool_))
    graph_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def adjacency(self) -> NDArray[np.float32]:
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        if self.edge_index.shape[1] > 0:
            adj[self.edge_index[0], self.edge_index[1]] = 1.0
        return adj

    def save(self, path: str | Path) -> None:
        super().save(path)
        directory = Path(path)
        np.savez_compressed(
            directory / "graph_edges.npz",
            edge_index=self.edge_index.astype(np.int64),
            edge_prob=self.edge_prob.astype(np.float32),
            node_ids=np.asarray(self.node_ids, dtype=np.int64),
        )
        packed, shape = pack_link_mask(self.link_mask)
        np.savez_compressed(directory / "dynamic_link_mask.npz", link_mask=packed, shape=shape)
        meta = {
            "node_ids": self.node_ids,
            "threshold": self.threshold,
            "edge_convention": "sender_to_receiver",
            "edge_count": int(self.edge_index.shape[1]),
            "num_nodes": self.num_nodes,
            "link_mask_shape": list(self.link_mask.shape),
            **self.graph_meta,
        }
        (directory / "dynamic_graph_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        legacy_adj = directory / "adjacency.npy"
        legacy_meta = directory / "graph_meta.json"
        if legacy_adj.exists():
            legacy_adj.unlink()
        if legacy_meta.exists():
            legacy_meta.unlink()

    @classmethod
    def load(cls, path: str | Path) -> GraphDataset:
        directory = Path(path)
        parent = InjectedDataset.load(directory)
        if (directory / "graph_edges.npz").exists():
            edges = np.load(directory / "graph_edges.npz")
            edge_index = edges["edge_index"].astype(np.int64)
            edge_prob = edges["edge_prob"].astype(np.float32)
            node_ids = [int(x) for x in edges["node_ids"].tolist()]
            link_payload = np.load(directory / "dynamic_link_mask.npz")
            link_mask = unpack_link_mask(link_payload["link_mask"], link_payload["shape"])
            meta = json.loads((directory / "dynamic_graph_meta.json").read_text())
            threshold = float(meta.get("threshold", 0.5))
        else:
            adjacency: NDArray[np.float32] = np.load(directory / "adjacency.npy")
            meta = json.loads((directory / "graph_meta.json").read_text())
            node_ids = [int(x) for x in meta["node_ids"]]
            edge_index = adjacency.astype(bool).nonzero()
            keep = edge_index[0] != edge_index[1]
            edge_index = np.asarray(edge_index, dtype=np.int64)[:, keep]
            edge_prob = np.ones((edge_index.shape[1],), dtype=np.float32)
            timestamps = sorted(parent.df["timestamp"].unique())
            link_mask = np.ones((len(timestamps), edge_index.shape[1]), dtype=np.bool_)
            threshold = float(meta["threshold"])

        return cls(
            df=parent.df,
            config=parent.config,
            feature_names=parent.feature_names,
            edge_index=edge_index,
            edge_prob=edge_prob,
            node_ids=node_ids,
            threshold=threshold,
            link_mask=link_mask,
            graph_meta=meta,
        )

    @classmethod
    def from_connectivity(
        cls,
        path: str | Path,
        connectivity_path: str | Path,
        threshold: float = 0.5,
        seed: int = 0,
        rho: float = 0.5,
        q_bad_base: float = 0.02,
        q_recover_base: float = 0.20,
        bad_success_floor: float = 0.05,
    ) -> GraphDataset:
        parent = InjectedDataset.load(path)
        df = parent.df
        group_col = parent.group_column
        node_ids = sorted(int(g) for g in df[group_col].unique())
        edge_index, edge_prob = load_directed_edges(connectivity_path, node_ids, threshold=threshold)
        timestamps = sorted(df["timestamp"].unique())
        link_mask = simulate_bursty_link_mask(
            len(timestamps),
            edge_index,
            edge_prob,
            seed=seed,
            rho=rho,
            q_bad_base=q_bad_base,
            q_recover_base=q_recover_base,
            bad_success_floor=bad_success_floor,
        )
        graph_meta = {
            "seed": seed,
            "timestamps": [str(ts) for ts in timestamps],
            "burst_params": {
                "rho": rho,
                "q_bad_base": q_bad_base,
                "q_recover_base": q_recover_base,
                "bad_success_floor": bad_success_floor,
            },
            "masks_applied": True,
        }
        return cls(
            df=df,
            config=parent.config,
            feature_names=parent.feature_names,
            edge_index=edge_index,
            edge_prob=edge_prob,
            node_ids=node_ids,
            threshold=threshold,
            link_mask=link_mask,
            graph_meta=graph_meta,
        )

    def prepare(
        self,
        window_config: WindowConfig | None = None,
        split_config: DataSplitConfig | None = None,
        features: list[str] | None = None,
        required_metadata: set[str] | None = None,
    ) -> WindowedSplits:
        if required_metadata is not None and "graph" not in required_metadata:
            return InjectedDataset.prepare(
                self,
                window_config=window_config,
                split_config=split_config,
                features=features,
                required_metadata=required_metadata,
            )
        wc = window_config if window_config is not None else WindowConfig()
        split = split_config if split_config is not None else DataSplitConfig(strategy="connectivity-chronological")
        selected_features = validate_features(features, self.feature_names)
        if selected_features != ["temp"]:
            raise ValueError('Dynamic graph preparation currently supports only features=["temp"]')

        df = self.df
        group_col = self.group_column
        node_ids = self.node_ids
        timestamps = sorted(df["timestamp"].unique())
        ts_index = {ts: i for i, ts in enumerate(timestamps)}
        node_index = {nid: i for i, nid in enumerate(node_ids)}
        T = len(timestamps)
        N = len(node_ids)

        X = np.zeros((T, N, 1), dtype=np.float32)
        y = np.full((T, N), -1, dtype=np.int32)
        node_mask = np.zeros((T, N), dtype=np.bool_)

        for row in df[["timestamp", group_col, "temp", "fault_state"]].itertuples(index=False):
            t = ts_index[getattr(row, "timestamp")]
            n = node_index[int(getattr(row, group_col))]
            X[t, n, 0] = np.float32(getattr(row, "temp"))
            y[t, n] = np.int32(getattr(row, "fault_state"))
            node_mask[t, n] = True

        edge_mask_all = self._available_edge_mask(node_mask)
        train_start, train_end, val_end, test_end = self._split_boundaries(T, edge_mask_all, wc, split)
        X_train, y_train, train_starts = create_windows_with_starts(
            X[train_start:train_end], y[train_start:train_end], wc.window_size, wc.train_stride
        )
        X_val, y_val, val_starts = create_windows_with_starts(
            X[train_end:val_end], y[train_end:val_end], wc.window_size, wc.test_stride
        )
        X_test, y_test, test_starts = create_windows_with_starts(
            X[val_end:test_end], y[val_end:test_end], wc.window_size, wc.test_stride
        )
        train_starts = train_starts + train_start
        val_starts = val_starts + train_end
        test_starts = test_starts + val_end

        metadata = GraphMetadata(
            edge_index=self.edge_index,
            edge_prob=self.edge_prob,
            node_ids=self.node_ids,
            num_nodes=self.num_nodes,
            threshold=self.threshold,
            dynamic_link_seed=self.graph_meta.get("seed"),
            burst_params=dict(self.graph_meta.get("burst_params", {})),
            timestamps=[str(ts) for ts in timestamps],
            link_mask_shape=tuple(self.link_mask.shape),
            adjacency=self.adjacency,
        )

        return WindowedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            metadata={"graph": metadata},
            node_mask_train=self._window_by_starts(node_mask, train_starts, wc.window_size),
            node_mask_val=self._window_by_starts(node_mask, val_starts, wc.window_size),
            node_mask_test=self._window_by_starts(node_mask, test_starts, wc.window_size),
            edge_mask_train=self._window_by_starts(edge_mask_all, train_starts, wc.window_size),
            edge_mask_val=self._window_by_starts(edge_mask_all, val_starts, wc.window_size),
            edge_mask_test=self._window_by_starts(edge_mask_all, test_starts, wc.window_size),
        )

    def _available_edge_mask(self, node_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if self.link_mask.shape != (node_mask.shape[0], self.edge_index.shape[1]):
            raise ValueError(
                f"link_mask shape {self.link_mask.shape} does not match "
                f"(T,E)=({node_mask.shape[0]},{self.edge_index.shape[1]})"
            )
        if self.edge_index.shape[1] == 0:
            return self.link_mask.copy()
        sender = self.edge_index[0]
        receiver = self.edge_index[1]
        return self.link_mask & node_mask[:, sender] & node_mask[:, receiver]

    def _split_boundaries(
        self,
        num_timestamps: int,
        edge_mask: NDArray[np.bool_],
        wc: WindowConfig,
        split: DataSplitConfig,
    ) -> tuple[int, int, int, int]:
        target_train_end, target_val_end = split_boundaries(num_timestamps, split)
        if split.strategy == "chronological":
            return 0, target_train_end, target_val_end, num_timestamps

        active_timesteps = np.flatnonzero(edge_mask.any(axis=1))
        if len(active_timesteps) == 0:
            msg = (
                "Unable to create connectivity-chronological graph split: no active graph edges were found. "
                f"num_timestamps={num_timestamps}, window_size={wc.window_size}, edge_count={edge_mask.shape[1]}"
            )
            raise ValueError(msg)

        active_start = int(active_timesteps[0])
        active_end = int(active_timesteps[-1]) + 1
        active_len = active_end - active_start
        train_end_rel, val_end_rel = split_boundaries(active_len, split)
        train_end = active_start + train_end_rel
        val_end = active_start + val_end_rel
        test_end = active_end

        train_ok = self._split_has_available_edges(edge_mask, active_start, train_end, wc.train_stride, wc.window_size)
        val_ok = self._split_has_available_edges(edge_mask, train_end, val_end, wc.test_stride, wc.window_size)
        test_ok = self._split_has_available_edges(edge_mask, val_end, test_end, wc.test_stride, wc.window_size)
        if not (train_ok and val_ok and test_ok):
            msg = (
                "Unable to create connectivity-chronological graph split: "
                "train, validation, and test splits must each contain at least one window with available graph edges. "
                f"num_timestamps={num_timestamps}, window_size={wc.window_size}, train_ratio={split.train_ratio}, "
                f"val_ratio={split.val_ratio}, test_ratio={split.test_ratio}, active_start={active_start}, "
                f"active_end={active_end}, train_end={train_end}, val_end={val_end}, "
                f"active_edge_timesteps={int(edge_mask.any(axis=1).sum())}, "
                f"train_has_edges={train_ok}, val_has_edges={val_ok}, test_has_edges={test_ok}"
            )
            raise ValueError(msg)
        return active_start, train_end, val_end, test_end

    @staticmethod
    def _split_has_available_edges(
        edge_mask: NDArray[np.bool_],
        start: int,
        end: int,
        stride: int,
        window_size: int,
    ) -> bool:
        if end - start < window_size:
            return False
        starts = range(start, end - window_size + 1, stride)
        return any(bool(edge_mask[i : i + window_size].any()) for i in starts)

    @staticmethod
    def _window_by_starts(
        values: NDArray[np.bool_],
        starts: NDArray[np.int64],
        window_size: int,
    ) -> NDArray[np.bool_]:
        if len(starts) == 0:
            return np.empty((0, window_size) + values.shape[1:], dtype=np.bool_)
        return np.stack([values[i : i + window_size] for i in starts]).astype(np.bool_)
