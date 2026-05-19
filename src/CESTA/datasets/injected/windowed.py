"""Unified windowed-split container and windowing utilities.

Provides ``WindowedSplits``, a single container that every data-preparation
path produces, plus the low-level sliding-window helper used internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from CESTA.schema.window import DataSplitConfig, WindowConfig


@dataclass
class WindowedSplits:
    """Unified container for windowed train/val/test arrays.

    Holds the numpy arrays produced by any data-preparation path
    (per-group windowing, graph-aligned windowing, etc.) together
    with arbitrary metadata that downstream consumers (e.g. model
    constructors) may need.

    Attributes:
        X_train: Training features ``(N, window_size, features)``.
        y_train: Training labels ``(N, window_size)``.
        X_val: Validation features.
        y_val: Validation labels.
        X_test: Test features.
        y_test: Test labels.
        metadata: Extra information (e.g. graph topology).
    """

    X_train: NDArray[np.float32]
    y_train: NDArray[np.int32]
    X_val: NDArray[np.float32]
    y_val: NDArray[np.int32]
    X_test: NDArray[np.float32]
    y_test: NDArray[np.int32]
    metadata: dict[str, Any] = field(default_factory=dict)
    split_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)
    node_mask_train: NDArray[np.bool_] | None = None
    node_mask_val: NDArray[np.bool_] | None = None
    node_mask_test: NDArray[np.bool_] | None = None
    edge_mask_train: NDArray[np.bool_] | None = None
    edge_mask_val: NDArray[np.bool_] | None = None
    edge_mask_test: NDArray[np.bool_] | None = None

    @property
    def input_size(self) -> int:
        """Return the feature dimension (last axis of X_train)."""
        if self.X_train.ndim == 4:
            return int(self.X_train.shape[-2] * self.X_train.shape[-1])
        return int(self.X_train.shape[-1])

    @property
    def has_val(self) -> bool:
        """Return whether a non-empty validation set exists."""
        return len(self.X_val) > 0

    @property
    def has_test(self) -> bool:
        """Return whether a non-empty test set exists."""
        return len(self.X_test) > 0


def create_windows(
    data: NDArray[np.float32],
    labels: NDArray[np.int32],
    window_size: int,
    stride: int,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Create sliding windows from contiguous data.

    Args:
        data: Feature array of shape ``(timesteps, features)``.
        labels: Label array of shape ``(timesteps,)`` or
            ``(timesteps, num_nodes)`` for per-node labels.
        window_size: Number of timesteps per window.
        stride: Step size between consecutive windows.

    Returns:
        Tuple of ``(X, y)`` where X has shape ``(num_windows, window_size, features)``
        and y has shape ``(num_windows, window_size, ...)``.
    """
    return create_windows_with_starts(data, labels, window_size, stride)[:2]


def create_windows_with_starts(
    data: NDArray[np.float32],
    labels: NDArray[np.int32],
    window_size: int,
    stride: int,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int64]]:
    if len(data) < window_size:
        y_shape = (0, window_size) + labels.shape[1:]
        return (
            np.empty((0, window_size) + data.shape[1:], dtype=np.float32),
            np.empty(y_shape, dtype=np.int32),
            np.empty((0,), dtype=np.int64),
        )

    starts = np.array(list(range(0, len(data) - window_size + 1, stride)), dtype=np.int64)
    X = np.stack([data[i : i + window_size] for i in starts])
    y = np.stack([labels[i : i + window_size] for i in starts])
    return X.astype(np.float32), y.astype(np.int32), starts


def split_boundaries(n: int, split: DataSplitConfig) -> tuple[int, int]:
    train_end = int(n * split.train_ratio)
    val_end = train_end + int(n * split.val_ratio) if split.val_ratio > 0 else train_end
    return train_end, val_end


def split_and_window(
    features: NDArray[np.float32],
    labels: NDArray[np.int32],
    wc: WindowConfig,
    split: DataSplitConfig,
    split_bounds: tuple[int, int, int, int] | None = None,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Chronologically split a single contiguous block and create windows."""
    if split_bounds is None:
        train_start = 0
        train_end, val_end = split_boundaries(len(features), split)
        test_end = len(features)
    else:
        train_start, train_end, val_end, test_end = split_bounds

    X_tr, y_tr = create_windows(features[train_start:train_end], labels[train_start:train_end], wc.window_size, wc.train_stride)
    X_va, y_va = create_windows(features[train_end:val_end], labels[train_end:val_end], wc.window_size, wc.test_stride)
    X_te, y_te = create_windows(features[val_end:test_end], labels[val_end:test_end], wc.window_size, wc.test_stride)

    return X_tr, y_tr, X_va, y_va, X_te, y_te


def validate_features(
    requested: list[str] | None,
    available: list[str],
) -> list[str]:
    """Validate and resolve the feature list.

    Args:
        requested: Feature names requested by the caller, or ``None`` to use all.
        available: Feature names available in the dataset.

    Returns:
        Resolved list of feature names.

    Raises:
        ValueError: If any requested name is not in *available*.
    """
    if requested is not None:
        unknown = set(requested) - set(available)
        if unknown:
            msg = f"Unknown features: {sorted(unknown)}. Available: {available}"
            raise ValueError(msg)
        return list(requested)
    return list(available)


def collect_splits(
    wc: WindowConfig,
    n_feat: int,
    train_X_parts: list[NDArray[np.float32]],
    train_y_parts: list[NDArray[np.int32]],
    val_X_parts: list[NDArray[np.float32]],
    val_y_parts: list[NDArray[np.int32]],
    test_X_parts: list[NDArray[np.float32]],
    test_y_parts: list[NDArray[np.int32]],
    label_trailing_shape: tuple[int, ...] = (),
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Concatenate per-group window parts into final arrays.

    Args:
        wc: Window configuration.
        n_feat: Number of features (last axis of X).
        train_X_parts: Per-group training feature windows.
        train_y_parts: Per-group training label windows.
        val_X_parts: Per-group validation feature windows.
        val_y_parts: Per-group validation label windows.
        test_X_parts: Per-group test feature windows.
        test_y_parts: Per-group test label windows.
        label_trailing_shape: Extra dimensions after ``window_size`` in labels
            (e.g. ``(num_nodes,)`` for per-node labels, ``()`` for scalar).

    Returns:
        Concatenated arrays; empty with correct shape when no windows exist.
    """
    y_empty = (0, wc.window_size) + label_trailing_shape
    X_train = np.concatenate(train_X_parts) if train_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_train = np.concatenate(train_y_parts) if train_y_parts else np.empty(y_empty, dtype=np.int32)
    X_val = np.concatenate(val_X_parts) if val_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_val = np.concatenate(val_y_parts) if val_y_parts else np.empty(y_empty, dtype=np.int32)
    X_test = np.concatenate(test_X_parts) if test_X_parts else np.empty((0, wc.window_size, n_feat), dtype=np.float32)
    y_test = np.concatenate(test_y_parts) if test_y_parts else np.empty(y_empty, dtype=np.int32)
    return X_train, y_train, X_val, y_val, X_test, y_test
