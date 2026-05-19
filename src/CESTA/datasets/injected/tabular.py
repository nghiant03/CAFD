"""Injected dataset container.

This module defines the InjectedDataset class which is the output of
the injection pipeline. It stores the full injected DataFrame and provides
a ``.prepare()`` method for windowed train/val/test splitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from CESTA.datasets.injected.windowed import (
    WindowedSplits,
    collect_splits,
    split_and_window,
    validate_features,
)
from CESTA.schema import InjectionConfig
from CESTA.schema.fault import FaultType
from CESTA.schema.manifest import DatasetInfo
from CESTA.schema.window import DataSplitConfig, WindowConfig
from CESTA.utils import sha256_file


@dataclass
class InjectedDataset:
    """Container for the injected dataset (raw DataFrame, no windowing).

    This is the output of the injection pipeline. The DataFrame contains
    the injected sensor data with a ``fault_state`` column holding per-row
    fault labels.

    Attributes:
        df: Full injected DataFrame including ``fault_state`` column.
        config: The configuration used to generate this dataset.
        feature_names: Names of feature columns (excludes group/fault_state).
    """

    df: pd.DataFrame
    config: InjectionConfig
    feature_names: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Save dataset to a directory with CSV data and JSON metadata."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(directory / "injected_data.csv", index=False)

        meta = {
            "config": self.config.model_dump(mode="json"),
            "feature_names": self.feature_names,
        }
        (directory / "injected_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> InjectedDataset:
        """Load dataset from directory."""
        directory = Path(path)

        meta_path = directory / "injected_meta.json"
        meta = json.loads(meta_path.read_text())
        config = InjectionConfig.model_validate(meta["config"])
        feature_names: list[str] = meta["feature_names"]

        df = pd.read_csv(directory / "injected_data.csv")

        for col in feature_names:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)
        if "fault_state" in df.columns:
            df["fault_state"] = df["fault_state"].astype(np.int32)

        return cls(
            df=df,
            config=config,
            feature_names=feature_names,
        )

    @property
    def group_column(self) -> str:
        """Return the group column name from config."""
        return self.config.group_column

    @property
    def num_groups(self) -> int:
        """Return the number of sensor groups."""
        if self.group_column in self.df.columns:
            return self.df[self.group_column].nunique()
        return 1

    @property
    def total_timesteps(self) -> int:
        """Return total number of timesteps."""
        return len(self.df)

    @property
    def num_features(self) -> int:
        """Return the number of features."""
        return len(self.feature_names)

    def describe(self, path: str | Path) -> DatasetInfo:
        """Return a :class:`DatasetInfo` snapshot of this dataset on disk.

        Hashes ``injected_data.csv`` and ``injected_meta.json`` under
        ``path`` (if present) and bundles feature/group counts that the
        dataset already knows.

        Args:
            path: Directory the dataset was loaded from / saved to.
        """
        directory = Path(path)
        data_file = directory / "injected_data.csv"
        meta_file = directory / "injected_meta.json"
        return DatasetInfo(
            path=str(directory.resolve()),
            data_sha256=sha256_file(data_file) if data_file.exists() else None,
            meta_sha256=sha256_file(meta_file) if meta_file.exists() else None,
            num_features=self.num_features,
            feature_names=list(self.feature_names),
            num_groups=self.num_groups,
            total_timesteps=self.total_timesteps,
        )

    def print_summary(self) -> None:
        """Print dataset summary statistics using rich formatting."""
        console = Console()

        info_table = Table(title="Injected Dataset Summary", show_header=True)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        info_table.add_row("Groups", str(self.num_groups))
        info_table.add_row("Total timesteps", f"{self.total_timesteps:,}")
        info_table.add_row("Features", str(self.num_features))
        info_table.add_row("Feature names", str(self.feature_names))

        if self.group_column in self.df.columns:
            group_lengths = self.df.groupby(self.group_column).size()
            info_table.add_row("Min group length", str(group_lengths.min()))
            info_table.add_row("Max group length", str(group_lengths.max()))
        console.print(info_table)

        if "fault_state" in self.df.columns:
            labels = self.df["fault_state"].to_numpy(dtype=np.int32)
            console.print("\n[bold]Class Distribution:[/bold]")
            console.print(self._build_class_dist_table(labels))

    def _build_class_dist_table(self, y: np.ndarray) -> Table:
        """Build a rich Table for class distribution."""
        table = Table(show_header=True)
        table.add_column("Fault Type", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        flat = y.flatten()
        total = len(flat)
        for ft in FaultType:
            count = int(np.sum(flat == ft.value))
            pct = 100.0 * count / total if total > 0 else 0.0
            table.add_row(ft.name, f"{count:,}", f"{pct:.2f}%")
        return table

    def prepare(
        self,
        window_config: WindowConfig | None = None,
        split_config: DataSplitConfig | None = None,
        features: list[str] | None = None,
        required_metadata: set[str] | None = None,
        split_bounds: tuple[int, int, int, int] | None = None,
    ) -> WindowedSplits:
        """Convert this dataset into windowed train/val/test arrays.

        For each group in the DataFrame:
            1. Chronologically split into train/val/test
            2. Extract sliding windows with appropriate stride

        The validation set is carved from the *end* of the training portion
        so that no overlapping windows can leak between splits.

        Args:
            window_config: Windowing configuration. Falls back to
                ``WindowConfig()``.
            features: Subset of feature names to use. When ``None``
                (default), all features from ``self.feature_names``
                are used.
            required_metadata: Metadata keys required by the consumer
                (e.g. a model). Subclasses may use this to choose a
                preparation strategy. Ignored for tabular datasets.

        Returns:
            WindowedSplits with windowed arrays.

        Raises:
            ValueError: If any name in *features* is not in the dataset.
        """
        wc = window_config if window_config is not None else WindowConfig()
        split = split_config if split_config is not None else DataSplitConfig()
        if split.strategy != "chronological":
            msg = f"InjectedDataset supports only chronological split strategy, got {split.strategy!r}"
            raise ValueError(msg)
        if split_bounds is not None and self.group_column in self.df.columns:
            msg = "Global split bounds are only supported for a single time-aligned tabular block"
            raise ValueError(msg)
        selected_features = validate_features(features, self.feature_names)
        group_col = self.group_column

        train_X_parts: list[NDArray[np.float32]] = []
        train_y_parts: list[NDArray[np.int32]] = []
        val_X_parts: list[NDArray[np.float32]] = []
        val_y_parts: list[NDArray[np.int32]] = []
        test_X_parts: list[NDArray[np.float32]] = []
        test_y_parts: list[NDArray[np.int32]] = []

        groups = self.df.groupby(group_col) if group_col in self.df.columns else [(None, self.df)]

        for _, group_df in groups:
            group_features = group_df[selected_features].to_numpy(dtype=np.float32)
            group_labels = group_df["fault_state"].to_numpy(dtype=np.int32)

            X_tr, y_tr, X_va, y_va, X_te, y_te = split_and_window(
                group_features, group_labels, wc, split, split_bounds=split_bounds
            )

            if len(X_tr) > 0:
                train_X_parts.append(X_tr)
                train_y_parts.append(y_tr)
            if len(X_va) > 0:
                val_X_parts.append(X_va)
                val_y_parts.append(y_va)
            if len(X_te) > 0:
                test_X_parts.append(X_te)
                test_y_parts.append(y_te)

        X_train, y_train, X_val, y_val, X_test, y_test = collect_splits(
            wc, len(selected_features),
            train_X_parts, train_y_parts,
            val_X_parts, val_y_parts,
            test_X_parts, test_y_parts,
        )

        windowed = WindowedSplits(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        if split_bounds is not None:
            train_start, train_end, val_end, test_end = split_bounds
            windowed.split_bounds = {
                "train": (train_start, train_end),
                "val": (train_end, val_end),
                "test": (val_end, test_end),
            }
        return windowed

    def get_class_weights(self) -> dict[int, float]:
        """Compute inverse frequency class weights for imbalanced learning.

        Returns:
            Dictionary mapping class index to weight.
        """
        if "fault_state" not in self.df.columns:
            return {ft.value: 1.0 for ft in FaultType}

        labels = self.df["fault_state"].to_numpy(dtype=np.int32)
        flat = labels.flatten()
        total = len(flat)
        weights = {}
        for ft in FaultType:
            count = int(np.sum(flat == ft.value))
            if count > 0:
                weights[ft.value] = total / (FaultType.count() * count)
            else:
                weights[ft.value] = 1.0
        return weights
