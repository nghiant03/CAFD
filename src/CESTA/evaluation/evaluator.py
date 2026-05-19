"""Evaluator for fault diagnosis models.

Runs inference on a dataset and computes classification metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from CESTA.batch import GraphWindowBatch, TemporalWindowBatch
from CESTA.datasets.injected.graph import GraphMetadata
from CESTA.logging import logger
from CESTA.models.base import BaseModel
from CESTA.schema import EvaluateConfig
from CESTA.schema.fault import FaultType
from CESTA.training.graph_batch import GraphWindowDataset, TemporalWindowDataset, collate_graph_batch, collate_temporal_batch

from .communication import aggregate_communication_stats
from .metrics import ClassMetrics, compute_class_metrics, confusion_matrix, macro_f1


@dataclass
class EvalResult:
    """Result container returned after evaluation.

    Attributes:
        loss: Average loss over the evaluation set.
        accuracy: Overall accuracy.
        macro_f1: Macro-averaged F1.
        class_metrics: Per-class precision, recall, F1, and support.
        y_true: Ground truth labels ``(total_timesteps,)``.
        y_pred: Predicted labels ``(total_timesteps,)``.
        y_prob: Predicted class probabilities ``(total_timesteps, num_classes)``.
    """

    loss: float
    accuracy: float
    macro_f1: float
    class_metrics: ClassMetrics
    y_true: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    y_pred: NDArray[np.int32] = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    y_prob: NDArray[np.float32] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    communication_metrics: dict[str, Any] | None = None

    def save(
        self,
        path: str | Path,
        train_config: dict[str, Any] | None = None,
        injection_config: dict[str, Any] | None = None,
    ) -> None:
        """Save evaluation results, predictions, and configs to a directory.

        Writes:
            - ``eval_metrics.json``: aggregate, per-class metrics, and embedded configs.
            - ``predictions.npz``: integer ``y_true``/``y_pred`` and ``y_prob`` arrays.
            - ``confusion_matrix.npy``: ``(num_classes, num_classes)`` integer matrix.

        Args:
            path: Directory to save into (created if needed).
            train_config: Training config dict to embed.
            injection_config: Injection config dict to embed.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        names = FaultType.names()
        per_class = {}
        for i, name in enumerate(names):
            if i < len(self.class_metrics.precision):
                per_class[name] = {
                    "precision": self.class_metrics.precision[i],
                    "recall": self.class_metrics.recall[i],
                    "f1": self.class_metrics.f1[i],
                    "support": self.class_metrics.support[i],
                }

        num_classes = len(names)
        cm = confusion_matrix(self.y_true, self.y_pred, num_classes)

        metrics_dict: dict[str, Any] = {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": per_class,
            "class_names": names,
            "confusion_matrix": cm.tolist(),
        }
        if train_config is not None:
            metrics_dict["train_config"] = train_config
        if injection_config is not None:
            metrics_dict["injection_config"] = injection_config

        (directory / "eval_metrics.json").write_text(json.dumps(metrics_dict, indent=2))

        np.savez_compressed(
            directory / "predictions.npz",
            y_true=self.y_true.astype(np.int32),
            y_pred=self.y_pred.astype(np.int32),
            y_prob=self.y_prob.astype(np.float32),
        )

        if self.communication_metrics is not None:
            (directory / "communication_metrics.json").write_text(
                json.dumps(self.communication_metrics, indent=2)
            )

    @classmethod
    def load(cls, path: str | Path) -> EvalResult:
        """Load evaluation results from a directory.

        Args:
            path: Directory containing ``eval_metrics.json`` and ``predictions.npz``.

        Returns:
            Reconstructed EvalResult.
        """
        directory = Path(path)
        meta = json.loads((directory / "eval_metrics.json").read_text())

        per_class = meta.get("per_class", {})
        names = FaultType.names()
        precision = [per_class.get(n, {}).get("precision", 0.0) for n in names]
        recall = [per_class.get(n, {}).get("recall", 0.0) for n in names]
        f1_scores = [per_class.get(n, {}).get("f1", 0.0) for n in names]
        support = [per_class.get(n, {}).get("support", 0) for n in names]

        npz_path = directory / "predictions.npz"
        if npz_path.exists():
            preds = np.load(npz_path)
            y_true = preds["y_true"].astype(np.int32)
            y_pred = preds["y_pred"].astype(np.int32)
            y_prob = preds["y_prob"].astype(np.float32)
        else:
            y_true = np.empty(0, dtype=np.int32)
            y_pred = np.empty(0, dtype=np.int32)
            y_prob = np.empty((0, 0), dtype=np.float32)

        communication_path = directory / "communication_metrics.json"
        communication_metrics = (
            json.loads(communication_path.read_text())
            if communication_path.exists()
            else None
        )

        return cls(
            loss=meta["loss"],
            accuracy=meta["accuracy"],
            macro_f1=meta["macro_f1"],
            class_metrics=ClassMetrics(
                precision=precision,
                recall=recall,
                f1=f1_scores,
                support=support,
            ),
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            communication_metrics=communication_metrics,
        )


class Evaluator:
    """Evaluates a fault-diagnosis model on a dataset.

    Args:
        config: Evaluation configuration.
        device: PyTorch device string. ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        config: EvaluateConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config if config is not None else EvaluateConfig()
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: BaseModel,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        criterion: nn.Module | None = None,
        metadata: dict[str, object] | None = None,
        node_mask: NDArray[np.bool_] | None = None,
        edge_mask: NDArray[np.bool_] | None = None,
    ) -> EvalResult:
        """Evaluate the model on the given data.

        Args:
            model: Trained model to evaluate.
            X: Feature array ``(N, seq_len, features)``.
            y: Label array ``(N, seq_len)``.
            criterion: Loss function. Defaults to ``CrossEntropyLoss``.

        Returns:
            :class:`EvalResult` with loss, accuracy, per-class metrics, and predictions.
        """
        model = model.to(self.device)
        model.eval()

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        loader = self._make_loader(X, y, metadata=metadata, node_mask=node_mask, edge_mask=edge_mask)

        num_classes = self._infer_num_classes(model, X, metadata)

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []
        communication_stats: list[dict[str, float]] = []

        for batch in loader:
            model_input, y_batch, batch_node_mask, batch_size = self._prepare_batch(batch)

            logits = model(model_input)
            loss = self._masked_loss(criterion, logits, y_batch, batch_node_mask)
            stats = getattr(model, "last_communication_stats", None)
            if isinstance(stats, dict):
                communication_stats.append({k: float(v) for k, v in stats.items()})

            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=-1)
            probs = torch.softmax(logits, dim=-1)
            valid_preds, valid_targets, valid_probs = self._valid_outputs(
                preds, y_batch, probs, batch_node_mask
            )
            correct += (valid_preds == valid_targets).sum().item()
            total += valid_targets.numel()

            all_preds.append(valid_preds.detach().cpu())
            all_targets.append(valid_targets.detach().cpu())
            all_probs.append(valid_probs.detach().cpu())

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        class_metrics = compute_class_metrics(all_preds, all_targets, num_classes)
        f1 = macro_f1(class_metrics)

        y_true = torch.cat(all_targets).numpy().astype(np.int32)
        y_pred = torch.cat(all_preds).numpy().astype(np.int32)
        y_prob = torch.cat(all_probs).numpy().astype(np.float32)
        communication_metrics = aggregate_communication_stats(
            {"test": communication_stats},
            model,
            metadata=metadata,
        )

        return EvalResult(
            loss=avg_loss,
            accuracy=accuracy,
            macro_f1=f1,
            class_metrics=class_metrics,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            communication_metrics=communication_metrics,
        )

    def log_results(self, result: EvalResult, split_name: str = "Test") -> None:
        """Log evaluation results.

        Args:
            result: Evaluation result to log.
            split_name: Name of the data split (e.g. "Test", "Validation").
        """
        logger.info(
            "{}: loss={:.4f} | acc={:.4f} | f1={:.4f}",
            split_name,
            result.loss,
            result.accuracy,
            result.macro_f1,
        )
        names = FaultType.names()
        cm = result.class_metrics
        logger.info("--- {} Per-Class Metrics ---", split_name)
        logger.info(
            "{:<10s}  {:>9s}  {:>9s}  {:>9s}  {:>9s}",
            "Class",
            "Precision",
            "Recall",
            "F1",
            "Support",
        )
        for i, name in enumerate(names):
            if i < len(cm.precision):
                logger.info(
                    "{:<10s}  {:>9.4f}  {:>9.4f}  {:>9.4f}  {:>9d}",
                    name,
                    cm.precision[i],
                    cm.recall[i],
                    cm.f1[i],
                    cm.support[i],
                )

    def _make_loader(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        metadata: dict[str, object] | None = None,
        node_mask: NDArray[np.bool_] | None = None,
        edge_mask: NDArray[np.bool_] | None = None,
    ) -> DataLoader[object]:
        """Create a DataLoader from numpy arrays."""
        graph_meta = (metadata or {}).get("graph")
        if isinstance(graph_meta, GraphMetadata) and node_mask is not None and edge_mask is not None:
            dataset = GraphWindowDataset(X, y, node_mask, edge_mask, graph_meta.edge_index)
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_graph_batch,
            )
        if isinstance(node_identity := (metadata or {}).get("node_identity"), dict):
            dataset = TemporalWindowDataset(X, y, node_identity.get("test_node_ids"))
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=collate_temporal_batch,
            )
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

    def _infer_num_classes(
        self,
        model: BaseModel,
        X: NDArray[np.float32],
        metadata: dict[str, object] | None,
    ) -> int:
        graph_meta = (metadata or {}).get("graph")
        if isinstance(graph_meta, GraphMetadata) and X.ndim == 4:
            sample = GraphWindowBatch(
                x=torch.zeros(1, X.shape[1], X.shape[2], X.shape[3], device=self.device),
                y=torch.zeros(1, X.shape[1], X.shape[2], dtype=torch.long, device=self.device),
                node_mask=torch.ones(1, X.shape[1], X.shape[2], dtype=torch.bool, device=self.device),
                edge_index=torch.tensor(graph_meta.edge_index, dtype=torch.long, device=self.device),
                edge_mask=torch.ones(1, X.shape[1], graph_meta.edge_index.shape[1], dtype=torch.bool, device=self.device),
            )
            return int(model(sample).size(-1))
        if isinstance((metadata or {}).get("node_identity"), dict):
            sample = TemporalWindowBatch(
                x=torch.zeros(1, X.shape[1], X.shape[2], device=self.device),
                y=torch.zeros(1, X.shape[1], dtype=torch.long, device=self.device),
                node_ids=torch.zeros(1, dtype=torch.long, device=self.device),
            )
            return int(model(sample).size(-1))
        return int(model(torch.zeros(1, X.shape[1], X.shape[2], device=self.device)).size(-1))

    def _prepare_batch(
        self,
        batch: object,
    ) -> tuple[torch.Tensor | GraphWindowBatch | TemporalWindowBatch, torch.Tensor, torch.Tensor | None, int]:
        if isinstance(batch, GraphWindowBatch):
            graph_batch = GraphWindowBatch(
                x=batch.x.to(self.device),
                y=batch.y.to(self.device),
                node_mask=batch.node_mask.to(self.device),
                edge_index=batch.edge_index.to(self.device),
                edge_mask=batch.edge_mask.to(self.device),
            )
            return graph_batch, graph_batch.y, graph_batch.node_mask, graph_batch.x.size(0)
        if isinstance(batch, TemporalWindowBatch):
            temporal_batch = TemporalWindowBatch(
                x=batch.x.to(self.device),
                y=batch.y.to(self.device),
                node_ids=batch.node_ids.to(self.device) if batch.node_ids is not None else None,
            )
            return temporal_batch, temporal_batch.y, None, temporal_batch.x.size(0)
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            raise TypeError("Expected a tensor batch, TemporalWindowBatch, or GraphWindowBatch")
        X_batch = batch[0].to(self.device)
        y_batch = batch[1].to(self.device)
        return X_batch, y_batch, None, X_batch.size(0)

    @staticmethod
    def _masked_loss(
        criterion: nn.Module,
        logits: torch.Tensor,
        targets: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        if node_mask is None:
            return criterion(flat_logits, flat_targets)
        valid = node_mask.reshape(-1) & (flat_targets >= 0)
        if not bool(valid.any()):
            return flat_logits.sum() * 0.0
        return criterion(flat_logits[valid], flat_targets[valid])

    @staticmethod
    def _valid_outputs(
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_preds = preds.reshape(-1)
        flat_targets = targets.reshape(-1)
        flat_probs = probs.reshape(-1, probs.size(-1))
        if node_mask is None:
            return flat_preds, flat_targets, flat_probs
        valid = node_mask.reshape(-1) & (flat_targets >= 0)
        return flat_preds[valid], flat_targets[valid], flat_probs[valid]
