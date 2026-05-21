"""Evaluator for fault diagnosis models.

Runs inference on a dataset and computes classification metrics.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from CESTA.logging import logger
from CESTA.models.base import BaseModel
from CESTA.schema import EvaluateConfig
from CESTA.schema.fault import FaultType
from CESTA.training.batch_utils import infer_num_classes, make_window_loader, prepare_batch
from CESTA.training.objectives import decode_predictions, masked_loss, valid_outputs

from .communication import aggregate_communication_stats
from .metrics import compute_class_metrics, macro_f1
from .result import EvalResult


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
            model_input, y_batch, batch_node_mask, batch_size = prepare_batch(batch, self.device)

            logits = model(model_input)
            loss = masked_loss(criterion, logits, y_batch, batch_node_mask)
            stats = getattr(model, "last_communication_stats", None)
            if isinstance(stats, dict):
                communication_stats.append({k: float(v) for k, v in stats.items()})

            total_loss += loss.item() * batch_size
            preds = decode_predictions(model, logits, batch_node_mask)
            probs = torch.softmax(logits, dim=-1)
            valid_preds, valid_targets, valid_probs = valid_outputs(
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
        return make_window_loader(
            X,
            y,
            self.config.batch_size,
            metadata=metadata,
            node_mask=node_mask,
            edge_mask=edge_mask,
            node_identity_split="test",
        )

    def _infer_num_classes(
        self,
        model: BaseModel,
        X: NDArray[np.float32],
        metadata: dict[str, object] | None,
    ) -> int:
        return infer_num_classes(model, X, metadata, self.device)

