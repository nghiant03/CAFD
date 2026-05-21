"""Trainer for fault diagnosis models.

Handles the full training loop including:
- Optional oversampling of minority classes
- Configurable loss function (cross-entropy or focal loss)
- Callback-driven logging, early stopping, and checkpointing
- Per-class precision, recall, F1 metrics
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

from CESTA.batch import GraphWindowBatch, TemporalWindowBatch
from CESTA.datasets.injected.graph import GraphMetadata
from CESTA.evaluation.metrics import ClassMetrics, compute_class_metrics, macro_f1
from CESTA.logging import logger
from CESTA.models.base import BaseModel
from CESTA.schema import TrainConfig
from CESTA.seed import seed_everything
from CESTA.training.callbacks import (
    LoggingCallback,
    TrainingCallback,
    TrainMetrics,
)
from CESTA.training.graph_batch import GraphWindowDataset, TemporalWindowDataset, collate_graph_batch, collate_temporal_batch
from CESTA.training.loss import FocalLoss
from CESTA.training.oversampling import oversample_minority


def build_loss(
    config: TrainConfig,
    device: torch.device,
) -> nn.Module:
    """Build the loss function from config.

    Args:
        config: Training configuration.
        device: Target device for tensors.

    Returns:
        Loss module ready for ``(logits, targets)`` inputs.
    """
    if config.use_focal_loss:
        alpha = (
            torch.tensor(config.focal_alpha, dtype=torch.float32).to(device)
            if config.focal_alpha is not None
            else None
        )
        logger.debug(
            "Using FocalLoss with gamma={}, alpha={}",
            config.focal_gamma,
            config.focal_alpha,
        )
        return FocalLoss(gamma=config.focal_gamma, alpha=alpha)
    logger.debug("Using CrossEntropyLoss")
    return nn.CrossEntropyLoss()


def _prepare_data(
    X: NDArray[np.float32],
    y: NDArray[np.int32],
    config: TrainConfig,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Apply oversampling if enabled.

    Args:
        X: Feature array ``(N, seq_len, features)``.
        y: Label array ``(N, seq_len)``.
        config: Training configuration.

    Returns:
        Possibly oversampled ``(X, y)`` tuple.
    """
    if config.oversample:
        logger.debug(
            "Oversampling minority classes with ratio={}, seed={}",
            config.oversample_ratio,
            config.seed,
        )
        X_out, y_out = oversample_minority(
            X, y, ratio=config.oversample_ratio, seed=config.seed
        )
        logger.info(
            "Oversampled: {} -> {} windows",
            len(X),
            len(X_out),
        )
        return X_out, y_out
    return X, y


@dataclass
class TrainResult:
    """Result container returned after training completes.

    Attributes:
        history: Per-epoch metrics collected during training.
        best_val_loss: Lowest validation loss seen (``None`` if no val data).
        stopped_epoch: Epoch at which training stopped (may be < total if early stopped).
    """

    history: list[TrainMetrics] = field(default_factory=list)
    best_val_loss: float | None = None
    stopped_epoch: int = 0


class Trainer:
    """Trains a fault-diagnosis model.

    Args:
        config: Training configuration (loss, oversampling, hyperparams).
        callbacks: Optional sequence of callbacks. If ``None``, a
            :class:`LoggingCallback` is used by default.
        device: PyTorch device string. ``None`` auto-selects CUDA if available.
    """

    def __init__(
        self,
        config: TrainConfig,
        callbacks: Sequence[TrainingCallback] | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.callbacks: list[TrainingCallback] = (
            list(callbacks) if callbacks is not None else [LoggingCallback()]
        )
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit(
        self,
        model: BaseModel,
        X_train: NDArray[np.float32],
        y_train: NDArray[np.int32],
        X_val: NDArray[np.float32] | None = None,
        y_val: NDArray[np.int32] | None = None,
        metadata: dict[str, object] | None = None,
        node_mask_train: NDArray[np.bool_] | None = None,
        edge_mask_train: NDArray[np.bool_] | None = None,
        node_mask_val: NDArray[np.bool_] | None = None,
        edge_mask_val: NDArray[np.bool_] | None = None,
    ) -> TrainResult:
        """Train the model.

        Args:
            model: Model instance to train (modified in-place).
            X_train: Training features ``(N, seq_len, features)``.
            y_train: Training labels ``(N, seq_len)``.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            :class:`TrainResult` with full training history.
        """
        seed_everything(self.config.seed)

        logger.debug(
            "Training data shape: X={}, y={}",
            X_train.shape,
            y_train.shape,
        )
        X_train, y_train = _prepare_data(X_train, y_train, self.config)

        if X_val is not None and y_val is not None:
            logger.info(
                "Using provided validation data: train={}, val={}",
                len(X_train),
                len(X_val),
            )

        train_loader = self._make_loader(
            X_train,
            y_train,
            shuffle=True,
            metadata=metadata,
            node_mask=node_mask_train,
            edge_mask=edge_mask_train,
        )
        val_loader = (
            self._make_loader(
                X_val,
                y_val,
                shuffle=False,
                metadata=metadata,
                node_mask=node_mask_val,
                edge_mask=edge_mask_val,
            )
            if X_val is not None and y_val is not None
            else None
        )
        logger.debug(
            "Train batches: {}, Val batches: {}",
            len(train_loader),
            len(val_loader) if val_loader is not None else 0,
        )

        model = model.to(self.device)

        num_classes = self._infer_num_classes(model, X_train, metadata)
        logger.info("Using device: {}", self.device)
        criterion = build_loss(self.config, self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        logger.debug("Optimizer: Adam(lr={})", self.config.learning_rate)

        result = TrainResult()

        for epoch in range(1, self.config.epochs + 1):
            self._maybe_anneal_gumbel_temperature(model, epoch)

            train_loss, train_acc, train_cm = self._train_epoch(
                model, train_loader, criterion, optimizer
            )
            train_class_metrics = compute_class_metrics(
                train_cm[0], train_cm[1], num_classes
            )
            train_macro_f1 = macro_f1(train_class_metrics)

            val_loss: float | None = None
            val_acc: float | None = None
            val_macro_f1: float | None = None
            val_class_metrics: ClassMetrics | None = None

            if val_loader is not None:
                val_loss, val_acc, val_cm = self._eval_epoch(
                    model, val_loader, criterion
                )
                val_class_metrics = compute_class_metrics(
                    val_cm[0], val_cm[1], num_classes
                )
                val_macro_f1 = macro_f1(val_class_metrics)

            metrics = TrainMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                train_macro_f1=train_macro_f1,
                val_macro_f1=val_macro_f1,
                train_class_metrics=train_class_metrics,
                val_class_metrics=val_class_metrics,
            )
            result.history.append(metrics)

            if val_loss is not None and (
                result.best_val_loss is None or val_loss < result.best_val_loss
            ):
                result.best_val_loss = val_loss

            should_continue = all(
                cb.on_epoch_end(metrics, model) for cb in self.callbacks
            )
            if not should_continue:
                result.stopped_epoch = epoch
                logger.info("Training stopped early at epoch {}", epoch)
                break
        else:
            result.stopped_epoch = self.config.epochs
            logger.info("Training completed all {} epochs", self.config.epochs)

        self._log_final_metrics(result)
        return result

    def _log_final_metrics(self, result: TrainResult) -> None:
        """Log a summary of final training metrics."""
        if not result.history:
            return

        last = result.history[-1]
        logger.info("--- Training Summary ---")
        logger.info(
            "Stopped at epoch {} | train_loss={:.4f} | train_acc={:.4f} | train_f1={:.4f}",
            result.stopped_epoch,
            last.train_loss,
            last.train_acc if last.train_acc is not None else 0.0,
            last.train_macro_f1 if last.train_macro_f1 is not None else 0.0,
        )
        if last.val_loss is not None:
            logger.info(
                "val_loss={:.4f} | val_acc={:.4f} | val_f1={:.4f} | best_val_loss={:.4f}",
                last.val_loss,
                last.val_acc if last.val_acc is not None else 0.0,
                last.val_macro_f1 if last.val_macro_f1 is not None else 0.0,
                result.best_val_loss
                if result.best_val_loss is not None
                else float("nan"),
            )
        if last.val_class_metrics is not None:
            self._log_class_metrics("Validation", last.val_class_metrics)
        elif last.train_class_metrics is not None:
            self._log_class_metrics("Training", last.train_class_metrics)

    @staticmethod
    def _log_class_metrics(split_name: str, cm: ClassMetrics) -> None:
        """Log per-class metrics table."""
        from CESTA.schema.fault import FaultType

        names = FaultType.names()
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

    def _maybe_anneal_gumbel_temperature(
        self,
        model: BaseModel,
        epoch: int,
    ) -> None:
        config = self.config
        if (
            config.gumbel_tau_anneal_epochs < 1
            or config.gumbel_tau_start == config.gumbel_tau_end
        ):
            return
        setter = getattr(model, "set_gumbel_temperature", None)
        if setter is None:
            return
        progress = min(float(epoch - 1) / config.gumbel_tau_anneal_epochs, 1.0)
        tau = config.gumbel_tau_start + progress * (
            config.gumbel_tau_end - config.gumbel_tau_start
        )
        setter(tau)

    def _make_loader(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int32],
        shuffle: bool,
        metadata: dict[str, object] | None = None,
        node_mask: NDArray[np.bool_] | None = None,
        edge_mask: NDArray[np.bool_] | None = None,
    ) -> DataLoader[object]:
        """Create a DataLoader from numpy arrays."""
        graph_meta = (metadata or {}).get("graph")
        if isinstance(graph_meta, GraphMetadata) and node_mask is not None and edge_mask is not None:
            dataset = GraphWindowDataset(X, y, node_mask, edge_mask, graph_meta.edge_index)
            collate_fn = collate_graph_batch
        elif isinstance(node_identity := (metadata or {}).get("node_identity"), dict):
            node_ids = node_identity.get("train_node_ids") if shuffle else node_identity.get("val_node_ids")
            dataset = TemporalWindowDataset(X, y, node_ids)
            collate_fn = collate_temporal_batch
        else:
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X_t, y_t)
            collate_fn = None
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator,
            collate_fn=collate_fn,
        )

    def _infer_num_classes(
        self,
        model: BaseModel,
        X_train: NDArray[np.float32],
        metadata: dict[str, object] | None,
    ) -> int:
        graph_meta = (metadata or {}).get("graph")
        if isinstance(graph_meta, GraphMetadata) and X_train.ndim == 4:
            sample = GraphWindowBatch(
                x=torch.zeros(1, X_train.shape[1], X_train.shape[2], X_train.shape[3], device=self.device),
                y=torch.zeros(1, X_train.shape[1], X_train.shape[2], dtype=torch.long, device=self.device),
                node_mask=torch.ones(1, X_train.shape[1], X_train.shape[2], dtype=torch.bool, device=self.device),
                edge_index=torch.tensor(graph_meta.edge_index, dtype=torch.long, device=self.device),
                edge_mask=torch.ones(1, X_train.shape[1], graph_meta.edge_index.shape[1], dtype=torch.bool, device=self.device),
            )
            return int(model(sample).size(-1))
        if isinstance((metadata or {}).get("node_identity"), dict):
            sample = TemporalWindowBatch(
                x=torch.zeros(1, X_train.shape[1], X_train.shape[2], device=self.device),
                y=torch.zeros(1, X_train.shape[1], dtype=torch.long, device=self.device),
                node_ids=torch.zeros(1, dtype=torch.long, device=self.device),
            )
            return int(model(sample).size(-1))
        return int(model(torch.zeros(1, X_train.shape[1], X_train.shape[2], device=self.device)).size(-1))

    def _train_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[object],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Run one training epoch.

        Returns:
            ``(avg_loss, accuracy, (all_preds, all_targets))`` over the epoch.
        """
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in loader:
            model_input, y_batch, node_mask, batch_size = self._prepare_batch(batch)

            optimizer.zero_grad()
            logits = model(model_input)

            loss = self._masked_loss(criterion, logits, y_batch, node_mask)
            loss = self._add_auxiliary_loss(model, loss)
            loss = self._add_boundary_loss(model, loss, y_batch, node_mask)
            loss = self._add_crf_loss(model, loss, logits, y_batch, node_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            preds = self._decode_predictions(model, logits, node_mask)
            valid_preds, valid_targets = self._valid_predictions(preds, y_batch, node_mask)
            correct += (valid_preds == valid_targets).sum().item()
            total += valid_targets.numel()

            all_preds.append(valid_preds.detach().cpu())
            all_targets.append(valid_targets.detach().cpu())

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, (all_preds, all_targets)

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
    def _valid_predictions(
        preds: torch.Tensor,
        targets: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_preds = preds.reshape(-1)
        flat_targets = targets.reshape(-1)
        if node_mask is None:
            return flat_preds, flat_targets
        valid = node_mask.reshape(-1) & (flat_targets >= 0)
        return flat_preds[valid], flat_targets[valid]

    def _add_auxiliary_loss(
        self,
        model: BaseModel,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.config.communication_penalty_weight
        if weight > 0.0:
            comm_loss = getattr(model, "communication_loss", None)
            if comm_loss is not None and isinstance(comm_loss, torch.Tensor):
                if self.config.communication_penalty_mode == "budget_hinge":
                    excess = torch.relu(
                        comm_loss - self.config.target_request_ratio
                    )
                    loss = loss + weight * (excess ** 2)
                else:
                    loss = loss + weight * comm_loss

        entropy_weight = self.config.gate_entropy_weight
        if entropy_weight > 0.0:
            gate_entropy = getattr(model, "gate_entropy", None)
            if gate_entropy is not None and isinstance(gate_entropy, torch.Tensor):
                loss = loss - entropy_weight * gate_entropy

        return loss

    def _add_boundary_loss(
        self,
        model: BaseModel,
        loss: torch.Tensor,
        targets: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        weight = self.config.boundary_loss_weight
        if weight <= 0.0:
            return loss
        boundary_logits = getattr(model, "last_boundary_logits", None)
        if not isinstance(boundary_logits, torch.Tensor):
            return loss
        boundary_targets, boundary_mask = self._boundary_targets(targets, node_mask, self.config.boundary_dilation)
        flat_logits = boundary_logits.reshape(-1)
        flat_targets = boundary_targets.reshape(-1)
        flat_mask = boundary_mask.reshape(-1)
        if not bool(flat_mask.any()):
            return loss
        selected_logits = flat_logits[flat_mask]
        selected_targets = flat_targets[flat_mask]
        pos_weight = None
        if self.config.boundary_positive_weight is not None:
            pos_weight = torch.tensor(self.config.boundary_positive_weight, dtype=selected_logits.dtype, device=selected_logits.device)
        boundary_loss = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, pos_weight=pos_weight, reduction="none")
        gamma = self.config.boundary_focal_gamma
        if gamma > 0.0:
            probs = torch.sigmoid(selected_logits)
            p_t = torch.where(selected_targets > 0.5, probs, 1.0 - probs)
            boundary_loss = ((1.0 - p_t).clamp_min(0.0) ** gamma) * boundary_loss
        return loss + weight * boundary_loss.mean()

    def _add_crf_loss(
        self,
        model: BaseModel,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        weight = self.config.crf_loss_weight
        if weight <= 0.0:
            return loss
        crf_nll = getattr(model, "crf_negative_log_likelihood", None)
        if not callable(crf_nll):
            return loss
        crf_loss = crf_nll(logits, targets, node_mask)
        if not isinstance(crf_loss, torch.Tensor):
            return loss
        return loss + weight * crf_loss

    @staticmethod
    def _decode_predictions(
        model: BaseModel,
        logits: torch.Tensor,
        node_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        decoder = getattr(model, "crf_decode", None)
        if callable(decoder) and bool(getattr(model, "use_crf", False)):
            decoded = decoder(logits, node_mask)
            if isinstance(decoded, torch.Tensor):
                return decoded
        return logits.argmax(dim=-1)

    @staticmethod
    def _boundary_targets(
        targets: torch.Tensor,
        node_mask: torch.Tensor | None,
        dilation: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = targets >= 0
        if node_mask is not None:
            valid = valid & node_mask
        boundary = torch.zeros_like(targets, dtype=torch.float32)
        transition_valid = valid[:, 1:] & valid[:, :-1]
        raw_boundary = ((targets[:, 1:] != targets[:, :-1]) & transition_valid).to(torch.float32)
        boundary[:, 1:] = raw_boundary
        boundary_mask = valid.clone()
        boundary_mask[:, 0] = False
        if dilation > 0:
            dilated = boundary.clone()
            for shift in range(1, dilation + 1):
                dilated[:, shift:] = torch.maximum(dilated[:, shift:], boundary[:, :-shift])
                dilated[:, :-shift] = torch.maximum(dilated[:, :-shift], boundary[:, shift:])
            boundary = dilated * boundary_mask.to(torch.float32)
        return boundary, boundary_mask

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: BaseModel,
        loader: DataLoader[object],
        criterion: nn.Module,
    ) -> tuple[float, float, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Run one evaluation epoch.

        Returns:
            ``(avg_loss, accuracy, (all_preds, all_targets))`` over the dataset.
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for batch in loader:
            model_input, y_batch, node_mask, batch_size = self._prepare_batch(batch)

            logits = model(model_input)
            loss = self._masked_loss(criterion, logits, y_batch, node_mask)
            loss = self._add_auxiliary_loss(model, loss)

            total_loss += loss.item() * batch_size
            preds = self._decode_predictions(model, logits, node_mask)
            valid_preds, valid_targets = self._valid_predictions(preds, y_batch, node_mask)
            correct += (valid_preds == valid_targets).sum().item()
            total += valid_targets.numel()

            all_preds.append(valid_preds.detach().cpu())
            all_targets.append(valid_targets.detach().cpu())

        avg_loss = total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy, (all_preds, all_targets)
