"""Evaluation result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from CESTA.evaluation.metrics import ClassMetrics, confusion_matrix
from CESTA.schema.fault import FaultType


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    class_metrics: ClassMetrics
    y_true: NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y_pred: NDArray[np.int32] = field(default_factory=lambda: np.empty(0, dtype=np.int32))
    y_prob: NDArray[np.float32] = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    communication_metrics: dict[str, Any] | None = None

    def save(
        self,
        path: str | Path,
        train_config: dict[str, Any] | None = None,
        injection_config: dict[str, Any] | None = None,
    ) -> None:
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
            (directory / "communication_metrics.json").write_text(json.dumps(self.communication_metrics, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EvalResult:
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
        communication_metrics = json.loads(communication_path.read_text()) if communication_path.exists() else None

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
