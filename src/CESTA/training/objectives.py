"""Shared objective and prediction helpers for training and evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn

from CESTA.models.base import BaseModel


def masked_loss(
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


def decode_predictions(
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


def valid_predictions(
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


def valid_outputs(
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
