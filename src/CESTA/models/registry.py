"""Model registry for dynamic model lookup.

Provides a central registry for model classes, allowing new architectures
to be added dynamically without modifying core training code.
"""

from __future__ import annotations

from typing import Any

from CESTA.models.base import BaseModel
from CESTA.models.spatial import CESTAClassifier, STGCNClassifier
from CESTA.models.temporal import (
    AutoformerClassifier,
    CNN1DClassifier,
    GRUClassifier,
    InformerClassifier,
    LSTMClassifier,
    ModernTCNClassifier,
    PatchTSTClassifier,
    TransformerClassifier,
)

_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(name: str, model_cls: type[BaseModel]) -> None:
    """Register a model class with a name.

    Args:
        name: Unique name for the model (used in CLI and configs).
        model_cls: The model class to register.

    Raises:
        ValueError: If name is already registered.
    """
    if name in _REGISTRY:
        raise ValueError(f"Model '{name}' is already registered")
    _REGISTRY[name] = model_cls


def get_model_class(name: str) -> type[BaseModel]:
    """Get a registered model class by name.

    Args:
        name: The registered model name.

    Returns:
        The model class.

    Raises:
        KeyError: If no model is registered with the given name.
    """
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(f"Model '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def create_model(
    name: str,
    *,
    input_size: int,
    num_classes: int,
    metadata: dict[str, Any] | None = None,
    **kwargs: object,
) -> BaseModel:
    """Create a model instance by name.

    Validates that all metadata required by the model architecture is
    present, extracts architecture-specific kwargs from the metadata,
    and constructs the model.

    Args:
        name: The registered model name.
        input_size: Number of input features per timestep.
        num_classes: Number of output classes.
        metadata: Dataset metadata from ``WindowedSplits.metadata``.
            Models declare required keys via ``required_metadata``.
        **kwargs: Additional arguments passed to the model constructor.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If required metadata keys are missing.
    """
    model_cls = get_model_class(name)
    metadata = metadata or {}

    required_metadata = model_cls.required_metadata
    optional_metadata = model_cls.optional_metadata & set(metadata.keys())
    usable_metadata = required_metadata | optional_metadata

    missing = required_metadata - set(metadata.keys())
    if missing:
        raise ValueError(
            f"Model '{name}' requires metadata keys {sorted(missing)}. "
            f"Available: {sorted(metadata.keys())}"
        )

    model_kwargs: dict[str, object] = {
        "input_size": input_size,
        "num_classes": num_classes,
        **kwargs,
    }

    model_kwargs.update(_extract_metadata_kwargs(metadata, usable_metadata))

    return model_cls(**model_kwargs)


def _extract_metadata_kwargs(
    metadata: dict[str, Any], required: set[str]
) -> dict[str, object]:
    """Extract model constructor kwargs from dataset metadata.

    Only extracts kwargs for metadata keys that the model actually requires.
    """
    kwargs: dict[str, object] = {}

    if "graph" in required:
        from CESTA.datasets.injected.graph import GraphMetadata

        graph_meta = metadata.get("graph")
        if isinstance(graph_meta, GraphMetadata):
            if graph_meta.adjacency is None:
                raise ValueError("Graph metadata is missing adjacency")
            edge_prob = graph_meta.adjacency.copy()
            if graph_meta.edge_index.shape[1] > 0:
                edge_prob[graph_meta.edge_index[0], graph_meta.edge_index[1]] = graph_meta.edge_prob
            kwargs["num_nodes"] = graph_meta.num_nodes
            kwargs["adjacency"] = graph_meta.adjacency.tolist()
            kwargs["edge_prob"] = edge_prob.tolist()

    node_identity = metadata.get("node_identity")
    if "node_identity" in required and isinstance(node_identity, dict):
        kwargs["num_nodes"] = int(node_identity["num_nodes"])

    return kwargs


def list_models() -> list[str]:
    """List all registered model names.

    Returns:
        List of registered model names.
    """
    return list(_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a model name is registered.

    Args:
        name: The model name to check.

    Returns:
        True if registered, False otherwise.
    """
    return name in _REGISTRY


register_model("lstm", LSTMClassifier)
register_model("gru", GRUClassifier)
register_model("autoformer", AutoformerClassifier)
register_model("transformer", TransformerClassifier)
register_model("informer", InformerClassifier)
register_model("patchtst", PatchTSTClassifier)
register_model("modern_tcn", ModernTCNClassifier)
register_model("stgcn", STGCNClassifier)
register_model("cesta", CESTAClassifier)
register_model("cnn1d", CNN1DClassifier)
