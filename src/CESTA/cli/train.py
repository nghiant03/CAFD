"""CLI subcommand for model training."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from CESTA.datasets import load_dataset
from CESTA.evaluation import Evaluator
from CESTA.logging import logger
from CESTA.models import create_model, get_model_class
from CESTA.schema import (
    EvaluateConfig,
    RunManifest,
    Timing,
    TrainConfig,
)
from CESTA.schema.config import load_config_file
from CESTA.schema.fault import FaultType
from CESTA.training import (
    CheckpointCallback,
    EarlyStoppingCallback,
    HistoryCallback,
    LoggingCallback,
    Trainer,
    build_loss,
)
from CESTA.utils import (
    collect_env_info,
    collect_git_info,
    generate_run_id,
    utc_now_iso,
)


def train(
    config_file: Annotated[
        Path,
        typer.Argument(help="Path to YAML/JSON training config file"),
    ],
    data: Annotated[
        Path,
        typer.Argument(help="Path to injected dataset directory"),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Parent directory for runs (default: runs/<model>). A new run subdirectory is created per invocation.",
        ),
    ] = None,
    early_stopping: Annotated[
        bool,
        typer.Option("--early-stopping/--no-early-stopping", help="Enable early stopping"),
    ] = False,
) -> None:
    """Train a fault diagnosis model."""
    config = TrainConfig.model_validate(load_config_file(config_file))
    logger.debug("TrainConfig: {}", config.model_dump(mode="json"))

    logger.info("Loading data from: {}", data)
    dataset = load_dataset(data)
    dataset.print_summary()

    model_cls = get_model_class(config.model)
    requested_metadata = set(model_cls.required_metadata)
    if config.model_kwargs.get("node_embedding_dim", 0):
        requested_metadata.add("node_identity")
    prepared = dataset.prepare(
        window_config=config.data.window,
        split_config=config.data.split,
        features=config.features,
        required_metadata=requested_metadata,
    )

    logger.debug(
        "Windowed shapes: X_train={}, y_train={}, X_val={}, y_val={}, X_test={}, y_test={}",
        prepared.X_train.shape,
        prepared.y_train.shape,
        prepared.X_val.shape,
        prepared.y_val.shape,
        prepared.X_test.shape,
        prepared.y_test.shape,
    )

    input_size = prepared.input_size
    num_classes = FaultType.count()
    logger.debug(
        "Creating model: arch={}, input_size={}, num_classes={}",
        config.model,
        input_size,
        num_classes,
    )

    net = create_model(
        config.model,
        input_size=input_size,
        num_classes=num_classes,
        metadata=prepared.metadata,
        **config.model_kwargs,
    )
    logger.info("Model: {} ({:,} parameters)", net.name, net.count_parameters())

    output_root = output if output is not None else Path(f"runs/{config.model}")
    git = collect_git_info()
    run_id = generate_run_id(config.model, config.seed, git)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run dir: {}", run_dir)

    callbacks = [
        LoggingCallback(),
        CheckpointCallback(
            save_path=run_dir,
            config_dict=config.model_dump(mode="json"),
            monitor=config.checkpoint_monitor,
        ),
        HistoryCallback(save_path=run_dir),
    ]

    if early_stopping:
        callbacks.append(EarlyStoppingCallback(patience=10, monitor=config.early_stopping_monitor))

    trainer = Trainer(config=config, callbacks=callbacks)

    env = collect_env_info(trainer.device)
    dataset_info = dataset.describe(data)

    logger.info(
        "Training for {} epochs | batch_size={} | lr={} | focal_loss={} | oversample={}",
        config.epochs,
        config.batch_size,
        config.learning_rate,
        config.use_focal_loss,
        config.oversample,
    )

    started_at = utc_now_iso()
    t0 = time.perf_counter()
    result = trainer.fit(
        model=net,
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_val=prepared.X_val if prepared.has_val else None,
        y_val=prepared.y_val if prepared.has_val else None,
        metadata=prepared.metadata,
        node_mask_train=prepared.node_mask_train,
        edge_mask_train=prepared.edge_mask_train,
        node_mask_val=prepared.node_mask_val if prepared.has_val else None,
        edge_mask_val=prepared.edge_mask_val if prepared.has_val else None,
    )
    duration = time.perf_counter() - t0
    ended_at = utc_now_iso()

    logger.info(
        "Training complete at epoch {} | best_val_loss={:.4f}",
        result.stopped_epoch,
        result.best_val_loss if result.best_val_loss is not None else float("nan"),
    )
    logger.info("Model saved to: {}", run_dir)

    if prepared.has_test:
        logger.info("--- Final Test Evaluation ---")
        weight_path = run_dir / "weight.pt"
        if weight_path.exists():
            import torch

            net.load_state_dict(torch.load(weight_path, map_location=trainer.device, weights_only=True))
            logger.info("Reloaded best checkpoint from {} for test evaluation", weight_path)
        else:
            logger.warning("No checkpoint at {}; evaluating final-epoch weights", weight_path)
        evaluator = Evaluator(
            config=EvaluateConfig(batch_size=config.batch_size),
            device=str(trainer.device),
        )
        criterion = build_loss(config, trainer.device)
        eval_result = evaluator.evaluate(
            net,
            prepared.X_test,
            prepared.y_test,
            criterion=criterion,
            metadata=prepared.metadata,
            node_mask=prepared.node_mask_test,
            edge_mask=prepared.edge_mask_test,
        )
        evaluator.log_results(eval_result)

        eval_result.save(
            run_dir,
            train_config=config.model_dump(mode="json"),
            injection_config=dataset.config.model_dump(mode="json"),
        )
        logger.info("Results saved to: {}", run_dir)

    manifest = RunManifest(
        run_id=run_id,
        kind="train",
        seed=config.seed,
        model=config.model,
        num_parameters=net.count_parameters(),
        git=git,
        env=env,
        dataset=dataset_info,
        timing=Timing(
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration,
            epochs_run=result.stopped_epoch,
        ),
        train_config=config.model_dump(mode="json"),
        injection_config=dataset.config.model_dump(mode="json"),
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest.model_dump(mode="json"), indent=2))
    logger.info("Manifest written to: {}", run_dir / "manifest.json")
