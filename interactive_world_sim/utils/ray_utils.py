"""Utilities for running training on Anyscale via Ray Train."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def _register_omegaconf_resolvers() -> None:
    """Re-register custom OmegaConf resolvers needed by the Hydra configs.

    Hydra is not initialized inside Ray workers, so the ``eval`` and ``torch``
    resolvers that ``main.py`` registers must be set up manually.
    """
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"np": np}))
    if not OmegaConf.has_resolver("torch"):
        OmegaConf.register_new_resolver("torch", lambda x: getattr(torch, x))



def build_ray_train_func(serialized_cfg: Dict[str, Any]) -> Any:
    """Return a training function that Ray Train workers will execute.

    The returned closure reuses the existing ``build_experiment`` /
    ``exec_task`` pipeline so that the model, dataset, and training loop
    are identical to a local run.

    Args:
        serialized_cfg: The Hydra config resolved and serialized via
            ``OmegaConf.to_container(cfg, resolve=True)``.
    """

    def train_func(config: dict) -> None:  # noqa: ARG001 – signature required by Ray
        from interactive_world_sim.experiments import build_experiment
        from interactive_world_sim.utils.wandb_utils import SpaceEfficientWandbLogger

        _register_omegaconf_resolvers()
        cfg = OmegaConf.create(serialized_cfg)
        OmegaConf.set_struct(cfg, False)

        # Mark this process as a Ray worker so exp_base uses Ray strategies.
        cfg._ray_worker = True  # noqa: SLF001

        # Set up W&B logger only on rank 0 to avoid duplicate runs.
        import ray.train

        logger = None
        rank = ray.train.get_context().get_world_rank()
        if rank == 0 and cfg.wandb.mode != "disabled":
            offline = cfg.wandb.mode != "online"
            logger = SpaceEfficientWandbLogger(
                name=cfg.name,
                save_dir=".",
                offline=offline,
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                log_model="all" if not offline else False,
                config=OmegaConf.to_container(cfg),
            )

        # Load checkpoint if specified.
        checkpoint_path = cfg.get("load", None)
        if checkpoint_path and not _is_run_id(checkpoint_path):
            pass  # Use the path directly.
        else:
            checkpoint_path = None

        experiment = build_experiment(cfg, logger, checkpoint_path)
        for task in cfg.experiment.tasks:
            experiment.exec_task(task)

    return train_func


def _is_run_id(value: str) -> bool:
    """Check whether a string looks like a W&B run ID (8-char alphanum)."""
    return len(value) == 8 and value.isalnum()


def submit_anyscale_job(cfg: DictConfig) -> Any:
    """Build and launch a Ray TorchTrainer from the Hydra config.

    Args:
        cfg: The full Hydra DictConfig (will be resolved and serialized).

    Returns:
        The ``ray.train.Result`` from ``TorchTrainer.fit()``.
    """
    from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer

    # Serialize the config for passing to Ray workers.
    # Use resolve=False to keep ${torch:float} as interpolation strings
    # (OmegaConf cannot serialize torch.dtype objects). The worker will
    # re-register resolvers and resolve them on access.
    _register_omegaconf_resolvers()
    serialized_cfg = OmegaConf.to_container(cfg, resolve=False)

    train_func = build_ray_train_func(serialized_cfg)

    scaling_config = ScalingConfig(
        num_workers=cfg.cluster.num_workers,
        use_gpu=cfg.cluster.use_gpu,
    )

    run_config = RunConfig(
        storage_path=cfg.cluster.storage_path,
        name=cfg.name,
        checkpoint_config=CheckpointConfig(num_to_keep=2),
        failure_config=FailureConfig(max_failures=3),
    )

    trainer = TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    return trainer.fit()
