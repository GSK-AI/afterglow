from copy import deepcopy
from pathlib import Path
from typing import Optional, Union
from warnings import warn

import torch
from pydantic import StrictInt, conint, validate_arguments
from torch import nn
from torch.utils.data import DataLoader

from ..trackers import CheckpointTracker, SWAGTracker

_IntGreaterThanOne = conint(gt=1)
TrajectoryTracker = Union[CheckpointTracker, SWAGTracker]


def _enable_tracking(
    tracker: TrajectoryTracker,
):
    if not tracker.module.training:
        warn(
            "enabling trajectory tracking for a model in eval mode. "
            "Trajectory will NOT be tracked unless you do model.train() first.",
            RuntimeWarning,
        )
    tracker.module.trajectory_tracker = tracker
    tracker.module.trajectory_tracking_enabled = True
    tracker.module.register_forward_hook(_forward_hook)


def _forward_hook(instance: nn.Module, *_):
    if instance.trajectory_tracking_enabled and instance.training:
        instance.trajectory_tracker._update_uncertainty_buffers()
        instance.trajectory_tracker.iterations += 1


def _create_swag_buffers(instance: nn.Module, max_cols: _IntGreaterThanOne):
    for name, parameter in instance.named_parameters():
        name = name.replace(".", "_")
        instance.register_buffer(f"{name}_mean", deepcopy(parameter))
        instance.register_buffer(f"{name}_squared_mean", torch.zeros_like(parameter))
        instance.register_buffer(
            f"{name}_D_block",
            torch.zeros((max_cols, *parameter.shape), device=parameter.device),

        )
    instance.register_buffer("num_snapshots_tracked", torch.tensor(0, dtype=int))


@validate_arguments(config={"pre": True, "arbitrary_types_allowed": True})
def load_swag_checkpoint(
    base_module: nn.Module,
    path: Union[str, Path],
    dataloader_for_batchnorm: Optional[DataLoader] = None,
    num_datapoints_for_bn_update: Optional[StrictInt] = None,
):
    """Loads the state dict of a SWAG-enabled model that was saved via
    :code:`model.trajectory_tracker.save` into :code:`base_model` after enabling
    SWAG on :code:`base_model`.

    Args:
        module: An instance of the module to load the swag checkpoint into.
        path: Path to the checkpoint
        dataloader_for_batchnorm: see `enable_swag_from_checkpoints`.
        num_datapoints_for_bn_update: see `enable_swag_from_checkpoints`.
    """

    checkpoint_dict = torch.load(path)
    tracker = SWAGTracker(
        base_module,
        start_iteration=0,
        update_period_in_iters=1,
        dataloader_for_batchnorm=dataloader_for_batchnorm,
        num_datapoints_for_bn_update=num_datapoints_for_bn_update,
        max_cols=checkpoint_dict["max_cols"],
    )
    _enable_tracking(tracker)
    _create_swag_buffers(base_module, max_cols=checkpoint_dict["max_cols"])
    base_module.load_state_dict(checkpoint_dict["state_dict"])
