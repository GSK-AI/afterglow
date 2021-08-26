from pathlib import Path
from typing import Callable, Optional, Union

import torch
from parse import parse
from pydantic import StrictInt, conint, validate_arguments
from torch import nn
from torch.utils.data import DataLoader

from ..trackers import CheckpointTracker, SWAGTracker
from .base import _create_swag_buffers, _enable_tracking
from .._types import SwagEnabledModule

_IntGreaterThanOne = conint(gt=1)


def _iter_from_filepath(filepath: Path):
    iter_str = parse("iter_{}", filepath.stem)[0]
    return int(iter_str)


@validate_arguments(config={"pre": True, "arbitrary_types_allowed": True})
def enable_swag_from_checkpoints(
    module: nn.Module,
    max_cols: _IntGreaterThanOne,
    checkpoint_dir: Path,
    start_iteration: StrictInt = 0,
    checkpoint_pattern: str = "*.ckpt",
    checkpoint_sort_key: Callable[[str], float] = _iter_from_filepath,
    dataloader_for_batchnorm: Optional[DataLoader] = None,
    num_datapoints_for_bn_update: Optional[StrictInt] = None,
) -> SwagEnabledModule:
    """Equips a model with SWAG-based uncertainty estimation by reconstructing
    the training trajectory from a series of saved checkpoints. Useful if you
    have non-SWAG-enabled checkpoints saved for an expensive-to-train model
    that you want to try SWAG on.

    Calling this on a model equips it with a :code:`trajectory_tracker` object which
    provides SWAG-sampling methods.
    Example usage:
    ::
        my_model = MyModel()
        enable_swag_from_checkpoints(
            my_model,
            max_cols=10,
            checkpoint_dir="./checkpoints",
            checkpoint_pattern="*.pt",
            checkpoint_sort_key=lambda x: int(str(x.stem)),
        ) # assuming your checkpoints are of the form "./checkpoints/<epoch-num>.pt"
        my_model.trajectory_tracker.predict_uncertainty(data)


    Args:
        module: The module to enable SWAG for.
        max_cols: Number of checkpoints to use in calculating the SWAG covariance
            matrix. Values between 10 and 20 are usually reasonable. See SWAG paper
            for details.
        checkpoint_dir: Directory where the checkpoints from the training run
            you want to apply SWAG to are found.
        start_iteration: iteration from which to begin recording snapshots.
        checkpoint_pattern: A glob pattern that, when applied to :code:`checkpoint_dir`,
            will select the checkpoints you want to include.
        checkpoint_sort_key: Function mapping from checkpoint filenames to a number,
            where the number can be used to order the checkpoints.
        dataloader_for_batchnorm: if this is is provided, we update the model's
            batchnorm running means and variances every time we sample a new set of
            parameters using the data in the dataloader. This is slow but can improve
            performance significantly. See SWAG paper, and
            :code:`torch.optim.swa_utils.update_bn`. Note that the
            assumptions made about what iterating over the dataloader returns are
            the same as those in :code:`torch.optim.swa_utils.update_bn`: it's
            assumed that iterating produces a sequence of (input_batch, label_batch)
            tuples.
        num_datapoints_for_bn_update: Number of training example to use to perfom the
            batchnorm update.
            If :code:`None`, we use the whole dataset, as in the original SWAG
            paper. It's better to better to set this value to 1 and increase the
            number of SWAG samples drawn when predicting in online mode
            (one example at a time) rather than in batch mode.
            If this is not None, dataloader_for_batchnorm must be
            initialised with :code:`shuffle=True`
    """
    tracker = SWAGTracker(
        module=module,
        start_iteration=0,
        update_period_in_iters=1,
        dataloader_for_batchnorm=dataloader_for_batchnorm,
        num_datapoints_for_bn_update=num_datapoints_for_bn_update,
        max_cols=max_cols,
    )
    _enable_tracking(
        tracker,
    )
    _create_swag_buffers(module, max_cols)
    _populate_uncertainty_buffers_from_checkpoints(
        module,
        checkpoint_dir,
        checkpoint_pattern,
        checkpoint_sort_key,
        start_iteration=start_iteration,
    )
    return module


def _populate_uncertainty_buffers_from_checkpoints(
    model: nn.Module,
    checkpoint_dir: Union[Path, str],
    checkpoint_pattern: str,
    checkpoint_sort_key: Callable[[str], float],
    start_iteration: int = 0,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = checkpoint_dir.glob(checkpoint_pattern)
    checkpoints_in_order = sorted(checkpoints, key=checkpoint_sort_key)
    for checkpoint in checkpoints_in_order[start_iteration:]:
        state_dict = torch.load(checkpoint)
        model.trajectory_tracker._update_tracked_state_dict(state_dict)
        model.trajectory_tracker._update_uncertainty_buffers()
        model.trajectory_tracker.iterations += 1


@validate_arguments(config={"pre": True, "arbitrary_types_allowed": True})
def enable_checkpointing(
    module: nn.Module,
    start_iteration: StrictInt,
    checkpoint_dir: Union[str, Path],
    update_period_in_iters: Optional[StrictInt] = None,
):
    """Convenience function to save checkpoints during a run in a format
    that will work easily with :code:`enable_swag_from_checkpoints`. If you use
    this function for checkpointing, you can call :code:`enable_swag_from_checkpoints`
    with :code:`checkpoint_pattern` and :code:`checkpoint_sort_key` left as the
    defaults.

    Args:
        module: the module to enable checkpointing for
        start_iteration: iteration at which to start saving checkpoints
        update_period_in_iters: how often to save the parameters, in interations
        checkpoint_dir: directory to save the checkpoints in. Need not exist.

    """
    tracker = CheckpointTracker(
        module=module,
        start_iteration=start_iteration,
        checkpoint_dir=checkpoint_dir,
        update_period_in_iters=update_period_in_iters,
    )
    _enable_tracking(tracker)
