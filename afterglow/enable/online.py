from typing import Optional

from pydantic import StrictInt, validate_arguments
from torch import nn
from torch.utils.data import DataLoader

from ..trackers import SWAGTracker
from .base import _create_swag_buffers, _enable_tracking

from .._types import _IntGreaterThanOne, SwagEnabledModule


@validate_arguments(config={"pre": True, "arbitrary_types_allowed": True})
def enable_swag(
    module: nn.Module,
    start_iteration: StrictInt,
    max_cols: _IntGreaterThanOne,
    update_period_in_iters: StrictInt,
    dataloader_for_batchnorm: Optional[DataLoader] = None,
    num_datapoints_for_bn_update: Optional[StrictInt] = None,
) -> SwagEnabledModule:
    """Enables online trajectory tracking. Models passed to this function
    and subsequently trained will update SWAG buffers during training, and
    will be equiped with the ability to sample from the SWAG posterior
    via a :code:`trajectory_tracker` object once training is done.

    Calling this on a model equips it with a `trajectory_tracker` object which
    provides SWAG-sampling methods.
    Example usage:
    ::
        my_model = MyModel()
        enable_swag(
            my_model,
            max_cols=10,
            update_period_in_iters: len(train_dataloader), # update once per epoch
        )
        trainer.fit(my_model, train_dataloader)
        my_model.trajectory_tracker.predict_uncertainty(data)

    Args:
        max_cols: Number of checkpoints to use in calculating the SWAG covariance
            matrix. Values between 10 and 20 are usually reasonable. See SWAG paper
            for details.
        update_period_in_iters: The interval between SWAG buffer updates, in iterations.
            This is usually set to the number of iterations per epoch, which you can
            get with :code:`len(train_dataloader)`.
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
        start_iteration=start_iteration,
        update_period_in_iters=update_period_in_iters,
        dataloader_for_batchnorm=dataloader_for_batchnorm,
        num_datapoints_for_bn_update=num_datapoints_for_bn_update,
        max_cols=max_cols,
    )
    _enable_tracking(
        tracker,
    )
    _create_swag_buffers(module, max_cols)
    return module
