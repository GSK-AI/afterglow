"""Adapted from torch.optim.swa_utils"""
from typing import Optional, Union
import torch


def update_bn(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    num_datapoints: Optional[int] = None,
):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader: dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model: model for which we seek to update BatchNorm
            statistics.
        device: If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
        num_datapoints: number of examples to use to perform the update.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    if num_datapoints is None:
        num_datapoints = len(loader.dataset)

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    datapoints_used_for_update = 0
    for input in loader:
        if datapoints_used_for_update == num_datapoints:
            break
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)
        input = input[: num_datapoints - datapoints_used_for_update]

        model(input)

        datapoints_used_for_update += len(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
