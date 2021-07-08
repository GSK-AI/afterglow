from copy import deepcopy
import os
import math
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pydantic import StrictInt, conint
import torch
from torch import nn
from torch.distributions import Normal
from .batchnorm import update_bn
from torch.utils.data import DataLoader, RandomSampler


_IntGreaterThanOne = conint(gt=1)


class SWAGTracker:
    """
    Models the parameter distribution over the training trajectory as a multivariate
    gaussian in a low-rank space. See SWAG paper: https://arxiv.org/abs/1902.02476.

    Args:
        module: module to enable tracking for.
        start_iteration: iteration from which to begin fitting the approx posterior.
        max_cols: the posterior covariance matrix is dimensionally reduced to this
            dimensionality. Must be greater than 1.
        update_period_in_iters: how often to observe the parameters, in interations
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

    def __init__(
        self,
        module: nn.Module,
        start_iteration: StrictInt,
        max_cols: _IntGreaterThanOne,
        update_period_in_iters: StrictInt,
        dataloader_for_batchnorm: Optional[DataLoader] = None,
        num_datapoints_for_bn_update: Optional[StrictInt] = None,
    ):
        self.iterations = 0
        self.module = module
        self.update_period_in_iters = update_period_in_iters
        self.max_cols = max_cols
        self.start_iteration = start_iteration
        self.dataloader_for_batchnorm = dataloader_for_batchnorm

        self.num_datapoints_for_bn_update = num_datapoints_for_bn_update

    def _get_buffer_for_param(self, param_name, buffer_name):
        safe_name = param_name.replace(".", "_")
        return getattr(self.module, f"{safe_name}_{buffer_name}")

    def _set_buffer_for_param(self, param_name, buffer_name, value):
        safe_name = param_name.replace(".", "_")
        setattr(self.module, f"{safe_name}_{buffer_name}", value)

    def predictive_samples(
        self,
        *args,
        num_samples: StrictInt = 1,
        dropout: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> torch.Tensor:
        """Produce samples from the model's output distribution given
                inputs :code:`args` and :code:`kwargs`.

        Args:
            *args: positional arguments to pass to the model
            **kwargs: keyword arguments to pass to the model
            num_samples: number of samples to draw
            dropout: whether to use mc-dropout estimation together
                with SWAG when sampling
            device: device on which the model lives. Only
                needed for the batchnorm update step, ignored if
                this doesn't happen. "cpu" by default.

            Returns:
                list of samples from the predictive distribution
        """
        samples = []
        with self._in_eval_mode():
            if dropout:
                self._set_dropout_to_train()
            for _ in range(num_samples):
                self.sample_state(device)
                with torch.no_grad():
                    samples.append(self.module(*args, **kwargs))
        return samples

    def predictive_samples_on_dataloader(
        self,
        dataloader: DataLoader,
        num_samples: StrictInt = 1,
        dropout: bool = False,
        prediction_key: Optional[str] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Produce samples from the model's output distribution on
        the examples contained in :code:`dataloader`. We assume that dataloader
        returns (input_batch, label_batch) tuples.

        Args:
            dataloader: contains the examples to sample from the output
                distribution on
            num_samples: number of samples to draw
            dropout: whether to use mc-dropout estimation together
                with SWAG when sampling
            prediction_key: if the model returns a dict, this specifices
                which key contains its predictions; uncertainty will be computed
                using the contents of this key. Must be provided if the model
                returns a dict, otherwise ignored.
            device: device on which the model lives.

        Returns:
            list of samples from the predictive distribution
        """
        samples = []
        with self._in_eval_mode():
            if dropout:
                self._set_dropout_to_train()
            for _ in range(num_samples):
                self.sample_state(device)
                with torch.no_grad():
                    predictions = []
                    for input_, _ in dataloader:
                        input_ = input_.to(device)
                        batch_predictions = self.module(input_)
                        if prediction_key is not None:
                            batch_predictions = batch_predictions[prediction_key]
                        predictions.append(batch_predictions)
                    samples.append(torch.cat(predictions))
        return samples

    def _set_dropout_to_train(self):
        for m in self.module.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    def _update_tracked_state_dict(self, state_dict: Dict[str, nn.Parameter]):
        # PyTorch uses OrderedDicts for state_dict because they can have
        # attributes. It gives state_dict a _metadata attribute which can
        # affect how the state_dict is loaded. We have to copy this here.
        full_state_dict = OrderedDict({**state_dict, **self._untracked_state_dict()})
        full_state_dict._metadata = getattr(self.module.state_dict(), "_metadata", None)

        self.module.load_state_dict(full_state_dict)

    def predict_uncertainty(
        self,
        *args,
        num_samples: StrictInt = 1,
        dropout: bool = False,
        prediction_key: Optional[str] = None,
        device: str = "cpu",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and standard deviation of predictive distribution
        when model inputs are :code:`args`, :code:`kwargs`.

        Args:
            *args: positional arguments to pass to the model
            **kwargs: keyword arguments to pass to the model
            num_samples: number of samples to draw
            dropout: whether to use mc-dropout estimation together
                with SWAG when sampling
            prediction_key: if the model returns a dict, this specifices
                which key contains its predictions; uncertainty will be computed
                using the contents of this key. Must be provided if the model
                returns a dict, otherwise ignored.
            device: device on which the model lives. Only
                needed for the batchnorm update step, ignored if
                this doesn't happen. "cpu" by default.


        Returns:
            mean and standard deviation of the predictive distribution
            at the inputs :code:`args`, :code:`kwargs`
        """
        origial_state_dict = deepcopy(self.module.state_dict())
        mean = None
        var = None
        for i in range(num_samples):
            prediction = self.predictive_samples(
                *args, num_samples=1, dropout=dropout, device=device, **kwargs
            )[0]
            if isinstance(prediction, tuple):
                mean, var = self.accumulate_mean_and_var_for_tuple(
                    prediction, mean, var, i
                )
            elif isinstance(prediction, dict):
                mean, var = self.accumulate_mean_and_var_for_dict(
                    prediction, mean, var, i, prediction_key
                )
            else:
                mean, var = self.accumulate_mean_and_var_for_scalar(
                    prediction, mean, var, i
                )
        if isinstance(mean, tuple):
            return mean, [channel_var.sqrt() for channel_var in var]
        self.module.load_state_dict(origial_state_dict)
        return mean, var.sqrt()

    def accumulate_mean_and_var_for_tuple(self, prediction, mean, var, step):
        if mean is None:
            mean = [0 for _ in prediction]
        if var is None:
            var = [0 for _ in prediction]

        new_stats_per_channel = [
            self.accumulate_mean_and_var_for_scalar(
                prediction[i], mean[i], var[i], step
            )
            for i in range(len(prediction))
        ]
        return zip(*new_stats_per_channel)

    def accumulate_mean_and_var_for_dict(
        self, prediction, mean, var, step, prediciton_key
    ):
        prediction = prediction[prediciton_key]
        return self.accumulate_mean_and_var_for_scalar(prediction, mean, var, step)

    @staticmethod
    def accumulate_mean_and_var_for_scalar(prediction, mean, var, step):
        if mean is None:
            mean = 0
        if var is None:
            var = 0
        mean = step / (step + 1) * mean + prediction / (step + 1)
        var = step / (step + 1) * var + ((prediction - mean) ** 2) / (step + 1)
        return mean, var

    def _mean_and_std_of_predictive_samples(
        self,
        predictive_samples: Union[
            List[torch.Tensor], List[Tuple[torch.Tensor, ...]], dict
        ],
        prediction_key: Optional[str] = None,
    ):
        if isinstance(predictive_samples[0], tuple):
            return self._mean_and_std_per_output_dim(predictive_samples)

        elif isinstance(predictive_samples[0], dict):
            if prediction_key is None:
                raise TypeError(
                    "When predicting uncertainty for a model that returns "
                    "a dict, you must pass 'prediction_key' (the key containing "
                    "predictions)."
                )
            try:
                predictive_samples = [
                    sample[prediction_key] for sample in predictive_samples
                ]
            except KeyError:
                raise KeyError(
                    f"'prediction_key' {prediction_key} not in model output."
                    f"Got keys {list(predictive_samples[0].keys())}"
                )

        predictive_samples = torch.stack(predictive_samples)
        return predictive_samples.mean(0), predictive_samples.std(0)

    @staticmethod
    def _mean_and_std_per_output_dim(predictive_samples):
        splits = zip(*predictive_samples)
        mean_and_stds = []
        for out_channel in splits:
            tensorized_out_channel = torch.stack(out_channel)
            mean_and_stds.append(
                (tensorized_out_channel.mean(0), tensorized_out_channel.std(0))
            )
        return tuple(mean_and_stds)

    def predict_uncertainty_on_dataloader(
        self,
        dataloader,
        num_samples: StrictInt = 1,
        dropout: bool = False,
        prediction_key: Optional[str] = None,
        max_unreduced_minibatches: Optional[int] = None,
        device: str = "cpu",
    ):
        """Predict the mean and standard deviation of the model's output
        distribution on the examples contained in :code:`dataloader`. We assume
        that dataloader returns (input_batch, label_batch) tuples.

        Args:
            dataloader: contains the examples to sample from the output
                distribution on
            num_samples: number of samples to draw
            dropout: whether to use mc-dropout estimation together
                with SWAG when sampling
            prediction_key: if the model returns a dict, this specifices
                which key contains its predictions; uncertainty will be computed
                using the contents of this key. Must be provided if the model
                returns a dict, otherwise ignored.
            max_unreduced_minibatches: the maximum number of minibatches whose
                samples to keep in memory before reducing to compute mean
                and standard deviation. Runs faster for larger values but
                takes more memory. If not provided, we accumulate samples
                for the whole dataloader before reducing.
            device: device on which the model lives. Only
                needed for the batchnorm update step, ignored if
                this doesn't happen. "cpu" by default.

        Returns:
            mean and standard deviation of the predictive distribution for
            the examples contained in :code:`dataloader`
        """
        origial_state_dict = deepcopy(self.module.state_dict())

        if max_unreduced_minibatches is None:
            max_unreduced_minibatches = len(dataloader)
        num_model_samples = math.ceil(max_unreduced_minibatches / len(dataloader))
        inputs_for_each_sample = [[] for _ in range(num_model_samples)]
        for i, (input_, _) in enumerate(dataloader):
            inputs_for_each_sample[i % num_model_samples].append(input_)

        means = []
        stds = []
        with self._in_eval_mode():
            if dropout:
                self._set_dropout_to_train()
            for input_batch_group in inputs_for_each_sample:
                predictive_samples = []
                for _ in range(num_samples):
                    self.sample_state(device)
                    with torch.no_grad():
                        predictions_for_this_sample = []
                        for input_ in input_batch_group:
                            input_ = input_.to(device)
                            batch_predictions = self.module(input_)
                            if prediction_key is not None:
                                batch_predictions = batch_predictions[prediction_key]
                            predictions_for_this_sample.append(batch_predictions)
                        predictions_for_this_sample = torch.cat(
                            predictions_for_this_sample
                        )
                        predictive_samples.append(predictions_for_this_sample)
                predictive_samples = torch.stack(predictive_samples)
                means.append(predictive_samples.mean(0))
                stds.append(predictive_samples.std(0))

        self.module.load_state_dict(origial_state_dict)
        return torch.cat(means), torch.cat(stds)

    def _untracked_state_dict(self):
        filtered_state_dict = {}
        tracked_keys = set(name for name, _ in self.module.named_parameters())
        for k, v in self.module.state_dict().items():
            if k not in tracked_keys:
                filtered_state_dict[k] = v
        return filtered_state_dict

    @contextmanager
    def _in_eval_mode(self):
        was_in_train_mode = self.module.training
        tracking_was_enabled = self.module.trajectory_tracking_enabled
        try:
            self.module.trajectory_tracking_enabled = False
            self.module.eval()
            yield
        finally:
            if was_in_train_mode:
                self.module.train()
            if tracking_was_enabled:
                self.module.trajectory_tracking_enabled = False

    def _bn_loader_does_not_shuffle(self):
        return hasattr(self.dataloader_for_batchnorm, "sampler") and isinstance(
            self.dataloader_for_batchnorm, RandomSampler
        )

    def _sample_state_dict(self) -> dict:
        if self.module.num_snapshots_tracked == 0:
            raise RuntimeError(
                "Attempted to sample weights using a tracker that has "
                "recorded no snapshots"
            )

        sampled = {}

        _, first_param = next(iter(self.module.named_parameters()))
        K_sample = (
            Normal(torch.zeros(self.max_cols), torch.ones(self.max_cols))
            .sample()
            .to(first_param.device)
        )

        for name, _ in self.module.named_parameters():
            mean = self._get_buffer_for_param(name, "mean")
            squared_mean = self._get_buffer_for_param(name, "squared_mean")
            d_block = self._get_buffer_for_param(name, "D_block")
            p1 = mean
            p2 = Normal(
                torch.zeros_like(mean),
                (0.5 * (squared_mean - mean.pow(2)).clamp(1e-30)).sqrt(),
            ).sample()
            shape = d_block.shape[1:]
            aux = d_block.reshape(self.max_cols, -1)
            p3 = torch.matmul(K_sample, aux).reshape(shape) / math.sqrt(
                2 * (self.max_cols - 1)
            )
            sampled[name] = p1 + p2 + p3
        return sampled

    def _update_uncertainty_buffers(self):
        if self.iterations >= self.start_iteration:
            if (
                self.iterations - self.start_iteration
            ) % self.update_period_in_iters == 0:
                if self.module.num_snapshots_tracked == 0:
                    with torch.no_grad():
                        for name, parameter in self.module.named_parameters():
                            mean = self._get_buffer_for_param(name, "mean")
                            squared_mean = self._get_buffer_for_param(
                                name, "squared_mean"
                            )
                            self._set_buffer_for_param(name, "mean", mean + parameter)
                            self._set_buffer_for_param(
                                name, "squared_mean", squared_mean + parameter.pow(2)
                            )
                else:
                    with torch.no_grad():
                        for name, parameter in self.module.named_parameters():
                            mean = self._get_buffer_for_param(name, "mean")
                            squared_mean = self._get_buffer_for_param(
                                name, "squared_mean"
                            )
                            d_block = self._get_buffer_for_param(name, "D_block")
                            self._set_buffer_for_param(
                                name,
                                "mean",
                                (self.module.num_snapshots_tracked * mean + parameter)
                                / (self.module.num_snapshots_tracked + 1),
                            )
                            self._set_buffer_for_param(
                                name,
                                "squared_mean",
                                (
                                    self.module.num_snapshots_tracked * squared_mean
                                    + parameter.pow(2)
                                )
                                / (self.module.num_snapshots_tracked + 1),
                            )
                            d_block = d_block.roll(1, dims=0)
                            d_block[0] = parameter - mean
                            self._set_buffer_for_param(name, "D_block", d_block)

                self.module.num_snapshots_tracked += 1

    def sample_state(self, device: str = "cpu"):
        """Update the state of the tracker's :code:`module` with a sample from
        the estimated distribution over parameters.

        Args:
            device: where to send the data duing batchnorm update. Ignored
                if we don't do batchnorm update.
        """
        sampled_state_dict = self._sample_state_dict()
        self._update_tracked_state_dict(sampled_state_dict)
        if self.dataloader_for_batchnorm is not None:
            tracking_was_enabled = self.module.trajectory_tracking_enabled
            self.module.trajectory_tracking_enabled = False
            update_bn(
                self.dataloader_for_batchnorm,
                self.module,
                device=device,
                num_datapoints=self.num_datapoints_for_bn_update,
            )
            self.module.trajectory_tracking_enabled = tracking_was_enabled

    def save(self, path: Union[str, Path]):
        """Save the uncertainty-enabled model so that it can be
        loaded using :code:`afterglow.load_swag_checkpoint`.

        Args:
            path: where to save the checkpoint
        """
        checkpoint_dict = {
            "state_dict": self.module.state_dict(),
            "max_cols": self.max_cols,
        }
        torch.save(checkpoint_dict, path)


class CheckpointTracker:
    """Records model state throughout training.

    This class is intended for convenient offline swag enabling, see
    :code:`afterglow.enable.offline.enable_swag_from_checkpoints`.

    Args:
        module: module to enable tracking for.
        start_iteration: iteration from which to begin recording snapshots.
        update_period_in_iters: how often to observe the parameters, in interations
        checkpoint_dir: directory in which to store the snapshots.
    """

    def __init__(
        self,
        module: nn.Module,
        start_iteration: int,
        update_period_in_iters: int,
        checkpoint_dir: Union[Path, str],
    ):
        self.iterations = 0
        self.module = module
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_dir = checkpoint_dir
        self.start_iteration = start_iteration
        self.update_period_in_iters = update_period_in_iters

    def _update_uncertainty_buffers(self):
        if self.iterations >= self.start_iteration:
            if (
                self.iterations - self.start_iteration
            ) % self.update_period_in_iters == 0:
                if _is_lead_process():
                    torch.save(
                        self.module.state_dict(),
                        self.checkpoint_dir / f"iter_{self.iterations}.ckpt",
                    )


def _is_lead_process():
    if "GLOBAL_RANK" in os.environ:
        return os.environ["GLOBAL_RANK"] == "0"
    return os.environ.get("LOCAL_RANK", "0") == "0"
