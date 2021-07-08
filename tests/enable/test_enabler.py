from copy import deepcopy
import math
from glob import glob

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from afterglow import (
    enable_checkpointing,
    enable_swag,
    enable_swag_from_checkpoints,
    load_swag_checkpoint,
)
from afterglow.trackers import SWAGTracker


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(1, 2)
        self.dropout = torch.nn.Dropout(0.5)
        self.batchnorm = torch.nn.BatchNorm1d(2)
        self.linear2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TupleReturningNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def forward(self, x):
        return self.net(x), 2 * self.net(x)


class DictReturningNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def forward(self, x):
        return {"logits": self.net(x), "something_else": torch.randn(10)}


def train_func(model, iterations):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(torch.randn(2, 1))
        if isinstance(output, tuple):
            output = output[0]
        elif isinstance(output, dict):
            output = output["logits"]
        loss = (output ** 2).sum()
        loss.backward()
        optimizer.step()


def test_output_type():

    model = Net()
    model = enable_swag(model, start_iteration=1, max_cols=2, update_period_in_iters=1)

    assert isinstance(model.trajectory_tracker, SWAGTracker)


def test_enable_uncertainty_initialises_swag_tracker_correctly():
    model = Net()
    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    assert model.trajectory_tracker.start_iteration == 20
    assert model.trajectory_tracker.update_period_in_iters == 10
    assert model.trajectory_tracker.max_cols == 30


@pytest.mark.parametrize("dropout", (True, False))
def test_sampling(dropout):
    model = Net()
    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    train_func(model, 50)

    samples = model.trajectory_tracker.predictive_samples(
        torch.tensor([[1.0], [1.0]]), num_samples=100, dropout=dropout
    )
    assert len(samples) == 100
    assert not_all_same(samples)


@pytest.mark.parametrize(
    "model, prediction_key", [(Net(), None), (DictReturningNet(), "logits")]
)
def test_uncertainty_prediction(model, prediction_key):
    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    train_func(model, 50)

    means, stds = model.trajectory_tracker.predict_uncertainty(
        torch.tensor([[1.0], [2.0]]), num_samples=100, prediction_key=prediction_key
    )
    assert not_all_same(means)
    assert not_all_same(stds)


def test_uncertainty_prediction_two_channels():
    model = TupleReturningNet()
    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    train_func(model, 50)

    means, stds = model.trajectory_tracker.predict_uncertainty(
        torch.tensor([[1.0], [2.0]]), num_samples=100
    )
    assert len(means[0]) == 2
    assert len(stds[0]) == 2
    assert not_all_same([mean[0] for mean in means])
    assert not_all_same([std[0] for std in stds])
    assert not_all_same([mean[1] for mean in means])
    assert not_all_same([std[1] for std in stds])


@pytest.mark.parametrize(
    "model, prediction_key, max_unreduced_minibatches",
    [(Net(), None, None), (DictReturningNet(), "logits", 2)],
)
def test_uncertainty_on_dataloader(model, prediction_key, max_unreduced_minibatches):
    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    train_func(model, 50)

    loader = DataLoader(
        [(torch.randn(1), torch.randn(1)) for _ in range(20)], batch_size=4
    )

    means, stds = model.trajectory_tracker.predict_uncertainty_on_dataloader(
        loader,
        num_samples=100,
        prediction_key=prediction_key,
        max_unreduced_minibatches=max_unreduced_minibatches,
    )
    assert len(means) == 20
    assert len(stds) == 20

    assert not_all_same(means)
    assert not_all_same(stds)


def test_weights_remain_constant_through_predicting_uncertainty():
    model = Net()

    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    train_func(model, 50)

    original_state_dict = deepcopy(model.state_dict())

    model.trajectory_tracker.predict_uncertainty(
        torch.tensor([[1.0], [2.0]]), num_samples=100
    )
    for key in model.state_dict():
        assert original_state_dict[key].equal(model.state_dict()[key])


def test_weights_remain_constant_through_predicting_uncertainty_on_dataloader():
    model = Net()

    model = enable_swag(
        model, start_iteration=20, update_period_in_iters=10, max_cols=30
    )

    train_func(model, 50)

    loader = DataLoader(
        [(torch.randn(1), torch.randn(1)) for _ in range(20)], batch_size=4
    )
    original_state_dict = deepcopy(model.state_dict())

    model.trajectory_tracker.predict_uncertainty_on_dataloader(loader, num_samples=100)
    for key in model.state_dict():
        assert original_state_dict[key].equal(model.state_dict()[key])


def test_swag_model_samples_same_before_and_after_loading():
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    elif hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)
    else:
        raise RuntimeError("This test doesn't work with your version of torch")
    torch.manual_seed(45)
    model = Net()
    model = enable_swag(model, start_iteration=0, max_cols=2, update_period_in_iters=1)
    train_func(model, 5)
    torch.manual_seed(45)
    before_load_samples = model.trajectory_tracker.predictive_samples(
        torch.tensor([[1.0], [1.0]]), num_samples=10
    )
    new_model = Net()
    new_model = enable_swag(
        new_model, start_iteration=0, max_cols=2, update_period_in_iters=1
    )
    new_model.load_state_dict(model.state_dict())
    torch.manual_seed(45)
    after_load_samples = new_model.trajectory_tracker.predictive_samples(
        torch.tensor([[1.0], [1.0]]), num_samples=10
    )
    assert all(
        [
            after_train_sample.equal(after_load_sample)
            for after_train_sample, after_load_sample in zip(
                before_load_samples, after_load_samples
            )
        ]
    )


def test_sampling_with_no_snapshots_tracked_raises_expcetion():
    model = Net()
    model = enable_swag(
        model, start_iteration=100, update_period_in_iters=1, max_cols=2
    )
    train_func(model, 40)
    with pytest.raises(RuntimeError):
        model.trajectory_tracker.predictive_samples(torch.tensor([[1.0], [1.0]]), 5)


@pytest.mark.parametrize("start_iteration, update_period", [(10, 5), (20, 3), (2, 8)])
def test_swag_tracker_takes_snapshots_at_correct_intervals(
    start_iteration, update_period
):
    model = Net()
    model = enable_swag(
        module=model,
        start_iteration=start_iteration,
        update_period_in_iters=update_period,
        max_cols=2,
    )
    num_train_iters = 40
    train_func(model, num_train_iters)
    num_iters_after_tracking = num_train_iters - start_iteration
    expected_num_tracked = math.ceil(num_iters_after_tracking / update_period)
    assert model.num_snapshots_tracked == expected_num_tracked


def test_checkpoint_uncertainty_saves_correct_checkpoints(tmp_path):
    tmp_path = tmp_path / "ckpts"
    model = Net()
    enable_checkpointing(
        module=model,
        start_iteration=10,
        update_period_in_iters=2,
        checkpoint_dir=tmp_path,
    )
    train_func(model, 20)
    checkpoints = sorted(glob(str(tmp_path / "*.ckpt")))
    expected_checkpoints = [str(tmp_path / f"iter_{i}.ckpt") for i in range(10, 20, 2)]
    assert checkpoints == expected_checkpoints

    saved_state_hashes = []
    for expected_checkpoint in expected_checkpoints:
        state_dict = torch.load(tmp_path / expected_checkpoint)
        model.load_state_dict(state_dict)
        with torch.no_grad():
            saved_state_hashes.append(
                np.sum([torch.sum(param) for param in model.parameters()])
            )

    assert len(set(saved_state_hashes)) == len(expected_checkpoints)


def test_offline_mode(tmp_path):
    tmp_path = tmp_path / "ckpts"
    model = Net()
    enable_checkpointing(
        module=model,
        start_iteration=0,
        update_period_in_iters=1,
        checkpoint_dir=tmp_path,
    )
    train_func(model, 20)
    model = Net()
    model = enable_swag_from_checkpoints(
        module=model, max_cols=20, checkpoint_dir=tmp_path
    )
    assert model.num_snapshots_tracked == 20

    samples = model.trajectory_tracker.predictive_samples(
        torch.tensor([[1.0], [1.0]]), num_samples=2
    )
    assert not_all_same(samples)


@pytest.mark.parametrize("num_datapoints_for_bn_update", (None, 2))
def test_tracker_updates_batchnorm(num_datapoints_for_bn_update):
    fake_dataloader = DataLoader(
        [(torch.randn(1), torch.randn(1)) for _ in range(12)], batch_size=3
    )
    model = Net()
    model = enable_swag(
        module=model,
        start_iteration=1,
        update_period_in_iters=1,
        max_cols=5,
        dataloader_for_batchnorm=fake_dataloader,
        num_datapoints_for_bn_update=num_datapoints_for_bn_update,
    )
    train_func(model, 30)
    pre_update_running_mean = model.batchnorm.running_mean.clone()
    model.trajectory_tracker.predictive_samples(
        torch.tensor([[1.0], [1.0]]), num_samples=2
    )
    assert not model.batchnorm.running_mean.equal(pre_update_running_mean)


def test_saving_and_loading_swag_checkpoint(tmp_path):
    model = Net()
    enable_swag(module=model, start_iteration=2, update_period_in_iters=2, max_cols=2)
    train_func(model, 20)
    pre_save_params = dict(model.named_parameters())
    model.trajectory_tracker.save(tmp_path / "swag.ckpt")
    model = Net()
    fake_dataloader = DataLoader(
        [(torch.randn(1), torch.randn(1)) for _ in range(10)], batch_size=5
    )
    load_swag_checkpoint(
        model, tmp_path / "swag.ckpt", dataloader_for_batchnorm=fake_dataloader
    )
    post_save_params = dict(model.named_parameters())
    for name in pre_save_params:
        assert post_save_params[name].equal(pre_save_params[name])


def not_all_same(samples):
    return not all([s.equal(samples[0]) for s in samples])
