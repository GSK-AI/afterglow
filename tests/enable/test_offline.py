from glob import glob

import numpy as np
import torch

from afterglow import enable_checkpointing, enable_swag_from_checkpoints
from afterglow._testing import Net, not_all_same


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
