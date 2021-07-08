import torch
from torch.utils.data import DataLoader

from afterglow import enable_swag, load_swag_checkpoint
from afterglow.trackers import SWAGTracker
from afterglow._testing import Net


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


def test_saving_and_loading_swag_checkpoint(tmp_path):
    model = Net()
    model = enable_swag(
        module=model, start_iteration=2, update_period_in_iters=2, max_cols=2
    )
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
