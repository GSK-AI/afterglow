import os

import pytest
import torch
from torch.distributed import init_process_group
from torch.multiprocessing import start_processes
from torch.nn.parallel import DistributedDataParallel

from afterglow import enable_swag


def train(model, iterations):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(torch.randn(2, 1))
        loss = (output ** 2).sum()
        loss.backward()
        optimizer.step()


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


def distributed_train_func(rank, model, iterations):
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "13243"
    init_process_group(backend="nccl", world_size=4)
    torch.cuda.empty_cache()
    torch.cuda.set_device(rank)
    model.to("cuda")
    buffer = model.trajectory_tracker.get_param_average("linear1.weight").clone()
    model = DistributedDataParallel(model, device_ids=[rank])
    train(model, iterations=iterations)
    assert model.module.trajectory_tracker.num_snapshots_tracked == 5
    assert not model.module.trajectory_tracker.get_param_average(
        "linear1.weight"
    ).equal(buffer)


@pytest.mark.integration
def test_swag_with_ddp():
    net = Net()
    enable_swag(net, start_iteration=0, max_cols=2, update_period_in_iters=2)
    start_processes(distributed_train_func, nprocs=4, args=(net, 10))
