import torch


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


def not_all_same(samples):
    return not all([s.equal(samples[0]) for s in samples])
