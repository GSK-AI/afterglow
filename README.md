![Afterglow Logo](static/img/afterglow.svg)

---

![Coverage](static/img/coverage.svg)

Afterglow provides your PyTorch models with uncertainty estimation capabilites. It's designed to work with any PyTorch model, with a minimum of fuss. It uses [SWAG](https://arxiv.org/abs/1902.02476) as its core uncertainty esitmation method.

With afterglow, you can transform code that trains point-estimating models into code that trains uncertainty-estimating models using a single line:

```python
from afterglow import enable_swag
enable_swag(
    model,
    start_iteration=100 * len(train_dataloader), # start tracking at epoch 100
    update_period_in_iters=len(train_dataloader), # update posterior every epoch
    max_cols=20,
)
```

After training your model as usual, you can obtain uncertainty estimates like so:

```python
mean, std = model.trajectory_tracker.predict_uncertainty(x, num_samples=30)
```

You can sample single instances of the model from the SWAG posterior:

```python
model.trajectory_tracker.sample_state()
sample_at_x = model(x)
```

You can efficiently predict on an entire dataloader, drawing one sample for each pass over the dataset:

```python
dataset_means, dataset_stds = model.trajectroy_tracker.predict_uncertainty_on_dataloader(
    dataloader=dataloder, num_samples=30
)
```

If you pass a dataloader to `enable_swag`, the SWAG batchnorm update step will be taken care of for you:

```python
from afterglow import enable_swag
enable_swag(
    model,
    start_iteration=100 * len(train_dataloader),
    update_period_in_iters=len(train_dataloader),
    max_cols=20,
    dataloader_for_batchnorm=train_dataloader, # now we'll do the bn update when we sample
)
```

You can speed online inference up by limiting the number of samples used to update batchnorm parameters:

```python
from afterglow import enable_swag
enable_swag(
    model,
    start_iteration=100 * len(train_dataloader),
    update_period_in_iters=len(train_dataloader),
    max_cols=20,
    dataloader_for_batchnorm=train_dataloader,
    num_datapoints_for_bn_update=100, # now we'll only use 100 examples fo the bn update
)
```

See the documentation, and the example app in `/example`, for more!
