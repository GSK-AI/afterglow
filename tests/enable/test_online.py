from afterglow import enable_swag
from afterglow.trackers import SWAGTracker

from afterglow._testing import Net


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
