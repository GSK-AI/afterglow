from abc import ABC
from pydantic import conint
from torch.nn import Module

from .trackers import SWAGTracker

_IntGreaterThanOne = conint(gt=1)


class SwagEnabledModule(Module):
    trajectory_tracker: SWAGTracker
