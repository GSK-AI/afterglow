from abc import ABC
from pydantic import conint

from .trackers import SWAGTracker

_IntGreaterThanOne = conint(gt=1)


class SwagEnabledModule(ABC):
    trajectory_tracker: SWAGTracker
