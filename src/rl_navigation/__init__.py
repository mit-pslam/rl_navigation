"""Main module for rl_navigation code."""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from rl_navigation.math_utilities import unit_vector, angle_between_vectors
from rl_navigation.disaster import (
    Pose,
    State,
    Twist,
    Simulator,
    Observer,
    ActionMapper,
    DiscreteActionMapper,
    ObservationMapper,
    DisasterEnv,
    GoalDisasterEnv,
)
