from rl_navigation.disaster import Twist, State, Pose
import numpy as np
import random
from typing import Tuple
from scipy.spatial.transform import Rotation as R


class Resetter:
    """Abstract class for resetting a gym state. """

    def sample(self) -> State:
        raise NotImplementedError


def randbounds(bounds: Tuple[float, float]):
    """Utility for sampling between two numbers."""
    return random.random() * (bounds[1] - bounds[0]) + bounds[0]


class BoundsResetter(Resetter):
    """ Resets the drone randomly within a given bounds. """

    def __init__(
        self,
        linear: Tuple = ((-4, 4), (-4, 4), (2, 9)),  # x, y, z
        angular: Tuple = ((0, 0), (0, 0), (0, 2 * np.pi)),  # x, y, z
    ):
        self.linear = linear
        self.angular = angular

    def sample(self) -> State:
        """Randomly samples a new environment state.

        Returns
        -------
        State

        """
        return State(
            ownship=Pose(
                position=np.array([randbounds(b) for b in self.linear]),
                orientation=R.from_euler(
                    "zyx", np.array([randbounds(b) for b in self.angular[::-1]])
                ).as_quat(),
            )
        )
