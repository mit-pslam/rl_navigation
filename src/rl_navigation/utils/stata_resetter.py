from rl_navigation.utils.resetter import Resetter, randbounds
import numpy as np
import os
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import random
from rl_navigation.disaster import Twist, State, Pose

# Load random collection of points from the stata ground floor scene
dir_path = os.path.dirname(os.path.realpath(__file__))
stata_ground_floor = np.genfromtxt(
    os.path.join(dir_path, "stata_ground_floor.csv"), delimiter=","
)

# Load random collection of points from the stata basement scene
stata_basement = np.genfromtxt(
    os.path.join(dir_path, "stata_basement.csv"), delimiter=","
)

stata_ground_floor_car = np.genfromtxt(
    os.path.join(dir_path, "stata_ground_floor_car.csv"), delimiter=","
)


class StataResetter(Resetter):
    """Resets the drone randomly within the Stata center.

    Note: The idea here was that the user would have a collection of points from the scene.
    Turns out this is more general than the Stata Center.
    Could consider making this more general in the future.
    """

    def __init__(
        self,
        points=stata_ground_floor,
        dx: float = 0.5,
        angular: Tuple = ((0, 0), (0, 0), (0, 2 * np.pi)),  # x, y, z
    ):
        """Initialize.

        Parameters
        ----------
        points: np.ndarray
            List of ENU points from the scene (shape: Nx3)

        dx: float = 0.5,
            Random sampling distance around each point

        angular: Tuple = ((0, 0), (0, 0), (0, 2 * np.pi))
            Bounds for angular sampling (x, y, z)

        Returns
        -------
        StataResetter

        """
        self.points = points
        self.dlinear: Tuple = ((-dx, dx), (-dx, dx), (dx, dx))  # x, y, z
        self.angular = angular

    def sample(self) -> State:
        """Randomly samples a new environment state.

        Returns
        -------
        State

        """
        return State(
            ownship=Pose(
                position=random.choice(self.points)
                + np.array([randbounds(b) for b in self.dlinear]),
                orientation=R.from_euler(
                    "zyx", np.array([randbounds(b) for b in self.angular[::-1]])
                ).as_quat(),
            )
        )
