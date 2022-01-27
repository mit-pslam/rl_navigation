from typing import Optional

import numpy as np
from gym import spaces

from rl_navigation.disaster import Twist, Twist2D, Twist2DWithDeclare


class ControlWithDeclareMapper:
    def __init__(self, action_mapper):
        self.action_mapper = action_mapper
        self.low = np.concatenate((self.action_mapper.low, np.array([-1])))
        self.high = np.concatenate((self.action_mapper.high, np.array([1])))

    def map(self, action: np.ndarray):
        control = self.action_mapper.map(action[..., :-1])
        declare = action[..., -1]
        return Twist2DWithDeclare(control.linear, control.angular, declare)

    def get_speed(self, action: Twist):
        return action.linear

    @property
    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.float32(self.low), high=np.float32(self.high), dtype=np.float32
        )


class DubinsActionMapper:
    """Action mapper for Dubins vehicle-like model
    (https://en.wikipedia.org/wiki/Dubins_path) used
    by the flightgoggles car sim
    (https://flightgoggles-documentation.scrollhelp.site/fg/Car-Dynamics.374996993.html).

    """

    def __init__(
        self,
        low: Optional[np.ndarray] = np.array([-1, -1]),
        high: Optional[np.ndarray] = np.array([1, 1]),
    ) -> None:
        """Initialize action mapper.

        Parameters
        ----------
        low: Optional[np.ndarray], shape=(2,), default=[-1, 1]
            Minimum velocity and steering angle.

        high: Optional[np.ndarray], shape=(2,), default=[1, 1]
            Maximum velocity and steering angle.
        """
        self.low = low
        self.high = high

    def map(self, action: np.ndarray) -> Twist2D:
        """Map input action to Twist2D.

        Parameters
        ----------
        action: np.ndarray, shape=(2,)
            Action consisting of velocity and steering angle.

        Returns
        -------
        Twist2D
            Twist2D object representing input `action`.
        """
        return Twist2D(linear=action[0], angular=action[1])

    def get_speed(self, action: Twist2D) -> float:
        """Get current vehicle speed."""
        return action.linear

    @property
    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.float32(self.low), high=np.float32(self.high), dtype=np.float32
        )


class ContinuousActionMapper:
    """Action mapper in Cartesian-like coordinates.
    Forward rate , strafe rate , vertical rate, and yaw rate.

    """

    def __init__(
        self, low=np.array([0.0, -0.1, -0.1, -0.1]), high=np.array([0.4, 0.1, 0.1, 0.1])
    ):
        self.low = low
        self.high = high

    def map(self, action) -> Twist:
        return Twist(linear=action[:3], angular=[0.0, 0.0, action[3]])

    def get_speed(self, action: Twist) -> float:
        return np.linalg.norm(action.linear)

    @property
    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.float32(self.low), high=np.float32(self.high), dtype=np.float32
        )


class PolarContinuousActionMapper:
    """Action mapper in Polar-like coordinates.
    Speed magnitude, speed angle, vertical rate, and yaw rate.

    """

    def __init__(
        self,
        low=np.array([0.0, -3.1415, -0.1, -0.1]),
        high=np.array([0.4, 3.1415, 0.1, 0.1]),
    ):
        self.low = low
        self.high = high

    def map(self, action) -> Twist:
        return Twist(
            linear=[
                action[0] * np.cos(action[1]),
                action[0] * np.sin(action[1]),
                action[2],
            ],
            angular=[0.0, 0.0, action[3]],
        )

    @property
    def space(self) -> spaces.Space:
        return spaces.Box(
            low=np.float32(self.low), high=np.float32(self.high), dtype=np.float32
        )
