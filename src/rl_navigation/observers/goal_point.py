import random
from typing import Any, Dict, Optional

import numpy as np
import sophus as sp
from rl_navigation.disaster import Observer, Pose, State
from rl_navigation.utils.resetter import Resetter
from scipy.spatial.transform import Rotation as R


def SE3_from_pose(pose: Pose):
    """Returns an SE(3) from a Pose."""
    return sp.SE3(
        R.from_quat(pose.orientation).as_matrix(),
        pose.position,
    )


class GoalPoint(Observer):
    """Observer that returns the goal point and the position/orientation of the drone.

    Observer allows for the origin to correspond to the reset location of the drone.
    It will keep track of transforms.
    """

    def __init__(
        self,
        resetter: Resetter,
        use_start_as_origin: Optional[bool] = True,
        p_reset_origin: Optional[float] = 1.0,
        max_range_from_ownship: Optional[float] = np.Inf,
    ):
        """Initialize.

        Parameters
        ----------
        resetter: Resetter
            rl_navigation utility for resetting the environment state. Used to sample new goal locations.

        use_start_as_origin: Optional[bool] = true,
            Used for tracking progress to goal point.
            Setting to true will set the coordinate origin to the start location of the ownship.
            Setting to false will use the world coordinate system.

        p_reset_origin: Optional[float] = 1.0,
            Parameter that specifies the likelihood the origin also resets when the environment resets.
            If the origin does not reset, it continues from the previous episode.

        max_range_from_ownship: Optional[flat] = np.Inf
            Parameter to specify a maximum range that the goal point will be from the ownship

        Returns
        -------
        GoalPoint

        """
        assert p_reset_origin >= 0.0 and p_reset_origin <= 1.0
        assert max_range_from_ownship > 0

        self.resetter = resetter
        self.use_start_as_origin = use_start_as_origin
        self.p_reset_origin = p_reset_origin
        self.max_range_from_ownship = max_range_from_ownship

        self.T_origin_from_world = None
        self.goal_position = None

        self.reset(State())

    def reset(self, state: State):
        """Resets the goal_point position and origin, if enabled."""

        if not self.use_start_as_origin:
            self.T_origin_from_world = sp.SE3()
        elif random.random() <= self.p_reset_origin:
            self.T_origin_from_world = SE3_from_pose(state.ownship).inverse()

        while True:
            pos = self.resetter.sample().ownship.position
            if (
                np.linalg.norm(state.ownship.position - pos)
                < self.max_range_from_ownship
            ):
                break

        self.goal_position = self.T_origin_from_world * pos

    def observe(
        self, state: State, observation: Optional[Dict[str, Any]] = dict()
    ) -> Dict[str, Any]:
        """Adds goal point observation to observation dictionary

        Parameters
        ----------
        state: State
            The world state.

        observation: Optional[Dict[str, Any]] = dict()
            Dictionary with any previously populated observations.

        Returns
        -------
        Dict[str, Any]
            Observation dictionary with an entry added for goal point information.

        """

        if state.time == 0:  # environment reset occurred
            self.reset(state)

        T_world_from_current = SE3_from_pose(state.ownship)
        T_origin_from_current = self.T_origin_from_world * T_world_from_current

        observation["goal_point"] = {
            "goal_position": self.goal_position,
            "ownship_position": T_origin_from_current.translation(),
            "ownship_orientation": R.from_matrix(
                T_origin_from_current.rotation_matrix()
            ).as_quat(),
        }

        return observation


def vector_angle(v1: np.array, v2: np.ndarray) -> float:
    """Calculate angle in degrees
    between vectors `v1` and `v1`."""
    v1 = v1 / np.linalg.norm(v1, ord=2)
    v2 = v2 / np.linalg.norm(v2, ord=2)
    return np.rad2deg(np.arccos(np.dot(v1, v2)))


def goal_rel_agent(
    ownship_position: np.ndarray,
    ownship_orientation: np.ndarray,
    goal_position: np.ndarray,
) -> float:
    """Get angle of goal point relative to agent.

    Parameters
    ----------
    ownship_position: np.ndarray, shape=(3,)
        Current agent (x, y, z) position.
    ownship_orientation: np.ndarray, shape=(4,)
        Current agent orientation as quaternion.
    goal_position: np.ndarray, shape=(3,)
        Goal (x, y, z) position.

    Returns
    -------
    np.ndarray
        Angle of goal realtive to agent in degrees.
    """
    T_current_from_origin = SE3_from_pose(
        Pose(position=ownship_position, orientation=ownship_orientation)
    ).inverse()
    goal_rel_ownship = T_current_from_origin * goal_position
    own_heading = np.array([1, 0])
    return vector_angle(own_heading, goal_rel_ownship[:2])
