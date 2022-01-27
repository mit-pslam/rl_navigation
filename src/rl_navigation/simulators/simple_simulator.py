from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import namedtuple

import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
import transforms3d
import sophus as sp
from rl_navigation.disaster import State, Twist, Pose, Simulator


class SimpleSimulator(Simulator):
    """A simple, discrete simulator, with an optional delay (input lag)
    specified as a queue length for actions."""

    def __init__(
        self, add_noise: bool = False, action_queue_size: Optional[int] = None
    ):
        """ Initialize.

        Parameters
        ----------

        add_noise:
            adds a random perturbation to linear-z and angular velocities
            with zero mean and fixed stdevs (see source)
        action_queue_size: Optional[int]
            If specified, the number of actions to accumulate in a FIFO queue

        Returns
        -------
        SimpleSimulator
        """
        super().__init__()
        if action_queue_size is not None and action_queue_size > 0:
            self.action_queue = deque(maxlen=action_queue_size + 1)
        else:
            self.action_queue = deque(maxlen=1)  # no delay
        self.add_noise = add_noise

    def step(self, state: State, action: Twist) -> Tuple[State, Dict[str, Any]]:
        """Steps world state forward one step.

        Parameters
        ----------
        state: State
            The world state.

        action: Twist
            The action to apply to the drone.

        Returns
        -------
        Tuple[State, Dict[str, Any]]
            A tuple that includes:
                - The next world state after applying action.
                - Drone observations (may be empty).

        """
        self.action_queue.append(action)
        if len(self.action_queue) < self.action_queue.maxlen:
            # return current state while action queue accumulates
            return (State(time=state.time + 1, ownship=state.ownship), dict())
        action = self.action_queue.popleft()

        T_world_from_agent = sp.SE3(
            R.from_quat(state.ownship.orientation).as_matrix(), state.ownship.position
        )

        velocity = np.concatenate([action.linear, action.angular])
        if self.add_noise:
            # only adds linear-z and angular noise
            z_noise = np.random.normal(0, 0.001)
            noise = np.concatenate(
                [[0, 0, z_noise], np.random.normal(0, 0.001, size=3)]
            )
            velocity = velocity + noise

        delta_time = 1.0  # hardcoding for now

        # https://strasdat.github.io/Sophus/class_Sophus_SE3.html#index-17
        new_state = T_world_from_agent * sp.SE3.exp(velocity * delta_time)

        position = new_state.translation()  # extract translation
        orientation = R.from_matrix(new_state.rotation_matrix()).as_quat()  # xyzw order

        return (
            State(
                time=state.time + 1,
                ownship=Pose(position=position, orientation=orientation),
            ),
            dict(),
        )
