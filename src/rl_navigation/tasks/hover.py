from typing import Any, Callable, Dict, Optional, Tuple, Union, List

from rl_navigation.disaster import (
    DisasterEnv,
    DiscreteActionMapper,
    ObservationMapper,
    Twist,
    State,
    Pose,
    Simulator,
)
from rl_navigation.simulators.simple_simulator import SimpleSimulator
from rl_navigation.observers.flight_goggles_renderer import FlightGogglesRenderer

from gym import spaces
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
import cv2


class FGObserver(ObservationMapper):
    """Observation Mapper that extracts the RGB image from the message that
    flight goggles returns."""

    def __init__(
        self,
        renderer: FlightGogglesRenderer,
        observation_height: int,
        observation_width: int,
    ):
        """
        Initializes an FGObserver.

        Parameters
        ----------
        renderer: FlightGogglesRenderer
            Used to extract observation details

        observation_height: int
            resizes observations from flightgoggles to specified height

        observation_width: int
            resizes observations from flightgoggles to specified width

        """
        self.height = observation_height
        self.width = observation_width
        self.channels = renderer.config["camera_model"][0]["channels"]

    def map(self, observation: Dict[str, Any]) -> np.ndarray:
        """Returns flight goggles image as np.ndarray.

        Parameters
        ----------
        observation: Dict[str, Any]
            Dictionary of all observation

        Returns
        -------
        np.ndarray
            Observation array

        """
        img = observation["fg"]["image"]
        # https://towardsdatascience.com/image-read-and-resize-with-opencv-tensorflow-and-pil-3e0f29b992be
        return cv2.resize(img, (self.width, self.height))

    @property
    def space(self) -> spaces.Box:
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space.

        """
        return spaces.Box(
            low=0,
            high=255,
            dtype=np.uint8,
            shape=(self.height, self.width, self.channels),
        )


def randbounds(bounds: Tuple[float, float]):
    """Utility for sampling between two numbers."""
    return random.random() * (bounds[1] - bounds[0]) + bounds[0]


# add yaw actions with: Twist(angular=[0.0, 0.0, 0.2]) etc
action_mapper = DiscreteActionMapper(
    [
        Twist(linear=[0.0, 0.0, 0.1]),
        Twist(linear=[0.0, 0.0, -0.1]),
        Twist(linear=[0.0, 0.0, 0.0]),  # no-op
    ]
)


class HoverEnv(DisasterEnv):
    """Goal of this environment is to hover at a target height.

    This is intended to be a very simple environment for integration testing.
    """

    def __init__(
        self,
        flight_goggles: Union[str, FlightGogglesRenderer],
        goal_height: float = 5.0,
        reset_bounds: Tuple = ((-4, 4), (-4, 4), (2, 9)),
        max_steps: int = 100,
        action_delay: int = 10,
        yaw_range: Optional[Tuple[float, float]] = None,
        terminate_outside_bounds: bool = True,
        observation_height: int = 192,
        observation_width: int = 256,
    ):
        """Initialize.

        Parameters
        ----------
        flight_goggles: Union[str, FlightGogglesRenderer]
            Path to flight goggles or a FlightGogglesRenderer object.

        goal_height: float = 5.0
            Target hovering height.

        reset_bounds: Tuple = ((-4, 4), (-4, 4), (2, 9))
            Bounds for randomly sampling a drone's position on reset (x, y, and z).
            Note that z is up.

        max_steps: int = 100
            Maximum steps per episode

        action_delay: int = 10
            Number of steps to queue up actions

        yaw_range: Optional[Tuple[float, float]] = None
            If specified, initialize drone with yaws in the provided range (radians)
            per episode, otherwise, initialize between [0, 2*pi)

        terminate_outside_bounds: bool = True
            End the episode if the agent leaves the yaw_range specifying height bounds

        observation_height: int = 192
            resizes observations from flightgoggles to specified height

        observation_width: int = 256
            resizes observations from flightgoggles to specified width


        Returns
        -------
        HoverEnv

        """
        self.goal_height = goal_height
        self.reset_bounds = reset_bounds
        self.max_steps = max_steps
        self.n_steps = 0
        self.yaw_range = yaw_range
        self.terminate_outside_bounds = terminate_outside_bounds
        assert action_delay >= 0

        if isinstance(flight_goggles, str):
            flight_goggles = FlightGogglesRenderer(flight_goggles)

        super().__init__(
            SimpleSimulator(add_noise=True, action_queue_size=action_delay),
            [flight_goggles],
            action_mapper,
            FGObserver(
                flight_goggles,
                observation_height=observation_height,
                observation_width=observation_width,
            ),
        )

        self.observation_height = observation_height
        self.observation_width = observation_width
        # TODO: set this to the actual response from reset()
        self._last_observation = np.zeros(
            (observation_height, observation_width, 3), dtype=np.uint8
        )

    def compute_extras(
        self,
        previous_state: State,
        action: Twist,
        next_state: State,
        observation: Dict[str, Any],
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Determines step reward, whether environment is done, and any additional information.

        Parameters
        ----------
        previous_state: State
            The previous gym state.

        action: Action
            The agent's action

        next_state: State
            The next (or current) gym state.

        observation: Dict[str, Any]
            The observation dictionary.

        Returns
        -------
        Tuple[float, bool, Dict[str, Any]]
            The tuple elements correspond to:
                1. amount of reward returned after previous action,
                2. whether the episode has ended, in which case further step() calls will return undefined results, and
                3. auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        self.n_steps += 1

        new_height = next_state.ownship.position[2]
        # only considers height
        outside_bounds_z = self.terminate_outside_bounds and (
            new_height < self.reset_bounds[2][0] or self.reset_bounds[2][1] < new_height
        )

        dh = new_height - self.goal_height
        reward = -abs(dh)

        is_done = (
            observation["fg"]["hasCameraCollision"]
            or self.n_steps >= self.max_steps
            or outside_bounds_z
        )

        # for video output
        self._last_observation = observation["fg"]["image"]

        return (reward, is_done, dict())

    def disaster_reset(self) -> Tuple[State, Dict]:
        """Resets the environment and randomly samples a new drone location.

        Returns
        -------
        Tuple[State, Dict]
            Tuple includes the initial state and observation.
        """

        self.n_steps = 0

        if self.yaw_range is None:
            new_orientation = R.from_euler("z", 2 * np.pi * random.random()).as_quat()
        else:
            new_orientation = R.from_euler(
                "z", np.random.uniform(self.yaw_range[0], self.yaw_range[1])
            ).as_quat()

        state = State(
            ownship=Pose(
                position=np.array([randbounds(b) for b in self.reset_bounds]),
                orientation=new_orientation,
            )
        )
        return (state, self.observation_mapper.map(self._observe(state, dict())))

    def render(self, mode="rgb_array"):
        img = cv2.cvtColor(self._last_observation, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.observation_width, self.observation_height))
