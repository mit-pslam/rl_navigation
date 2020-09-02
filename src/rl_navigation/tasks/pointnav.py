from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from yacs.config import CfgNode
from rl_navigation.config import get_cfg_defaults
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
import rl_navigation.fg_msgs as fg_msgs

from gym import spaces
import numpy as np
import random
import cv2
from scipy.spatial.transform import Rotation as R


class FGObserver(ObservationMapper):
    """Observation Mapper that extracts the RGB image from the message that
    flight goggles returns."""

    def _zero_mean(self, img):
        return cv2.normalize(
            img.astype(np.float), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX
        )

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
        return self._zero_mean(observation["fg"].images[0])

    @property
    def space(self) -> spaces.Dict:
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Dict
            The observation space.

        """
        return spaces.Box(
            -1, 1, dtype=np.float, shape=(fg_msgs.HEIGHT, fg_msgs.WIDTH, 3)
        )
        # TODO(mmaz) 'the model does not support input space of type Dict'
        # return spaces.Dict(
        #     {
        #         "observation": spaces.Box(
        #             -1, 1, dtype=np.float, shape=(fg_msgs.HEIGHT, fg_msgs.WIDTH, 3)
        #         ),
        #         "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,)),
        #         "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,)),
        #     }
        # )


# TODO(mmaz) change to GoalDisasterEnv
class PointnavEnv(DisasterEnv):
    """Goal of this environment is to navigate to a desired location
    """

    def __init__(self, configuration_file: Optional[str]):
        """Pointgoal Navigation.

        Parameters
        ----------
        configuration_file: Optional[str]
            YACS Config filepath

        Returns
        -------
        PointgoalEnv

        """
        cfg = get_cfg_defaults()

        if configuration_file is not None:
            cfg.merge_from_file(configuration_file)
        cfg.freeze()

        self.config = cfg
        if self.config.FLIGHTGOGGLES.BINARY == "":
            raise ValueError(
                "Please specify a valid binary path to FlightGoggles in experiment.yaml"
            )

        self.max_steps = self.config.POINTNAV.MAXIMUM_EPISODE_LENGTH
        self.minimum_reward_success = self.config.POINTNAV.MINIMUM_REWARD_SUCCESS
        self.starting_poses = np.load(self.config.INITIAL_CONDITIONS.STARTING_POSES)
        self.goal_poses = np.load(self.config.INITIAL_CONDITIONS.STARTING_POSES)
        self.goal_position = None

        self.n_steps = 0

        super().__init__(
            SimpleSimulator(),
            [
                FlightGogglesRenderer(
                    flight_goggles_path=self.config.FLIGHTGOGGLES.BINARY,
                    pose_port=self.config.FLIGHTGOGGLES.POSE_PORT,
                    video_port=self.config.FLIGHTGOGGLES.VIDEO_PORT,
                    publish_period=self.config.FLIGHTGOGGLES.PUBLISH_PERIOD,
                ),
                # PointnavObserver() #TODO(mmaz) can't access self.goal_position
            ],
            # TODO(mmaz) move to ROS ENU convention
            DiscreteActionMapper(
                [
                    Twist(linear=[0.0, 0.0, 0.5]),
                    Twist(angular=[0.0, 0.2, 0.0]),
                    Twist(angular=[0.0, -0.2, 0.0]),
                ]
            ),
            FGObserver(),
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

        reward = self.compute_reward(
            achieved_goal=next_state.ownship.position,
            desired_goal=self.goal_position,
            info=dict(),
        )
        is_done = (
            observation["fg"].renderMetadata.hasCameraCollision
            or self.n_steps >= self.max_steps
            or reward > self.minimum_reward_success
        )

        return (reward, is_done, dict())

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(desired_goal - achieved_goal)
        return 1.0 - np.log10(distance) - (0.1 * self.n_steps / self.max_steps)

    def disaster_reset(self) -> Tuple[State, Dict]:
        """Resets the environment and randomly samples a new drone location.

        Returns
        -------
        Tuple[State, Dict]
            Tuple includes the initial state and observation.
        """

        self.n_steps = 0

        while True:
            i, j = np.random.randint(0, self.starting_poses.shape[0], (2,))
            if i != j:
                break
        starting_position = np.copy(self.starting_poses[i, :3])
        self.goal_position = np.copy(self.starting_poses[j, :3])

        state = State(
            ownship=Pose(
                position=starting_position,
                orientation=R.from_euler("y", 2 * np.pi * random.random()).as_quat(),
            )
        )
        return (state, self.observation_mapper.map(self._observe(state, dict())))
        # TODO(mmaz) this gets lost afaik
        # return (
        #     state,
        #     self.observation_mapper.map(
        #         self._observe(
        #             state,
        #             {
        #                 "achieved_goal": starting_position,
        #                 "desired_goal": self.goal_position,
        #             },
        #         )
        #     ),
        # )
