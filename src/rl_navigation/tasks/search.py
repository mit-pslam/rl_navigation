from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces
from rl_navigation.disaster import (
    ActionMapper,
    ObservationMapper,
    Observer,
    Pose,
    Simulator,
    State,
    Twist,
)
from rl_navigation.observers.goal_point import SE3_from_pose, goal_rel_agent
from rl_navigation.simulators.actions import ControlWithDeclareMapper
from rl_navigation.tasks.pointnav import (
    GoalPointEnv,
    get_image_observation_space,
    get_image_observation,
)
from rl_navigation.utils.resetter import Resetter


class SearchObserver(ObservationMapper):
    def __init__(
        self,
        position_mode: str = "compact",
        renderer: str = "",
        fields=[],
        shape: Tuple[int, int] = (192, 256),
        use_fast_depth_estimate: bool = False,
    ):
        assert position_mode in ["compact", "full"]

        assert renderer in ["", "flight_goggles", "fg", "flightgoggles", "tesse"]
        if renderer == "flight_goggles" or renderer == "flightgoggles":
            renderer = "fg"

        for field in fields:
            assert field in ["image", "depth", "grayscale"]

        self.position_mode = position_mode
        self.renderer = renderer
        self.fields = fields
        self.shape = shape
        self.use_fast_depth_estimate = use_fast_depth_estimate

        self.last_observation = None

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
        self.last_observation = observation
        return get_image_observation(
            self.fields,
            observation,
            self.shape,
            self.renderer,
            self.use_fast_depth_estimate,
        )

    @property
    def space(self) -> spaces.Box:
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space.

        """
        return spaces.Dict(get_image_observation_space(self.fields, self.shape))


class TesseGymSearchObserver(SearchObserver):
    """Observation Mapper that is a placeholder for future tesse-gym-based models.
    Currently, these models have specific dictionary keys that are required, which may be relaxed.

    TODO duplicate code.
    """

    def remap(self, x):
        if "image" in self.fields:
            x["RGB_LEFT"] = x.pop("image")
        if "depth" in self.fields:
            x["DEPTH"] = x.pop("depth")

        return x

    def map(self, observation: Dict[str, Any]) -> np.ndarray:
        out = super().map(observation)
        return self.remap(out)

    @property
    def space(self) -> spaces.Box:
        out = {}
        for field in self.fields:
            if field == "image":
                out["RGB_LEFT"] = spaces.Box(
                    low=0,
                    high=1,
                    dtype=np.float64,
                    shape=(self.shape[0], self.shape[1], 3),
                )
            elif field == "depth":
                out["DEPTH"] = spaces.Box(
                    low=0,
                    high=1,
                    dtype=np.float64,
                    shape=(self.shape[0], self.shape[1], 1),
                )

        return spaces.Dict(out)


class SearchEnv(GoalPointEnv):
    SUCCESS_KEY = "reached_goal"  # for dev
    CAM_FOV = 70

    def __init__(
        self,
        simulator: Simulator,
        observer_chain: List[Observer],
        action_mapper: ActionMapper,
        observation_mapper: ObservationMapper,
        resetter: Resetter,
        max_steps: int = 100,
        success_dist: float = 1,
        success_reward: float = 10.0,
        collision_reward: float = -10.0,
        false_positive_reward: float = -10.0,
        angular_weight: float = 0.0,
        change_twist_linear_weight: float = 0.0,
        change_twist_angular_weight: float = 0.0,
        state: Optional[Union[State, None]] = None,
        metadata: Optional[Dict[str, Any]] = {
            "render.modes": ["rgb_array"],
            "video.frames_per_second": 5,
        },
        reward_range: Optional[Tuple[float, float]] = (-float("inf"), float("inf")),
        end_on_false_positive: Optional[bool] = True,
        declare_threshold: Optional[float] = 0,
        enforce_target_in_fov: Optional[bool] = False,
    ):
        """Initialize target search environment.

        Parameters
        ----------
        simulator: Simulator
            Simulator model to use.

        observer_chain: List[Observer]
            List of Observers to populate scene observations

        action_mapper: ActionMapper
            Maps some action space into a Twist

        observation_mapper: ObservationMapper
            Maps observations from simulator and observation chain into observation space

        resetter: Resetter = BoundsResetter()
            Simulator ressetter

        max_steps: int = 100
            Maximum steps per episode

        success_dist: float = 0.25
            Distance in meters for successful point navigation to be declared

        success_reward: float = 10.0
            Reward for getting to goal point

        collision_reward: float = -10.0
            Reward (typically <0) for colliding

        false_positive_reward: float = -1.0
            Reward for false positive. Only applicable if using a declare action.

        change_range_weight: float = 1.0
            Reward weight for reducing range from goal point

        angular_weight: float = 0.0
            Reward weight (typically <=0) applied to angular twist magnitude

        change_twist_linear_weight: float = 0.0
            Reward weight (typically <=0) applied to a change in linear twist

        change_twist_angular_weight: float = 0.0
            Reward weight (typically <=0) applied to a change in angular twist

        state: Optional[Union[State, None]] = None
            Initial state.

        metadata: Optional[Dict[str, Any]] = {"render.modes": ["rgb_array"]}
            OpenAI gym metadata

        reward_range: Optional[Tuple[float, float]] = (-float("inf"), float("inf"))
            Reward range used for training

        end_on_false_positive: Optional[bool] = True,
            End episode on false positive.

        enforece_target_in_fov: Optional[bool] = False
            Target must be in agent's field of view to
            be considered found.

        Returns
        -------
        SearchEnv
        """
        self.resetter = resetter
        self.max_steps = max_steps
        self.success_dist = success_dist
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.angular_weight = angular_weight
        self.change_twist_linear_weight = change_twist_linear_weight
        self.change_twist_angular_weight = change_twist_angular_weight
        self.false_positive_reward = false_positive_reward
        self.end_on_false_positive = end_on_false_positive
        self.declare_threshold = declare_threshold
        self.enforce_target_in_fov = enforce_target_in_fov
        self.declare_target_found = isinstance(action_mapper, ControlWithDeclareMapper)

        super().__init__(
            simulator,
            observer_chain,
            action_mapper,
            observation_mapper,
            state=state,
            metadata=metadata,
            reward_range=reward_range,
        )

    def is_target_found(
        self, observation: Dict[str, Any], action: Twist
    ) -> Tuple[bool, float, bool, Dict[str, Any]]:
        """Compute if the agent has found the target.

        This method also provides the opportunity to
        provide a reward, end the episode, or
        provide any additional info.

        Parameters
        ----------
        observation: Dict[str, Any]
            Agent's observation.
        action: Twist
            Agent's current action

        Returns
        --------
            Tuple[bool, float, bool, Dict[str, Any]]
            - True if target is found
            - Reward
            - Episode has ended
            - Any additional info
        """
        reward = 0.0
        info = {}
        end = False

        dist_from_goal = np.linalg.norm(
            observation["goal_point"]["goal_position"]
            - observation["goal_point"]["ownship_position"],
            ord=2,
        )

        target_angle = goal_rel_agent(
            observation["goal_point"]["ownship_position"],
            observation["goal_point"]["ownship_orientation"],
            observation["goal_point"]["goal_position"],
        )

        within_target_range = dist_from_goal <= self.success_dist

        # angle is centroid of target. Add FOV buffer
        # to account for target size
        target_in_fov = target_angle < self.CAM_FOV + 5

        # initial found condition: within range
        found_target = within_target_range

        # optionally enforce target in agent fov
        if self.enforce_target_in_fov and not target_in_fov:
            found_target = False

        # optionally enforce declare action
        if self.declare_target_found:
            info["false_positive"] = False
            declare = action.reached > self.declare_threshold
            if declare and not found_target:  # false position
                reward += self.false_positive_reward
                info["false_positive"] = True
                end = self.end_on_false_positive
            elif not declare and found_target:  # false negative
                found_target = False

        return found_target, reward, end, info

    def compute_extras(
        self,
        previous_state: State,
        action: Twist,
        next_state: State,
        observation: Dict[str, Any],
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Determines step reward, whether environment
        is done, and any additional information.

        Reward is a function of task success
        (i.e., found  target), action, and collision.

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
                2. whether the episode has ended, in which case further
                    step() calls will return undefined results, and
                3. auxiliary diagnostic information (helpful for
                    debugging, and sometimes learning)
        """
        self.n_steps += 1

        self.ownship_position_history[self.n_steps, :] = observation["goal_point"][
            "ownship_position"
        ]

        found_target, reward, end_from_reward, reward_info = self.is_target_found(
            observation, action
        )

        if found_target:
            reward += self.success_reward

        # action related rewards
        reward += self.angular_weight * np.linalg.norm(action.angular)
        reward += self.change_twist_linear_weight * np.linalg.norm(
            np.subtract(action.linear, self.prev_action.linear)
        )
        reward += self.change_twist_angular_weight * np.linalg.norm(
            np.subtract(action.angular, self.prev_action.angular)
        )

        self.prev_action = action

        # Determine and award collision
        collided = self.is_collided(observation)
        if collided:
            reward += self.collision_reward

        is_done = (
            collided
            or found_target
            or self.n_steps >= self.max_steps
            or end_from_reward
        )

        info = {
            "collided": collided,
            self.SUCCESS_KEY: found_target,
            "speed": self.action_mapper.get_speed(action),
            "action": action,
        }
        info.update(reward_info)
        self.last_info = info

        return reward, is_done, info

    def disaster_reset(self) -> Tuple[State, Dict]:
        """Resets the environment and randomly samples a
        new drone location.

        Returns
        -------
        Tuple[State, Dict]
            Tuple includes the initial state and observation.
        """
        self.n_steps = 0
        self.last_info = None

        state = self.resetter.sample()
        self.simulator.reset(state)
        obs = self._observe(state, dict())

        # Store for rendering
        self.ownship_position_history = np.zeros((self.max_steps + 1, 3))
        self.ownship_position_history[self.n_steps, :] = obs["goal_point"][
            "ownship_position"
        ]

        return (state, self.observation_mapper.map(obs))
