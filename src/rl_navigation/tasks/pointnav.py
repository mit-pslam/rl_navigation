from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import sophus as sp
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation as R

from rl_navigation.disaster import (
    ActionMapper,
    DisasterEnv,
    ObservationMapper,
    Observer,
    Simulator,
    State,
    Twist,
)
from rl_navigation.utils.image import depth_to_image, hconcat_resize_min
from rl_navigation.utils.resetter import BoundsResetter, Resetter


class PointnavObserver(ObservationMapper):
    """Observation Mapper that extracts the RGB image from the message that
    flight goggles returns."""

    def __init__(
        self,
        position_mode: str = "compact",
        renderer: str = "",
        fields=[],
        shape: Tuple[int, int] = (192, 256),
        use_fast_depth_estimate: bool = False,
    ):
        """Initialize.

        Parameters
        ----------
        position_mode: str = "compact"
            String specifying how relative position of the goal is observed.
            Valid values are "compact" or "full".
            - compact: 3 elements that define the relative position of the goal in ownship coordinates.
            - full: 9 elements that define the goal position (3), ownship position (3), and ownship euler angles (3).
            Note: compact trains well, whereas full has not demonstrated the same performance in early testing.

        include_rgb: bool = False
            Include RGB image in observation

        shape: Tuple[int, int, int] = (192, 256, 3)
            Target shape for image

        Returns
        -------
        PointnavObserver

        """

        assert position_mode in ["compact", "full"]

        assert renderer in ["", "flight_goggles", "fg", "flightgoggles", "tesse"]
        if renderer == "flight_goggles" or renderer == "flightgoggles":
            renderer = "fg"

        for field in fields:
            assert field in ["image", "depth"]

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

        if self.position_mode is "full":
            goal_point_obs = np.concatenate(
                (
                    observation["goal_point"]["goal_position"],
                    observation["goal_point"]["ownship_position"],
                    R.from_quat(
                        observation["goal_point"]["ownship_orientation"]
                    ).as_euler("zyx"),
                )
            )
        elif self.position_mode is "compact":
            T_ownship_from_world = sp.SE3(
                R.from_quat(
                    observation["goal_point"]["ownship_orientation"]
                ).as_matrix(),
                observation["goal_point"]["ownship_position"],
            ).inverse()

            goal_point_obs = np.array(
                T_ownship_from_world * observation["goal_point"]["goal_position"]
            )

        obs_out = {"goal_point": goal_point_obs}
        obs_out.update(
            get_image_observation(
                self.fields,
                observation,
                self.shape,
                self.renderer,
                self.use_fast_depth_estimate,
            )
        )

        return obs_out

    @property
    def space(self) -> spaces.Box:
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Box
            The observation space.

        """
        if self.position_mode is "full":
            goal_point_space = spaces.Box(
                low=-np.Inf, high=np.Inf, dtype=float, shape=(9,)
            )
        elif self.position_mode is "compact":
            goal_point_space = spaces.Box(
                low=-np.Inf, high=np.Inf, dtype=float, shape=(3,)
            )

        out = {"goal_point": goal_point_space}
        out.update(get_image_observation_space(self.fields, self.shape))
        return spaces.Dict(out)


def get_image_observation(
    fields: List[str],
    observation: Dict[str, Any],
    shape: np.ndarray,
    renderer: str,
    use_fast_depth_estimate: bool,
):
    """Get dictionary of image observations specified in `fields`.

    Parameters
    ----------
    fields: List[str]
        Observation field names. Only images are considered.
    observation: Dict[str, Any]
        Observations. Images must be np arrays.
    shape: np.ndarray, shape=(H, W, C)
        Image shapes.
    renderer: str
        Either `fg` or `tesse`.
    use_fast_depth_estimate: bool
        Use fast depth estimate instead of gt depth.

    Returns
    -------
    Dict[str, np.ndarray]
        Requested observations of shape `shape`.
    """
    obs = {}
    for field in fields:
        if field == "depth" and use_fast_depth_estimate:
            img = observation["fast-depth-estimate"]
            interpolation = cv2.INTER_NEAREST
        else:
            img = observation[renderer][field]
            interpolation = cv2.INTER_LINEAR  # cv2 default
        img = cv2.resize(img, (shape[1], shape[0]), interpolation=interpolation)

        if img.ndim == 2:
            obs[field] = np.expand_dims(img, axis=2)
        else:
            obs[field] = img

        if img.dtype == np.uint8:
            obs[field] = img / np.float32(255.0)

    return obs


def get_image_observation_space(
    fields: List[str], img_shape: np.ndarray
) -> Dict[str, spaces.Box]:
    """Get observation spaces associated with images.

    Parameters
    ----------
    fields: List[str]
        Observation field names. Only images are considered.
    shape: np.ndarray, shape=(H, W, C)
        Expected image shape.

    Returns
    -------
    Dict[str, spaces.Box]
        Relevant fields from `fields` with associated observation spaces.
    """
    out = {}
    for field in fields:
        if field == "image":
            out["image"] = spaces.Box(
                low=0,
                high=1,
                dtype=np.float64,
                shape=(img_shape[0], img_shape[1], 3),
            )
        elif field == "depth":
            out["depth"] = spaces.Box(
                low=0,
                high=1,
                dtype=np.float64,
                shape=(img_shape[0], img_shape[1], 1),
            )
    return out


class TesseGymPointnavObserver(PointnavObserver):
    """Observation Mapper that is a placeholder for future tesse-gym-based models.
    Currently, these models have specific dictionary keys that are required, which may be relaxed.
    """

    def remap(self, x):
        x["POSE"] = x.pop("goal_point")
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
        if self.position_mode is "full":
            goal_point_space = spaces.Box(
                low=-np.Inf, high=np.Inf, dtype=float, shape=(9,)
            )
        elif self.position_mode is "compact":
            goal_point_space = spaces.Box(
                low=-np.Inf, high=np.Inf, dtype=float, shape=(3,)
            )

        out = {"POSE": goal_point_space}
        for field in self.fields:
            if field == "image":
                out["RGB_LEFT"] = spaces.Box(
                    low=0,
                    high=1,
                    dtype=float,
                    shape=(self.shape[0], self.shape[1], 3),
                )
            elif field == "depth":
                out["DEPTH"] = spaces.Box(
                    low=0,
                    high=1,
                    dtype=float,
                    shape=(self.shape[0], self.shape[1], 1),
                )

        return spaces.Dict(out)


class GoalPointEnv(DisasterEnv):
    """Env with functionality for rendering goal points (e.g., navigation
    objectives, target locations)."""

    def __init__(
        self,
        simulator: Simulator,
        observer_chain: List[Observer],
        action_mapper: ActionMapper,
        observation_mapper: ObservationMapper,
        state: Optional[Union[State, None]] = None,
        metadata: Optional[Dict[str, Any]] = {"render.modes": ["rgb_array"]},
        reward_range: Optional[Tuple[float, float]] = (-float("inf"), float("inf")),
    ):
        """Initialize goal environment, which provides functionality for
        tasks involving a goal point (e.g., navigating to a point, finding a target).

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

        state: Optional[Union[State, None]] = None
            Initial state.

        metadata: Optional[Dict[str, Any]] = {"render.modes": ["rgb_array"]}
            OpenAI gym metadata

        reward_range: Optional[Tuple[float, float]] = (-float("inf"), float("inf"))
            Reward range used for training


        Returns
        -------
        GoalPointEnv
        """
        # internal state
        self.n_steps = 0
        self.ownship_position_history = np.zeros((self.max_steps + 1, 3))
        self.last_info = None  # do we need this?
        self.prev_action = Twist()

        super().__init__(
            simulator,
            observer_chain,
            action_mapper,
            observation_mapper,
            state=state,
            metadata=metadata,
            reward_range=reward_range,
        )

    def is_collided(self, observation: Dict[str, Any]) -> bool:
        """Return true if agent is in collision."""
        return observation.get("fg", dict()).get(
            "hasCameraCollision", False
        ) or observation.get("tesse", dict()).get("collision", False)

    def build_render(self, obs):
        out = []
        if "image" in obs:
            out.append(obs["image"])

        if "depth" in obs:
            out.append(depth_to_image(obs["depth"]))

        if "fast-depth-estimate" in self.observation_mapper.last_observation:
            out.append(
                depth_to_image(
                    self.observation_mapper.last_observation["fast-depth-estimate"]
                )
            )

        out.append(self.render_trajectory())
        return hconcat_resize_min(out)

    def render(self, mode="rgb_array"):
        if "fg" in self.observation_mapper.last_observation:
            img = self.build_render(self.observation_mapper.last_observation["fg"])
        elif "tesse" in self.observation_mapper.last_observation:
            img = self.build_render(self.observation_mapper.last_observation["tesse"])
        else:
            img = self.render_trajectory()

        if self.last_info is not None and self.last_info["reached_goal"]:
            return np.uint8([0, 1, 0]) * img  # Add green mask
        elif (
            self.last_info is not None
            and "false_positive" in self.last_info
            and self.last_info["false_positive"]
        ):
            return np.uint8([1, 0, 0]) * img  # add red mask
        elif self.last_info is not None and self.last_info["collided"]:
            return np.uint8([1, 1, 0]) * img  # add yellow
        else:
            return img

    def render_trajectory(self):
        # See https://matplotlib.org/gallery/user_interfaces/
        # canvasagg.html#sphx-glr-gallery-user-interfaces-canvasagg-py

        fig = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        canvas = FigureCanvasAgg(fig)

        ax = fig.add_subplot(3, 1, (1, 2), xlabel="X", ylabel="Y")
        ax.axis("equal")
        ax.plot(
            self.ownship_position_history[: self.n_steps, 0],
            self.ownship_position_history[: self.n_steps, 1],
        )

        gp_obs = self.observation_mapper.last_observation["goal_point"]
        yaw = R.from_quat(gp_obs["ownship_orientation"]).as_euler("zyx")[0]
        ax.arrow(
            gp_obs["ownship_position"][0],
            gp_obs["ownship_position"][1],
            0.1 * np.cos(yaw),
            0.1 * np.sin(yaw),
            width=0.05,
        )
        ax.plot(
            gp_obs["goal_position"][0],
            gp_obs["goal_position"][1],
            marker="x",
            color="red",
        )

        ax = fig.add_subplot(3, 1, 3, xlabel="time", ylabel="Z")
        ax.plot(self.ownship_position_history[: self.n_steps, 2])
        ax.plot(
            [0, self.max_steps],
            [gp_obs["goal_position"][2], gp_obs["goal_position"][2]],
            color="red",
            linestyle="--",
        )

        canvas.draw()
        buf = canvas.buffer_rgba()
        return np.asarray(buf)[:, :, :3]


class PointnavEnv(GoalPointEnv):
    """Goal of this environment is to navigate to a desired location"""

    def __init__(
        self,
        simulator: Simulator,
        observer_chain: List[Observer],
        action_mapper: ActionMapper,
        observation_mapper: ObservationMapper,
        resetter: Resetter = BoundsResetter(),
        max_steps: int = 100,
        success_dist: float = 0.25,
        success_reward: float = 10.0,
        collision_reward: float = -10.0,
        change_range_weight: float = 1.0,
        angular_weight: float = 0.0,
        change_twist_linear_weight: float = 0.0,
        change_twist_angular_weight: float = 0.0,
        state: Optional[Union[State, None]] = None,
        metadata: Optional[Dict[str, Any]] = {
            "render.modes": ["rgb_array"],
            "video.frames_per_second": 5,
        },
        reward_range: Optional[Tuple[float, float]] = (-float("inf"), float("inf")),
    ):
        """Initialize.

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


        Returns
        -------
        PointnavEnv

        """
        # Parameters
        self.resetter = resetter
        self.max_steps = max_steps
        self.success_dist = success_dist
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.change_range_weight = change_range_weight
        self.angular_weight = angular_weight
        self.change_twist_linear_weight = change_twist_linear_weight
        self.change_twist_angular_weight = change_twist_angular_weight

        # Internal State
        self.n_steps = 0
        self.prev_dist_from_goal = None
        self.prev_action = Twist()
        self.ownship_position_history = np.zeros(
            (self.max_steps + 1, 3)
        )  # for plotting
        self.last_info = None

        super().__init__(
            simulator,
            observer_chain,
            action_mapper,
            observation_mapper,
            state=state,
            metadata=metadata,
            reward_range=reward_range,
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

        # Store for rendering
        self.ownship_position_history[self.n_steps, :] = observation["goal_point"][
            "ownship_position"
        ]

        # Calculate rewards
        # Reflects https://github.mit.edu/TESS/tesse-gym pointgoal reward model
        reward = 0.0  # no time penalty

        # Determine and reward progress towards goal
        dist_from_goal = np.linalg.norm(
            observation["goal_point"]["goal_position"]
            - observation["goal_point"]["ownship_position"],
            ord=2,
        )

        if self.prev_dist_from_goal is not None:
            reward += self.change_range_weight * (
                self.prev_dist_from_goal - dist_from_goal
            )

        self.prev_dist_from_goal = dist_from_goal

        reached_goal = False
        if dist_from_goal <= self.success_dist:
            reached_goal = True
            reward += self.success_reward

        # Determine and award collision
        collided = self.is_collided(observation)
        if collided:
            reward += self.collision_reward

        # Determine reward with turning
        reward += self.angular_weight * np.linalg.norm(action.angular)

        # Determine and award change in Action
        reward += self.change_twist_linear_weight * np.linalg.norm(
            np.subtract(action.linear, self.prev_action.linear)
        )
        reward += self.change_twist_angular_weight * np.linalg.norm(
            np.subtract(action.angular, self.prev_action.angular)
        )
        self.prev_action = action

        # Determine if done criteria is true
        is_done = (
            collided
            or self.n_steps >= self.max_steps
            or reached_goal  # agent successfully declares at goal point
        )

        info = {
            "collided": collided,
            "reached_goal": reached_goal,
            "speed": np.linalg.norm(action.linear),
            "action": action,
        }
        self.last_info = info

        return (reward, is_done, info)

    def disaster_reset(self) -> Tuple[State, Dict]:
        """Resets the environment and randomly samples a new drone location.

        Returns
        -------
        Tuple[State, Dict]
            Tuple includes the initial state and observation.
        """

        self.n_steps = 0
        self.prev_dist_from_goal = None
        self.last_info = None

        state = self.resetter.sample()
        obs = self._observe(state, dict())

        # Store for rendering
        self.ownship_position_history = np.zeros((self.max_steps + 1, 3))
        self.ownship_position_history[self.n_steps, :] = obs["goal_point"][
            "ownship_position"
        ]

        return (state, self.observation_mapper.map(obs))
