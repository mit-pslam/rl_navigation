from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import Env as GymEnv
from gym import error, logger, spaces

Pose = namedtuple(
    "Pose",
    ["position", "orientation"],
    defaults=(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])),
)
State = namedtuple("State", ["time", "ownship"], defaults=(0.0, Pose()))

Twist = namedtuple(
    "Twist",
    ["linear", "angular"],
    defaults=(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
)

Twist2D = namedtuple(
    "Twist2d", ["linear", "angular"], defaults=(np.array([0]), np.array([0]))
)

Twist2DWithDeclare = namedtuple(
    "Twist2DWithDeclare",
    ["linear", "angular", "reached"],
    defaults=(Twist2D(), np.array([0])),
)


class Simulator:
    """Abstract class defining interface for drone + world simulators."""

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

        raise NotImplementedError

    def reset(self, state: State) -> None:
        """Reset simulator, if needed.

        Parameters
        ----------
        state: State
            State of new episode.
        """
        pass


class Observer:
    """Abstract class for defining drone observers."""

    def observe(
        self, state: State, observation: Optional[Dict[str, Any]] = dict()
    ) -> Dict[str, Any]:
        """Adds any observations defined by the class.

        Parameters
        ----------
        state: State
            The world state.

        observation: Optional[Dict[str, Any]] = dict()
            Dictionary with any previously populated observations.

        Returns
        -------
        Dict[str, Any]
            Observation dictionary with an entry added for this class observation.

        """
        raise NotImplementedError


class ActionMapper:
    """Abstract class for mapping gym-like actions into Twists."""

    def map(self, action) -> Twist:
        """Converts gym-like action to a twist.

        Parameters
        ----------
        action
            Allowable actions spaces can be found at https://github.com/openai/gym/tree/master/gym/spaces.

        Returns
        -------
        Twist
            Command input to the drone.

        """
        raise NotImplementedError

    @property
    def space(self) -> spaces.Space:
        """Returns the action space.

        Returns
        -------
        gym.spaces.Space
            The action space.

        """
        raise NotImplementedError


class DiscreteActionMapper:
    """Maps discrete gym actions to discrete Twists."""

    def __init__(self, actions: List[Twist]):
        """Initialize.

        Parameters
        ----------
        actions: List[Twist]
            List of discrete twists. Number of actions corresponds to list length.

        Returns
        -------
        DiscreteActionMapper


        """
        self.actions = actions

    def map(self, action: int) -> Twist:
        """Map discrete action to a Twist.

        Parameters
        ----------
        action: int
            Discrete action

        Returns
        -------
        Twist
            The discrete action represented as a Twist

        """
        return self.actions[action]

    @property
    def space(self) -> spaces.Discrete:
        """Returns the action space.

        Returns
        -------
        gym.spaces.Discrete
            The action space.

        """
        return spaces.Discrete(len(self.actions))


class ObservationMapper:
    """Observation mapper to convert observations into a gym Space."""

    def map(self, observation: Dict[str, Any]) -> np.ndarray:
        """Map observation dictionary in np.ndarray.

        Parameters
        ----------
        observation: Dict[str, Any]
            Dictionary of all observation

        Returns
        -------
        np.ndarray
            Observation array

        """
        raise NotImplementedError

    @property
    def space(self) -> spaces.Space:
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Space
            The observation space.

        """
        raise NotImplementedError


class DisasterEnv(GymEnv):
    """OpenAI Gym environment class for the AIIA SUAS program."""

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
        """Initialize.

        Parameters
        ----------
        simulator: Simulator
            Simulator to be used for the gym environment.

        observer_chain: List[Observer]
            List of observers for the gym environment.

        action_mapper: ActionMapper
            ActionMapper for the gym environment.

        observation_mapper: ObservationMapper
            ObservationMapper for the gym environment.

        state: Optional[Union[State, None]] = None
            Initial state for the environment.

        metadata: Optional[Dict[String, Any]] = {"render.modes": ["rgb_array"]}
            Metadata for the gym environment

        reward_range: Optional[Tuple[float, float]] = (-float("inf"), float("inf"))
            Reward range.

        Returns
        -------
        DisasterEnv

        """
        self.simulator = simulator
        self.observer_chain = observer_chain
        self.action_mapper = action_mapper
        self.observation_mapper = observation_mapper

        if state is None:
            self.state, _ = self.disaster_reset()
        else:
            self.state = state

        self.metadata = metadata
        self.reward_range = reward_range

        self.done = False

    @property
    def observation_space(self):
        """Returns the observation space.

        Returns
        -------
        gym.spaces.Space
            The observation space.

        """
        return self.observation_mapper.space

    @property
    def action_space(self):
        """Returns the action space.

        Returns
        -------
        gym.spaces.Discrete
            The action space.

        """
        return self.action_mapper.space

    def _observe(self, state: State, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the observation chain and returns an observation dictionary.

        Parameters
        ----------
        state: State
            The current gym state.

        observation: Dict[str, Any]
            The observation dictionary.

        Returns
        -------
        Dict[str, Any]
            The observation dictionary.

        """
        for observer in self.observer_chain:
            observation = observer.observe(state, observation)

        return observation

    def _step(self, action: Twist) -> Dict[str, Any]:
        """Steps the environment forward and returns an observation dictionary

        Parameters
        ----------
        action: Twist
            The agent's action represented as a twist.

        Returns
        -------
        Dict[str, Any]
            The observation dictionary.

        """
        (self.state, observation) = self.simulator.step(self.state, action)
        return self._observe(self.state, observation)

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
        raise NotImplementedError

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Parameters
        ----------
        action: gym.spaces.Space
            The agent's action

        Returns
        -------
        Tuple[np.ndarray, float, bool, Dict[str, Any]]
            The tuple elements correspond to:
                1. The observation,
                2. amount of reward returned after previous action,
                3. whether the episode has ended, in which case further step() calls will return undefined results, and
                4. auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        if self.done:
            logger.warn("You are calling 'step()' after done=True")

        previous_state = self.state
        mapped_action = self.action_mapper.map(action)
        observation = self._step(mapped_action)

        (reward, done, info) = self.compute_extras(
            previous_state, mapped_action, self.state, observation
        )
        self.done = done

        return (self.observation_mapper.map(observation), reward, done, info)

    def disaster_reset(self) -> Tuple[State, Dict[str, Any]]:
        """Resets the state of the environment and returns the initial state and observation.

        RETURNS
        -------
        Tuple[State, Dict[str, Any]]
            Fields correspond to:
                1. the initial state, and
                2. the initial observation.

        """
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns an initial observation.

        RETURNS
        -------
        np.ndarray
            The initial observation.

        """
        (self.state, observation) = self.disaster_reset()
        self.done = False
        return observation

    def render(self, mode: str = "rgb_array"):
        """Renders the environment."""
        raise NotImplementedError


class GoalDisasterEnv(DisasterEnv):
    """Convenience class modified from https://github.com/openai/gym/blob/master/gym/core.py#L156.

    A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """

    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error(
                "GoalEnv requires an observation space of type gym.spaces.Dict"
            )
        for key in ["observation", "achieved_goal", "desired_goal"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    'GoalDisasterEnv requires the "{}" key to be part of the observation dictionary.'.format(
                        key
                    )
                )
        super().reset()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        raise NotImplementedError
