from typing import Dict, Union

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from rl_navigation.observers.flight_goggles_renderer import FlightGogglesRenderer
from rl_navigation.observers.goal_point import GoalPoint
from rl_navigation.rllib.utils import (
    get_action_mapper,
    get_bounds_resetter,
    get_flightgoggles_env_args,
    get_flightgoggles_resetter,
    get_tesse_renderer,
)
from rl_navigation.simulators.simple_simulator import SimpleSimulator
from rl_navigation.tasks.pointnav import PointnavEnv, PointnavObserver
from rl_navigation.utils.flight_goggles_cfgs import (
    cfg_v3_depth_basement_defaults,
    cfg_v3_depth_defaults,
)


class PointnavWrapperEnv(PointnavEnv):
    """Wrapper environment for Pointnav that uses a config dictionary.

    This is intended to be used with ray.rllib
    """

    def __init__(self, config: Union[Dict, EnvContext]):
        if type(config) is not EnvContext:
            config = EnvContext(config, 0)

        max_steps = config.get("max_steps", 100)
        success_dist = config.get("success_dist", 0.25)
        success_reward = config.get("success_reward", 10.0)
        collision_reward = config.get("collision_reward", -10.0)
        change_range_weight = config.get("change_range_weight", 1.0)
        angular_weight = config.get("angular_weight", 0.0)
        change_twist_linear_weight = config.get("change_twist_linear_weight", 0.0)
        change_twist_angular_weight = config.get("change_twist_angular_weight", 0.0)
        if "change_action_weight" in config.keys():
            change_twist_linear_weight = config["change_action_weight"]
            change_twist_angular_weight = config["change_action_weight"]

        base_port = config.get("base_port", 8000)

        # Build Observer Chain
        renderer = config.get("renderer", "")
        flight_goggles_scene = config.get("flight_goggles_scene", "basement")
        fields = config.get("fields", [])
        max_range_from_ownship = config.get("max_range_from_ownship", 20)  # meters

        rank = (
            config.vector_index + config.worker_index * 10
        )  # TODO: How can we make sure the rank is always unique?

        if renderer == "tesse":
            tesse = get_tesse_renderer(config["tesse_path"], fields, rank, base_port)
            resetter = tesse
        elif renderer == "flight_goggles":
            resetter = get_flightgoggles_resetter(flight_goggles_scene)
        else:
            resetter = get_bounds_resetter(max_range_from_ownship)

        observer_chain = [
            GoalPoint(resetter, max_range_from_ownship=max_range_from_ownship)
        ]

        if renderer == "flight_goggles":
            env = get_flightgoggles_env_args(config)

            observer_chain.append(
                FlightGogglesRenderer(
                    config["flight_goggles_path"],
                    pose_port=base_port + 2 * rank,
                    video_port=base_port + 1 + 2 * rank,
                    config=cfg_v3_depth_basement_defaults()
                    if flight_goggles_scene == "basement"
                    else cfg_v3_depth_defaults(),
                    env=env,
                )
            )
        elif renderer == "tesse":
            observer_chain.append(tesse)

        if config.get("use_fast_depth_estimate", False):
            assert (
                "fast_depth_ckpt_file" in config
            ), 'config["fast_depth_ckpt_file"] needs to be specified.'
            
            from rl_navigation.observers.fast_depth_estimation import FastDepthEstimator

            observer_chain.append(FastDepthEstimator(config["fast_depth_ckpt_file"]))

        # Specificy observation mapper
        observation_mapper = PointnavObserver(  # TODO: Upgrade to TesseGymPointnavObserver when available for wide usage.
            position_mode=config.get("position_mode", "compact"),
            renderer=renderer,
            fields=fields,
            use_fast_depth_estimate=config.get("use_fast_depth_estimate", False),
        )

        # Specify Action Mapper
        action_mapper = get_action_mapper(config.get("action_mapper", "continuous"))

        super().__init__(
            SimpleSimulator(add_noise=True, action_queue_size=0),
            observer_chain,
            action_mapper,
            observation_mapper,
            resetter,
            max_steps,
            success_dist,
            success_reward=success_reward,
            collision_reward=collision_reward,
            change_range_weight=change_range_weight,
            angular_weight=angular_weight,
            change_twist_linear_weight=change_twist_linear_weight,
            change_twist_angular_weight=change_twist_angular_weight,
        )


class PointnavCallbacks(DefaultCallbacks):
    """Custom Callbacks for training metrics and video creation.

    See following links for details:
    - https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
    - https://docs.ray.io/en/master/_modules/ray/rllib/evaluation/metrics.html?highlight=RolloutMetrics
    """

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        episode.user_data["total_speed"] = 0
        episode.user_data["actions"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["total_speed"] += episode.last_info_for()["speed"]
            episode.user_data["actions"].append(episode.last_info_for()["action"])

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs
    ):
        info = episode.last_info_for()

        episode.custom_metrics["collided"] = info["collided"]
        episode.custom_metrics["reached_goal"] = info["reached_goal"]
        episode.custom_metrics["speed"] = (
            episode.user_data["total_speed"] / episode.length
        )

        # Calculate how much actions are changing step-to-step
        episode.custom_metrics["linear_action_change"] = np.mean(
            [
                np.linalg.norm(np.subtract(x.linear, y.linear))
                for x, y in zip(
                    episode.user_data["actions"], episode.user_data["actions"][1:]
                )
            ]
        )
        episode.custom_metrics["angular_action_change"] = np.mean(
            [
                np.linalg.norm(np.subtract(x.angular, y.angular))
                for x, y in zip(
                    episode.user_data["actions"], episode.user_data["actions"][1:]
                )
            ]
        )

        episode.custom_metrics["angular_action_magnitude"] = np.mean(
            [np.linalg.norm(x.angular) for x in episode.user_data["actions"]]
        )
