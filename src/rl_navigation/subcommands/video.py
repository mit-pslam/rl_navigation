"""Module to handle creating video."""
from rl_navigation.fg import FlightGogglesHeadingEnv
from rl_navigation.config import get_cfg_defaults

import stable_baselines.common.vec_env as stb_env
from stable_baselines import PPO2

from typing import Optional
import datetime


def make_unity_env(config, num_env):
    """Create a wrapped Unity environment."""

    def make_env(rank):
        def _thunk():
            env = FlightGogglesHeadingEnv(config)
            return env

        return _thunk

    return stb_env.DummyVecEnv([make_env(i) for i in range(num_env)])


def run_video(
    input_model: str,
    configuration_file: Optional[str] = None,
    video_length: int = 6000,
    output_directory: str = "videos",
    output_prefix: str = "",
    **kwargs
):
    """Make a video from the results."""
    cfg = get_cfg_defaults()

    if configuration_file is not None:
        cfg.merge_from_file(configuration_file)

    cfg.freeze()

    env = make_unity_env(cfg, 1)

    model = PPO2.load(input_model)

    name_prefix = "{}_{}".format(
        output_prefix, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    video_env = stb_env.VecVideoRecorder(
        env,
        video_folder=output_directory,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix=name_prefix,
    )

    obs = video_env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = video_env.step(action)
