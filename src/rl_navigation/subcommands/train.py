"""Module to handle training a policy."""
from rl_navigation.fg import FlightGogglesHeadingEnv
from rl_navigation.config import get_cfg_defaults

import stable_baselines.common.policies as stb_policies
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


def run_train(
    configuration_file: Optional[str] = None,
    quiet: bool = True,
    input_model: Optional[str] = None,
    output_prefix: str = "",
    **kwargs
):
    """Run training for a policy."""
    cfg = get_cfg_defaults()

    if configuration_file is not None:
        cfg.merge_from_file(configuration_file)
    cfg.freeze()

    env = make_unity_env(cfg, 1)

    model = PPO2(
        stb_policies.CnnLstmPolicy,
        env,
        gamma=cfg.TRAINING.HYP_GAMMA,
        verbose=1,
        nminibatches=cfg.TRAINING.HYP_MINIBATCHES,
        tensorboard_log="./tensorboard/",
    )

    if input_model is not None:
        model.load(input_model)

    model.learn(total_timesteps=cfg.TRAINING.TOTAL_TIMESTEPS)

    output = "{}_{}.policy".format(
        output_prefix, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    model.save(output)
