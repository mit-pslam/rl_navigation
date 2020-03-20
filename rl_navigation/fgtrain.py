from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
from stable_baselines import PPO2

import numpy as np

from fg import FlightGogglesHeadingEnv
import time

from config import get_cfg_defaults


def make_unity_env(config, num_env):
    """ Create a wrapped Unity environment. """

    def make_env(rank):
        def _thunk():
            env = FlightGogglesHeadingEnv(config)
            return env

        return _thunk

    return DummyVecEnv([make_env(i) for i in range(num_env)])


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    # cfg.merge_from_file("experiment.yaml")
    cfg.freeze()

    env = make_unity_env(cfg, 1)

    # TODO(MMAZ) add hyperparams to config.py
    model = PPO2(
        CnnLstmPolicy, env, gamma=0.9, verbose=1, nminibatches=1, tensorboard_log="./tensorboard/"
    )
    # model.load("loopfg.policy")

    model.learn(total_timesteps=cfg.TRAINING.TOTAL_TIMESTEPS)

    model.save("loopfg_heading4.policy")
