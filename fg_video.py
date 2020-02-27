from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
from stable_baselines import PPO2

import numpy as np

from fg import FlightGogglesHeadingEnv
import time
import os

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


    model = PPO2.load("loopfg_twist.policy")

    video_length = 6000

    video_env = VecVideoRecorder(
        env,
        video_folder="videos",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="loopfg",
    )

    obs = video_env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = video_env.step(action)

    # os.rename(
    #     "/home/mark/relate/habitat_rl/fg/videos/loopfg-step-0-to-step-2500.mp4",
    #     "/home/mark/relate/habitat_rl/fg/videos/loopfg.mp4",
    # )



