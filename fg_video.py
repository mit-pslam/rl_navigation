from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv
from stable_baselines import PPO2

import numpy as np

from fg import FlightGogglesHeadingEnv
import time
import os


pose_port = "10253"
video_port = "10254"
report_port = "5556"

def make_unity_env(num_env):
    """ Create a wrapped Unity environment. """

    def make_env(rank):
        def _thunk():
            env = FlightGogglesHeadingEnv(pose_port, video_port, report_port)
            return env

        return _thunk

    return DummyVecEnv([make_env(i) for i in range(num_env)])


env = make_unity_env(1)


model = PPO2.load("../../fg_policies/loopfg_heading4.policy")

video_length = 2000

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



