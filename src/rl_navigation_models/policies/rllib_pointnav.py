import logging
import yaml

import rl_navigation_ros.policy_interface as rlnav_policy
from rl_navigation.rllib.pointnav import PointnavWrapperEnv
from rl_navigation import Twist, Pose

from ray.rllib.agents import ppo
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
import ray

ray.init()


class RllibPointNavPolicy(rlnav_policy.AbstractPolicy):
    def __init__(self, config):
        """Make the policy."""
        super(RllibPointNavPolicy, self).__init__(config)

        self.agent = ppo.PPOTrainer(env=PointnavWrapperEnv, config=config["ppo_config"])
        self.agent.restore(config["checkpoint_path"])
        self.env = PointnavWrapperEnv(config["ppo_config"]["env_config"])

        self.state = self.agent.get_policy().get_initial_state()
        self.prev_action = self.env.action_space.sample()

    def _finalize_intialization(self, config):
        """Set up the policy."""
        pass

    def __call__(self, observation_dict):
        if ("goal" not in observation_dict) or ("pose" not in observation_dict):
            logging.warn("Either 'goal' or 'pose' not provided. Returning no-op.")
            return Twist()

        if observation_dict["goal"][0] != "geometry_msgs.msg.Pose":
            raise rlnav_policy.PolicyEvalError(
                "goal message type is {}, expected geometry_msgs.msg.Pose.".format(
                    observation_dict["goal"][0]
                )
            )

        if observation_dict["pose"][0] != "geometry_msgs.msg.Pose":
            raise rlnav_policy.PolicyEvalError(
                "pose message type is {}, expected geometry_msgs.msg.Pose.".format(
                    observation_dict["pose"][0]
                )
            )

        # Reorganize dictionary to meet what pointnav observation mapper expects
        reorganized_obs = {
            "goal_point": {
                "goal_position": observation_dict["goal"][1].position,
                "ownship_position": observation_dict["pose"][1].position,
                "ownship_orientation": observation_dict["pose"][1].orientation,
            },
            "fg": {},
        }

        ## Add RGB and Depth images, while checking correct type.
        if "image" in observation_dict:
            if observation_dict["image"][0] != "sensor_msgs.msg.Image":
                raise rlnav_policy.PolicyEvalError("image message type is not correct.")
            reorganized_obs["fg"]["image"] = observation_dict["image"][1]

        if "depth" in observation_dict:
            if observation_dict["image"][0] != "sensor_msgs.msg.Image":
                raise rlnav_policy.PolicyEvalError("depth message type is not correct.")
            reorganized_obs["fg"]["depth"] = observation_dict["depth"][1]

        # Use Environment and Policy to return action
        obs = self.env.observation_mapper.map(reorganized_obs)

        if self.env.observation_space.contains(obs):
            if len(self.state)>0:  # using lstm
                self.prev_action, self.state, _ = self.agent.compute_action(obs, state=self.state, prev_action=flatten_to_single_ndarray(self.prev_action))
            else:
                self.prev_action = self.agent.compute_action(obs)
            return self.env.action_mapper.map(self.prev_action)
        else:
            logging.warn(
                "Observation ({}) not in observation space ({}). Returning no-op.".format(
                    obs, self.env.observation_space
                )
            )
            return Twist()


if __name__ == "__main__":
    """
    This is for some basic testing/debugging.
    The only argument is a path to the policy.
    """
    import sys
    import rl_navigation_ros.message_conversions as rlnav_conversions
    from rl_navigation_ros.message_converter import MessageConverter, setup_conversions
    import numpy as np

    # Setup converters
    setup_conversions("noetic")
    handlers = [
        rlnav_conversions.TwistHandler(),
        rlnav_conversions.PoseHandler(),
        rlnav_conversions.ImageHandler(),
    ]
    converter = MessageConverter(handlers)
    print("Setup converters")

    # Setup policy
    env_config = {
        "renderer": "",
        "fields": [],
        "max_steps": 300,
        "action_mapper": "continuous",
    }

    if len(sys.argv)>2 and sys.argv[2]=="lstm":
        model_config = {
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "tanh",
            "use_lstm": True,
            "max_seq_len": 1,
            "lstm_cell_size": 128,  # good
            "lstm_use_prev_action": True,
        }
    else:
        model_config = {"fcnet_hiddens": [512, 256], "fcnet_activation": "tanh"}

    ppo_config = {
        "env_config": env_config,  # config to pass to env class
        "num_gpus": 1,
        "num_workers": 0,
        "model": model_config,
        "framework": "torch",
    }

    config = {
        "checkpoint_path": sys.argv[1],
        "ppo_config": ppo_config,
    }

    print(config)

    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            yaml.dump(config, f)

    policy = RllibPointNavPolicy(config)
    print("Setup policy")

    goals = (
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [-5.0, 0.0, 0.0],
        [-10.0, 0.0, 0.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, -5.0],
    )
    # Do a quick test
    for goal in goals:
        observation_dict = {
            "goal": (
                "geometry_msgs.msg.Pose",
                Pose(position=np.array(goal)),
            ),
            "pose": (
                "geometry_msgs.msg.Pose",
                Pose(),
            ),
            "image": (
                "sensor_msgs.msg.Image",
                np.random.randint(256, size=(1024, 756, 3), dtype=np.uint8),
            ),
            "depth": (
                "sensor_msgs.msg.Image",
                np.random.default_rng().random((1024, 756), dtype=np.float32),
            ),
        }

        desired_twist = policy(observation_dict)

        msg = {"msg": converter.serialize("geometry_msgs.msg.Twist", desired_twist)}

        print("-------------------------------")
        print("Goal: {}".format(goal))
        print("Action: {}".format(desired_twist))
        print(msg)
