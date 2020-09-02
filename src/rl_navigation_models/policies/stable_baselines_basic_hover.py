import rl_navigation_ros.policy_interface as rlnav_policy
from rl_navigation.tasks.hover import action_mapper

from stable_baselines import PPO2
import cv2
import numpy as np

class StableBaselinesBasicHoverPolicy(rlnav_policy.AbstractPolicy):

    def __init__(self, config):
        """Make the policy."""
        super(StableBaselinesBasicHoverPolicy, self).__init__(config)
        self.model = PPO2.load(config['model_path'])
        self.state = None  # state for lstm policy
        self.height = config['height']
        self.width = config['width']

    def _finalize_intialization(self, config):
        """Set up the policy."""
        pass

    def __call__(self, observation_dict):
        if "image" not in observation_dict:
            raise rlnav_policy.PolicyEvalError("image not provided to the policy.")
        if observation_dict["image"][0] != "sensor_msgs.msg.Image":
            raise rlnav_policy.PolicyEvalError("image message type is not correct.")


        img = observation_dict["image"][1]
        img = cv2.resize(img.astype(np.uint8), (self.width, self.height))
        action, self.state = self.model.predict(
            np.expand_dims(img, axis=0),  # "Vectorize" input
            state=self.state,
            deterministic=True,
        )
        return action_mapper.map(action[0])

if __name__=="__main__":
    """
    This is for some basic testing/debugging.
    The only argument is a path to the policy.
    """
    import sys
    import rl_navigation_ros.message_conversions as rlnav_conversions
    from rl_navigation_ros.message_converter import MessageConverter, setup_conversions
    import numpy as np
    
    # Setup converters
    setup_conversions("kinetic")
    handlers = [rlnav_conversions.TwistHandler(), rlnav_conversions.ImageHandler()]
    converter = MessageConverter(handlers)
    print("Setup converters")

    # Setup policy
    config = {"model_path": sys.argv[1],
              "height": 192,
              "width": 256
    }
    policy = StableBaselinesBasicHoverPolicy(config)
    print("Setup policy")

    # Do a quick test
    for i in range(5):
        observation_dict = {
            "image": ("sensor_msgs.msg.Image",
                      np.random.randint(256, size=(1024,756,3), dtype=np.uint8)
                     )
        }

        desired_twist = policy(observation_dict)

        msg = {"msg": converter.serialize("geometry_msgs.msg.Twist", desired_twist)}
        print("Step {}".format(i))
        print(desired_twist)
        print(msg)
