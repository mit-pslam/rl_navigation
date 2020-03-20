from yacs.config import CfgNode as CN

_C = CN()

_C.FLIGHTGOGGLES = CN()

_C.FLIGHTGOGGLES.BINARY = (
    "/home/ma23705/catkin_ws/devel/.private/flightgoggles/lib/flightgoggles/FlightGoggles.x86_64"
)
# these should be strings (they are passed in as arguments)
_C.FLIGHTGOGGLES.POSE_PORT = "10253"
_C.FLIGHTGOGGLES.VIDEO_PORT = "10254"
# TODO(MMAZ): increasing this rate may result in dropped frames
_C.FLIGHTGOGGLES.PUBLISH_RATE = 1 / 20.0  # 20Hz, how frequently new poses are sent to the renderer

_C.INITIAL_CONDITIONS = CN()
_C.INITIAL_CONDITIONS.STARTING_POSES = "small_starting_set.npy"
_C.INITIAL_CONDITIONS.IDEAL_CURVE = "data_reward_fg__unity.npy"

_C.AGENT_MOVEMENT = CN()
# TODO(MMAZ) add logic to support CONTINUOUS=False (currently commented out)
_C.AGENT_MOVEMENT.CONTINUOUS = True
# how far away can an agent stray from the ideal curve before episode termination
_C.AGENT_MOVEMENT.MAX_DIST = 10.0
# TODO(MMAZ) this is not explicitly tracked against a clock
_C.AGENT_MOVEMENT.FORWARD_VELOCITY = 1.15  # meters/second

_C.TRAINING = CN()
_C.TRAINING.TOTAL_TIMESTEPS = 1400000
# to observe agent movement and rewards in realtime (see plot_train_fg.py)
_C.TRAINING.REPORT_PORT = 5556


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for the simulator"""
    return _C.clone()
