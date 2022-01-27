import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Union

from ray.rllib.env.env_context import EnvContext
from ray.tune.logger import UnifiedLogger
from rl_navigation.disaster import ActionMapper, DiscreteActionMapper, Simulator, Twist
from rl_navigation.observers.tesse_renderer import TesseRenderer
from rl_navigation.simulators.actions import (
    ContinuousActionMapper,
    ControlWithDeclareMapper,
    DubinsActionMapper,
    PolarContinuousActionMapper,
)
from rl_navigation.simulators.car_simulator import CarDynamicsSimulator
from rl_navigation.simulators.simple_simulator import SimpleSimulator
from rl_navigation.utils.resetter import BoundsResetter
from rl_navigation.utils.stata_resetter import (
    StataResetter,
    stata_basement,
    stata_ground_floor,
    stata_ground_floor_car,
)
from tesse.msgs import Camera, Channels, Compression
from tesse_gym.core.utils import get_network_config


def get_simulator(simulator: str) -> Simulator:
    if simulator == "simple":
        return SimpleSimulator(add_noise=True, action_queue_size=0)
    elif simulator == "dubins-car":
        return CarDynamicsSimulator()
    else:
        raise ValueError(f"Incorrect simulator value: {simulator}")


def get_action_mapper(action_mapper: str) -> ActionMapper:
    if action_mapper == "continuous":
        action_mapper = ContinuousActionMapper()
    elif action_mapper == "continuous-polar":
        action_mapper = PolarContinuousActionMapper()
    elif action_mapper == "discrete":
        action_mapper = DiscreteActionMapper(
            [
                Twist(linear=[0.1, 0.0, 0.0]),  # forward
                Twist(linear=[-0.1, 0.0, 0.0]),  # back
                Twist(linear=[0.0, 0.1, 0.0]),  # right
                Twist(linear=[0.0, -0.1, 0.0]),  # left
                Twist(linear=[0.0, 0.0, 0.1]),  # up
                Twist(linear=[0.0, 0.0, -0.1]),  # down
                Twist(angular=[0.0, 0.0, 0.1]),  # spin left
                Twist(angular=[0.0, 0.0, -0.1]),  # spin right
            ]
        )
    elif action_mapper == "dubins-car":
        return DubinsActionMapper()
    elif action_mapper == "dubins-car-declare":
        return ControlWithDeclareMapper(DubinsActionMapper())
    else:
        raise ValueError("Incorrect action_mapper value: ", action_mapper)
    return action_mapper


def get_tesse_renderer(tesse_path, fields: List[str], rank, base_port) -> TesseRenderer:
    cameras = []
    for field in fields:
        if field == "image":
            cameras.append((Camera.RGB_LEFT, Compression.OFF, Channels.THREE))
        elif field == "depth":
            cameras.append((Camera.DEPTH, Compression.OFF, Channels.THREE))
        elif field == "segmentation":
            cameras.append((Camera.SEGMENTATION, Compression.OFF, Channels.THREE))

    tesse = TesseRenderer(
        tesse_path,
        network_config=get_network_config(
            simulation_ip="localhost",
            own_ip="localhost",
            worker_id=rank,
            base_port=base_port,
        ),
        cameras=cameras,
    )
    return tesse


def get_flightgoggles_resetter(flight_goggles_scene: str) -> StataResetter:
    scene = stata_ground_floor
    if flight_goggles_scene == "basement":
        scene = stata_basement
    elif flight_goggles_scene == "ground_floor_car":
        scene = stata_ground_floor_car
    return StataResetter(scene)


def get_bounds_resetter(max_range_from_ownship: float) -> BoundsResetter:
    return BoundsResetter(
        (
            (-max_range_from_ownship, max_range_from_ownship),
            (-max_range_from_ownship, max_range_from_ownship),
            (0.5, 2.5),
        )
    )


def get_flightgoggles_env_args(config: EnvContext) -> Union[Dict[str, str], None]:
    if "display" in config.keys() and "screens" in config.keys():
        display = config["display"]
        display = (
            ":" + display if display[0] is not ":" else display
        )  # Add ":" if not at beginning
        screen = config["screens"][config.worker_index % len(config["screens"])]
        env = {
            "ENABLE_DEVICE_CHOOSER_LAYER": "1",
            "VULKAN_DEVICE_INDEX": "{}".format(screen),
            "DISPLAY": "{}.{}".format(display, screen),
        }
        print(env)
    else:
        env = None


def get_logger_creator(dir_path: str, dir_name: str) -> UnifiedLogger:
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(dir_name, timestr)

    def logger_creator(config: Dict[str, Any]):

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=dir_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator
