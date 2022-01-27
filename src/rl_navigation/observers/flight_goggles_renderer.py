import subprocess
import os
import numpy as np
import atexit
import zmq
import time
import json
import yaml
import signal
from yacs.config import CfgNode as CN

from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import namedtuple

import threading, queue

from rl_navigation.disaster import Observer, State
from rl_navigation.utils.flight_goggles_cfgs import (
    cfg_v3_defaults,
    cfg_v3_depth_defaults,
    dump,
)

from flightgoggles_client import FlightGogglesClient as fg_client
import flightgoggles

from scipy.spatial.transform import Rotation as R
import sophus as sp


T_ned_from_enu = sp.SE3(
    np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
)
T_enu_from_ned = T_ned_from_enu.inverse()


class FlightGogglesRenderer(Observer):
    """Observer that renders an RGB of what the drone would see."""

    def __init__(
        self,
        flight_goggles_path: str,
        pose_port: int = 10253,
        video_port: int = 10254,
        state: State = State(),
        config: CN = cfg_v3_depth_defaults(),
        screen_quality: int = 2,
        connection_timeout_seconds: int = 9999,  # FG v3.2
        env: Union[Dict, None] = None,
    ):
        """Initialize.

        Parameters
        ----------
        flight_goggles_path: str
            Path to flight goggles.

        pose_port: int = 10253
            Port to use for sending poses to flight goggles.

        video_port: int = 10254
            Port to use for receiving images from flight goggles.

        state: State = State()
            Initial state for initial drone placement in environment.

        config: CfgNode
            FG configuration. See rl_navigation/utils/flightgoggles_cfg.py.

        env: Union[Dict, None]
            Environment specific parameters.

        Returns
        -------
        FlightGogglesRenderer
        """
        # Launch flightgoggles
        self.proc = subprocess.Popen(
            [
                flight_goggles_path,
                "-input-port",
                str(pose_port),
                "-output-port",
                str(video_port),
                "-screen-quality",
                str(screen_quality),
                "-connection-timeout-seconds",
                str(connection_timeout_seconds),
            ],
            cwd=os.path.dirname(flight_goggles_path),
            env=env,
        )

        config_file = dump(config)

        with open(config_file, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.client = fg_client(
            yaml_path=config_file,
            input_port=str(pose_port),
            output_port=str(video_port),
        )

        for i, camera in enumerate(self.config["camera_model"].values()):
            self.client.addCamera(
                camera["ID"],
                np.int(i),
                np.int(camera["outputShaderType"]),
                bool(camera["hasCollisionCheck"]),
            )

        atexit.register(self.close)

    def observe(
        self, state: State, observation: Optional[Dict[str, Any]] = dict()
    ) -> Dict[str, Any]:
        """Adds flight goggles observation to observation dictuionary

        Parameters
        ----------
        state: State
            The world state.

        observation: Optional[Dict[str, Any]] = dict()
            Dictionary with any previously populated observations.

        Returns
        -------
        Dict[str, Any]
            Observation dictionary with an entry added for flight goggles.

        """
        signal.signal(signal.SIGALRM, self.__timeout_handler__)
        p_ned, q_ned = self.toFGClientInputs(state)
        while True:
            try:  # TODO: Why is this necessary????
                try:
                    signal.alarm(1)

                    for i in range(len(self.config["camera_model"].values())):
                        self.client.setCameraPose(p_ned, q_ned, i)

                    self.client.setStateTime(self.client.getTimestamp())
                    self.client.requestRender()
                    imgs, isInCollision, _, _ = self.client.getImage()

                except Exception:
                    continue
            except Exception:
                continue
            signal.alarm(0)
            break

        observation["fg"] = {
            "hasCameraCollision": isInCollision,
        }
        for camera in self.config["camera_model"].values():
            if camera["ID"] == "RGB":
                observation["fg"]["image"] = np.reshape(
                    imgs[camera["ID"]],
                    (
                        self.config["state"]["camHeight"],
                        self.config["state"]["camWidth"],
                        camera["channels"],
                    ),
                ).astype(np.uint8)[
                    :, :, ::-1
                ]  # Switch BGR to RGB (supposedly)
            elif camera["ID"] == "Depth":
                depth = (
                    np.reshape(
                        imgs[camera["ID"]],
                        (
                            self.config["state"]["camHeight"],
                            self.config["state"]["camWidth"],
                        ),
                    ).astype(float)
                    / 65535
                )  # 2**16-1, note: multiply by 100 for actual depth (in m's)

                # If no depth is detected, within 100 m, then 0 is reported.
                # We move anything at 0 out to the farthest distance possible (1.0)
                observation["fg"]["depth"] = np.where(depth == 0.0, 1.0, depth)
            else:
                pass

        return observation

    def toFGClientInputs(self, state: State):
        """Converts state to flight goggles client inputs, which expects North-East-Down

        Parameters
        ----------
        state: State
            The world state.

        Returns
        -------
        Tuple[]
            Position (as vector) and orientation (as quaternion)

        """
        T_enu_from_agent = sp.SE3(
            R.from_quat(state.ownship.orientation).as_matrix(), state.ownship.position
        )
        T_ned_from_agent = T_ned_from_enu * T_enu_from_agent * T_enu_from_ned

        p_ned = T_ned_from_agent.translation()
        q = R.from_matrix(T_ned_from_agent.rotation_matrix()).as_quat()
        q_ned = [q[x] for x in [-1, 0, 1, 2]]  # put w at front

        return (p_ned, q_ned)

    def close(self):
        """Callback to close flight goggles."""
        self.proc.kill()
        self.client.terminate()
        del self.client

    def __timeout_handler__(self, signum, frame):
        raise Exception
