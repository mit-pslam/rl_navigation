import subprocess
import os
import numpy as np
import atexit
import zmq
import time
import json
import yaml
import signal

from typing import Any, Callable, Dict, Optional, Tuple, Union
from collections import namedtuple

import threading, queue

from rl_navigation.disaster import Observer, State

from flightgoggles_client import FlightGogglesClient as fg_client
import flightgoggles


class FlightGogglesRenderer(Observer):
    """Observer that renders an RGB of what the drone would see."""

    def __init__(
        self,
        flight_goggles_path: str,
        pose_port: int = 10253,
        video_port: int = 10254,
        state: State = State(),
        config_file: str = os.path.join(
            flightgoggles.__path__[0], "..", "config", "FlightGogglesClient.yaml"
        ),
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

        config_file: str
            Path to FlightGoggles config file.

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
            ],
            cwd=os.path.dirname(flight_goggles_path),
        )

        with open(config_file, "r") as stream:
            self.config = yaml.safe_load(stream)

        self.client = fg_client(
            yaml_path=config_file,
            input_port=str(pose_port),
            output_port=str(video_port),
        )
        self.client.addCamera(
            self.config["camera_model"][0]["ID"],
            np.int(self.config["camera_model"][0]["channels"]),
            np.bool(self.config["camera_model"][0]["isDepth"]),
            np.int(self.config["camera_model"][0]["outputIndex"]),
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
        while True:
            try:  # TODO: Why is this necessary????
                try:
                    signal.alarm(1)

                    orientation = [
                        state.ownship.orientation[x] for x in [-1, 0, 1, 2]
                    ]  # put w at front
                    self.client.setCameraPose(state.ownship.position, orientation, 0)
                    self.client.setStateTime(self.client.getTimestamp())
                    self.client.requestRender()
                    imgs = self.client.getImage()

                except Exception:
                    continue
            except Exception:
                continue
            signal.alarm(0)
            break

        observation["fg"] = {
            "image": np.reshape(
                imgs[self.config["camera_model"][0]["ID"]],
                (
                    self.config["state"]["camHeight"],
                    self.config["state"]["camWidth"],
                    self.config["camera_model"][0]["channels"],
                ),
            ).astype(np.uint8),
            "hasCameraCollision": False,  # TODO: Determine how we can get collision field
        }
        return observation

    def close(self):
        """Callback to close flight goggles."""
        self.proc.kill()
        self.client.terminate()
        del self.client

    def __timeout_handler__(self, signum, frame):
        raise Exception
