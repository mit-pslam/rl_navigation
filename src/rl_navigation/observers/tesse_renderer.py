from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import defusedxml.ElementTree as ET

from rl_navigation.disaster import Observer, State, Pose
from rl_navigation.utils.resetter import Resetter, randbounds

from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import (
    NetworkConfig,
    TesseConnectionError,
    get_network_config,
    response_nonetype_check,
    set_all_camera_params,
)

from tesse.msgs import *


def reposition_message(state: State) -> Reposition:

    # See https://github.com/siemens/ros-sharp/blob/master/Unity3D/Assets/RosSharp/Scripts/Extensions/TransformExtensions.cs for conventions
    return Reposition(
        -state.ownship.position[1],
        state.ownship.position[2],
        state.ownship.position[0],
        state.ownship.orientation[1],
        -state.ownship.orientation[2],
        -state.ownship.orientation[0],
        state.ownship.orientation[3],
    )


class TesseRenderer(TesseGym, Observer, Resetter):
    """Observer that renders with TESSE."""

    def __init__(
        self,
        sim_path: Union[str, None],
        network_config: Optional[NetworkConfig] = get_network_config(),
        cameras: Optional = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.DEPTH, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
        ],
        scene_id: Optional[int] = None,
        sample_height: Optional[Tuple[float, float]] = (0.5, 2.5),
    ) -> None:

        TesseGym.__init__(
            self,
            sim_path,
            network_config,
            scene_id,
            999999999999,  # episode length
            200,  # frame rate
            set_all_camera_params,
            True,
            no_collisions=False,
        )

        self.sample_height = sample_height
        self.cameras = cameras

    def observe(
        self, state: State, observation: Optional[Dict[str, Any]] = dict()
    ) -> Dict[str, Any]:
        """Adds tesse observation to observation dictionary

        Parameters
        ----------
        state: State
            The world state.

        observation: Optional[Dict[str, Any]] = dict()
            Dictionary with any previously populated observations.

        Returns
        -------
        Dict[str, Any]
            Observation dictionary with an entry added for tesse.

        """
        self.env.send(reposition_message(state))  # update position in tesse

        # TODO: check whether this messes up vehicle position
        self.env.send(StepWithTransform())  # allow collision to register

        result = self._data_request(DataRequest(metadata=True, cameras=self.cameras))

        tesse = dict()
        for (i, image) in enumerate(result.images):
            if self.cameras[i][0] == Camera.RGB_LEFT:
                tesse["image"] = image
            if self.cameras[i][0] == Camera.DEPTH:
                tesse["depth"] = np.where(image == 0.0, 1.0, image)
            if self.cameras[i][0] == Camera.SEGMENTATION:
                tesse["segmentation"] = image

        observation["tesse"] = tesse

        observation["tesse"]["collision"] = (
            ET.fromstring(result.metadata).find("collision").attrib["status"].lower()
            == "true"
        )

        return observation

    def sample(self) -> State:
        """Returns a random initial state."""
        self.env.send(Respawn())

        result = self._data_request(MetadataRequest())

        # See https://github.com/siemens/ros-sharp/blob/master/Unity3D/Assets/RosSharp/Scripts/Extensions/TransformExtensions.cs for conventions
        pos = self._get_agent_position(result.metadata)
        (x, y, z, w) = self._get_agent_rotation(result.metadata, False)
        return State(
            ownship=Pose(
                position=np.array([pos[2], -pos[0], randbounds(self.sample_height)]),
                orientation=np.array([-z, x, -y, w]),
            )
        )
