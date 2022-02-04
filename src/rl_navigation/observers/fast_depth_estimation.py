from fast_depth_estimation import FastDepthEstimator as FDE
# Note: fast_depth_estimation can be installed with an optional argument (i.e., pip install -e rl_navigation[fast_depth])

import numpy as np

from rl_navigation.disaster import Observer, State
from typing import Any, Callable, Dict, Optional, Tuple, Union


class FastDepthEstimator(Observer):
    def __init__(
        self,
        ckpt_path: str,
        shape: Tuple[int, int] = (384, 512),
        max_depth: float = 100.0,
    ):
        """Initialize.

        Parameters
        ----------
        ckpt_path: str
            Path to fast depth estimator check point.

        shape: Tuple[int, int] = (384, 512)
            Requested shape for depth image.

        max_depth: float = 100.0
            Max depth. Images are normalized by this value.

        Returns
        -------
        FastDepthEstimator

        """
        self.fde = FDE(shape, ckpt_path)
        self.max_depth = max_depth

    def observe(
        self, state: State, observation: Optional[Dict[str, Any]] = dict()
    ) -> Dict[str, Any]:
        """Adds fast-depth-estimation observation to observation dictuionary

        Parameters
        ----------
        state: State
            The world state.

        observation: Optional[Dict[str, Any]] = dict()
            Dictionary with any previously populated observations.

        Returns
        -------
        Dict[str, Any]
            Observation dictionary with an entry added for fast depth estimator.
            Requires that FlightGoggles image is already in the dictionary.

        """
        assert (
            "fg" in observation and "image" in observation["fg"]
        ), 'observation["fg"]["image"] missing in observation dictionary'

        observation["fast-depth-estimate"] = np.clip(
            np.squeeze(
                self.fde.estimate_depth(observation["fg"]["image"])
                / np.float32(self.max_depth),
            ),
            0.0,
            1.0, # in case model reports values beyond maxdepth
        )

        return observation
