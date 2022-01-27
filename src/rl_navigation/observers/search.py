from typing import Any, Dict, Optional, Union

import numpy as np
from rl_navigation.disaster import State
from rl_navigation.observers.flight_goggles_renderer import FlightGogglesRenderer
from rl_navigation.observers.goal_point import SE3_from_pose
from rl_navigation.utils.flight_goggles_cfgs import cfg_v3_depth_defaults, dump
from scipy.spatial.transform import Rotation as R
from yacs.config import CfgNode as CN


class FlightGogglesSearchRenderer(FlightGogglesRenderer):
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
        super().__init__(
            flight_goggles_path=flight_goggles_path,
            pose_port=pose_port,
            video_port=video_port,
            state=state,
            config=config,
            screen_quality=screen_quality,
            connection_timeout_seconds=connection_timeout_seconds,
            env=env,
        )

        # parent class creates FG client
        assert "objects" in self.config, "No targets given"
        for object in self.config["objects"].values():
            self.client.addObject(
                object["ID"],
                object["prefabID"],
                np.double(object["size_x"]),
                np.double(object["size_y"]),
                np.double(object["size_z"]),
            )
            self.client.setObjectPose([0, 0, 0], [1, 0, 0, 0], 0)
        self.spawn_state = State()

    @staticmethod
    def euler_to_quat(euler: np.ndarray) -> np.ndarray:
        """Return [w, x, y, z] quaternion"""
        return R.from_euler("zyx", euler).as_quat()[[3, 0, 1, 2]]

    def reset(self, state: State) -> None:
        self.spawn_state = state

    def observe(
        self, state: State, observation: Optional[Dict[str, Any]] = dict()
    ) -> Dict[str, Any]:
        assert (
            "goal_point" in observation
        ), "Observation dict does not contain goal point"
        if state.time == 0:
            goal_point_rel_agent = observation["goal_point"]["goal_position"]
            T_world_from_origin = SE3_from_pose(state.ownship)
            goal_point_world = T_world_from_origin * goal_point_rel_agent
            goal_point_world *= np.array([1, -1, -1])  # to NED coords?

            # heuristic to make target easier to see in basement
            goal_point_world[1] += 1.25
            goal_point_world[2] += 1
            rotation = np.random.uniform(-np.pi / 2, np.pi / 2)
            object_q_ned = self.euler_to_quat(np.array([0, -np.pi / 2, 0]))

            self.client.setObjectPose(goal_point_world, object_q_ned, 0)

        return super().observe(state, observation=observation)
