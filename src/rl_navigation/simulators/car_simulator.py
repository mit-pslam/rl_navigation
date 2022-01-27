from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from car_dynamics_sim import CarDynamicsSim
from rl_navigation.disaster import Pose, Simulator, State, Twist2D


class CarDynamicsSimulator(Simulator):
    def __init__(self, freq: Optional[float] = 5.0):
        """Initialize Dubins vehicle-like car simulator.
        
        See flightgoggles docs for more detail on the 
        vehicle model:
        https://flightgoggles-documentation.scrollhelp.site/fg/Car-Dynamics.374996993.html

        Parameters
        ----------
        freq: Optional[float], default = 5.0
            Vehicle simulator frequency.
        """

        # default FG args
        self.car_sim = CarDynamicsSim(
            maxSteeringAngle=1.0,
            minSpeed=0.0,
            maxSpeed=5.0,
            velProcNoiseAutoCorr=[0.001, 0.001],
        )
        self.car_sim.setVehicleState(
            position=np.zeros(2), velocity=np.zeros(2), heading=0
        )
        self.freq = freq

    def reset(self, state: State) -> None:
        """Reset car model position, velociy, 
        and heading.
        
        Parameters
        ----------
        state: State
            Desired state.
        """
        heading = R.from_quat(state.ownship.orientation).as_euler("xyz")[2]
        self.car_sim.setVehicleState(
            position=state.ownship.position[:2], velocity=np.zeros(2), heading=heading
        )

    def step(self, state: State, action: Twist2D) -> Tuple[State, Dict[str, Any]]:
        """Simulate one vehicle step.
        
        Parameters
        ----------
        state: State
            Current vehicle  state.

        action: Twist2D
            Desired forward velocity and steering angle.

        Returns
        -------
        Tupel[State, Dict[str, Any]]
            New state and empty dictionary.
        """
        self.car_sim.proceedState_ExplicitEuler(
            1.0 / self.freq, action.linear, action.angular
        )
        car_state = self.car_sim.getVehicleState()

        position = np.append(car_state["position"], state.ownship.position[2])
        orientation = R.from_euler(
            "xyz", np.append(np.zeros(2), self.car_sim.getVehicleState()["heading"])
        ).as_quat()

        return (
            State(
                time=state.time + 1,
                ownship=Pose(position=position, orientation=orientation),
            ),
            dict(),
        )
