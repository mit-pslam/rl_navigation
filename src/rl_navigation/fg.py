"""Module that handles making a gym environment for training."""
import rl_navigation.fg_msgs as fg_msgs
import rl_navigation.math_utilities as math_utils

import atexit
import numpy as np
import zmq
import json
import time
import cv2
import os

from gym import Env as GymEnv, spaces, logger

from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from scipy.spatial import cKDTree
import transforms3d
import sophus as sp

import queue

# the triumvirate of potential bugs
import subprocess
import threading
import multiprocessing


HWM = 6


# TODO(nathan) turn into int enum
class RenderState:
    """Enum for what FG is doing."""

    NOOP = 0  # default
    WAITING = 1  # action to send on wire
    SUBMITTED = 2  # action sent on wire
    # RECEIVED = 3 # RGB image returned


class DroneState:
    """Keep track of the drone state."""

    def __init__(self, config):
        """Make the drone state."""
        # self.init_pos__unity = np.array([0, 3, 0])
        # self.init_rot__unity = np.array([0, 0, 0, 1])  # xyzw
        self.config = config

        # initial locations for the agent
        pose_path = os.path.join(
            self.config.INITIAL_CONDITIONS.RESOURCE_DIR,
            self.config.INITIAL_CONDITIONS.STARTING_POSES,
        )
        self.starting_poses = np.load(pose_path)

        # Ideal Curve - train the agent to follow this trajectory
        self._SMOOTH = 1.5
        self._SAMPLING_INTERVAL = 1000
        ideal_curve_path = os.path.join(
            self.config.INITIAL_CONDITIONS.RESOURCE_DIR,
            self.config.INITIAL_CONDITIONS.IDEAL_CURVE,
        )
        ideal__unity = np.load(ideal_curve_path)
        tck, u = interpolate.splprep(ideal__unity.T, s=self._SMOOTH, per=1)
        u_new = np.linspace(u.min(), u.max(), self._SAMPLING_INTERVAL)
        self.x_new, self.y_new, self.z_new = interpolate.splev(u_new, tck, der=0)
        self._tree = cKDTree(np.c_[self.x_new, self.y_new, self.z_new])

        self.lock = threading.Lock()
        self.reset()

        self.renderState = RenderState.NOOP

    def get_current_position(self):
        """Get the current pose from FG."""
        pose = self.get_state()
        return pose[0:3, 3]

    def get_state(self):
        """Get the current unity pose."""
        with self.lock:
            # TODO(MMAZ) is it ok to return a reference instead? i dont think so?
            # no guarantee it wont be changed out from under you
            return np.copy(self._T__world_from_agent__unity.matrix())

    def set_state(self, new_state):
        """Set the current unity pose."""
        with self.lock:
            self._T__world_from_agent__unity = new_state
            self.renderState = RenderState.WAITING

    def pose_on_wire(self):
        """Set that we've asked for a render."""
        with self.lock:
            self.renderState = RenderState.SUBMITTED

    def image_returned(self):
        """Set that we've gotten an image after a request."""
        with self.lock:
            self.renderState = RenderState.NOOP

    def reset(self):
        """Reset the pose of the agent."""
        # select a random initial orientation
        random_ix = np.random.randint(0, self.starting_poses.shape[0])
        random_pose = self.starting_poses[random_ix, :]

        self.current_pos__unity = np.copy(random_pose[:3])
        self.current_rot__unity = np.copy(random_pose[3:])
        self.no_zoom = np.ones(3)
        # TODO(MMAZ) choose better notation to keep track of transformations
        # https://git.io/JvCVY
        # should keep track of right-handed (ROS) vs left-handed (Unity)
        with self.lock:
            self._T__world_from_agent__unity = sp.SE3(
                R.from_quat(self.current_rot__unity).as_dcm(), self.current_pos__unity
            )
            self.renderState = RenderState.WAITING

    def next_state_continuous(self, action):
        """
        Update pose using an angular velocity (nominally in radians/sec).

        TODO(MMAZ): Angular velocity change is applied instantaneously (there are
        no vehicle dynamics currently)

        Returns: tuple(next state, current state)
        """
        # TODO(MMAZ) switch from Unity's coordinate system
        # TODO(MMAZ) can be treated as radians per second, but this is not explicitly
        # tracked against a clock. ideally, should be multiplied by delta_time below in the
        # exponentiation operation
        # vz: forward along agent's nose in Unity coordinate
        # wy: angular velocity for yawing in Unity coordinates
        new_angular_velocity = np.array(
            [0, 0, self.config.AGENT_MOVEMENT.FORWARD_VELOCITY, 0, action, 0]
        )

        # first_change_heading = transforms3d.affines.compose(
        #     T=np.zeros(3), R=first_change_heading, Z=self.no_zoom
        # )
        # next_move_forward = transforms3d.affines.compose(
        #     T=np.array([0, 0, DiscreteActions.FWD_AMT]), R=np.eye(3), Z=self.no_zoom
        # )

        # T_ti_tj__unity = np.dot(first_change_heading, next_move_forward)

        # TODO(MMAZ) should be a safe read of _T__world_from_agent__unity, this method and
        # self.reset() are the only ones to modify state
        new_state = self._T__world_from_agent__unity * sp.SE3.exp(
            new_angular_velocity * self.config.FLIGHTGOGGLES.PUBLISH_RATE
        )
        return new_state, self._T__world_from_agent__unity

    def act_discrete(self, action):
        """Make discrete command choices."""
        if action == DiscreteActions.FORWARD:
            pos = np.array([0, 0, DiscreteActions.FWD_AMT])  # unity down camera
            rot = R.from_euler("xyz", [0, 0, 0])
        if action == DiscreteActions.TURN_LEFT:
            pos = np.array([0, 0, 0])
            rot = R.from_euler(
                "xyz", [0, -np.deg2rad(DiscreteActions.TURN_AMT), 0]
            )  # LHS
        if action == DiscreteActions.TURN_RIGHT:
            pos = np.array([0, 0, 0])
            rot = R.from_euler(
                "xyz", [0, np.deg2rad(DiscreteActions.TURN_AMT), 0]
            )  # LHS

        T_ti_tj__unity = transforms3d.affines.compose(
            T=pos, R=rot.as_dcm(), Z=self.no_zoom
        )

        # TODO(MMAZ) should be a safe read of _T__world_from_agent__unity, this method and
        # self.reset() are the only ones to modify state
        new_state = np.dot(self._T__world_from_agent__unity, T_ti_tj__unity)
        self.set_state(new_state=new_state)


class FGRenderer:
    """Class that requests images from FG."""

    def __init__(self, config, drone_state):
        """Make the renderer."""
        self.config = config
        atexit.register(self._close)

        self.pose_port = self.config.FLIGHTGOGGLES.POSE_PORT
        self.video_port = self.config.FLIGHTGOGGLES.VIDEO_PORT

        if self.config.FLIGHTGOGGLES.BINARY == "":
            raise ValueError(
                "Please specify a valid binary path to FlightGoggles in experiment.yaml"
            )

        # https://github.com/mit-fast/FlightGogglesRenderer/blob/eac129e3816ab74ee5a2c9d08c426e333afb8785/FlightGoggles/Scripts/CameraController.cs#L131-L132
        self.proc = subprocess.Popen(
            [
                self.config.FLIGHTGOGGLES.BINARY,
                "-input-port",
                self.pose_port,
                "-output-port",
                self.video_port,
            ],
            cwd=os.path.dirname(self.config.FLIGHTGOGGLES.BINARY),
        )
        self.drone_state = drone_state

        self.context = zmq.Context()

        self.upload_socket = self.context.socket(zmq.PUB)  # zeromq publisher
        self.upload_socket.setsockopt(
            zmq.SNDHWM, HWM
        )  # "send highwatermark" - 1 -> do not queue up messages
        self.upload_socket.bind("tcp://*:%s" % self.pose_port)

        self.img_queue = multiprocessing.Queue()

        # set up subscriber
        self._sub()
        # set up publisher
        self._pub()
        # set up image consumer
        self._consume()

        self.observe_queue = queue.Queue()

    def _close(self):
        self.proc.kill()
        self._subscriber.terminate()
        # daemon threads should auto-exit

    @classmethod
    def _recv_proc(cls, img_queue, video_port):
        # runs in separate Subscriber Procees (sp)
        sp_context = zmq.Context()
        download_socket = sp_context.socket(zmq.SUB)  # subscriber socket
        download_socket.setsockopt(
            zmq.RCVHWM, HWM
        )  # "rcv highwatermark" - 1 -> do not queue up messages

        # only receive the latest message, does not work TODO
        # download_socket.setsockopt(zmq.CONFLATE, 1)

        download_socket.setsockopt(zmq.SUBSCRIBE, b"")  # no message filter
        # socket.setsockopt(zmq.RCVTIMEO, 1000) # wait 1sec before raising EAGAIN
        download_socket.bind("tcp://*:%s" % video_port)

        # print("\n\n\nsubscriber process init\n\n\n")
        while True:
            msg = download_socket.recv_multipart()  # copy=True by default
            # print("received")
            render_output = fg_msgs.RenderOutput(msg)
            img_queue.put(render_output)

    def _sub(self):
        self._subscriber = multiprocessing.Process(
            target=FGRenderer._recv_proc, args=(self.img_queue, self.video_port)
        )
        self._subscriber.start()

    def _pub_thread(self):
        # runs on separate thread
        while True:
            # publish
            state = self.drone_state.get_state()
            pos = state[0:3, 3]  # extract translation
            rot = state[0:3, 0:3]  # extract rotation matrix
            rqx = R.from_dcm(rot).as_quat()  # xyzw order

            c = fg_msgs.Camera(pos, rqx)
            time_since_epoch_in_nanoseconds = int(round(time.time() * 1000000000))
            s = fg_msgs.State(ntime=time_since_epoch_in_nanoseconds, cameras=[c])

            jmsg = json.dumps(s.asJsonDict())
            self.upload_socket.send_multipart([str.encode("Pose"), str.encode(jmsg)])

            if self.drone_state.renderState == RenderState.WAITING:
                self.drone_state.pose_on_wire()

            time.sleep(self.config.FLIGHTGOGGLES.PUBLISH_RATE)  # eg 20Hz

    def _pub(self):
        # as thread, to access shared state
        self._publisher = threading.Thread(target=self._pub_thread, daemon=True)
        self._publisher.start()

    def _consume_thread(self):
        # runs on separate thread
        # imgid = 0
        while True:
            r = self.img_queue.get()
            # c = r.renderMetadata.hasCameraCollision
            # cs = "0" if not c else "1"
            # with open("baz.log", "w") as fh:
            #     fh.write("{} {}\n".format(imgid, cs))
            # imgid += 1

            if self.drone_state.renderState == RenderState.SUBMITTED:
                self.drone_state.image_returned()
                self.observe_queue.put(r)

            # cv2.imshow("fg", r.images[0])
            # cv2.waitKey(1)
            # if c:
            #     self.drone_state.reset()

    def _consume(self):
        # as thread, to access shared state
        self._consumer = threading.Thread(target=self._consume_thread, daemon=True)
        self._consumer.start()


# TODO(nathan) switch to enum
class DiscreteActions:
    """Possible discrete actions."""

    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

    TURN_AMT = 5.0  # degrees
    FWD_AMT = 0.15  # meters

    @classmethod
    def interpret(cls, action):
        """Verify actions from policy are within enum."""
        if action == cls.FORWARD:
            return cls.FORWARD
        if action == cls.TURN_LEFT:
            return cls.TURN_LEFT
        if action == cls.TURN_RIGHT:
            return cls.TURN_RIGHT
        raise NotImplementedError("This action does not exist")

    @classmethod
    def report(cls, action):
        """Turn enum to string."""
        if action == cls.FORWARD:
            return "FORWARD"
        if action == cls.TURN_LEFT:
            return "TURN_LEFT"
        if action == cls.TURN_RIGHT:
            return "TURN_RIGHT"


# https://docs.python.org/3/howto/curses.html
def drive(stdscr, drone_state):
    """Allow the user to drive around in FG."""
    time.sleep(2)
    stdscr.clear()
    while True:
        c = stdscr.getch()
        action = None
        if c == ord("w"):
            action = DiscreteActions.FORWARD
        elif c == ord("a"):
            action = DiscreteActions.TURN_LEFT
        elif c == ord("d"):
            action = DiscreteActions.TURN_RIGHT
        else:
            continue
        drone_state.act(action)
        # stdscr.addstr("pressed {}\n".format(Actions.report(action)))
        # time.sleep(1)
        # stdscr.refresh()


class FlightGogglesHeadingEnv(GymEnv):
    """Gym environment for Flight Goggles."""

    max_steps = 1500

    def __init__(self, config):
        """Make the gym environment."""
        self.config = config
        self.drone_state = DroneState(self.config)
        self.renderer = FGRenderer(self.config, self.drone_state)
        self.shape = (fg_msgs.HEIGHT, fg_msgs.WIDTH, 3)

        self.metadata = {"render.modes": ["rgb_array"]}
        self.reward_range = (-float("inf"), float("inf"))

        # assume images are normalized between -1 and +1
        self.observation_space = spaces.Box(-1, 1, dtype=np.float, shape=self.shape)

        # self.position_queue = queue.deque(maxlen=40)

        # discrete
        # self.action_space = spaces.Discrete(3)
        # continuous
        self.SCALE_YAW = 12.0
        self.action_space = spaces.Box(-1, 1, dtype=np.float32, shape=(1,))

        self.MAX_DIST = self.config.AGENT_MOVEMENT.MAX_DIST

        self.USE_FILTER = False
        if self.USE_FILTER:
            # self.action_queue = queue.deque(maxlen=4) 1M
            self.action_queue = queue.deque(maxlen=5)
            # self.ii_queue = queue.deque(maxlen=40)

        self._last_observation = self.reset()

        self.save_json = False
        if self.save_json:
            self.jdata = []
            atexit.register(self.write_json)
        self.report_zmq = True
        # zmq position reporting
        if self.report_zmq:
            context = zmq.Context()
            self.report_socket = context.socket(zmq.PUB)  # zeromq publisher
            self.report_socket.setsockopt(
                zmq.SNDHWM, 1
            )  # "send highwatermark" - do not queue up messages
            self.report_socket.bind("tcp://*:%s" % self.config.TRAINING.REPORT_PORT)

    def _zero_mean(self, img):
        return cv2.normalize(
            img.astype(np.float), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX
        )

    def dist_reward_fxn(self, distance_to_loop, floor_dist_reward, ceil_dist_reward):
        """Reward for agent based on distance traveled."""
        EXP = 2.0
        if np.isclose(distance_to_loop, 0):
            return ceil_dist_reward
        if distance_to_loop < 1:
            dist_reward = 1.0 / distance_to_loop
        else:
            dist_reward = 2 + -1 * (distance_to_loop ** EXP)
        return np.clip(dist_reward, a_min=floor_dist_reward, a_max=ceil_dist_reward)

    def clip_dist_reward(self, distance_to_loop):
        """Adjust reward for agent to normalize."""
        # TODO(MMAZ) normalize or weight rewards to fall between -/+ 1
        # TODO(MMAZ) add these to config.py?
        mindist = 3.0
        floor_reward = -400
        ceil_reward = 1.0
        EXP = 3
        weighting = 0.2
        if distance_to_loop < mindist:
            return weighting * ceil_reward
        dist_reward = ceil_reward + -1 * ((distance_to_loop - mindist) ** EXP)
        return weighting * np.clip(dist_reward, a_min=floor_reward, a_max=ceil_reward)

    def _calculate_loop_reward(self, next_state, current_state):

        agent_current_position = current_state[0:3, 3]
        agent_next_position = next_state[0:3, 3]

        dd, ii = self.drone_state._tree.query(
            (
                agent_current_position[0],
                agent_current_position[1],
                agent_current_position[2],
            )
        )

        # self.ii_queue.append(ii)
        # ii_min = np.min(self.ii_queue)
        # ii_max = np.max(self.ii_queue)
        # #if ii > ii_max: add logic? TODO(MMAZ)
        # if (ii_max - ii_min) < 30:
        #     stuck_reward = -10
        # else:
        #     stuck_reward = 0

        # projection of current agent position onto ideal loop
        loop_current_position = np.array(
            [
                self.drone_state.x_new[ii],
                self.drone_state.y_new[ii],
                self.drone_state.z_new[ii],
            ]
        )
        # the +1 moves it to the next point on the parametric curve
        # the sampling interval mods it by 1000
        jj = (ii + 1) % self.drone_state._SAMPLING_INTERVAL
        loop_next_position = np.array(
            [
                self.drone_state.x_new[jj],
                self.drone_state.y_new[jj],
                self.drone_state.z_new[jj],
            ]
        )

        v_agent_heading = agent_next_position - agent_current_position
        v_ideal_heading = loop_next_position - loop_current_position

        angle_between = math_utils.angle_between_vectors(
            v_agent_heading, v_ideal_heading
        )

        # minimize angle between agent heading and ideal loop heading
        # TODO(MMAZ): normalize or weight rewards to be between -/+ 1
        heading_reward = self.dist_reward_fxn(
            angle_between, floor_dist_reward=-500, ceil_dist_reward=50
        )
        # heading_reward = -np.abs(angle_between)

        # minimize distance to ideal loop
        # dist_reward = 0.2 * (-dd)
        # dist_reward = 0.8 * self.dist_reward_fxn(dd, floor_dist_reward=-500, ceil_dist_reward=1)
        dist_reward = self.clip_dist_reward(dd)

        # keepalive = 1
        # dist_reward = dist_reward + keepalive

        result = {
            "loop_reward": heading_reward + dist_reward,
            "loop_dist": dd,
            "heading_angle": angle_between,
            "heading_reward": heading_reward,
            "agent_current_pos": agent_current_position.tolist(),
            "agent_next_pos": agent_next_position.tolist(),
            "ideal_current_pos": loop_current_position.tolist(),
            "ideal_next_pos": loop_next_position.tolist(),
            "ii": ii,
        }
        return result

    def step(self, action):
        """Step the agent forward in time."""
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        if self.done:
            logger.warn("You are calling 'step()' after done=True")

        # discrete
        # action_to_execute = Actions.interpret(action)
        # self.drone_state.act_discrete(action_to_execute)

        # continuous

        if self.USE_FILTER:
            # moving average filter:
            self.action_queue.append(action[0])
            action = np.mean(self.action_queue)
        else:
            action = action[0]

        (next_state, current_state) = self.drone_state.next_state_continuous(action)
        self.drone_state.set_state(next_state)

        render = self.renderer.observe_queue.get()
        # TODO(MMAZ) is this really the 'last' observation or the 'next' observation?
        self._last_observation = render.images[0]

        reward = 0.0

        result = self._calculate_loop_reward(
            next_state.matrix(), current_state.matrix()
        )
        reward += result["loop_reward"]

        self.steps += 1
        if self.steps > self.max_steps:
            self.done = True

        if result["loop_dist"] > self.MAX_DIST:
            # TODO(MMAZ)
            reward = np.minimum(-1.0, reward)
            self.done = True

        didcollide = render.renderMetadata.hasCameraCollision
        if didcollide:
            # TODO(MMAZ)
            # discrete
            # reward -= 1.0
            # continuous
            reward = np.minimum(-1.0, reward)
            self.done = True

        zm = self._zero_mean(self._last_observation)

        if self.report_zmq:
            self.report_socket.send_json(result)
        if self.save_json:
            self.jdata.append(result)

        return (zm, reward, self.done, {})

    def reset(self):
        """Reset the environment."""
        self.done = False
        self.steps = 0
        if self.USE_FILTER:
            self.action_queue.clear()
            # self.ii_queue.clear()

        self.drone_state.reset()
        render = self.renderer.observe_queue.get()

        self._last_observation = render.images[0]
        return self._zero_mean(self._last_observation)

    def render(self, mode="rgb_array"):
        """Save the observation somewhere for latter debugging."""
        if self.save_json:
            destination = "/home/mark/relate/fgout/"
            idx = len(self.jdata)
            fn = destination + "render{:05d}.jpg".format(idx)
            cv2.imwrite(fn, self._last_observation)
        return cv2.cvtColor(self._last_observation, cv2.COLOR_BGR2RGB)

    def write_json(self):
        """Write json to send to FG."""
        name = "data_fg.json"
        if self.save_json:
            with open(name, "w") as fh:
                json.dump(self.jdata, fh)
