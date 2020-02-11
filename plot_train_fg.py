import numpy as np
import zmq

# based on: https://stackoverflow.com/questions/41602588/matplotlib-3d-scatter-animations
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation
from scipy import interpolate
from scipy.spatial import cKDTree

import threading
import queue
from collections import deque
import csv
import os

# https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# https://stackoverflow.com/a/13849249
def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def sock(q):
    port = "5556"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)  # subscriber socket
    socket.setsockopt(zmq.CONFLATE, 1)  # only receive the latest message
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # no message filter
    socket.setsockopt(zmq.RCVTIMEO, 1000)  # wait 1sec before raising EAGAIN
    socket.connect("tcp://127.0.0.1:%s" % port)

    # zmq receives habitat coordinates
    # the python queue this publishes to uses matplotlib coordinates
    while True:
        try:
            # msg = socket.recv()
            msg = socket.recv_json()
            q.put(msg)

            # RECORD_CSV = True
            # if RECORD_CSV:
            #     # from yzx_mpl to xyz_habitat
            #     orientation_habitat = msg['orientation_habitat']
            #     rot_x_habitat, rot_y_habitat, rot_z_habitat, rot_w_habitat = float(orientation_habitat['x']), float(orientation_habitat['y']), float(orientation_habitat['z']), float(orientation_habitat['w'])

            #     pos_quat = [current_episode_id, pos_x__habitat, pos_y__habitat, pos_z__habitat, rot_x_habitat, rot_y_habitat, rot_z_habitat, rot_w_habitat, reward]
            #     with open(csvfilename, 'a', newline='') as fh:
            #         writer = csv.writer(fh)
            #         writer.writerow(pos_quat)

        except zmq.Again:
            # print("no msg")
            continue  # try to recv again
    # cleanup zmq
    socket.close()
    context.term()

fmt_report = """Reward: {:06.3f}
Distance {:06.3f}
Angle {:06.3f}
u: {:04d}"""

def mktext(reward=0, dist=0, angle=0, ii=0):
    return fmt_report.format(reward, dist, angle, ii)

if __name__ == "__main__":
    _SMOOTH = 1.5
    _SAMPLING_INTERVAL = 1000
    ideal__unity = np.load("data_reward_fg__unity.npy")
    tck, u = interpolate.splprep(ideal__unity.T, s=_SMOOTH, per=1)
    u_new = np.linspace(u.min(), u.max(), _SAMPLING_INTERVAL)
    x_new, y_new, z_new = interpolate.splev(u_new, tck, der=0)
    tree = cKDTree(np.c_[x_new, y_new, z_new])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim(-12, 12)
    ax.set_ylim(-2, 53)
    ax.set_zlim(2, 4)

    # reward curve (remains on the graph during animations)
    ax.plot(z_new, -x_new, y_new, "g--")

    USE_ALL_DATA = False
    POSITION_Q_SIZE = 25

    if USE_ALL_DATA:
        xdata, ydata, zdata = [], [], []
    else:
        xdata, ydata, zdata = (
            deque(maxlen=POSITION_Q_SIZE),
            deque(maxlen=POSITION_Q_SIZE),
            deque(maxlen=POSITION_Q_SIZE),
        )

    mq = queue.Queue()
    s = threading.Thread(target=sock, args=(mq,), daemon=True)
    s.start()

    graph, = ax.plot(xdata, ydata, zdata, linestyle="", marker="o", color=(0, 0, 1, 0.1))

    # https://matplotlib.org/examples/animation/subplots.html

    # current agent's distance to reward curve
    xline, yline, zline = [], [], []

    # ideal heading
    xideal, yideal, zideal = [], [], []
    # agent heading
    xagent, yagent, zagent = [], [], []

    # NOTE: the , is critical!!!
    dist_line, = ax.plot(xline, yline, zline, linestyle=":", color="r", linewidth=2)
    ideal_line, = ax.plot(xideal, yideal, zideal, linestyle="-", color=(1,0,1), linewidth=2)
    agent_line, = ax.plot(xagent, yagent, zagent, linestyle="-", color=(0,0,1), linewidth=2)


    # https://stackoverflow.com/questions/18274137/how-to-animate-text-in-matplotlib

    # https://matplotlib.org/3.1.1/gallery/mplot3d/text3d.html

    # https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/fancytextbox_demo.html

    # dist_report = ax.text(17, -3, -1, "Distance: {:06.3f}".format(0)
    # mp3d textbox location
    dist_report = ax.text(
        0,
        12,
        4.55,
        mktext(),
        size=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8)),
    )

    def update_plot(frame_idx):
        try:
            msg = mq.get_nowait()
            # x, y, z, reward = (
            #     float(msg["x"]),
            #     float(msg["y"]),
            #     float(msg["z"]),
            #     float(msg["reward"]),
            # )
            agent_current_position = np.array(msg['agent_current_pos'])
            agent_next_position = np.array(msg['agent_next_pos'])
            loop_current_pos = np.array(msg['ideal_current_pos'])
            loop_next_pos = np.array(msg['ideal_next_pos'])
            reward = msg['loop_reward']
            heading_angle = np.rad2deg(msg['heading_angle'])
            ii = msg['ii']

            x,y,z = agent_current_position[0], agent_current_position[1], agent_current_position[2]

            # convert to ROS
            # ROS: x, y, z = UNITY: z, -x, y 
            xdata.append(z)
            ydata.append(-x)
            zdata.append(y)

            # determine distance to reward curve
            dd, ii = tree.query((x, y, z))
            # print("dist", dd)
            # dist_report.set_text("Distance: {:06.3f}".format(dd))
            dist_report.set_text(mktext(reward, dd, heading_angle, ii))

            # visualize distance to reward curve
            # format: [xstart, xend], [ystart, yend], [zstart, zend]
            (xline, yline, zline) = (
                [-x, -tree.data[ii][0]],
                [y, tree.data[ii][1]],
                [z, tree.data[ii][2]],
            )
            dist_line.set_data(zline, xline)
            dist_line.set_3d_properties(yline)

            SCALE_UNIT = 3.0

            # visualize ideal heading
            v_unit_ideal = unit_vector(loop_next_pos - loop_current_pos)
            loop_next_pos = loop_current_pos + (v_unit_ideal * SCALE_UNIT)
            (xideal, yideal, zideal) = (
                [-loop_current_pos[0], -loop_next_pos[0]],
                [loop_current_pos[1], loop_next_pos[1]],
                [loop_current_pos[2], loop_next_pos[2]],
            )
            ideal_line.set_data(zideal, xideal)
            ideal_line.set_3d_properties(yideal)

            # visualize agent heading
            v_unit_agent = unit_vector(agent_next_position - agent_current_position)
            agent_next_position = agent_current_position + (v_unit_agent * SCALE_UNIT)
            (xagent, yagent, zagent) = (
                [-agent_current_position[0], -agent_next_position[0]],
                [agent_current_position[1], agent_next_position[1]],
                [agent_current_position[2], agent_next_position[2]],
            )
            agent_line.set_data(zagent, xagent)
            agent_line.set_3d_properties(yagent)

        except queue.Empty:
            pass

        # visualize points agent has (recently) traversed
        graph.set_data(xdata, ydata)
        graph.set_3d_properties(zdata)

        return graph, dist_line, ideal_line, agent_line, dist_report

    ani = matplotlib.animation.FuncAnimation(fig, update_plot, 19, interval=40, blit=True)

    plt.show()
