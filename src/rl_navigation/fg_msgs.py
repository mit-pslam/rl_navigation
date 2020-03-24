"""Messages to transmit to and from FG."""
import json
import cv2
import numpy as np

WIDTH = 256
HEIGHT = 256


class Landmark:
    """Location of landmark in the simulator."""

    def __init__(self, j):
        """Make a landmark."""
        self.ID = j["ID"]  # string
        self.position = j["position"]  # list[double]


class RenderMetadata:
    """Render information."""

    def __init__(self, j):
        """Make render information."""
        self.ntime = j["ntime"]  # np.int64
        self.camWidth = j["camWidth"]  # int
        self.camHeight = j["camHeight"]  # int
        self.camDepthScale = j["camDepthScale"]  # double
        self.cameraIDs = j["cameraIDs"]  # list[string]
        self.channels = j["channels"]  # list[int]
        self.hasCameraCollision = j["hasCameraCollision"]  # bool
        self.landmarksInView = [Landmark(x) for x in j["landmarksInView"]]  # list[Landmark]
        self.lidarReturn = j["lidarReturn"]  # float


class RenderOutput:
    """Render output."""

    def __init__(self, msg):
        """Make render output."""
        # flightgoggles returns a multipart message:
        # https://pyzmq.readthedocs.io/en/latest/api/zmq.html#zmq.Socket.recv_multipart
        # https://netmq.readthedocs.io/en/latest/message/#creating-multipart-messages
        # https://github.com/mit-fast/FlightGogglesRenderer/blob/eac129e3816ab74ee5a2c9d08c426e333afb8785/FlightGoggles/Scripts/CameraController.cs#L1069

        self.renderMetadata = RenderMetadata(json.loads(msg[0]))  # RenderMetadata

        # unpack image
        # https://github.com/mit-fast/FlightGoggles/blob/8a0af6cdcd4d5fa9cc002959b294186008e82dd2/flightgoggles_ros_bridge/src/Common/FlightGogglesClient.cpp#L160-L183
        # TODO unpack multiple images
        imleft = np.frombuffer(msg[1], dtype=np.uint8)
        # print(imleft.shape) == 2359296
        # imleft = np.reshape(imleft[:,0,0], (768,1024,3))
        imleft = np.reshape(imleft, (HEIGHT, WIDTH, 3))
        imleft = cv2.flip(imleft, 0)
        imleft = cv2.cvtColor(imleft, cv2.COLOR_RGB2BGR)
        #         imleft_n_right = np.frombuffer(msg[1], dtype=np.uint8)
        #         imleft = imleft_n_right[:921600]
        #         imright = imleft_n_right[921600:]
        #         imleft = np.reshape(imleft, (HEIGHT,WIDTH,3))
        #         imleft = cv2.flip(imleft,0)
        #         imleft = cv2.cvtColor(imleft, cv2.COLOR_RGB2BGR)
        #         imright = np.reshape(imright, (HEIGHT,WIDTH,3))
        #         imright = cv2.flip(imright,0)
        #         imright = cv2.cvtColor(imright, cv2.COLOR_RGB2BGR)

        self.images = [imleft]  # list[cv::Mat]


# class Object_t:
#     def __init__(self,j):
#         self.ID


class Camera:
    """Camera information."""

    def __init__(self, position, rotation, camera_id="Camera_RGB_left"):
        """Make camera information."""
        self.ID = camera_id
        # flightgoggles_ros_bridge/src/Common/jsonMessageSpec.hpp
        # Position and rotation use Unity left-handed coordinates.
        # Z North, X East, Y up.
        # E.G. East, Up, North.
        self.position = position
        self.rotation = rotation
        # flightgoggles_ros_bridge/src/ROSClient/ROSClient.cpp
        self.channels = 3
        self.isDepth = False
        self.outputIndex = 0
        self.hasCollisionCheck = True
        self.doesLandmarkVisCheck = True

    def asJsonDict(self):
        """Make json object from camera."""
        j = {
            "ID": self.ID,
            "position": self.position.tolist(),
            "rotation": self.rotation.tolist(),
            "channels": self.channels,
            "isDepth": self.isDepth,
            "outputIndex": self.outputIndex,
            "hasCollisionCheck": self.hasCollisionCheck,
            "doesLandmarkVisCheck": self.doesLandmarkVisCheck,
        }
        return j


class State:
    """State information."""

    def __init__(self, ntime, cameras, objects=[]):
        """Make state of FG."""
        # flightgoggles_ros_bridge/src/ROSClient/ROSClient.hpp
        self.sceneIsInternal = True
        self.sceneFilename = "Abandoned_Factory_Morning"
        self.ntime = ntime
        self.camWidth = WIDTH
        self.camHeight = HEIGHT
        # flightgoggles_ros_bridge/src/Common/jsonMessageSpec.hpp
        self.camFOV = 70.0
        self.camDepthScale = 0.20  # 0.xx corresponds to xx cm resolution
        self.cameras = cameras
        self.objects = objects

    def asJsonDict(self):
        """Make dictionary of FG state."""
        j = {
            "sceneIsInternal": self.sceneIsInternal,
            "sceneFilename": self.sceneFilename,
            "ntime": self.ntime,
            "camWidth": self.camWidth,
            "camHeight": self.camHeight,
            "camFOV": self.camFOV,
            "camDepthScale": self.camDepthScale,
            "cameras": [j.asJsonDict() for j in self.cameras],
            "objects": self.objects,
        }
        return j
