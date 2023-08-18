from yacs.config import CfgNode as CN
import re
import tempfile
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum


class Scene(Enum):
    Abandoned_Factory_Morning = "Abandoned_Factory_Morning"
    Stata_Basement = "Stata_Basement"
    Stata_GroundFloor = "Stata_GroundFloor"

    def sceneFilename(self) -> str:
        return self.value


class CameraShader(Enum):
    RGB = ("RGB", -1, 3, False)
    InstanceID = ("InstanceID", 0, 1, False)
    SemanticID = ("SemanticID", 1, 1, False)
    DepthCompressed = ("DepthCompressed", 2, 1, True)
    DepthMultiChannel = ("DepthMultiChannel", 3, 1, True)
    SurfaceNormals = ("SurfaceNormals", 4, 3, False)  #  3 channels?
    grayscale = ("grayscale", 5, 3, False)
    opticalFlow = ("opticalFlow", 6, 3, False)  # 3 channels?

    def ID(self) -> str:
        return self.value[0]

    def outputShaderType(self) -> int:
        return self.value[1]

    def channels(self) -> int:
        return self.value[2]

    def isDepth(self) -> bool:
        return self.value[3]


class Object(Enum):
    Gate = "gate"
    Blackeagle = "Blackeagle"
    Square_Gate = "Square_Gate"
    Circular_Gate = "Circular_Gate"

    def prefabID(self) -> str:
        return self.value


def scene(
    scene: Scene = Scene.Stata_Basement,
    initialPose: List[float] = [50.0, -49.0, -2.0, 0, 0, 0, 1],
    imu_freq: float = 200,
    cam_freq: float = 30,
    camWidth: int = 1024,
    camHeight: int = 768,
    camFOV: float = 70.0,
    camDepthScale: float = 0.20,
    cameras: List[CameraShader] = [CameraShader.RGB, CameraShader.DepthMultiChannel],
    cameraInitialPose: List[float] = [0, 0, 0, 1, 0, 0, 0],
    objects: List[Tuple[Object, str, float, float, float]] = [
        (Object.Gate, "gate1", 100, 100, 100)
    ],
    inputPort: str = "10253",
    outputPort: str = "10254",
) -> CN:

    C = CN()

    C.state = CN()
    C.state.sceneFilename = scene.sceneFilename()
    C.state.camWidth = camWidth
    C.state.camHeight = camHeight
    C.state.camFOV = camFOV
    C.state.camDepthScale = camDepthScale

    # # Setup objects, which may not be needed
    C.objects = CN()
    for i, object in enumerate(objects):
        C.objects[str(i)] = CN()
        C.objects[str(i)].prefabID = object[0].prefabID()
        C.objects[str(i)].ID = object[1]
        C.objects[str(i)].size_x = object[2]
        C.objects[str(i)].size_y = object[3]
        C.objects[str(i)].size_z = object[4]

    # Setup renderer
    C.renderer = CN()

    C.renderer["0"] = CN()
    C.renderer["0"].inputPort = str(inputPort)
    C.renderer["0"].outputPort = str(outputPort)

    # Setup vehicle model
    C.vehicle_model = CN()
    C.vehicle_model.uav1 = CN()
    C.vehicle_model.uav1.type = "uav"
    C.vehicle_model.uav1.initialPose = initialPose
    C.vehicle_model.uav1.imu_freq = imu_freq

    C.vehicle_model.uav1.cameraInfo = CN()

    # Setup camera model
    C.camera_model = CN()

    for i, camera in enumerate(cameras):
        C.camera_model[str(i)] = CN()

        C.camera_model[str(i)].ID = camera.ID()
        C.camera_model[str(i)].channels = camera.channels()
        C.camera_model[str(i)].isDepth = camera.isDepth()
        C.camera_model[str(i)].outputIndex = i
        C.camera_model[str(i)].hasCollisionCheck = (
            i == 0
        )  # Only setup collisions with the 0th camera
        C.camera_model[str(i)].doesLandmarkVisCheck = False
        C.camera_model[
            str(i)
        ].initialPose = (
            cameraInitialPose.copy()
        )  # .copy() needed to prevent weirdness in *.cfg
        C.camera_model[str(i)].renderer = 0
        C.camera_model[str(i)].freq = cam_freq
        C.camera_model[str(i)].outputShaderType = camera.outputShaderType()

        C.vehicle_model.uav1.cameraInfo[camera.ID()] = CN()
        C.vehicle_model.uav1.cameraInfo[
            camera.ID()
        ].relativePose = (
            cameraInitialPose.copy()
        )  # .copy() needed to prevent weirdness in *.cfg
        C.vehicle_model.uav1.cameraInfo[camera.ID()].freq = cam_freq

    return C


def cfg_v3_defaults():
    return scene(scene=Scene.Stata_GroundFloor, cameras=[CameraShader.RGB])


def cfg_v3_depth_defaults():
    return scene(
        scene=Scene.Stata_GroundFloor,
        cameras=[CameraShader.RGB, CameraShader.DepthMultiChannel],
    )


def cfg_v3_basement_defaults():
    return scene(scene=Scene.Stata_Basement, cameras=[CameraShader.RGB])


def cfg_v3_depth_basement_defaults():
    return scene(
        scene=Scene.Stata_Basement,
        cameras=[CameraShader.RGB, CameraShader.DepthMultiChannel],
    )


def cfg_v3_depth_ground_floor_defaults():
    return cfg_v3_depth_defaults()


def dump(cfg: CN, filename: str = None):
    x = cfg.dump()

    # The style for FG configuration files uses integers as keys.
    # This is a little hack to replace string keys in the cfg dump as integers
    # 10 is a "magical" upperbound for max keys (expect to be 2 or 3 max, typically)
    for i in range(10):
        x = re.sub("'{}':".format(i), "{}:".format(i), x)

    if filename is None:
        # This 'temporary file' should actually stay 'forever', i.e. until
        # deleted by the user.
        (fd, filename) = tempfile.mkstemp(suffix=".yaml", prefix="FlightGoggles")
        fl = os.fdopen(fd, "w")
    else:
        fl = open(filename, "w")

    fl.write(x)
    fl.close()
    return filename


def get_fg_config(scene: str, depth: bool = True) -> CN:
    if scene == "" and depth:
        return cfg_v3_depth_defaults()
    if scene == "basement" and depth:
        return cfg_v3_depth_basement_defaults()
    elif scene == "basement" and not depth:
        return cfg_v3_basement_defaults()
    elif scene == "ground_floor" and depth:
        return cfg_v3_depth_ground_floor_defaults()
    elif scene == "ground_floor_car" and depth:
        return cfg_v3_depth_ground_floor_defaults()
    else:
        raise ValueError(f"scene: {scene} with depth: {depth} not supported")
