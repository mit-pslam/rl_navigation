from yacs.config import CfgNode as CN
import re
import tempfile
import os

_C = CN()

_C.state = CN()
_C.state.sceneFilename = "Abandoned_Factory_Morning"
_C.state.camWidth = 1024
_C.state.camHeight = 768
_C.state.camFOV = 70.0
_C.state.camDepthScale = 0.20

_C.state.map = CN()
_C.state.map.filename = "Stata_GroundFloor.png"
_C.state.map.x_min = -31.0
_C.state.map.x_max = 78.5
_C.state.map.y_min = -68.1
_C.state.map.y_max = 78.8
_C.state.map.map_margin = 0
_C.state.map.map_scale = 5

_C.renderer = CN()

_C.renderer["0"] = CN()
_C.renderer["0"].inputPort = "10253"
_C.renderer["0"].outputPort = "10254"

_C.camera_model = CN()

_C.camera_model["0"] = CN()
_C.camera_model["0"].ID = "RGB"
_C.camera_model["0"].channels = 3
_C.camera_model["0"].isDepth = False
_C.camera_model["0"].outputIndex = 0
_C.camera_model["0"].hasCollisionCheck = True
_C.camera_model["0"].doesLandmarkVisCheck = False
_C.camera_model["0"].initialPose = [0, 0, 0, 1, 0, 0, 0]
_C.camera_model["0"].renderer = 0
_C.camera_model["0"].freq = 30
_C.camera_model["0"].outputShaderType = -1

_C.vehicle_model = CN()

_C.vehicle_model.uav1 = CN()
_C.vehicle_model.uav1.type = "uav"
_C.vehicle_model.uav1.initialPose = [50.0, -49.0, -2.0, 0, 0, 0, 1]
_C.vehicle_model.uav1.imu_freq = 200

_C.vehicle_model.uav1.cameraInfo = CN()

_C.vehicle_model.uav1.cameraInfo.RGB = CN()
_C.vehicle_model.uav1.cameraInfo.RGB.relativePose = [0, 0, 0, 1, 0, 0, 0]
_C.vehicle_model.uav1.cameraInfo.RGB.freq = 30

_C.objects = CN()
_C.objects["0"] = CN()
_C.objects["0"].ID = "gate1"
_C.objects["0"].prefabID = "gate"
_C.objects["0"].size_x = 100
_C.objects["0"].size_y = 100
_C.objects["0"].size_z = 100

_C_v2 = _C.clone()


def cfg_v2_defaults():
    return _C_v2.clone()


# Flight Goggles Version 3
_C_v3 = _C.clone()
_C_v3.state.sceneFilename = "Stata_GroundFloor"


def cfg_v3_defaults():
    return _C_v3.clone()


# Flight Goggles Version 3 with depth
_C_v3_depth = _C.clone()
_C_v3_depth.state.sceneFilename = "Stata_GroundFloor"

_C_v3_depth.camera_model["1"] = CN()
_C_v3_depth.camera_model["1"].ID = "Depth"
_C_v3_depth.camera_model["1"].channels = 1
_C_v3_depth.camera_model["1"].isDepth = True
_C_v3_depth.camera_model["1"].outputIndex = 1
_C_v3_depth.camera_model["1"].hasCollisionCheck = False
_C_v3_depth.camera_model["1"].doesLandmarkVisCheck = False
_C_v3_depth.camera_model["1"].initialPose = [0, 0, 0, 1, 0, 0, 0]
_C_v3_depth.camera_model["1"].renderer = 0
_C_v3_depth.camera_model["1"].freq = 30
_C_v3_depth.camera_model["1"].outputShaderType = 3

_C_v3_depth.vehicle_model.uav1.cameraInfo.Depth = CN()
_C_v3_depth.vehicle_model.uav1.cameraInfo.Depth.relativePose = [0, 0, 0, 1, 0, 0, 0]
_C_v3_depth.vehicle_model.uav1.cameraInfo.Depth.freq = 30


def cfg_v3_depth_defaults():
    return _C_v3_depth.clone()


# Flight Goggles version 3 in Basement
_C_v3_basement = _C_v3.clone()
_C_v3_basement.state.sceneFilename = "Stata_Basement"


def cfg_v3_basement_defaults():
    return _C_v3_basement.clone()


# Flight Goggles version 3 in Basement with Depth
_C_v3_depth_basement = _C_v3_depth.clone()
_C_v3_depth_basement.state.sceneFilename = "Stata_Basement"


def cfg_v3_depth_basement_defaults():
    return _C_v3_depth_basement.clone()


# Flight Goggles version 3 in Stata first floor with depth
_C_v3_depth_ground_floor = _C_v3_depth.clone()
_C_v3_depth_ground_floor.state.sceneFilename = "Stata_GroundFloor"


def cfg_v3_depth_ground_floor_defaults():
    return _C_v3_depth_ground_floor.clone()


# Freeze all of the configs so a user doesn't modify directly
_C_v2.freeze()
_C_v3.freeze()
_C_v3_depth.freeze()
_C_v3_basement.freeze()
_C_v3_depth_basement.freeze()
_C_v3_depth_ground_floor.freeze()


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
