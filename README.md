## rl_navigation

This repository provides gym environments for training policies with a particular emphasis on using [FlightGoggles](https://flightgoggles.mit.edu/).

### Dependencies

These install instructions assume Ubuntu and Python 3.7.

It also assumes that you have the [FlightGoggles renderer](https://flightgoggles-documentation.scrollhelp.site/fg/Installation.392888321.html).
This package uses the FlightGoggles Python interface. It will be installed automatically below.

First, install a few items with `apt`:
```bash
sudo apt install cmake libeigen3-dev libopencv-dev libzmqpp-dev
```

**[Optional]** Install the [vulkan device chooser layer](https://github.com/aejsmith/vkdevicechooser) if you anticipate training on a machine with multiple GPUs and want to use all of them.
FlightGoggles uses [vulkan](https://developer.nvidia.com/vulkan), which has limited Mult-GPU support by default (see [this particular issue](https://www.reddit.com/r/linux_gaming/comments/c3gk5v/force_specific_gpu_in_vulkan_applicationsgames/) and a [solution](https://github.com/aejsmith/vkdevicechooser) for more details).
```bash
sudo apt install libvulkan-dev vulkan-validationlayers-dev
```

## Setup

### Setup Environment (Optional, but Recommended)

**The venv way**

You can create a virtual environment with `venv` using something like the following.
```bash
python3.7 -m venv </path/to/environment>
```
Then, activate that environment with:
```bash
source </path/to/environment/bin/activate>
```

**The conda way**

```bash
conda create --name navigation python=3.7 meson pkgconfig numpy
conda activate navigation
```

### Install

**[Important Note]** We recommend training with [rllib](https://docs.ray.io/en/stable/rllib.html) and [pytorch](https://pytorch.org).
However, `rllib` is not a requirement for using this package.
If you choose to use a separate training pipeline, modify the following installation steps as appropriate.

Firstly, install the proper version of [pytorch](https://pytorch.org/get-started/locally/) for your machine.
It is important that you install a version compatible with CUDA on your machine.

For basic usage, install this repository with `pip`:
```bash
pip install "rl_navigation[rllib] @ git+ssh://git@github.com/mit-pslam/rl_navigation.git"
```

If cloning to develop, certainly use something like:
```bash
git clone git@github.com/mit-pslam/rl_navigation.git
pip install -e rl_navigation[rllib]
```

Note that you can install the following additional features:
```
pip install -e rl_navigation[doc,test,rllib,fast_depth,ros] #see extras_require
```
- **doc**: feature to support autogenerated documentation.
- **test**: feature to support testing.
- **rllib**: feature to support using [rllib](https://docs.ray.io/en/latest/rllib.html).
- **fast_depth**: feature to support a [fast-depth-estimation](https://github.mit.edu/aiia-suas-disaster-response/fast-depth-estimation) observer.
Note that access to the `fast-depth-estimation` repository is needed to use this feature.
- **ros**: feature to support running in ROS using [rl_navigation_ros](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation_ros).
Again, note that access to this repository is needed to use this feature.


## Overview

The aim of this repository is to combine loosely-coupled dynamic simulators with one or more observation models.
The simulator updates the position and orientation of a vehicle based on action inputs from a policy.
The observation model is made up of a **chain** of observers.
Each one adds to an observation dictionary.
Examples of observers include FlightGoggles and Kimera.

See [disaster.py](src/rl_navigation/disaster.py) for an overview of how these are all combined together.
An example of a simple hover gym environment can be found in [hover.py](src/rl_navigation/tasks/hover.py).

## More Resources

**Running FlightGoggles or TESSE in headless mode**

See details [here](https://github.com/mit-aera/FlightGoggles/wiki/Running-Flightgoggles-in-AWS) or [here](https://github.com/MIT-TESSE/tesse-core#running-tesse-headless) for running a simulator in headless mode.

**Evaluation in ROS**

You can run policies in [ROS](https://www.ros.org/) using [rl_navigation_ros](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation_ros) (assuming you have access to this repository).
A trained policy can be ran as described [here](src/rl_navigation_models/policies/README.md).

## Acknowledgement
Research was sponsored by the United States Air Force Research Laboratory and the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
