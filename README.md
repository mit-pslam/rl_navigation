## rl_navigation

This repository provides gym environments for training policies with a particular emphasis on using [FlightGoggles](https://github.mit.edu/aiia-suas-disaster-response/FlightGoggles).

### Dependencies

First, install a few items with `apt`:
```bash
sudo apt install python3.7* libeigen3-dev libzmqpp-dev
```

Next, install [OpenCV](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html).

Optionally, install the [vulkan device chooser layer](https://github.com/aejsmith/vkdevicechooser) if you anticipate training on a machine with multiple GPUs and want to use all of them.
See [this issue](https://github.mit.edu/aiia-suas-disaster-response/FlightGoggles/issues/1) for a detailed discussion.
```bash
sudo apt install libvulkan-dev vulkan-validationlayers-dev
```

## Setup

You can use pip and python3.7 to install this.
Note you may need to add the deadsnakes ppa to get python3.7.

### Setup Environment (Optional)

**The venv way**

You can create a virtual environment with `venv` using something like the following.
```bash
python3.7 -m venv /path/to/environment
```
Then, activate that environment with:
```bash
source /path/to/environment/bin/activate
```

**The conda way**

```bash
conda create --name aia python=3.7 meson pkgconfig numpy
conda activate aia
```

### Install

Firstly, install the proper version of [pytorch](https://pytorch.org/get-started/locally/) for your machine.
It is important that you install a version compatible with CUDA on your machine.
Note that [fast-depth-estimation](https://github.mit.edu/aiia-suas-disaster-response/fast-depth-estimation) has a dependency of `torchvision==0.9.1`, which is used by this package.

**Note:** The installation for this package will also install [pyFlightGoggles](https://github.mit.edu/aiia-suas-disaster-response/pyFlightGoggles/tree/opencv4-compatible) in your `site-packages`.
If you anticipate needing to do any development on [pyFlightGoggles](https://github.mit.edu/aiia-suas-disaster-response/pyFlightGoggles), then you should install in your development environment.
Please install the [opencv4-compatible](https://github.mit.edu/aiia-suas-disaster-response/FlightGoggles-Python/tree/opencv4-compatible) branch and follow instructions there.

```bash
pip install git+https://github.mit.edu/aiia-suas-disaster-response/rl_navigation.git
```

If cloning to develop, certainly use something like:
```bash
git clone git@github.mit.edu:aiia-suas-disaster-response/rl_navigation.git
pip install -e rl_navigation
```

Note that you can install the following additional features:
```
pip install -e rl_navigation[doc,test,rllib] #see extras_require
```
- **doc**: feature to support autogenerated documentation.
- **test**: feature to support testing.
- **hover**: feature to support hover policy learning and demonstration using [stable-baselines](https://stable-baselines.readthedocs.io/en/master/).
- **rllib**: feature to support using [rllib](https://docs.ray.io/en/latest/rllib.html), required for point navigation.
- **ros**: feature to support running in ROS.

So, if you want to run point navigation experiments in ROS, you should run `pip install -e rl_navigation[rllib,ros]`.

**NOTE:** If using `rllib`, please check out [their documentation](https://docs.ray.io/en/latest/rllib.html#running-rllib).
You will need to install `pytorch` or `tensorflow`.
Note that we do already require `pytorch`.

## Usage

The aim of this repository is to combine loosely-coupled dynamic simulators with one or more observation models.
The simulator updates the position and orientation of a vehicle based on action inputs from a policy.
The observation model is made up of a **chain** of observers.
Each one adds to an observation dictionary.
Examples of observers include FlightGoggles and Kimera.

See [disaster.py](src/rl_navigation/disaster.py) for an overview of how these are all combined together.
An example of a simple hover gym environment can be found in [hover.py](src/rl_navigation/tasks/hover.py).

See [here](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation/tree/develop/src/rl_navigation_models/policies) for details about running the hover demonstraion.

### Training Example

Here is an example to train a policy to hover in the FlightGoggles warehouse.
This example requires [stable-baselines](https://stable-baselines.readthedocs.io/en/master/).

First, import all necessary packages.
```python
from rl_navigation.tasks.hover import HoverEnv 
import time
import matplotlib.pyplot as plt
from pathlib import Path
import stable_baselines.common.policies as stb_policies
import stable_baselines.common.vec_env as stb_env
from stable_baselines import PPO2
```

Second, set `fg_path` to the path for FlightGoggles (e.g., `fg_path = /home/disaster/flightgoggles/FlightGoggles_Linux64_v2.0.3/FlightGoggles.x86_64`).

Next, setup the training environment.
```python
def make_unity_env(num_env):
    """Create a wrapped Unity environment."""

    def make_env(rank):
        def _thunk():
            env = HoverEnv(fg_path,
                           goal_height=3.2,
                           reset_bounds=((-3, 3), (-3, 3), (.2, 6.2)),
                           action_delay=0,
                           terminate_outside_bounds=False,
                           max_steps=50
            )
            return env

        return _thunk

    return stb_env.DummyVecEnv([make_env(i) for i in range(num_env)])

env = make_unity_env(1)  # Define environment
```

Now you can define the policy model.
```python
model = PPO2(
        stb_policies.CnnPolicy,
        env,
        gamma=0.95,
        learning_rate=0.000125,
        verbose=1,
        nminibatches=1,
        tensorboard_log="./tensorboard/",
    )
```

Finally, launch model learning.
```python
model.learn(total_timesteps=4e6)
```

If you have [tensorboard](https://www.tensorflow.org/tensorboard), you can launch it from a new terminal.
```sh
tensorboard --logdir ./tensorboard/
```

### Evaluation in ROS
You can run policies in [ROS](https://www.ros.org/) using [rl_navigation_ros](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation_ros). A pre-trained policy (generated using the above) can be ran as described [here](src/rl_navigation_models/policies/README.md).


## Next Steps

* Vehicle dynamics model
* Train depth-based policies
* Increase FPS for training (disable motion blur?)
* Note: there are several `TODO`s in the codebase indicating areas for improvement.
