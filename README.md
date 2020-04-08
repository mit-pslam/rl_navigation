## Reinforcement Learning Experiments for FlightGoggles

This repository contains experiments for training trajectory-following and obstacle-avoiding policies in the FlightGoggles warehouse environment. By default, the action space is continuous, with actions interpreted as an angular velocity between `(-1, 1) rad/"step"`. Discrete policies are also supported.

### Installation

You can use pip and python3.7 to install this.  You may need to add the deadsnakes ppa to get python3.7.

```bash
sudo apt install python3.7* libeigen3-dev
python3.7 -m venv /path/to/environment
source /path/to/environment/bin/activate
cd /to/some/desired/directory
git clone git@github.mit.edu/aiia-suav-distaster-response/rl_navigation
pip install rl_navigation
```

**WARNING: use conda at your own risk.**

For developing with conda: though not recommended, you can nest a `venv` inside a conda env, e.g.,:

```bash
conda create --name aiia python=3.7 meson pkgconfig numpy
conda activate aiia
python -m venv ./env/
source env/bin/activate
pip install -e rl_navigation[doc,test] #see extras_require
```

## Interaction

Once installed (and the environment you used is active), you should be able to use the command line interface:

```bash
rl_navigation -h
```

You have a few options

```bash
rl_navigation train  # train a model
rl_navigation plot   # show what's going on during training
rl_navigation movie  # export the result of running a policy as a movie
```

For training, you'll need to pass in a valid path to the FlightGoggles Binary using a config file, for example:

```bash
rl_navigation train --configuration_file binary.yaml
```

`binary.yaml` should contain something like this:

```yaml
FLIGHTGOGGLES:
  BINARY: "/home/mark/catkin_ws/devel/.private/flightgoggles/lib/flightgoggles/FlightGoggles.x86_64"
```

(Remember not to check in your `binary.yaml` file)

## Next steps

* Vehicle dynamics model
* Train depth-based policies
* Increase FPS for training (disable motion blur?)
* Note: there are several `TODO`s in the codebase indicating areas for improvement.
