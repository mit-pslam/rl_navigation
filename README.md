## Reinforcement Learning Experiments for FlightGoggles

This repository contains experiments for training trajectory-following and obstacle-avoiding policies in the FlightGoggles warehouse environment. By default, the action space is continuous, with actions interpreted as an angular velocity between `(-1, 1) rad/"step"`. Discrete policies are also supported.

### Installation

**WARNING: use conda at your own risk.**

You can use pip and python3.7 to install this.  You may need to add the deadsnakes ppa to get python3.7.

```bash
sudo apt install python3.7*
python3.7 -m venv /path/to/environment
source /path/to/environment/bin/activate
cd /to/some/desired/directory
git clone git@github.mit.edu/aiia-suav-distaster-response/rl_navigation
pip install rl_navigation
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

## Next steps

* Vehicle dynamics model
* Train depth-based policies
* Increase FPS for training (disable motion blur?)
* Note: there are several `TODO`s in the codebase indicating areas for improvement.
