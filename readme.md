## Reinforcement Learning Experiments for FlightGoggles

This repository contains experiments for training trajectory-following and obstacle-avoiding policies in the FlightGoggles warehouse environment. By default, the action space is continuous, with actions interpreted as an angular velocity between `(-1, 1) rad/"step"`. Discrete policies are also supported.

### Installation

Anaconda is recommended:

```bash
conda install conda-build
conda create -n fgrl --python=3.7
cd /to/some/desired/directory
git clone git@github.mit.edu/aiia-suav-distaster-response/rl_navigation
cd rl_navigation
conda build conda.recipes && conda install -n fgrl --use-local --force-reinstall -y rl_navigation
```

You can also do this with pip and python3.7.  You may need to add the deadsnakes ppa for this.

**WARNING: Do not mix and match these methods.**

```bash
sudo apt install python3.7*
python3.7 -m venv /path/to/environment
source /path/to/environment/bin/activate
cd /to/some/desired/directory
git clone git@github.mit.edu/aiia-suav-distaster-response/rl_navigation
pip install -e rl_navigation
```

### Training

[Create an `experiment.yml` file using `yacs`](https://github.com/rbgirshick/yacs#usage) (you will need to call `cfg.merge_from_file("experiment.yml")` in `fgtrain.py`) or edit `config.py` and specify the location of your FlightGoggles binary:

```yaml
_C.FLIGHTGOGGLES.BINARY = "/path/to/FlightGoggles.x86_64"
```

Then, activate your conda environment and start training:

```bash
$ conda activate fgrl
(fgrl) python fgtrain.py
```

If you want to view the agent's performance in realtime, then in a second terminal run:

```bash
$ conda activate fgrl
(fgrl) python plot_train_fg.py
```

You should see the following output:

![](doc/fgrl.gif)


## Next steps

* Vehicle dynamics model
* Train depth-based policies
* Increase FPS for training (disable motion blur?)
* Note: there are several `TODO`s in the codebase indicating areas for improvement.
