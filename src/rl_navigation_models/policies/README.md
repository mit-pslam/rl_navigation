## Running Policies in ROS

![AI](https://media.kasperskycontenthub.com/wp-content/uploads/sites/67/2016/09/10032917/artificial-intelligence.jpg)

You may be thinking, *Why must I be surrounded by frickin' idiots?*
We don't know that answer to that question.
But, we can help you run the latest and greatest RL policies.

We assume you are familiar with installation instructions for [this package](../../../README.md), as well as [rl_navigation_ros](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation_ros).



## Stable Baselines Basic Hover Policy

When installing this package in your python virtual environment (i.e., `venv`), consider using `pip install -e rl_navigation[hover,ros]`.
This will install [stable-baselines](https://stable-baselines.readthedocs.io/en/master/), and (attempt to) install tensorflow correctly.
You may need to clean up the install.

In order to run a policy, you'll need to download the file that specifies the model.
Then, you'll need to generate a configuration file.

1. Navigate to a directory where you want to store the model file. 
Download [hover-20200806.zip](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation/releases/download/v0.1-alpha/hover-20200806.zip) to that directory.
Note that you'll have to use a browser that is logged into github.mit.edu.
Sorry, `wget` won't work.

2. Create a configuration file for the hover policy.
```sh
echo "model_path: \"$(pwd)/hover-20200806.zip\"
height: 192
width: 256" > hover-policy.yaml
```

3. Now you are ready to run in ROS. Execute the following `roslaunch` command.
```sh
roslaunch rl_navigation_ros rl_navigation_pipeline.launch \
environment_path:=/path/to/your/environment input_map_file:=flightgoggles_map.yaml \
module:=policies policy:=StableBaselinesBasicHoverPolicy model_config_file:=$(pwd)/hover-policy.yaml
```
**Note:** this assumes you are running from the same directory where you saved `hover-policy.yaml`.
If running from elsewhere, you will need to specify the full path.


## RLlib Point Navigation Policies

For Point Navigation, we started using [RLlib](https://docs.ray.io/en/master/rllib.html).
It's a more flexible, more powerful reinforcement learning package.

When installing this package in your python virtual environment (i.e., `venv`), consider using `pip install -e rl_navigation[rllib,ros]`.
This will install [RLlib](https://docs.ray.io/en/master/rllib.html), and some dependencies.
Note, however, that you will **first** need to install [pytorch](https://pytorch.org/get-started/locally/) following their directions for your system configuration.

We intend to release multipe versions of point navigation.
These are spelled out below.

First, a note on connecting this up to different ROS topics.
You specify how topics are used with a yaml file, such as those in [rl_navigation_ros/config](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation_ros/tree/master/config).
Point navigation will require ownship **pose** and **goal** point (note: that we expect a pose message, and orientation is not used).
Point navigation may also require RGB **image** and/or **depth** image, depending on the policy being used.
The following table shows message requirements.

| policy | pose | goal | image | depth |
| --- | --- | --- | --- | --- |
| pointnav-pose-20210212 | x | x | - | - |
| pointnav-pose-20210713 | x | x | - | - |

### pointnav-pose-20210212

**Overview**: This is the first point navigation policy that only uses pose information.
The source for pose information could be Kimera VIO, directly from a simulator, or from mo-cap.
The policy does not provide collision avoidance capabilities.
It was optimized to get to the goal point as quickly as possible.
Will it come in hot??

1. Navigate to a directory where you want to store the model file. 
Download [pointnav-pose-20210212.zip](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation/releases/download/v0.1-alpha/pointnav-pose-20210212.zip) to that directory.
Note that you'll have to use a browser that is logged into github.mit.edu.
Sorry, `wget` won't work.
Unzip the folder: `unzip pointnav-pose-20210212.zip`.

2. Create a configuration file for the policy.
```sh
echo "checkpoint_path: $(pwd)/pointnav-pose-20210212/pointnav-pose
ppo_config:
  env_config:
    action_mapper: continuous
    fields: []
    max_steps: 300
    renderer: ''
  framework: torch
  model:
    fcnet_activation: tanh
    fcnet_hiddens:
    - 512
    - 256
  num_gpus: 1
  num_workers: 0" > pointnav-pose-20210212.yaml
```

3. Now you are ready to run in ROS. Execute the following `roslaunch` command.
```sh
roslaunch rl_navigation_ros rl_navigation_pipeline.launch \
environment_path:=/path/to/your/environment input_map_file:=the-pointnav-map-you-want-use.yaml \
module:=policies policy:=RllibPointNavPolicy model_config_file:=$(pwd)/pointnav-pose-20210212.yaml
```
**Note:** this assumes you are running from the same directory where you saved `pointnav-pose-20210212.yaml`.
If running from elsewhere, you will need to specify the file's full path.


### pointnav-pose-20210713

**Overview**: This is the second point navigation policy that only uses pose information.
The source for pose information could be Kimera VIO, directly from a simulator, or from mo-cap.
The policy does not provide collision avoidance capabilities.
It was optimized to get to the goal point quickly while also trying to generate a *smooth* trajectory.

1. Navigate to a directory where you want to store the model file.
Download [pointnav-pose-20210713.zip](https://github.mit.edu/aiia-suas-disaster-response/rl_navigation/releases/download/v0.1-alpha/pointnav-pose-20210713.zip) to that directory.
Note that you'll have to use a browser that is logged into github.mit.edu.
Sorry, `wget` won't work.
Unzip the folder: `unzip pointnav-pose-20210713.zip`.

2. Create a configuration file for the policy.
```sh
echo "checkpoint_path: $(pwd)/pointnav-pose-20210713/pointnav-pose
ppo_config:
  env_config:
    action_mapper: continuous
    fields: []
    max_steps: 300
    renderer: ''
  framework: torch
  model:
    fcnet_activation: tanh
    fcnet_hiddens:
    - 128
    - 128
    lstm_cell_size: 128
    lstm_use_prev_action: true
    max_seq_len: 1
    use_lstm: true
  num_gpus: 1
  num_workers: 0" > pointnav-pose-20210713.yaml
```

3. Now you are ready to run in ROS. Execute the following `roslaunch` command.
```sh
roslaunch rl_navigation_ros rl_navigation_pipeline.launch \
environment_path:=/path/to/your/environment input_map_file:=the-pointnav-map-you-want-use.yaml \
module:=policies policy:=RllibPointNavPolicy model_config_file:=$(pwd)/pointnav-pose-20210713.yaml
```
**Note:** this assumes you are running from the same directory where you saved `pointnav-pose-20210713.yaml`.
If running from elsewhere, you will need to specify the file's full path.
