## Running Policies in ROS

![Brace yourselves](https://memegenerator.net/img/instances/68268988/brace-yourselves-artificial-intelligence-is-coming.jpg)

### Stable Baselines Basic Hover Policy

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
