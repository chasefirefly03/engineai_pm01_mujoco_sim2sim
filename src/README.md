# sim2sim仿真
## 基础行走
1. 启动mujoco仿真模块`ros2 launch mujoco_simulator mujoco_simulator.launch.py`

2. 启动仿真`ros2 run pm01_deploy pm01_controller --ros-args -p config_file:=src/pm01_deploy/config/pm01.yaml`

或者可以直接运行python程序进行仿真`python3 src/pm01_deploy/script/deploy_mujoco.py`,记得修改`config_file`

## AMP
`python3 src/pm01_deploy/script/deploy_mujoco_amp.py`

## Beyond Minic
`python3 src/pm01_deploy/script/deploy_mujoco_minic.py`

# sim2real
`ros2 run pm01_deploy pm01_controller --ros-args -p config_file:=/src/pm01_deploy/config/pm01.yaml`

# 现有的问题
## AMP
### 机器人步态跨步太大
1. 动作数据的步幅太大，导致机器人在低速的时候动作很差
2. 机器人在给0速度下还是向前走或者踏步
