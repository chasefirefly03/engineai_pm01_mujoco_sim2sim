- manager base env的C++部署还存在问题
- direct env 不能原地旋转

# 编译
```
colcon build
source install/setup.bash
```

# Sim2Sim
> 1. 运行之前需要编译
> 2. 修改pm01_deploy/config/param下yaml文件的机器人模型路径`robot_xml_path`为绝对路径
## base walking
###  C++
> C++仿真代码和实机部署的代码一样的,但是需要安装onnxruntime(见Sim2Real一节）
1. 启动mujoco仿真模块
```            
ros2 launch mujoco_simulator mujoco_simulator.launch.py
```

2. 启动仿真
- manager base env????????
```
ros2 run pm01_deploy pm01_controller --ros-args -p config_file:=src/pm01_deploy/config/param/pm01_real_base_manager.yaml -p policy_file:=src/pm01_deploy/config/policy/base_walking/manager_base_env.onnx
```
- direct env 
```
ros2 run pm01_deploy pm01_controller --ros-args -p config_file:=src/pm01_deploy/config/param/pm01_real_direct_env.yaml -p policy_file:=src/pm01_deploy/config/policy/base_walking/dirct_env.onnx
```
3. 启动手柄??????
```
python src/pm01_deploy/script/gamepad_publisher.py
```

### Python
> 需要安装pytorch
- manager base env
```
python3 src/pm01_deploy/script/deploy_mujoco_base.py --config_file src/pm01_deploy/config/param/pm01_mujoco_base_manager.yaml --policy_file src/pm01_deploy/config/policy/base_walking/manager_base_env.pt
```
- direct env
```        
python3 src/pm01_deploy/script/deploy_mujoco_base.py --config_file src/pm01_deploy/config/param/pm01_mujoco_base_direct.yaml --policy_file src/pm01_deploy/config/policy/base_walking/dirct_env.pt
```
- 手柄控制机器人

？？？？？？？？？？


## AMP
`python3 src/pm01_deploy/script/deploy_mujoco_amp.py`
 
## Beyond Minic
```
python3 src/pm01_deploy/script/deploy_mujoco_minic.py
```

# Sim2Real
## 安装onnxruntime

## 在机器人运行
```
ros2 run pm01_deploy pm01_controller --ros-args -p config_file:=/src/pm01_deploy/config/pm01.yaml
```

# 现有的问题
## AMP
### 机器人步态跨步太大
1. 动作数据的步幅太大，导致机器人在低速的时候动作很差
2. 机器人在给0速度下还是向前走或者踏步
