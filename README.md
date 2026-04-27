# 编译
1. 修改`src/pm01_deploy/config/param`下的`robot_xml_path`为绝对路径
2. C++仿真与部署依赖onnxruntimr:在[onnxruntime](https://github.com/microsoft/onnxruntime/releases)下载适当架构的onnxruntime-linux-x64-1.24.2并解压复制到/usr/include/onnxruntime-linux-x64-1.24.2
3. python仿真依赖pytorch:`pip install pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple`
4. 编译构建工作空间
```bash
colcon build
source install/setup.bash
```

# RL Walking
## 仿真
> 待添加文档说明
###  C++
> 需要在[onnxruntime](https://github.com/microsoft/onnxruntime/releases)下载onnxruntime-linux-x64-1.24.2并解压复制到/usr/include/onnxruntime-linux-x64-1.24.2
1. 启动mujoco仿真模块
```bash            
ros2 launch mujoco_simulator mujoco_simulator.launch.py
```

2. 启动仿真
- manager base env
```bash
ros2 run pm01_deploy pm01_controller_rl_walking --ros-args -p config_file:=src/pm01_deploy/config/param/pm01_real_base_manager.yaml -p policy_file:=src/pm01_deploy/config/policy/base_walking/manager_base_env.onnx
```
- direct env 
```bash
ros2 run pm01_deploy pm01_controller_rl_walking --ros-args -p config_file:=src/pm01_deploy/config/param/pm01_real_direct_env.yaml -p policy_file:=src/pm01_deploy/config/policy/base_walking/dirct_env.onnx
```
3. 启动手柄控制
```bash
python src/pm01_deploy/script/gamepad_publisher.py
```

### Python
> 需要安装pytorch：`pip install pytorch -i https://pypi.tuna.tsinghua.edu.cn/simple`
- manager base env
```bash
python3 src/pm01_deploy/script/deploy_mujoco_base.py --config_file src/pm01_deploy/config/param/pm01_mujoco_base_manager.yaml --policy_file src/pm01_deploy/config/policy/base_walking/manager_base_env.pt
```
- direct env
```bash        
python3 src/pm01_deploy/script/deploy_mujoco_base.py --config_file src/pm01_deploy/config/param/pm01_mujoco_base_direct.yaml --policy_file src/pm01_deploy/config/policy/base_walking/dirct_env.pt
```
- 手柄控制机器人

> 待完善

## Sim2Real
> 待完善

```bash
ros2 run pm01_deploy pm01_controller_rl_walking --ros-args -p config_file:=src/pm01_deploy/config/param/pm01_mujoco_base_direct.yaml
```

 
# Beyond Minic
## 仿真
### Python
```bash
python3 src/pm01_deploy/script/deploy_mujoco_minic.py --config_file src/pm01_deploy/config/param/pm01_mujoco_minic_python.yaml
```
### C++
> 1. CMakeLists文件可能存在mujoco路径问题，这个为C++直接运行在mujoco，与众擎官方仓库无关
> 2. BUG：在Mujoco中点击Reset后仿真会加速；执行完动作后不会停止，会重新加载动作数据并继续执行
> 3. 策略和动作可在`src/pm01_deploy/config/param/pm01_mujoco_minic_cpp.yaml`中修改
```bash
ros2 run minic_mujoco pm01_mujoco_sim
```
## 部署
！！！！执行完动作后会退出程序，假如最后时刻的动作使得机器人不平衡，可能摔倒导致机器人受损！！！！
1. 部署依赖onnxruntime，可以不在机器人上安装pytorch
2. 动作和策略可在`src/pm01_deploy/config/param/pm01_real_minic.yaml`中修改
3. cpp文件在`src/pm01_deploy/src/pm01_minic_deploy.cpp`
```python
ros2 run pm01_deploy pm01_minic_deploy
```

# AMP
> 待完善
`python3 src/pm01_deploy/script/deploy_mujoco_amp.py`