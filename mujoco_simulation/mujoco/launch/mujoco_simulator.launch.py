from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import OpaqueFunction
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Get package directory
    package_dir = get_package_share_directory('mujoco_simulator')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description=(
            '若 true：节点使用 ROS /clock（仿真时间）。本栈未在 MuJoCo 内发布 /clock，'
            '与墙钟 50Hz 的 minic 推理节点联仿时建议保持 false，避免 header 时间与物理脱节。'
            '若你自行接入 /clock 并与策略同用仿真时间，可改为 true。'
        ),
    )

    # Create environment variables
    env_vars = [
        SetEnvironmentVariable('PRODUCT', 'pm_v2'),
        SetEnvironmentVariable('MUJOCO_ASSETS_PATH',
                               PathJoinSubstitution([package_dir, 'assets'])),
        SetEnvironmentVariable('LD_LIBRARY_PATH', [
                               '/opt/engineai_robotics_third_party/lib:/opt/ros/humble/lib:', EnvironmentVariable(name='LD_LIBRARY_PATH', default_value='')])
    ]

    # 根据headless参数创建节点配置
    def launch_setup(context, *args, **kwargs):

        # 准备节点参数
        args = []

        # 定义MuJoCo模拟器节点
        mujoco_node = Node(
            package='mujoco_simulator',
            executable='mujoco_simulator',
            name='mujoco_simulator',
            output='screen',
            emulate_tty=True,
            arguments=args,
            parameters=[
                {
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'), value_type=bool
                    ),
                },
            ]
        )

        return [mujoco_node]

    # 使用OpaqueFunction来获取上下文中的参数值
    mujoco_launch = OpaqueFunction(function=launch_setup)

    # Return launch description
    return LaunchDescription([
        declare_use_sim_time,
        *env_vars,
        mujoco_launch
    ])
