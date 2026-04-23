import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import argparse

config_file="src/pm01_deploy/config/param/pm01_mujoco_minic.yaml"

anchor_body_name="LINK_TORSO_YAW"
MOTION_BODY_INDEX = 3

def load_motion(motion_file):                
    motion = np.load(motion_file)            
    motion_pos = motion["body_pos_w"]        
    motion_quat = motion["body_quat_w"]      
    motion_input_pos = motion["joint_pos"]   
    motion_input_vel = motion["joint_vel"]   
    return motion_pos, motion_quat, motion_input_pos, motion_input_vel 

def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):  # 计算 A 到 B 的相对位姿  
    rotm_a = np.zeros(9)  # 初始化旋转矩阵扁平数组  
    mujoco.mju_quat2Mat(rotm_a, quat_a)  # 四元数转旋转矩阵  
    rotm_a = rotm_a.reshape(3, 3)  # 重塑为 3x3 矩阵  
    rel_pos = rotm_a.T @ (pos_b - pos_a)  # 计算相对位置向量  
    rel_quat = quaternion_multiply(quaternion_conjugate(quat_a), quat_b)  # 计算相对旋转四元数  
    rel_quat = rel_quat / np.linalg.norm(rel_quat)  # 归一化四元数  
    return rel_pos, rel_quat  # 返回相对位置和相对姿态  

def quaternion_conjugate(q):  # 计算四元数共轭  
    return np.array([q[0], -q[1], -q[2], -q[3]])  # 标量不变向量取负 

def quaternion_multiply(q1, q2):  # 计算四元数乘积  
    w1, x1, y1, z1 = q1  # 解包第一个四元数  
    w2, x2, y2, z2 = q2  # 解包第二个四元数  
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # 计算结果 w 分量  
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # 计算结果 x 分量  
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # 计算结果 y 分量  
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # 计算结果 z 分量  
    
    return np.array([w, x, y, z])  # 返回乘积四元数


def get_yaw_matrix_from_quat(quat_wxyz):  # 与 pm01_controller.cpp / GetYawMatrixFromQuat 一致
    rotm = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotm, quat_wxyz.astype(np.float64))
    R = rotm.reshape(3, 3)
    x_axis_rotated = R[:, 0]
    x_proj = np.array([x_axis_rotated[0], x_axis_rotated[1]], dtype=np.float64)
    n = np.linalg.norm(x_proj)
    if n < 1e-4:
        return np.eye(3, dtype=np.float64)
    x_proj = x_proj / n
    new_x = np.array([x_proj[0], x_proj[1], 0.0], dtype=np.float64)
    new_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    new_y = np.cross(new_z, new_x)
    return np.stack([new_x, new_y, new_z], axis=1)


def rotation_matrix_to_quat_wxyz(R):  # MuJoCo w,x,y,z
    quat = np.zeros(4, dtype=np.float64)
    mat9 = np.ascontiguousarray(R.reshape(9), dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat9)
    return quat.astype(np.float64)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", type=str, help="config file name in the config folder")
    # args = parser.parse_args()
    # config_file = args.config_file

    xml_to_policy = [0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
    policy_to_xml = [0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21, 2, 5, 10, 14, 18, 22, 6, 11, 15, 19, 23, 7]

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_file"]
        xml_path = config["robot_xml_path"]
        motion_file = config["motion_file_path"]
        get_info = config["get_info"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["joint_kp"], dtype=np.float32)
        kds = np.array(config["joint_kd"], dtype=np.float32)

        default_angles = np.array(config["default_joint_pos"], dtype=np.float32)
        
        ang_vel_scale = config["observation_scale_base_ang_vel"]
        dof_pos_scale = config["observation_scale_joint_pos"]
        dof_vel_scale = config["observation_scale_joint_vel"]
        action_scale = config["action_scale"]

        # num_actions = config["num_actions"]
        # num_obs = config["num_observations"]
        num_actions = 24
        num_obs = 129
        
    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)   

    timestep = 0
    motion_body_index = MOTION_BODY_INDEX
    body_name = anchor_body_name
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)  # 获取锚点 ID 
    _, motion_quat, motion_input_pos, motion_input_vel = load_motion(motion_file)
    init_to_world_quat = None  # 首帧 yaw 对齐：将参考动作首帧水平朝向与仿真锚点对齐

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        # time.sleep(5)
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                quaternion = d.xquat[body_id]  # 获取仿真中锚点刚体姿态  
                motion_input = np.concatenate(  # 拼接目标关节位置与速度  
                    (motion_input_pos[timestep, :], motion_input_vel[timestep, :]),  # 使用 npz 原始关节顺序  
                    axis=0,  # 进行拼接  
                )  # 拼接结束  
                motion_quat_current = motion_quat[timestep, motion_body_index, :]  # 读取当前参考锚点动作姿态  
                # yaw 对齐：init_to_world = R_yaw_robot @ R_yaw_motion.T，首帧冻结（同 pm01_controller / rl_beyondmimic.cc）
                if timestep == 0:
                    yaw_motion = get_yaw_matrix_from_quat(motion_quat[0, motion_body_index])
                    yaw_robot = get_yaw_matrix_from_quat(quaternion)
                    init_to_world_mat = yaw_robot @ yaw_motion.T
                    init_to_world_quat = rotation_matrix_to_quat_wxyz(init_to_world_mat)
                motion_frame_alignment = quaternion_multiply(init_to_world_quat, motion_quat_current)
                anchor_quat = quaternion_multiply(quaternion_conjugate(quaternion), motion_frame_alignment)
                anchor_quat = anchor_quat / np.linalg.norm(anchor_quat)
                anchor_ori = np.zeros(9)  # 初始化锚点旋转矩阵  
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)  # 仿真锚点相对于参考锚点的相对姿态四元数转旋转矩阵  
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]  # 取旋转矩阵前两列  
                motion_anchor_ori_b = anchor_ori.reshape(-1,)  # 展平为向量  

                joint_pos = d.qpos[7:31]
                joint_vel = d.qvel[6:30]

                base_ang_vel = d.qvel[3:6]

                qj = (joint_pos[xml_to_policy] - default_angles[xml_to_policy]) * dof_pos_scale
                dqj = (joint_vel[xml_to_policy]) * dof_vel_scale

                obs[:48] = motion_input
                obs[48:54] = motion_anchor_ori_b
                obs[54:57] = base_ang_vel * ang_vel_scale
                obs[57:81] = qj
                obs[81:105] = dqj
                obs[105:129] = action[xml_to_policy]

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[policy_to_xml]
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                timestep += 1
                
                if get_info:
                    print("--------------------------------")
                    # print("motion_command     \n", (obs[0:48]))
                    print("motion_anchor_ori_b\n", (obs[48:54]))
                    print("base_ang_vel       \n",(obs[54:57]))
                    print("joint_pos_rel      \n",(obs[57:81]))
                    # print("joint_vel_rel      \n",(obs[81:105]))
                    # print("last_action        \n",(obs[105:129]))
                    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep * 20 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)