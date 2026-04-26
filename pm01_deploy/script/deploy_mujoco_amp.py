import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import argparse


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="config file name in the config folder")
    # args = parser.parse_args()
    # config_file = args.config_file
    config_file="src/pm01_deploy/config/param/pm01_mujoco_amp.yaml"

    xml_to_policy = [0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
    policy_to_xml = [0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21, 2, 5, 10, 14, 18, 22, 6, 11, 15, 19, 23, 7]

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_file"]
        xml_path = config["robot_xml_path"]
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
        base_quat_w_scale = config["observation_scale_base_quat_w"]
        # cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = 24
        num_obs = 425  # config["num_observations"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    
    # buffers
    policy_history_len = 5
    buffer_dof_pos = np.zeros((policy_history_len, 24), dtype=np.float32)
    buffer_dof_vel = np.zeros((policy_history_len, 24), dtype=np.float32)
    # buffer_action size should be consistent with action size which is 24 (num_actions from config)
    buffer_action = np.zeros((policy_history_len, 24), dtype=np.float32) 
    buffer_base_ang_vel = np.zeros((policy_history_len, 3), dtype=np.float32)
    buffer_base_quat_w = np.zeros((policy_history_len, 4), dtype=np.float32)
    buffer_cmd = np.zeros((policy_history_len, 3), dtype=np.float32)
    buffer_proj_grav = np.zeros((policy_history_len, 3), dtype=np.float32)

    counter = 0

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)        

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
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

                joint_pos = d.qpos[7:31]
                joint_vel = d.qvel[6:30]

                base_ang_vel = d.qvel[3:6]
                base_quat_w = d.qpos[3:7]

                velocity_commands = cmd
                projected_gravity = get_gravity_orientation(base_quat_w)
                # gait_phase

                count = counter * simulation_dt

                qj = (joint_pos - default_angles) * dof_pos_scale
                dqj = joint_vel * dof_vel_scale

                # update buffers
                buffer_dof_pos = np.roll(buffer_dof_pos, -1, axis=0)
                buffer_dof_pos[-1] = qj[xml_to_policy]

                buffer_dof_vel = np.roll(buffer_dof_vel, -1, axis=0)
                buffer_dof_vel[-1] = dqj[xml_to_policy]

                buffer_action = np.roll(buffer_action, -1, axis=0)
                buffer_action[-1] = action[xml_to_policy]

                buffer_base_ang_vel = np.roll(buffer_base_ang_vel, -1, axis=0)
                buffer_base_ang_vel[-1] = base_ang_vel * ang_vel_scale

                buffer_base_quat_w = np.roll(buffer_base_quat_w, -1, axis=0)
                buffer_base_quat_w[-1] = base_quat_w * base_quat_w_scale

                buffer_cmd = np.roll(buffer_cmd, -1, axis=0)
                buffer_cmd[-1] = velocity_commands

                buffer_proj_grav = np.roll(buffer_proj_grav, -1, axis=0)
                buffer_proj_grav[-1] = projected_gravity

                # construct obs
                idx = 0
                obs[idx:idx+120] = buffer_dof_pos.flatten()
                idx += 120
                obs[idx:idx+120] = buffer_dof_vel.flatten()
                idx += 120
                obs[idx:idx+120] = buffer_action.flatten()
                idx += 120
                obs[idx:idx+15] = buffer_base_ang_vel.flatten()
                idx += 15
                obs[idx:idx+20] = buffer_base_quat_w.flatten()
                idx += 20
                obs[idx:idx+15] = buffer_cmd.flatten()
                idx += 15
                obs[idx:idx+15] = buffer_proj_grav.flatten()
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[policy_to_xml]
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                if get_info:
                    print("--------------------------------")
                    print("joint_pos         \n", obs[0:120])
                    print("joint_vel         \n", obs[120:240])
                    print("actions           \n", obs[240:360])
                    print("base_ang_vel      \n", obs[360:375])
                    print("base_quat_w       \n", obs[375:395])
                    print("velocity_commands \n", obs[395:410])
                    print("projected_gravity \n", obs[410:425])
                    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep * 20 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)