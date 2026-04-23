#ifndef PM01_UTILS_HPP
#define PM01_UTILS_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace pm01_utils
{
    // 消除初始yaw的偏差
    inline void updateFirstFrameYawAlignment(Eigen::Quaterniond imu_data, Eigen::Quaterniond ref_anchor_ori_quat_w, Eigen::Matrix3d &ref_init_yaw_rot, Eigen::Matrix3d &body_init_yaw_rot)
    {
        // Get the robot's current orientation from IMU
        Eigen::Matrix3d R_local = imu_data.toRotationMatrix();

        // Get the reference trajectory's first-frame body orientation (quaternion: w, x, y, z)
        Eigen::Matrix3d ref_anchor_ori_rot_w = ref_anchor_ori_quat_w.toRotationMatrix();

        // Extract yaw angles (rotation about Z-axis) from both orientations
        double ref_yaw = std::atan2(ref_anchor_ori_rot_w(1, 0), ref_anchor_ori_rot_w(0, 0));
        double body_yaw = std::atan2(R_local(1, 0), R_local(0, 0));

        // Store pure yaw rotation matrices for coordinate frame alignment in observations
        ref_init_yaw_rot = Eigen::AngleAxisd(ref_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        body_init_yaw_rot = Eigen::AngleAxisd(body_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }

    // 提取出yaw
    inline Eigen::Quaternionf yawQuaternion(const Eigen::Quaternionf &q)
    {
        float yaw = std::atan2(2.0f * (q.w() * q.z() + q.x() * q.y()), 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z()));
        float half_yaw = yaw * 0.5f;
        Eigen::Quaternionf ret(std::cos(half_yaw), 0.0f, 0.0f, std::sin(half_yaw));
        return ret.normalized();
    }

    // 计算机器人躯干实时朝向
    inline Eigen::Quaternionf get_torso_quat_w(const Eigen::Quaternionf &root_quat_w, float torso_yaw)
    {
        Eigen::Quaternionf torso_quat_w = root_quat_w * Eigen::AngleAxisf(torso_yaw, Eigen::Vector3f::UnitZ());
        return torso_quat_w;
    }

    // 计算参考动作实时朝向
    inline Eigen::Quaternionf get_anchor_quat_w(float torso_yaw, const Eigen::Quaternionf &motion_root_quat)
    {
        Eigen::Quaternionf anchor_quat_w = motion_root_quat * Eigen::AngleAxisf(torso_yaw, Eigen::Vector3f::UnitZ());
        return anchor_quat_w;
    }

    // 获取机器人初始朝向和参考动作之间的yaw旋转差
    inline Eigen::Quaternionf get_init_quat(const Eigen::Quaternionf &motion_root_quat_w, const Eigen::Quaternionf &real_torso_quat_w)
    {
        auto ref_yaw = yawQuaternion(motion_root_quat_w);
        auto robot_yaw = yawQuaternion(real_torso_quat_w);
        Eigen::Quaternionf init_quat = robot_yaw * ref_yaw.inverse();
        return init_quat;
    }

    // 消除机器人当前朝向与参考动作初始朝向之间的Yaw偏差,用于obs
    inline Eigen::VectorXf get_motion_anchor_ori_b(
        const Eigen::Quaternionf &root_quat_w,
        const Eigen::Quaternionf &init_quat,
        float &torso_yaw_degree,
        const Eigen::Quaternionf &motion_root_quat)
    {
        auto real_quat_w = get_torso_quat_w(root_quat_w, torso_yaw_degree);
        auto ref_quat_w = get_anchor_quat_w(torso_yaw_degree, motion_root_quat);

        Eigen::Quaternionf rot_quat = (init_quat * ref_quat_w).conjugate() * real_quat_w;
        Eigen::Matrix3f rot = rot_quat.toRotationMatrix().transpose();

        Eigen::VectorXf motion_anchor_ori_b(6);
        motion_anchor_ori_b << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
        return motion_anchor_ori_b;
    }

    inline std::vector<std::vector<float>> load_csv(const std::string &filename)
    {
        std::vector<std::vector<float>> data;
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ','))
            {
                try
                {
                    row.push_back(std::stof(value));
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Invalid value in file: " << value << std::endl;
                }
            }
            if (!row.empty()) data.push_back(row);
        }
        return data;
    }
}
#endif // PM01_UTILS_HPP
