#ifndef PM01_CONTROLLER_H
#define PM01_CONTROLLER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <stdexcept>


#include "components/message_handler.hpp"
#include "pm01_utils.hpp"


class MotionLoader_
{
public:
    float dt;
    int num_frames;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    MotionLoader_(std::string motion_file, float fps)
        : dt(1.0f / fps)
    {
        auto data = pm01_utils::load_csv(motion_file);
        if (data.empty())
        {
            throw std::runtime_error("Could not open motion file or empty: " + motion_file);
        }

        num_frames = static_cast<int>(data.size());

        for (int i = 0; i < num_frames; ++i)
        {
            const auto &row = data[i];
            if (row.size() < 7)
            {
                throw std::runtime_error(
                    "Motion CSV row " + std::to_string(i) + " has fewer than 7 columns");
            }
            root_positions.push_back(
                Eigen::VectorXf(Eigen::Map<const Eigen::VectorXf>(row.data(), 3)));
            root_quaternions.push_back(Eigen::Quaternionf(row[6], row[3], row[4], row[5]));
            dof_positions.push_back(Eigen::VectorXf(Eigen::Map<const Eigen::VectorXf>(
                row.data() + 7, static_cast<Eigen::Index>(row.size() - 7))));
        }
        dof_velocities = _comupte_raw_derivative(dof_positions);
    }

    std::vector<Eigen::VectorXf> _comupte_raw_derivative(const std::vector<Eigen::VectorXf>& data)
    {
        std::vector<Eigen::VectorXf> derivative;
        for(size_t i = 0; i < data.size() - 1; ++i) {
            derivative.push_back((data[i + 1] - data[i]) / dt);
        }
        derivative.push_back(derivative.back());
        return derivative;
    }
};

class pm01_controller : public rclcpp::Node
{
	public:
        bool Initialize();
        pm01_controller();
        void ControlCallback();

        enum class ControlState {
            ZERO_TORQUE,
            MOVE_TO_DEFAULT,
            RL_CONTROL,
            DAMP
        };

    private:
        void ZeroTorqueState();
        void MoveToDefaultPos();
        void RLControl();
        void DampState();
    
        Eigen::VectorXf get_motion_anchor_ori(const Eigen::Quaternionf& root_quat_w, int index);
        Eigen::VectorXf get_motion(int timestep);
        ControlState current_state_;
        rclcpp::Time state_start_time_;
        Eigen::VectorXf move_to_default_start_pos_;

        bool info_get_action_output;
        bool info_get_joint_command_output;
        bool info_get_obs;

        float time = 0.0f;
        int motion_index = 0;
        /** Accumulates fractional motion frames so playback matches motion_data_fps vs control_frequency. */
        float motion_frame_accumulator_ = 0.0f;
        int timestep = 0;
        float torso_yaw_degree = 0.0f;
        Eigen::Quaternionf init_quat;

        rclcpp::TimerBase::SharedPtr control_timer_; 
        std::shared_ptr<MessageHandler> message_handler_;
        std::shared_ptr<interface_protocol::msg::JointCommand> joint_command_;

        std::string policy_file;
        std::string config_file;
        std::string motion_file_csv;
        
        std::vector<float> default_joint_pos;
        Eigen::VectorXf initial_joint_pos;
        std::vector<float> joint_kp;
        std::vector<float> joint_kd;
        Eigen::VectorXf joint_pos;
        Eigen::VectorXf joint_vel;
        Eigen::VectorXd joint_pos_cmd;
        
        float observation_scale_linear_vel;
        float observation_scale_base_ang_vel;
        float observation_scale_base_quat_w;   
        float observation_scale_joint_pos;
        float observation_scale_joint_vel;

        float num_observations;
        float num_actions;
        float action_scale;
        // float num_include_obs_steps;
        float fps;

        float control_frequency;
        
        Eigen::VectorXf obs;
		Eigen::VectorXf act;

        std::vector<int> xml_to_policy;
        std::vector<int> policy_to_xml;

        std::string policy_path;
        Ort::Env env;
        Ort::Session session;
        
        std::vector<const char*> input_node_names;
        std::vector<const char*> output_node_names;
        std::vector<std::string> input_node_names_str;
        std::vector<std::string> output_node_names_str;

        std::shared_ptr<MotionLoader_> motion;


        Eigen::Matrix3d ref_init_yaw_rot_;
        Eigen::Matrix3d body_init_yaw_rot_;
};

#endif
