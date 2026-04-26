#ifndef PM01_CONTROLLER_H
#define PM01_CONTROLLER_H

#include <Eigen/Core>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>


#include "components/message_handler.hpp"


class pm01_controller : public rclcpp::Node
{
	public:

        bool Initialize();
        pm01_controller();
        void ControlCallback();
        Eigen::Vector3f projected_gravity_b(Eigen::Quaternionf quat, Eigen::Vector3f vec);

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

        ControlState current_state_;
        rclcpp::Time state_start_time_;
        Eigen::VectorXf move_to_default_start_pos_;

        bool info_get_action_output;
        bool info_get_joint_command_output;
        bool info_get_obs;
        Eigen::Vector3f set_body_vel;
        Eigen::Vector3f gravity_world;

        float action_clip_limit;
        float cycle_time;
        float period = .6;
        float time = 0;
        float global_phase_;

        rclcpp::TimerBase::SharedPtr control_timer_; 
        std::shared_ptr<MessageHandler> message_handler_;
        std::shared_ptr<interface_protocol::msg::JointCommand> joint_command_;


        std::string policy_file;
        std::string config_file;
        
        std::vector<float> default_joint_pos;
        Eigen::VectorXf initial_joint_pos;
        std::vector<float> joint_kp;
        std::vector<float> joint_kd;
        Eigen::VectorXf joint_pos;
        Eigen::VectorXf joint_vel;
        Eigen::VectorXd joint_pos_cmd;


        // std::vector<float> action_scale;

        float observation_scale_linear_vel;
        float observation_scale_base_ang_vel;
        float observation_scale_base_quat_w;   
        float observation_scale_joint_pos;
        float observation_scale_joint_vel;


        float num_observations;
        float num_actions;
        float action_scale;
        // float num_include_obs_steps;

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
};

#endif
