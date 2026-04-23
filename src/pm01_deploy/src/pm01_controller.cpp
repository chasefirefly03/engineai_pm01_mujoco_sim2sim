#include <chrono>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/parameter.hpp>
#include <thread>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "components/message_handler.hpp"
#include "pm01_controller.hpp"

pm01_controller::pm01_controller():Node("pm01_controller"), session(nullptr){
    this->declare_parameter<std::string>("config_file", "");
    this->get_parameter("config_file", config_file);
    // this->declare_parameter<std::string>("policy_file", "");
    // this->get_parameter("policy_file", policy_file);    
    // this->declare_parameter<std::string>("motion_file_npz", "");
    // this->get_parameter("motion_file_npz", motion_file_npz);

    YAML::Node yaml_node = YAML::LoadFile(config_file);

    policy_file = yaml_node["policy_file"].as<std::string>();
    motion_file_csv = yaml_node["motion_file_path"].as<std::string>();

	default_joint_pos = yaml_node["default_joint_pos"].as<std::vector<float>>();
    joint_kp = yaml_node["joint_kp"].as<std::vector<float>>();
    joint_kd = yaml_node["joint_kd"].as<std::vector<float>>();
    observation_scale_linear_vel = yaml_node["observation_scale_linear_vel"].as<float>();
    observation_scale_base_ang_vel = yaml_node["observation_scale_base_ang_vel"].as<float>();
    observation_scale_base_quat_w = yaml_node["observation_scale_base_quat_w"].as<float>();   
    observation_scale_joint_pos = yaml_node["observation_scale_joint_pos"].as<float>();
    observation_scale_joint_vel = yaml_node["observation_scale_joint_vel"].as<float>();
    num_observations = yaml_node["num_observations"].as<float>();
    num_actions = yaml_node["num_actions"].as<float>();
    action_scale = yaml_node["action_scale"].as<float>();

    info_get_action_output = yaml_node["info_get_action_output"].as<bool>();
    info_get_joint_command_output = yaml_node["info_get_joint_command_output"].as<bool>();
    info_get_obs = yaml_node["info_get_obs"].as<bool>();
    fps = yaml_node["motion_data_fps"].as<float>();
    control_frequency = yaml_node["control_frequency"].as<float>();
    if (fps <= 0.f || control_frequency <= 0.f) {
        throw std::runtime_error(
            "motion_data_fps and control_frequency must be positive (got motion_data_fps=" +
            std::to_string(fps) + ", control_frequency=" + std::to_string(control_frequency) + ")");
    }

    obs.setZero(num_observations);
	act.setZero(num_actions);

    xml_to_policy = {0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22};
    policy_to_xml = {0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21, 2, 5, 10, 14, 18, 22, 6, 11, 15, 19, 23, 7};
    
    // Initialize ONNX Runtime
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "pm01_controller");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    time = 0.0f;
    timestep = 0;
    torso_yaw_degree = 0.0f;
    init_quat.setIdentity();

    // Check if policy file exists
    std::ifstream f(policy_file.c_str());
    if (!f.good()) {
        RCLCPP_ERROR(this->get_logger(), "Policy file not found: %s", policy_file.c_str());
        // Handle error?
    }
    
    try {
        session = Ort::Session(env, policy_file.c_str(), session_options);
    } catch (const Ort::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load ONNX model: %s", e.what());
    }

    // Determine input and output names dynamically
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Inputs
    size_t num_input_nodes = session.GetInputCount();
    input_node_names_str.resize(num_input_nodes);
    input_node_names.resize(num_input_nodes);
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto name_ptr = session.GetInputNameAllocated(i, allocator);
        input_node_names_str[i] = name_ptr.get();
        input_node_names[i] = input_node_names_str[i].c_str();
        RCLCPP_INFO(this->get_logger(), "Model Input %zu: %s", i, input_node_names[i]);
    }

    // Outputs
    size_t num_output_nodes = session.GetOutputCount();
    output_node_names_str.resize(num_output_nodes);
    output_node_names.resize(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        output_node_names_str[i] = name_ptr.get();
        output_node_names[i] = output_node_names_str[i].c_str();
        RCLCPP_INFO(this->get_logger(), "Model Output %zu: %s", i, output_node_names[i]);
    }

    // loading motion data
    motion = std::make_shared<MotionLoader_>(motion_file_csv, fps);

    // ?????????????????????????????????????
    // switch to zero torque state
    // current_state_ = ControlState::ZERO_TORQUE;
    current_state_ = ControlState::RL_CONTROL;
    
}

bool pm01_controller::Initialize() {
    try {
        // Initialize message handler
        message_handler_ = std::make_shared<MessageHandler>(shared_from_this());
        joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();
        message_handler_->Initialize();
        // Wait for first motion state
        while (!message_handler_->GetLatestMotionState() ||
               !message_handler_->GetLatestImu() ||
               !message_handler_->GetLatestJointState() ||
                message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for joint bridge state and sensor data...");
        }
        RCLCPP_INFO(get_logger(), "Already in joint bridge state");
        // Get initial joint positions
        auto initial_state = message_handler_->GetLatestJointState();
        if (!initial_state) {
        RCLCPP_ERROR(get_logger(), "Failed to get initial joint state");
        return false;
        }
        initial_joint_pos = Eigen::Map<const Eigen::VectorXd>(initial_state->position.data(), initial_state->position.size()).cast<float>();
        
        RCLCPP_INFO(get_logger(), "Starting control loop");
        control_timer_ = create_wall_timer(std::chrono::duration<float>(1.0/control_frequency),
                                         std::bind(&pm01_controller::ControlCallback, this));   

        // 获取机器人初始朝向和参考动作之间的yaw旋转差
        auto imu = message_handler_->GetLatestImu();
        Eigen::Quaternionf robot_init_base_link_quaternion(
            (float)imu->quaternion.w, 
            (float)imu->quaternion.x, 
            (float)imu->quaternion.y, 
            (float)imu->quaternion.z);
        auto joint_init_position = message_handler_->GetLatestJointState();
        Eigen::Quaternionf motion_first_quaternion = motion->root_quaternions[0];
        Eigen::Quaternionf robot_init_torso_quaternion = pm01_utils::get_torso_quat_w(robot_init_base_link_quaternion,torso_yaw_degree);
        init_quat = pm01_utils::get_init_quat(motion_first_quaternion,robot_init_torso_quaternion);

        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
        return false;
    }
}

void pm01_controller::RLControl() {   
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") return;

    auto joint_state = message_handler_->GetLatestJointState();
    joint_pos = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size()).cast<float>();
    joint_vel = Eigen::Map<const Eigen::VectorXd>(joint_state->velocity.data(), joint_state->velocity.size()).cast<float>();

    auto imu = message_handler_->GetLatestImu();
    Eigen::Quaternionf base_link_quat(
        (float)imu->quaternion.w, 
        (float)imu->quaternion.x, 
        (float)imu->quaternion.y, 
        (float)imu->quaternion.z);
    
    // Vector for observation loop [w, x, y, z]
    Eigen::Vector4f base_quat_w(base_link_quat.w(), base_link_quat.x(), base_link_quat.y(), base_link_quat.z());
    Eigen::Vector3f base_ang_vel = Eigen::Vector3f(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z).cast<float>(); 

    if (!motion || motion->num_frames <= 0) {
        return;
    }
    const int nf = motion->num_frames;
    const int motion_idx = motion_index % nf;

    Eigen::VectorXf motion_commnad(48);
    motion_commnad.segment(0, 24) = motion->dof_positions[motion_idx];
    motion_commnad.segment(24, 24) = motion->dof_velocities[motion_idx];

    Eigen::VectorXf motion_anchor_ori_b = pm01_utils::get_motion_anchor_ori_b(
        base_link_quat, init_quat, torso_yaw_degree, motion->root_quaternions[motion_idx]);

    timestep = timestep + 1;

    // Align motion frame advance with timeline: each control tick is 1/control_frequency [s],
    // one motion row spans 1/motion_data_fps [s] => advance motion_frame_accumulator_ by fps/control_frequency.
    motion_frame_accumulator_ += fps / control_frequency;
    while (motion_frame_accumulator_ >= 1.0f) {
        motion_frame_accumulator_ -= 1.0f;
        motion_index = (motion_index + 1) % nf;
    }
    
    // [INFO] Observation Manager: <ObservationManager> contains 2 groups.
    // +-----------------------------------------------------------+
    // | Active Observation Terms in Group: 'policy' (shape: (129,)) |
    // +-----------+-----------------------------------+-----------+
    // |   Index   | Name                              |   Shape   |
    // +-----------+-----------------------------------+-----------+
    // |     0     | motion_command                    |   (48,)   |
    // |     1     | motion_anchor_ori_b               |    (6,)   |
    // |     2     | base_ang_vel                      |    (3,)   |
    // |     3     | joint_pos_rel                     |   (24,)   |
    // |     4     | joint_vel_rel                     |   (24,)   |
    // |     5     | last_action                       |   (24,)   |
    // +-----------+-----------------------------------+-----------+

    obs.segment(0, 48) = motion_commnad;
    obs.segment(48, 6) = motion_anchor_ori_b;
    obs.segment(54, 3) = base_ang_vel * observation_scale_base_ang_vel;
    
    for (int i = 0; i < 24; i++)
    {
        // joint_pos_rel
        obs(57 + i) = (joint_pos[xml_to_policy[i]] - default_joint_pos[xml_to_policy[i]]) * observation_scale_joint_pos;
        // joint_vel_rel
        obs(81 + i) = joint_vel[xml_to_policy[i]] * observation_scale_joint_vel;
    }
    // last_action
    obs.segment(105, 24) = act;

    // policy forward
    std::vector<int64_t> input_node_dims = {1, obs.size()};
    size_t input_tensor_size = obs.size();
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, obs.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_node_names.data(),
        &input_tensor, 
        1, 
        output_node_names.data(), 
        1
    );

    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    // Assume output is (1, num_actions)
    std::memcpy(act.data(), floatarr, static_cast<size_t>(num_actions) * sizeof(float));

    joint_command_->position.resize(24);
    for (int i = 0; i < 24; i++){
        joint_command_->position[i] = act(policy_to_xml[i]) * action_scale + default_joint_pos[i];
    }

    // 发送关节命令
    joint_command_->velocity = std::vector<double>(act.size(), 0.0);
    joint_command_->feed_forward_torque = std::vector<double>(act.size(), 0.0);
    joint_command_->torque = std::vector<double>(act.size(), 0.0);
    joint_command_->stiffness = std::vector<double>(joint_kp.data(), joint_kp.data() + joint_kp.size());
    joint_command_->damping = std::vector<double>(joint_kd.data(), joint_kd.data() + joint_kd.size());
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    
    message_handler_->PublishJointCommand(*joint_command_);
    
    if (info_get_obs){std::cout << "obs: \n" << obs.transpose() << std::endl;};
    if (info_get_action_output){std::cout << "action: \n" << act.transpose() << std::endl;};
    if (info_get_joint_command_output){
        std::cout << "joint_command_output: \n";
        for (const auto& val : joint_command_->position) std::cout << val << " ";
        std::cout << std::endl;
    };

     // Check transition to DAMP
    auto gamepad = message_handler_->GetLatestGamepad();
    if (gamepad && gamepad->digital_states[6] == 1) { // SELECT/BACK
         RCLCPP_INFO(get_logger(), "Switching to DAMP");
         current_state_ = ControlState::DAMP;
    }
}

void pm01_controller::ControlCallback() {   
    // Dispatch based on state
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") return;

     switch (current_state_) {
        case ControlState::ZERO_TORQUE:
            ZeroTorqueState();
            break;
        case ControlState::MOVE_TO_DEFAULT:
            MoveToDefaultPos();
            break;
        case ControlState::RL_CONTROL:
            RLControl();
            break;
        case ControlState::DAMP:
            DampState();
            break;
    }
}

void pm01_controller::ZeroTorqueState() {
    auto gamepad = message_handler_->GetLatestGamepad();
    
    // Send zero torque command
    joint_command_->position.assign(24, 0.0);
    joint_command_->velocity.assign(24, 0.0);
    joint_command_->feed_forward_torque.assign(24, 0.0);
    joint_command_->torque.assign(24, 0.0);
    joint_command_->stiffness.assign(24, 0.0); 
    joint_command_->damping.assign(24, 0.0);
    
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    message_handler_->PublishJointCommand(*joint_command_);

    // Check transition: Start button (7)
    if (gamepad && gamepad->digital_states[7] == 1) { 
        RCLCPP_INFO(get_logger(), "Switching to MOVE_TO_DEFAULT");
        current_state_ = ControlState::MOVE_TO_DEFAULT;
        state_start_time_ = this->now();
        
        // Capture start position for interpolation
        auto joint_state = message_handler_->GetLatestJointState();
        if (joint_state) {
            move_to_default_start_pos_ = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size()).cast<float>();
        } else {
            move_to_default_start_pos_ = Eigen::VectorXf::Zero(24);
        }
    }
}

void pm01_controller::MoveToDefaultPos() {
    auto gamepad = message_handler_->GetLatestGamepad();
    
    float transition_duration = 2.0f; // 2 seconds
    float elapsed = (this->now() - state_start_time_).seconds();
    float phase = std::clamp(elapsed / transition_duration, 0.0f, 1.0f);
    
    joint_command_->position.resize(24);
    for (int i = 0; i < 24; i++){
        joint_command_->position[i] = (1.0f - phase) * move_to_default_start_pos_[i] + phase * default_joint_pos[i];
    }
    
    joint_command_->velocity.assign(24, 0.0);
    joint_command_->feed_forward_torque.assign(24, 0.0);
    joint_command_->torque.assign(24, 0.0);
    // Use configured KP/KD for holding position
    joint_command_->stiffness = std::vector<double>(joint_kp.begin(), joint_kp.end());
    joint_command_->damping = std::vector<double>(joint_kd.begin(), joint_kd.end());
    
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    message_handler_->PublishJointCommand(*joint_command_);

    // Check transition: A button (2)
    if (phase >= 1.0f && gamepad && gamepad->digital_states[2] == 1) { 
         RCLCPP_INFO(get_logger(), "Switching to RL_CONTROL");
         current_state_ = ControlState::RL_CONTROL;
         time = 0.0f; // Reset RL time
         
         // Update initial_joint_pos so RL smoothing doesn't jump
         auto joint_state = message_handler_->GetLatestJointState();
         if(joint_state) {
           initial_joint_pos = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size()).cast<float>();
         }

         // Initialize init_quat for motion tracking
         auto imu = message_handler_->GetLatestImu();
         if (imu && motion && motion->num_frames > 0) {
            Eigen::Quaternionf base_quat(
                (float)imu->quaternion.w, 
                (float)imu->quaternion.x, 
                (float)imu->quaternion.y, 
                (float)imu->quaternion.z);
            auto real_torso_quat_w = pm01_utils::get_torso_quat_w(base_quat, torso_yaw_degree);
            auto motion_root_quat_w = motion->root_quaternions[0];
            init_quat = pm01_utils::get_init_quat(motion_root_quat_w, real_torso_quat_w);
         }
    }
}

void pm01_controller::DampState() {
    joint_command_->position.assign(24, 0.0);
    joint_command_->velocity.assign(24, 0.0);
    joint_command_->feed_forward_torque.assign(24, 0.0);
    joint_command_->torque.assign(24, 0.0);
    joint_command_->stiffness.assign(24, 0.0); 
    joint_command_->damping.assign(24, 8.0); // kd = 8.0
    
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    message_handler_->PublishJointCommand(*joint_command_);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
	auto controller = std::make_shared<pm01_controller>();
    if (controller->Initialize()) {
        rclcpp::spin(controller);
    }

    rclcpp::shutdown();
	return 0;
}




