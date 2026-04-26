#include "pm01_minic_deploy.hpp"

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <interface_protocol/msg/parallel_parser_type.hpp>

namespace {

constexpr int kNumJoints = 24;

Eigen::Quaternionf AnchorQuatFromImuBaseWaist(const Eigen::Quaternionf& q_base_imu_wxyz, float waist_theta,
                                             bool from_base_waist, const Eigen::Vector3f& waist_axis_base) {
  if (!from_base_waist) {
    return q_base_imu_wxyz.normalized();
  }
  Eigen::Vector3f u = waist_axis_base;
  if (u.squaredNorm() < 1e-12f) {
    return q_base_imu_wxyz.normalized();
  }
  u.normalize();
  return (q_base_imu_wxyz.normalized() * Eigen::AngleAxisf(waist_theta, u)).normalized();
}

Eigen::Matrix3f GetYawMatrixFromQuat(const Eigen::Quaternionf& q) {
  const Eigen::Vector3f x_axis_rotated = q * Eigen::Vector3f::UnitX();
  Eigen::Vector2f x_proj(x_axis_rotated.x(), x_axis_rotated.y());
  if (x_proj.norm() < 1e-4f) {
    return Eigen::Matrix3f::Identity();
  }
  x_proj.normalize();
  const Eigen::Vector3f new_x(x_proj.x(), x_proj.y(), 0.0f);
  const Eigen::Vector3f new_z = Eigen::Vector3f::UnitZ();
  const Eigen::Vector3f new_y = new_z.cross(new_x);
  Eigen::Matrix3f R_yaw;
  R_yaw.col(0) = new_x;
  R_yaw.col(1) = new_y;
  R_yaw.col(2) = new_z;
  return R_yaw;
}

Eigen::Quaternionf RotationMatrixToQuat(const Eigen::Matrix3f& R) {
  return Eigen::Quaternionf(R).normalized();
}

Eigen::VectorXf MotionAnchorOriB6(const Eigen::Quaternionf& anchor_quat) {
  const Eigen::Quaternionf n = anchor_quat.normalized();
  const Eigen::Matrix3f R = n.toRotationMatrix();
  Eigen::VectorXf out(6);
  out << R(0, 0), R(0, 1), R(1, 0), R(1, 1), R(2, 0), R(2, 1);
  return out;
}

}  // namespace

std::string Pm01ControllerMujoco::ResolvePath(const std::string& path) {
  if (path.empty()) {
    return path;
  }
  if (path[0] == '/') {
    return path;
  }
  return path;
}

std::string Pm01ControllerMujoco::ResolvePolicyOnnx(const YAML::Node& yaml_node) {
  if (yaml_node["onnx_policy_file"]) {
    return yaml_node["onnx_policy_file"].as<std::string>();
  }
  if (yaml_node["policy_file"]) {
    const std::string p = yaml_node["policy_file"].as<std::string>();
    if (p.size() > 5 && p.substr(p.size() - 5) == ".onnx") {
      return p;
    }
  }
  throw std::runtime_error(
      "需要 ONNX 策略：在 YAML 中设置 onnx_policy_file，或将 policy_file 指向 .onnx"
      "（.pt 请先用 torch 导出为 ONNX）。");
}

void Pm01ControllerMujoco::ParseYaml(const YAML::Node& y) {
  if (y["motion_file_path"]) {
    motion_file_ = ResolvePath(y["motion_file_path"].as<std::string>());
  } else if (y["motion_file"]) {
    motion_file_ = ResolvePath(y["motion_file"].as<std::string>());
  } else {
    throw std::runtime_error("YAML 缺少 motion_file_path 或 motion_file");
  }

  if (y["motion_body_index"]) {
    motion_body_index_ = y["motion_body_index"].as<int>();
  }
  if (y["anchor_quat_from_base_waist"]) {
    anchor_quat_from_base_waist_ = y["anchor_quat_from_base_waist"].as<bool>();
  } else {
    anchor_quat_from_base_waist_ = true;
  }

  if (y["waist_yaw_joint_state_index"]) {
    waist_yaw_joint_state_index_ = y["waist_yaw_joint_state_index"].as<int>();
  }
  if (y["waist_yaw_axis_in_base"]) {
    const auto ax = y["waist_yaw_axis_in_base"].as<std::vector<float>>();
    if (ax.size() != 3) {
      throw std::runtime_error("waist_yaw_axis_in_base 须为长度 3 的数组");
    }
    waist_yaw_axis_in_base_ = Eigen::Vector3f(ax[0], ax[1], ax[2]);
  }

  if (y["control_frequency"]) {
    const double hz = y["control_frequency"].as<double>();
    if (hz > 1e-6) {
      control_period_sec_ = 1.0 / hz;
    }
  } else {
    // 与 deploy_mujoco_minic：每 control_decimation 个仿真步跑一次策略一致；无 MuJoCo 时用墙钟等效该频率。
    const double simulation_dt = y["simulation_dt"] ? y["simulation_dt"].as<double>() : 0.001;
    int decimation = 20;
    if (y["control_decimation"]) {
      decimation = y["control_decimation"].as<int>();
    }
    if (decimation < 1) {
      decimation = 1;
    }
    control_period_sec_ = simulation_dt * static_cast<double>(decimation);
  }

  get_info_ = y["get_info"] && y["get_info"].as<bool>();

  if (y["imu_topic"]) {
    imu_topic_ = y["imu_topic"].as<std::string>();
  }
  if (y["joint_state_topic"]) {
    joint_state_topic_ = y["joint_state_topic"].as<std::string>();
  }
  if (y["publish_robot_joint_command"]) {
    publish_robot_joint_command_ = y["publish_robot_joint_command"].as<bool>();
  }
  if (y["joint_command_topic"]) {
    joint_command_topic_ = y["joint_command_topic"].as<std::string>();
  }

  default_joint_pos_ = y["default_joint_pos"].as<std::vector<float>>();
  joint_kp_ = y["joint_kp"].as<std::vector<float>>();
  joint_kd_ = y["joint_kd"].as<std::vector<float>>();

  observation_scale_base_ang_vel_ =
      y["observation_scale_base_ang_vel"] ? y["observation_scale_base_ang_vel"].as<float>() : 1.f;
  observation_scale_joint_pos_ =
      y["observation_scale_joint_pos"] ? y["observation_scale_joint_pos"].as<float>() : 1.f;
  observation_scale_joint_vel_ =
      y["observation_scale_joint_vel"] ? y["observation_scale_joint_vel"].as<float>() : 1.f;
  action_scale_ = y["action_scale"] ? y["action_scale"].as<float>() : 1.f;

  num_observations_ = y["num_observations"] ? y["num_observations"].as<int>() : 129;
  num_actions_ = y["num_actions"] ? y["num_actions"].as<int>() : 24;

  obs_.setZero(num_observations_);
  act_.setZero(num_actions_);
}

void Pm01ControllerMujoco::LoadOnnxSession(const std::string& onnx_path) {
  const std::filesystem::path p{onnx_path};
  if (!std::filesystem::exists(p)) {
    throw std::runtime_error("ONNX 策略文件不存在: " + onnx_path);
  }
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  const size_t num_input_nodes = session_->GetInputCount();
  input_node_names_str_.resize(num_input_nodes);
  input_node_names_.resize(num_input_nodes);
  for (size_t i = 0; i < num_input_nodes; ++i) {
    auto name_ptr = session_->GetInputNameAllocated(i, allocator);
    input_node_names_str_[i] = name_ptr.get();
    input_node_names_[i] = input_node_names_str_[i].c_str();
    RCLCPP_INFO(get_logger(), "模型输入 %zu: %s", i, input_node_names_[i]);
  }
  const size_t num_output_nodes = session_->GetOutputCount();
  output_node_names_str_.resize(num_output_nodes);
  output_node_names_.resize(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; ++i) {
    auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
    output_node_names_str_[i] = name_ptr.get();
    output_node_names_[i] = output_node_names_str_[i].c_str();
    RCLCPP_INFO(get_logger(), "模型输出 %zu: %s", i, output_node_names_[i]);
  }
}

Pm01ControllerMujoco::Pm01ControllerMujoco(const std::string& config_file)
    : Node("pm01_minic_inference"),
      config_file_(config_file),
      env_(ORT_LOGGING_LEVEL_WARNING, "pm01_minic_inference"),
      xml_to_policy_({0,  6,  12, 1,  7,  13, 18, 23, 2,  8,  14, 19, 3,  9,  15, 20, 4,  10, 16, 21,
                      5,  11, 17, 22}),
      policy_to_xml_({0,  3,  8,  12, 16, 20, 1,  4,  9,  13, 17, 21, 2,  5,  10, 14, 18, 22, 6,  11,
                      15, 19, 23, 7}) {
  const YAML::Node y = YAML::LoadFile(config_file_);
  ParseYaml(y);

  const float motion_fps_fallback = y["motion_fps"] ? y["motion_fps"].as<float>() : 50.f;
  motion_ = std::make_shared<MotionLoader_>(motion_file_, motion_fps_fallback, motion_body_index_);

  const std::string onnx_path = ResolvePath(ResolvePolicyOnnx(y));
  LoadOnnxSession(onnx_path);

  if (static_cast<int>(default_joint_pos_.size()) != kNumJoints ||
      static_cast<int>(joint_kp_.size()) != kNumJoints ||
      static_cast<int>(joint_kd_.size()) != kNumJoints) {
    throw std::runtime_error("default_joint_pos / joint_kp / joint_kd 须各为 24 个元素");
  }

  rclcpp::QoS qos(3);
  qos.best_effort();
  qos.durability_volatile();

  imu_sub_ = create_subscription<interface_protocol::msg::ImuInfo>(
      imu_topic_, qos, std::bind(&Pm01ControllerMujoco::ImuCallback, this, std::placeholders::_1));
  joint_sub_ = create_subscription<interface_protocol::msg::JointState>(
      joint_state_topic_, qos, std::bind(&Pm01ControllerMujoco::JointStateCallback, this, std::placeholders::_1));

  if (publish_robot_joint_command_) {
    joint_cmd_pub_ = create_publisher<interface_protocol::msg::JointCommand>(joint_command_topic_, qos);
  }

  const double hz = control_period_sec_ > 1e-12 ? 1.0 / control_period_sec_ : 0.0;
  RCLCPP_INFO(get_logger(),
              "策略/发布周期 %.6f s → %.3f Hz（create_wall_timer，墙钟，不受 use_sim_time 影响）。"
              "IMU=%s joint_state=%s。请保证 MuJoCo 约 100%% 实时，否则 motion 帧与物理时间易错位。",
              control_period_sec_, hz, imu_topic_.c_str(), joint_state_topic_.c_str());
}

void Pm01ControllerMujoco::ImuCallback(const interface_protocol::msg::ImuInfo::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(sensor_mutex_);
  latest_imu_ = msg;
}

void Pm01ControllerMujoco::JointStateCallback(const interface_protocol::msg::JointState::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(sensor_mutex_);
  latest_joint_ = msg;
}

bool Pm01ControllerMujoco::Initialize() {
  RCLCPP_INFO(get_logger(), "等待首帧 IMU 与 joint_state（各至少 %d 维）...", kNumJoints);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
  while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(shared_from_this());
    interface_protocol::msg::ImuInfo::SharedPtr imu_copy;
    interface_protocol::msg::JointState::SharedPtr j_copy;
    {
      std::lock_guard<std::mutex> lock(sensor_mutex_);
      imu_copy = latest_imu_;
      j_copy = latest_joint_;
    }
    if (imu_copy && j_copy && j_copy->position.size() >= static_cast<size_t>(kNumJoints) &&
        j_copy->velocity.size() >= static_cast<size_t>(kNumJoints)) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "仍等待传感器...");
  }
  {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    if (!latest_imu_ || !latest_joint_ || latest_joint_->position.size() < static_cast<size_t>(kNumJoints) ||
        latest_joint_->velocity.size() < static_cast<size_t>(kNumJoints)) {
      RCLCPP_ERROR(get_logger(), "120s 内未收到完整 IMU/joint_state");
      return false;
    }
    if (waist_yaw_joint_state_index_ < 0 ||
        waist_yaw_joint_state_index_ >= static_cast<int>(latest_joint_->position.size())) {
      RCLCPP_ERROR(get_logger(), "waist_yaw_joint_state_index 越界");
      return false;
    }
  }

  const auto period = std::chrono::duration<double>(control_period_sec_);
  control_timer_ = create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(period),
                                     std::bind(&Pm01ControllerMujoco::ControlTimerCallback, this));
  RCLCPP_INFO(get_logger(), "已开始推理控制循环");
  return true;
}

void Pm01ControllerMujoco::ControlTimerCallback() {
  interface_protocol::msg::ImuInfo::SharedPtr imu;
  interface_protocol::msg::JointState::SharedPtr jst;
  {
    std::lock_guard<std::mutex> lock(sensor_mutex_);
    imu = latest_imu_;
    jst = latest_joint_;
  }
  if (!imu || !jst || jst->position.size() < static_cast<size_t>(kNumJoints) ||
      jst->velocity.size() < static_cast<size_t>(kNumJoints)) {
    return;
  }

  const int t = policy_timestep_ % motion_->num_frames;

  Eigen::VectorXf motion_command(48);
  motion_command.segment(0, 24) = motion_->dof_positions[t];
  motion_command.segment(24, 24) = motion_->dof_velocities[t];

  const Eigen::Quaternionf q_imu(static_cast<float>(imu->quaternion.w), static_cast<float>(imu->quaternion.x),
                                   static_cast<float>(imu->quaternion.y), static_cast<float>(imu->quaternion.z));
  const float waist_q =
      static_cast<float>(jst->position[static_cast<size_t>(waist_yaw_joint_state_index_)]);
  const Eigen::Quaternionf quat_for_anchor =
      AnchorQuatFromImuBaseWaist(q_imu, waist_q, anchor_quat_from_base_waist_, waist_yaw_axis_in_base_);

  const Eigen::Vector3f base_ang_vel(static_cast<float>(imu->angular_velocity.x),
                                     static_cast<float>(imu->angular_velocity.y),
                                     static_cast<float>(imu->angular_velocity.z));

  Eigen::VectorXf joint_pos(kNumJoints);
  Eigen::VectorXf joint_vel(kNumJoints);
  for (int i = 0; i < kNumJoints; ++i) {
    joint_pos[i] = static_cast<float>(jst->position[static_cast<size_t>(i)]);
    joint_vel[i] = static_cast<float>(jst->velocity[static_cast<size_t>(i)]);
  }

  const Eigen::Quaternionf motion_quat_current = motion_->root_quaternions[t].normalized();

  if (!init_to_world_set_) {
    const Eigen::Matrix3f yaw_robot = GetYawMatrixFromQuat(quat_for_anchor);
    const Eigen::Matrix3f yaw_motion = GetYawMatrixFromQuat(motion_->root_quaternions[0].normalized());
    const Eigen::Matrix3f init_to_world_mat = yaw_robot * yaw_motion.transpose();
    init_to_world_quat_ = RotationMatrixToQuat(init_to_world_mat);
    init_to_world_set_ = true;
  }

  const Eigen::Quaternionf motion_frame_alignment =
      (init_to_world_quat_ * motion_quat_current).normalized();
  const Eigen::Quaternionf anchor_quat =
      (quat_for_anchor.conjugate() * motion_frame_alignment).normalized();

  const Eigen::VectorXf motion_anchor_ori_b = MotionAnchorOriB6(anchor_quat);

  obs_.segment(0, 48) = motion_command;
  obs_.segment(48, 6) = motion_anchor_ori_b;
  obs_.segment(54, 3) = base_ang_vel * observation_scale_base_ang_vel_;

  for (int i = 0; i < kNumJoints; ++i) {
    const int xi = xml_to_policy_[static_cast<size_t>(i)];
    obs_(57 + i) =
        (joint_pos[xi] - default_joint_pos_[static_cast<size_t>(xi)]) * observation_scale_joint_pos_;
    obs_(81 + i) = joint_vel[xi] * observation_scale_joint_vel_;
  }
  for (int i = 0; i < kNumJoints; ++i) {
    obs_(105 + i) = act_(i);
  }

  const std::vector<int64_t> input_dims = {1, static_cast<int64_t>(obs_.size())};
  const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, obs_.data(), static_cast<size_t>(obs_.size()), input_dims.data(), input_dims.size());

  const auto output_tensors =
      session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1,
                    output_node_names_.data(), 1);

  const float* out_ptr = output_tensors.front().GetTensorData<float>();
  std::memcpy(act_.data(), out_ptr, static_cast<size_t>(num_actions_) * sizeof(float));

  Eigen::VectorXf target_dof_pos(kNumJoints);
  for (int i = 0; i < kNumJoints; ++i) {
    const int pi = policy_to_xml_[static_cast<size_t>(i)];
    target_dof_pos[i] = act_[pi] * action_scale_ + default_joint_pos_[static_cast<size_t>(i)];
  }

  if (publish_robot_joint_command_ && joint_cmd_pub_) {
    interface_protocol::msg::JointCommand msg;
    msg.position.resize(kNumJoints);
    for (int i = 0; i < kNumJoints; ++i) {
      msg.position[static_cast<size_t>(i)] = static_cast<double>(target_dof_pos[static_cast<Eigen::Index>(i)]);
    }
    msg.velocity.assign(kNumJoints, 0.0);
    msg.feed_forward_torque.assign(kNumJoints, 0.0);
    msg.torque.assign(kNumJoints, 0.0);
    msg.stiffness.assign(kNumJoints, 0.0);
    msg.damping.assign(kNumJoints, 0.0);
    for (int i = 0; i < kNumJoints; ++i) {
      msg.stiffness[static_cast<size_t>(i)] = static_cast<double>(joint_kp_[static_cast<size_t>(i)]);
      msg.damping[static_cast<size_t>(i)] = static_cast<double>(joint_kd_[static_cast<size_t>(i)]);
    }
    msg.parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    joint_cmd_pub_->publish(msg);
  }

  if (get_info_) {
    std::stringstream ss;
    ss << "motion_anchor_ori_b " << obs_.segment(48, 6).transpose() << " | base_ang_vel "
       << obs_.segment(54, 3).transpose();
    RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());
  }

  ++policy_timestep_;
}

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    const std::string config =
        (argc > 1) ? argv[1]
                   : "src/pm01_deploy/config/param/pm01_real_minic.yaml";
    auto node = std::make_shared<Pm01ControllerMujoco>(config);
    if (!node->Initialize()) {
      rclcpp::shutdown();
      return 1;
    }
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
