#ifndef PM01_CONTROLLER_MUJOCO_H
#define PM01_CONTROLLER_MUJOCO_H

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include "interface_protocol/msg/imu_info.hpp"
#include "interface_protocol/msg/joint_command.hpp"
#include "interface_protocol/msg/joint_state.hpp"
#include "motion_loader.hpp"

/**
 * Minic 动作模仿策略的纯推理节点（无 MuJoCo）：订阅 IMU + joint_state，按固定频率
 * 组观测、ONNX 推理、发布 JointCommand。观测与原先 deploy_mujoco_minic / 仿真版一致。
 */
class Pm01ControllerMujoco : public rclcpp::Node {
 public:
  explicit Pm01ControllerMujoco(const std::string& config_file);
  /** 阻塞等待首帧传感器后启动控制定时器。 */
  bool Initialize();

 private:
  void ImuCallback(const interface_protocol::msg::ImuInfo::SharedPtr msg);
  void JointStateCallback(const interface_protocol::msg::JointState::SharedPtr msg);
  void ControlTimerCallback();

  void LoadOnnxSession(const std::string& onnx_path);
  static std::string ResolvePolicyOnnx(const YAML::Node& yaml_node);
  static std::string ResolvePath(const std::string& path);
  void ParseYaml(const YAML::Node& y);

  std::string config_file_;

  std::string motion_file_;
  int motion_body_index_{3};

  bool anchor_quat_from_base_waist_{true};
  /** joint_state.position 中与腰 yaw 对应下标（与 MJCF 7+idx 顺序一致）。 */
  int waist_yaw_joint_state_index_{12};
  /** 腰 yaw 旋转轴在 baselink 坐标系下的单位向量（与 MJCF jnt_axis 一致）。 */
  Eigen::Vector3f waist_yaw_axis_in_base_{0.f, 0.f, 1.f};

  double control_period_sec_{0.02};

  bool get_info_{false};

  std::string imu_topic_{"/hardware/imu_info"};
  std::string joint_state_topic_{"/hardware/joint_state"};
  bool publish_robot_joint_command_{true};
  std::string joint_command_topic_{"/hardware/joint_command"};

  rclcpp::Subscription<interface_protocol::msg::ImuInfo>::SharedPtr imu_sub_;
  rclcpp::Subscription<interface_protocol::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<interface_protocol::msg::JointCommand>::SharedPtr joint_cmd_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::mutex sensor_mutex_;
  interface_protocol::msg::ImuInfo::SharedPtr latest_imu_;
  interface_protocol::msg::JointState::SharedPtr latest_joint_;

  std::vector<float> default_joint_pos_;
  std::vector<float> joint_kp_;
  std::vector<float> joint_kd_;

  float observation_scale_base_ang_vel_{1.f};
  float observation_scale_joint_pos_{1.f};
  float observation_scale_joint_vel_{1.f};
  float action_scale_{1.f};

  int num_observations_{129};
  int num_actions_{24};

  Eigen::VectorXf obs_;
  Eigen::VectorXf act_;

  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;

  std::vector<int> xml_to_policy_;
  std::vector<int> policy_to_xml_;

  std::shared_ptr<MotionLoader_> motion_;
  std::vector<const char*> input_node_names_;
  std::vector<const char*> output_node_names_;
  std::vector<std::string> input_node_names_str_;
  std::vector<std::string> output_node_names_str_;

  int policy_timestep_{0};
  bool init_to_world_set_{false};
  Eigen::Quaternionf init_to_world_quat_{1.f, 0.f, 0.f, 0.f};
};

#endif
