#ifndef PM01_CONTROLLER_MUJOCO_H
#define PM01_CONTROLLER_MUJOCO_H

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include "motion_loader.hpp"

namespace mujoco {
class Simulate;
}  // namespace mujoco

/**
 * MuJoCo + ONNX 部署仿真（与 deploy_mujoco_minic.py 观测/PD/锚点 yaw 对齐逻辑一致）。
 * 策略需为 ONNX；Python 的 .pt 需先导出为同名 .onnx 后在 YAML 中配置。
 */
class Pm01ControllerMujoco {
 public:
  explicit Pm01ControllerMujoco(const std::string& config_file);
  /** 阻塞运行仿真直至 simulation_duration 或窗口关闭。 */
  void Run();

 private:
  void RunHeadless();
  void RunWithGui();
  void MainPhysicsLoop(mjModel* m, mjData* d, int anchor_body_id, mujoco::Simulate* sim_for_sync);
  void LoadOnnxSession(const std::string& onnx_path);
  static std::string ResolvePolicyOnnx(const YAML::Node& yaml_node);
  static std::string ResolvePath(const std::string& path);
  void ParseYaml(const YAML::Node& y);

  std::string config_file_;

  std::string robot_xml_path_;
  std::string motion_file_;
  /**
   * motion/观测锚点体（回退为直接 d->xquat 的 body，通常与 mocap torso 一致）。
   * 若 anchor_quat_from_base_waist_ 为 true，仿真侧不直接取该四元数，而由 base+腰关节合成。
   */
  std::string anchor_body_name_{"LINK_TORSO_YAW"};
  int motion_body_index_{3};

  /** 若 true：q_anchor = R_world_base * R_waist(J12)，与 IMU 在 baselink 再叠加腰 yaw 的约定一致。 */
  bool anchor_quat_from_base_waist_{true};
  std::string base_link_name_{"LINK_BASE"};
  std::string waist_yaw_joint_name_{"J12_WAIST_YAW"};

  double simulation_duration_{600.0};
  double simulation_dt_{0.001};
  int control_decimation_{20};

  /** 与官方 simulate 被动模式 + RenderLoop 对应 Python launch_passive；为 false 时无窗口。 */
  bool use_gui_{false};

  /**
   * use_gui 时：以 d->time 对齐墙钟(约 1:1)，与 Python 脚本带 viewer 时的 time.sleep 类似。
   * 无头模式下 MainPhysicsLoop 不睡眠，此项无效。默认 true。
   */
  bool realtime_sync_{true};

  bool get_info_{false};

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
};

#endif
