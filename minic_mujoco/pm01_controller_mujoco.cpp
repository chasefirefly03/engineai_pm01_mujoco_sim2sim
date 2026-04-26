#include "pm01_controller_mujoco.hpp"

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>

#include "simulate/glfw_adapter.h"
#include "simulate/simulate.h"
namespace mj = mujoco;

namespace {

constexpr int kSimulateSyncEveryNSteps = 16;


Eigen::Quaternionf MjQuatToEigen(const mjtNum* q) {
  return Eigen::Quaternionf(static_cast<float>(q[0]), static_cast<float>(q[1]),
                              static_cast<float>(q[2]), static_cast<float>(q[3]));
}

/**
 * 与 pm01_controller(ROS) 中 base * AngleAxis(腰) 一致：R_world_torsou ≈ R_base * R_joint(axis, q)
 * 关节轴用模型中的 jnt_axis，与 J12 等 MJCF 一致。
 */
Eigen::Quaternionf SimAnchorBodyQuatForObs(const mjModel* m, const mjData* d, bool from_base_waist,
                                            const std::string& base_name, const std::string& joint_name,
                                            int anchor_body_id_fallback) {
  if (!from_base_waist) {
    return MjQuatToEigen(d->xquat + 4 * anchor_body_id_fallback);
  }
  const int bb = mj_name2id(m, mjOBJ_BODY, base_name.c_str());
  const int jid = mj_name2id(m, mjOBJ_JOINT, joint_name.c_str());
  if (bb < 0) {
    throw std::runtime_error("找不到 base body: " + base_name);
  }
  if (jid < 0) {
    throw std::runtime_error("找不到腰 yaw 关节: " + joint_name);
  }
  if (m->jnt_type[jid] != mjJNT_HINGE) {
    throw std::runtime_error("腰关节 " + joint_name + " 需为 hinge（mjJNT_HINGE）。");
  }
  const mjtNum theta = d->qpos[m->jnt_qposadr[jid]];

  Eigen::Vector3f u(static_cast<float>(m->jnt_axis[3 * jid + 0]), static_cast<float>(m->jnt_axis[3 * jid + 1]),
                    static_cast<float>(m->jnt_axis[3 * jid + 2]));
  if (u.squaredNorm() < 1e-12f) {
    return MjQuatToEigen(d->xquat + 4 * anchor_body_id_fallback);
  }
  u.normalize();
  const Eigen::Quaternionf q_b = MjQuatToEigen(d->xquat + 4 * bb);
  return (q_b * Eigen::AngleAxisf(static_cast<float>(theta), u)).normalized();
}

/** 与 Python get_yaw_matrix_from_quat / pm01 GetYawMatrixFromQuat 一致。 */
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

/** 行主序 3x3 → wxyz，与 mujoco mju_mat2Quat 一致。 */
void MatRowMajor9ToQuatWxyz(const Eigen::Matrix3f& R, mjtNum* quat_wxyz) {
  mjtNum mat9[9];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      mat9[r * 3 + c] = static_cast<mjtNum>(R(r, c));
    }
  }
  mju_mat2Quat(quat_wxyz, mat9);
}

Eigen::Quaternionf RotationMatrixToQuatWxyz(const Eigen::Matrix3f& R) {
  mjtNum q4[4];
  MatRowMajor9ToQuatWxyz(R, q4);
  return Eigen::Quaternionf(static_cast<float>(q4[0]), static_cast<float>(q4[1]),
                            static_cast<float>(q4[2]), static_cast<float>(q4[3]))
      .normalized();
}

Eigen::VectorXf MotionAnchorOriB6(const Eigen::Quaternionf& anchor_quat) {
  const Eigen::Quaternionf n = anchor_quat.normalized();
  const Eigen::Matrix3f R = n.toRotationMatrix();
  Eigen::VectorXf out(6);
  out << R(0, 0), R(0, 1), R(1, 0), R(1, 1), R(2, 0), R(2, 1);
  return out;
}

void PdTorques(mjtNum* ctrl, int nu, mjData* d, const Eigen::VectorXf& target_xml,
               const std::vector<float>& kp, const std::vector<float>& kd) {
  if (nu != static_cast<int>(kp.size()) || nu != static_cast<int>(kd.size())) {
    throw std::runtime_error("PD gains size must match nu");
  }
  for (int i = 0; i < nu; ++i) {
    const float q = static_cast<float>(d->qpos[7 + i]);
    const float dq = static_cast<float>(d->qvel[6 + i]);
    ctrl[i] = kp[static_cast<size_t>(i)] * (target_xml[i] - q) +
              kd[static_cast<size_t>(i)] * (0.f - dq);
  }
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
      "C++ MuJoCo 需要 ONNX 策略：在 YAML 中设置 onnx_policy_file，或将 policy_file 指向 .onnx"
      "（.pt 请先用 torch 导出为 ONNX）。");
}

void Pm01ControllerMujoco::ParseYaml(const YAML::Node& y) {
  robot_xml_path_ = ResolvePath(y["robot_xml_path"].as<std::string>());
  if (y["motion_file_path"]) {
    motion_file_ = ResolvePath(y["motion_file_path"].as<std::string>());
  } else if (y["motion_file"]) {
    motion_file_ = ResolvePath(y["motion_file"].as<std::string>());
  } else {
    throw std::runtime_error("YAML 缺少 motion_file_path 或 motion_file");
  }

  if (y["anchor_body_name"]) {
    anchor_body_name_ = y["anchor_body_name"].as<std::string>();
  }
  if (y["motion_body_index"]) {
    motion_body_index_ = y["motion_body_index"].as<int>();
  }
  if (y["anchor_quat_from_base_waist"]) {
    anchor_quat_from_base_waist_ = y["anchor_quat_from_base_waist"].as<bool>();
  } else {
    anchor_quat_from_base_waist_ = true;
  }
  if (y["base_link_name"]) {
    base_link_name_ = y["base_link_name"].as<std::string>();
  }
  if (y["waist_yaw_joint_name"]) {
    waist_yaw_joint_name_ = y["waist_yaw_joint_name"].as<std::string>();
  }

  simulation_duration_ = y["simulation_duration"] ? y["simulation_duration"].as<double>() : 600.0;
  simulation_dt_ = y["simulation_dt"] ? y["simulation_dt"].as<double>() : 0.001;
  if (y["control_rate_hz"]) {
    const double rate = y["control_rate_hz"].as<double>();
    if (rate > 1e-6 && simulation_dt_ > 0.0) {
      const double steps_per_sec = 1.0 / simulation_dt_;
      control_decimation_ = static_cast<int>(std::lround(steps_per_sec / rate));
      if (control_decimation_ < 1) {
        control_decimation_ = 1;
      }
    }
  } else if (y["control_decimation"]) {
    control_decimation_ = y["control_decimation"].as<int>();
  } else {
    control_decimation_ = 20;
  }

  get_info_ = y["get_info"] && y["get_info"].as<bool>();

  if (y["use_gui"]) {
    use_gui_ = y["use_gui"].as<bool>();
  }
  if (y["realtime_sync"]) {
    realtime_sync_ = y["realtime_sync"].as<bool>();
  } else {
    realtime_sync_ = true;
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
    std::cout << "模型输入 " << i << ": " << input_node_names_[i] << "\n";
  }
  const size_t num_output_nodes = session_->GetOutputCount();
  output_node_names_str_.resize(num_output_nodes);
  output_node_names_.resize(num_output_nodes);
  for (size_t i = 0; i < num_output_nodes; ++i) {
    auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
    output_node_names_str_[i] = name_ptr.get();
    output_node_names_[i] = output_node_names_str_[i].c_str();
    std::cout << "模型输出 " << i << ": " << output_node_names_[i] << "\n";
  }
}

Pm01ControllerMujoco::Pm01ControllerMujoco(const std::string& config_file)
    : config_file_(config_file),
      env_(ORT_LOGGING_LEVEL_WARNING, "pm01_controller_mujoco"),
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
}

void Pm01ControllerMujoco::Run() {
  if (use_gui_) {
    RunWithGui();
  } else {
    RunHeadless();
  }
}

void Pm01ControllerMujoco::RunHeadless() {
  char load_err[1024] = {0};
  mjModel* m = mj_loadXML(robot_xml_path_.c_str(), nullptr, load_err, sizeof(load_err));
  if (!m) {
    throw std::runtime_error(std::string("mj_loadXML 失败: ") + load_err);
  }
  mjData* d = mj_makeData(m);
  m->opt.timestep = simulation_dt_;

  const int anchor_body_id = mj_name2id(m, mjOBJ_BODY, anchor_body_name_.c_str());
  if (anchor_body_id < 0) {
    mj_deleteData(d);
    mj_deleteModel(m);
    throw std::runtime_error("找不到 body: " + anchor_body_name_);
  }
  if (anchor_quat_from_base_waist_) {
    if (mj_name2id(m, mjOBJ_BODY, base_link_name_.c_str()) < 0) {
      mj_deleteData(d);
      mj_deleteModel(m);
      throw std::runtime_error("找不到 baselink body: " + base_link_name_);
    }
    if (mj_name2id(m, mjOBJ_JOINT, waist_yaw_joint_name_.c_str()) < 0) {
      mj_deleteData(d);
      mj_deleteModel(m);
      throw std::runtime_error("找不到腰偏航关节: " + waist_yaw_joint_name_);
    }
  }

  if (m->nu != 24 || static_cast<int>(joint_kp_.size()) != 24) {
    mj_deleteData(d);
    mj_deleteModel(m);
    throw std::runtime_error("期望 24 个 motor 与 24 组 PD 增益，实际 nu=" + std::to_string(m->nu));
  }

  std::cout << "MuJoCo 仿真开始(无头)，时长 " << simulation_duration_ << " s，dt=" << simulation_dt_
            << "，control_decimation=" << control_decimation_ << "\n";
  try {
    MainPhysicsLoop(m, d, anchor_body_id, nullptr);
  } catch (...) {
    mj_deleteData(d);
    mj_deleteModel(m);
    throw;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
  std::cout << "MuJoCo 仿真结束。\n";
}

void Pm01ControllerMujoco::RunWithGui() {
  mjvCamera cam{};
  mjv_defaultCamera(&cam);
  mjvOption opt{};
  mjv_defaultOption(&opt);
  mjvPerturb pert{};
  mjv_defaultPerturb(&pert);

  auto sim = std::make_unique<mj::Simulate>(std::make_unique<mj::GlfwAdapter>(), &cam, &opt, &pert, true);

  mjModel* m = nullptr;
  mjData* d = nullptr;

  std::thread phys_thread(
      [this, s = sim.get(), &m, &d]() {
        char ebuf[1024] = {0};
        m = mj_loadXML(robot_xml_path_.c_str(), nullptr, ebuf, sizeof(ebuf));
        if (m == nullptr) {
          std::cerr << "mj_loadXML 失败: " << ebuf << "\n";
          s->exitrequest.store(1);
          return;
        }
        d = mj_makeData(m);
        m->opt.timestep = simulation_dt_;
        if (m->nu != 24 || static_cast<int>(joint_kp_.size()) != 24) {
          std::cerr << "期望 nu=24，实际 nu=" << m->nu << "\n";
          s->exitrequest.store(1);
          mj_deleteData(d);
          d = nullptr;
          mj_deleteModel(m);
          m = nullptr;
          return;
        }
        const int anchor = mj_name2id(m, mjOBJ_BODY, anchor_body_name_.c_str());
        if (anchor < 0) {
          std::cerr << "找不到 body: " << anchor_body_name_ << "\n";
          s->exitrequest.store(1);
          mj_deleteData(d);
          d = nullptr;
          mj_deleteModel(m);
          m = nullptr;
          return;
        }
        if (anchor_quat_from_base_waist_) {
          if (mj_name2id(m, mjOBJ_BODY, base_link_name_.c_str()) < 0) {
            std::cerr << "找不到 baselink: " << base_link_name_ << "\n";
            s->exitrequest.store(1);
            mj_deleteData(d);
            d = nullptr;
            mj_deleteModel(m);
            m = nullptr;
            return;
          }
          if (mj_name2id(m, mjOBJ_JOINT, waist_yaw_joint_name_.c_str()) < 0) {
            std::cerr << "找不到腰偏航关节: " << waist_yaw_joint_name_ << "\n";
            s->exitrequest.store(1);
            mj_deleteData(d);
            d = nullptr;
            mj_deleteModel(m);
            m = nullptr;
            return;
          }
        }
        const std::string show_name = robot_xml_path_.empty() ? "model" : robot_xml_path_;
        s->Load(m, d, show_name.c_str());

        if (simulation_duration_ <= 0.0) {
          std::cout << "MuJoCo 仿真，GUI=on，墙钟不限制(仅关窗口或 UI Quit)，dt=" << simulation_dt_
                    << "，control_decimation=" << control_decimation_ << "\n";
        } else {
          std::cout << "MuJoCo 仿真，GUI=on，最长 " << simulation_duration_ << " s，dt=" << simulation_dt_
                    << "，control_decimation=" << control_decimation_ << "\n";
        }
        try {
          MainPhysicsLoop(m, d, anchor, s);
        } catch (const std::exception& e) {
          std::cerr << "物理/策略线程: " << e.what() << "\n";
        }
        s->exitrequest.store(1);
      });

  sim->RenderLoop();
  sim->exitrequest.store(1);
  if (phys_thread.joinable()) {
    phys_thread.join();
  }
  if (d != nullptr) {
    mj_deleteData(d);
  }
  if (m != nullptr) {
    mj_deleteModel(m);
  }
  std::cout << "MuJoCo 仿真结束。\n";
}

void Pm01ControllerMujoco::MainPhysicsLoop(mjModel* m, mjData* d, int anchor_body_id,
                                            mujoco::Simulate* sim_for_sync) {
  Eigen::VectorXf target_dof_pos(24);
  for (int i = 0; i < 24; ++i) {
    target_dof_pos[i] = default_joint_pos_[static_cast<size_t>(i)];
  }

  int counter = 0;
  int policy_timestep = 0;
  bool init_to_world_set = false;
  Eigen::Quaternionf init_to_world_quat(1.f, 0.f, 0.f, 0.f);

  const auto wall_start = std::chrono::steady_clock::now();
  const auto wall_limit = std::chrono::duration<double>(simulation_duration_);
  const bool with_gui = (sim_for_sync != nullptr);
  const bool gui_unlimited = with_gui && (simulation_duration_ <= 0.0);

  for (;;) {
    if (with_gui) {
      if (sim_for_sync->exitrequest.load() != 0) {
        break;
      }
      if (!gui_unlimited) {
        if (std::chrono::steady_clock::now() - wall_start >= wall_limit) {
          sim_for_sync->exitrequest.store(1);
          break;
        }
      }
    } else {
      if (std::chrono::steady_clock::now() - wall_start >= wall_limit) {
        break;
      }
    }

    PdTorques(d->ctrl, m->nu, d, target_dof_pos, joint_kp_, joint_kd_);
    mj_step(m, d);
    ++counter;

    if (sim_for_sync != nullptr) {
      if (counter == 1 || (counter % kSimulateSyncEveryNSteps) == 0) {
        sim_for_sync->Sync();
      }
    }

    if (counter % control_decimation_ == 0) {
      const int t = policy_timestep % motion_->num_frames;

      Eigen::VectorXf motion_command(48);
      motion_command.segment(0, 24) = motion_->dof_positions[t];
      motion_command.segment(24, 24) = motion_->dof_velocities[t];

      const Eigen::Quaternionf quat_sim = SimAnchorBodyQuatForObs(
          m, d, anchor_quat_from_base_waist_, base_link_name_, waist_yaw_joint_name_, anchor_body_id);
      const Eigen::Quaternionf motion_quat_current = motion_->root_quaternions[t].normalized();

      if (!init_to_world_set) {
        const Eigen::Matrix3f yaw_robot = GetYawMatrixFromQuat(quat_sim);
        const Eigen::Matrix3f yaw_motion = GetYawMatrixFromQuat(motion_->root_quaternions[0].normalized());
        const Eigen::Matrix3f init_to_world_mat = yaw_robot * yaw_motion.transpose();
        init_to_world_quat = RotationMatrixToQuatWxyz(init_to_world_mat);
        init_to_world_set = true;
      }

      const Eigen::Quaternionf motion_frame_alignment =
          (init_to_world_quat * motion_quat_current).normalized();
      const Eigen::Quaternionf anchor_quat =
          (quat_sim.conjugate() * motion_frame_alignment).normalized();

      const Eigen::VectorXf motion_anchor_ori_b = MotionAnchorOriB6(anchor_quat);

      const Eigen::Vector3f base_ang_vel(static_cast<float>(d->qvel[3]), static_cast<float>(d->qvel[4]),
                                       static_cast<float>(d->qvel[5]));

      Eigen::VectorXf joint_pos(24);
      Eigen::VectorXf joint_vel(24);
      for (int i = 0; i < 24; ++i) {
        joint_pos[i] = static_cast<float>(d->qpos[7 + i]);
        joint_vel[i] = static_cast<float>(d->qvel[6 + i]);
      }

      obs_.segment(0, 48) = motion_command;
      obs_.segment(48, 6) = motion_anchor_ori_b;
      obs_.segment(54, 3) = base_ang_vel * observation_scale_base_ang_vel_;

      for (int i = 0; i < 24; ++i) {
        const int xi = xml_to_policy_[static_cast<size_t>(i)];
        obs_(57 + i) =
            (joint_pos[xi] - default_joint_pos_[static_cast<size_t>(xi)]) * observation_scale_joint_pos_;
        obs_(81 + i) = joint_vel[xi] * observation_scale_joint_vel_;
      }
      // 与 deploy_mujoco_minic.py: obs[105:129] = action[xml_to_policy] 一致。此时 action 已是
      // a_x[j]=a_p[policy_to_xml[j]]，有 a_x[xml_to_policy[i]] = a_p[i]；ort 输出 act_ 即 a_p
      for (int i = 0; i < 24; ++i) {
        obs_(105 + i) = act_(i);
      }

      const std::vector<int64_t> input_dims = {1, static_cast<int64_t>(obs_.size())};
      const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, obs_.data(), static_cast<size_t>(obs_.size()), input_dims.data(),
          input_dims.size());

      const auto output_tensors =
          session_->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1,
                        output_node_names_.data(), 1);

      const float* out_ptr = output_tensors.front().GetTensorData<float>();
      std::memcpy(act_.data(), out_ptr, static_cast<size_t>(num_actions_) * sizeof(float));

      for (int i = 0; i < 24; ++i) {
        const int pi = policy_to_xml_[static_cast<size_t>(i)];
        target_dof_pos[i] =
            act_[pi] * action_scale_ + default_joint_pos_[static_cast<size_t>(i)];
      }

      if (get_info_) {
        std::cout << "--------------------------------\n";
        std::cout << "motion_anchor_ori_b\n"
                  << obs_.segment(48, 6).transpose() << "\n";
        std::cout << "base_ang_vel\n"
                  << obs_.segment(54, 3).transpose() << "\n";
        std::cout << "joint_pos_rel\n"
                  << obs_.segment(57, 24).transpose() << "\n";
      }

      ++policy_timestep;
    }

    // 每步末尾对齐墙钟：在 Sync/策略之后 sleep，使本步的 ORT 等耗时也计入（约 1:1 实时）
    if (with_gui && realtime_sync_) {
      const auto t_target = wall_start + std::chrono::duration<double>(static_cast<double>(d->time));
      if (t_target > std::chrono::steady_clock::now()) {
        std::this_thread::sleep_until(t_target);
      }
    }
  }
}

int main(int argc, char** argv) {
  try {
    const std::string config =
        (argc > 1) ? argv[1]
                   : "src/pm01_deploy/config/param/pm01_mujoco_minic_cpp.yaml";
    Pm01ControllerMujoco ctrl(config);
    ctrl.Run();
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
