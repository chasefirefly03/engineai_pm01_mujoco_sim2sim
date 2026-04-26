#ifndef MOTION_LOADER_HPP
#define MOTION_LOADER_HPP

#include "cnpy.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace motion_loader_detail {

inline float read_fps_from_npz(const cnpy::npz_t& npz, float fallback_fps)
{
    auto it = npz.find("fps");
    if (it == npz.end())
    {
        return fallback_fps;
    }
    const cnpy::NpyArray& arr = it->second;
    if (arr.num_vals < 1)
    {
        return fallback_fps;
    }
    if (arr.word_size == sizeof(float))
    {
        const float* p = arr.data<float>();
        if (!p)
        {
            return fallback_fps;
        }
        float v = p[0];
        return (v > 1e-6f) ? v : fallback_fps;
    }
    if (arr.word_size == sizeof(int64_t))
    {
        const int64_t* p = arr.data<int64_t>();
        if (!p)
        {
            return fallback_fps;
        }
        float v = static_cast<float>(p[0]);
        return (v > 1e-6f) ? v : fallback_fps;
    }
    return fallback_fps;
}

inline Eigen::Quaternionf row_quat_wxyz(const float* q)
{
    return Eigen::Quaternionf(q[0], q[1], q[2], q[3]).normalized();
}

}  // namespace motion_loader_detail

/**
 * 从 NPZ 加载 Mimic 参考轨迹（字段与 deploy_mujoco_minic / Beyond 导出一致）。
 * 需要: joint_pos, joint_vel, body_quat_w（可为 [T,4] 或 [T,num_bodies,4]）。
 * 可选: body_pos_w（[T,3] 或 [T,N,3]）、fps（标量，覆盖构造函数中的 fps）。
 */
class MotionLoader_
{
public:
    float dt{0.02f};
    int num_frames{0};

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    float fps;

    MotionLoader_(const std::string& motion_file, float fallback_fps, int motion_body_index);
};

inline MotionLoader_::MotionLoader_(const std::string& motion_file, float fallback_fps, int motion_body_index)
{
    cnpy::npz_t npz = cnpy::npz_load(motion_file);

    auto it_pos = npz.find("joint_pos");
    auto it_vel = npz.find("joint_vel");
    auto it_quat = npz.find("body_quat_w");
    if (it_pos == npz.end() || it_vel == npz.end() || it_quat == npz.end())
    {
        throw std::runtime_error(
            "NPZ must contain keys: joint_pos, joint_vel, body_quat_w — file: " + motion_file);
    }

    const cnpy::NpyArray& ap = it_pos->second;
    const cnpy::NpyArray& av = it_vel->second;
    const cnpy::NpyArray& aq = it_quat->second;

    if (ap.word_size != sizeof(float) || av.word_size != sizeof(float) || aq.word_size != sizeof(float))
    {
        throw std::runtime_error("NPZ arrays must be float32");
    }

    const float* jp = ap.data<float>();
    const float* jv = av.data<float>();
    const float* bq = aq.data<float>();

    if (ap.shape.size() != 2 || av.shape.size() != 2)
    {
        throw std::runtime_error("joint_pos / joint_vel must be 2D [num_frames, num_dof]");
    }

    const size_t T = ap.shape[0];
    const size_t npos = ap.shape[1];
    const size_t nvel = av.shape[0];
    const size_t nd_vel = av.shape[1];

    if (T != nvel || npos != nd_vel)
    {
        throw std::runtime_error("joint_pos and joint_vel shape mismatch");
    }
    if (npos != 24 || nd_vel != 24)
    {
        throw std::runtime_error(
            "joint_pos/joint_vel must have 24 columns for pm01 mimc deploy (got " +
            std::to_string(npos) + ")");
    }

    fps = motion_loader_detail::read_fps_from_npz(npz, fallback_fps);
    if (fps <= 1e-6f)
    {
        throw std::runtime_error("motion fps must be positive");
    }
    dt = 1.0f / fps;

    num_frames = static_cast<int>(T);

    Eigen::VectorXf pos_row(static_cast<int>(npos));
    Eigen::VectorXf vel_row(static_cast<int>(nd_vel));

    root_positions.reserve(num_frames);
    root_quaternions.reserve(num_frames);
    dof_positions.reserve(num_frames);
    dof_velocities.reserve(num_frames);

    int b_idx = motion_body_index;
    if (b_idx < 0)
    {
        b_idx = 0;
    }

    for (int t = 0; t < num_frames; ++t)
    {
        for (size_t j = 0; j < npos; ++j)
        {
            pos_row(static_cast<int>(j)) = jp[t * npos + j];
        }
        for (size_t j = 0; j < nd_vel; ++j)
        {
            vel_row(static_cast<int>(j)) = jv[t * nd_vel + j];
        }
        dof_positions.push_back(pos_row);
        dof_velocities.push_back(vel_row);

        if (aq.shape.size() == 2)
        {
            if (aq.shape[0] != T || aq.shape[1] != 4)
            {
                throw std::runtime_error("body_quat_w with 2 dims must be [num_frames, 4]");
            }
            const float* qq = bq + static_cast<size_t>(t) * 4;
            root_quaternions.push_back(motion_loader_detail::row_quat_wxyz(qq));
            Eigen::VectorXf rp(3);
            rp.setZero();
            root_positions.push_back(rp);
        }
        else if (aq.shape.size() == 3)
        {
            const size_t nb = aq.shape[1];
            const size_t nc = aq.shape[2];
            if (aq.shape[0] != T || nc != 4)
            {
                throw std::runtime_error("body_quat_w with 3 dims must be [T, num_bodies, 4]");
            }
            if (static_cast<size_t>(b_idx) >= nb)
            {
                throw std::runtime_error("motion_body_index out of range for body_quat_w");
            }
            const size_t off = (static_cast<size_t>(t) * nb + static_cast<size_t>(b_idx)) * 4;
            root_quaternions.push_back(motion_loader_detail::row_quat_wxyz(bq + off));

            Eigen::VectorXf rp(3);
            rp.setZero();
            auto it_bp = npz.find("body_pos_w");
            if (it_bp != npz.end())
            {
                const cnpy::NpyArray& bp = it_bp->second;
                if (bp.word_size == sizeof(float) && bp.shape.size() == 3 && bp.shape[0] == T &&
                    bp.shape[2] == 3 && static_cast<size_t>(b_idx) < bp.shape[1])
                {
                    const float* pp = bp.data<float>();
                    const size_t poff =
                        (static_cast<size_t>(t) * bp.shape[1] + static_cast<size_t>(b_idx)) * 3;
                    rp << pp[poff], pp[poff + 1], pp[poff + 2];
                }
                else if (bp.word_size == sizeof(float) && bp.shape.size() == 2 && bp.shape[0] == T &&
                         bp.shape[1] == 3)
                {
                    const float* pp = bp.data<float>();
                    const size_t poff = static_cast<size_t>(t) * 3;
                    rp << pp[poff], pp[poff + 1], pp[poff + 2];
                }
            }
            root_positions.push_back(rp);
        }
        else
        {
            throw std::runtime_error("body_quat_w must be [T,4] or [T,num_bodies,4]");
        }
    }
}

#endif
