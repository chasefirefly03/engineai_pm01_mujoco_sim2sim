cmake_minimum_required(VERSION 3.16)
project(minic_mujoco)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interface_protocol REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED NO_MODULE)
find_package(Eigen3 REQUIRED)
# simulate/ 子工程使用 EIGEN3_INCLUDE_DIRS（与 mujoco_simulator 包一致）
if(NOT EIGEN3_INCLUDE_DIRS)
  if(DEFINED EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIRS "${EIGEN3_INCLUDE_DIR}")
  else()
    get_target_property(_e3_inc Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    set(EIGEN3_INCLUDE_DIRS "${_e3_inc}")
  endif()
endif()
find_package(ZLIB REQUIRED)

# MuJoCo + 仓库内 simulate 源码（同 mujoco_simulator）
set(ENGINEAI_ROBOTICS_THIRD_PARTY "/opt/engineai_robotics_third_party"
  CACHE PATH "Third-party MuJoCo / dependencies root")
if(NOT IS_DIRECTORY ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include/mujoco)
  message(FATAL_ERROR "MuJoCo headers not found under ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include. "
    "Set ENGINEAI_ROBOTICS_THIRD_PARTY to a prefix containing include/mujoco/mujoco.h .")
endif()
set(SIMULATION_MUJOCO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mujoco_simulation/mujoco")
if(NOT IS_DIRECTORY ${SIMULATION_MUJOCO_DIR}/simulate)
  message(FATAL_ERROR "simulate sources not at ${SIMULATION_MUJOCO_DIR}/simulate (clone mujoco_simulation 或保持目录相对位置)。")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ENGINEAI_ROBOTICS_THIRD_PARTY})
# 直接定位 MuJoCo 的 config（避免在复杂 CMAKE_PREFIX_PATH 下未被 discover）
set(mujoco_DIR "${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib/cmake/mujoco")
list(APPEND CMAKE_MODULE_PATH ${SIMULATION_MUJOCO_DIR}/simulate/cmake)
find_package(mujoco REQUIRED)
find_package(glfw3 REQUIRED)
# 构建与 mujoco_simulator 相同的 static lib（GlFW/viewer 等），供本包链接
add_subdirectory(${SIMULATION_MUJOCO_DIR}/simulate _minic_mujoco_simulate)
set(MUJOCO_LIB "${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib/libmujoco.so")

set(ONNXRUNTIME_ROOT_DIR "/usr/include/onnxruntime-linux-x64-1.24.2" CACHE PATH "ONNX Runtime root")
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT_DIR}/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT_DIR or install headers/libs.")
endif()

set(CNPY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pm01_deploy/third_party/cnpy")

add_executable(pm01_mujoco_sim
  pm01_controller_mujoco.cpp
  ${CNPY_DIR}/cnpy.cpp
)

ament_target_dependencies(pm01_mujoco_sim
  rclcpp
  interface_protocol
)

target_include_directories(pm01_mujoco_sim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${SIMULATION_MUJOCO_DIR}
  ${SIMULATION_MUJOCO_DIR}/simulate
  ${ENGINEAI_ROBOTICS_THIRD_PARTY}/include
  ${CNPY_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# 与 mujoco_simulator 一致：子目录里 target_link(… mujoco) 会生成 -lmujoco，需能搜到 third_party 下的 libmujoco.so
target_link_directories(pm01_mujoco_sim PRIVATE ${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib)

target_link_libraries(pm01_mujoco_sim
  libsimulate
  ${MUJOCO_LIB}
  glfw
  yaml-cpp
  Eigen3::Eigen
  ZLIB::ZLIB
  ${ONNXRUNTIME_LIBRARY}
  ${CMAKE_DL_LIBS}
  pthread
)

# 安装后 ros2 run 不依赖 LD_LIBRARY_PATH：对 libmujoco 等非系统路径的 .so 写入 RPATH
set(_minic_mujoco_third_libdir "${ENGINEAI_ROBOTICS_THIRD_PARTY}/lib")
set_target_properties(pm01_mujoco_sim PROPERTIES
  BUILD_RPATH "${_minic_mujoco_third_libdir}"
  INSTALL_RPATH "${_minic_mujoco_third_libdir}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

set_property(TARGET pm01_mujoco_sim PROPERTY CXX_STANDARD 20)

install(TARGETS pm01_mujoco_sim DESTINATION lib/${PROJECT_NAME})

ament_package()
