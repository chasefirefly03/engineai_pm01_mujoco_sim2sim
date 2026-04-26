cmake_minimum_required(VERSION 3.8)
project(pm01_deploy)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(interface_protocol REQUIRED)
find_package(yaml-cpp 0.6 REQUIRED NO_MODULE)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
# find_package(Torch REQUIRED)
# ============修改为你的电脑或机器人上的onnx库的路径
set(ONNXRUNTIME_ROOT_DIR "/usr/include/onnxruntime-linux-x64-1.24.2")
# ============修改为你的电脑或机器人上的onnx库的路径
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h HINTS ${ONNXRUNTIME_ROOT_DIR}/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)

if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found!")
endif()

include_directories(
  include
  third_party/cnpy
  ${EIGEN3_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
  src
)

add_executable(pm01_controller_rl_walking
  src/pm01_controller_rl_walking.cpp
  src/components/message_handler.cc
  third_party/cnpy/cnpy.cpp
)
ament_target_dependencies(pm01_controller_rl_walking
  rclcpp
  interface_protocol
)
target_link_libraries(pm01_controller_rl_walking
  # 修复2：yaml-cpp链接目标（去掉命名空间别名）
  yaml-cpp
  Eigen3::Eigen
  ZLIB::ZLIB
  ${ONNXRUNTIME_LIBRARY}
)
set_property(TARGET pm01_controller_rl_walking PROPERTY CXX_STANDARD 14)

add_executable(pm01_minic_deploy
  src/pm01_minic_deploy.cpp
  third_party/cnpy/cnpy.cpp
)
ament_target_dependencies(pm01_minic_deploy
  rclcpp
  interface_protocol
)
target_link_libraries(pm01_minic_deploy
  yaml-cpp
  Eigen3::Eigen
  ZLIB::ZLIB
  ${ONNXRUNTIME_LIBRARY}
  ${CMAKE_DL_LIBS}
  pthread
)
set_property(TARGET pm01_minic_deploy PROPERTY CXX_STANDARD 17)

install(TARGETS
  pm01_controller_rl_walking
  pm01_minic_deploy
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()