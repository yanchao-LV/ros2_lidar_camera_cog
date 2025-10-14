cmake_minimum_required(VERSION 3.5)
project(cabbage_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

# 先查找 ROS 2 依赖（关键：确保 vision_msgs 被正确找到）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)       # 显式依赖 vision_msgs
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)

# PCL 1.12.1 路径配置
set(PCL_INCLUDE_DIR "/usr/local/pcl-1.12/include/pcl-1.12")
set(PCL_LIB_DIR "/usr/local/pcl-1.12/lib")
if(NOT EXISTS ${PCL_INCLUDE_DIR})
  message(FATAL_ERROR "PCL 1.12.1 头文件路径不存在: ${PCL_INCLUDE_DIR}")
endif()
if(NOT EXISTS ${PCL_LIB_DIR})
  message(FATAL_ERROR "PCL 1.12.1 库文件路径不存在: ${PCL_LIB_DIR}")
endif()

# 头文件和库路径
include_directories(
  include
  ${PCL_INCLUDE_DIR}
)
link_directories(${PCL_LIB_DIR})

# 编译节点
add_executable(fusion_node src/fusion_node.cpp)
ament_target_dependencies(fusion_node
  rclcpp
  sensor_msgs
  vision_msgs          # 链接 vision_msgs
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  visualization_msgs
)
target_link_libraries(fusion_node
  ${PCL_LIB_DIR}/libpcl_common.so
  ${PCL_LIB_DIR}/libpcl_filters.so
  ${PCL_LIB_DIR}/libpcl_search.so
  ${PCL_LIB_DIR}/libpcl_kdtree.so
  ${PCL_LIB_DIR}/libpcl_segmentation.so
)

# 安装节点
install(TARGETS
  fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

