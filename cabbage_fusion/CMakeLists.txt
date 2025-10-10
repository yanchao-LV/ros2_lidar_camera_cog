cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(cabbage_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

# 依赖（完全移除tf2_geometry_msgs）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)  # 仅用于点云转换，已安装
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters search kdtree segmentation)
find_package(vision_msgs REQUIRED)

add_executable(fusion_node 
  src/fusion_node.cpp
)

target_include_directories(fusion_node PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
)

# 链接依赖（移除tf2_geometry_msgs）
ament_target_dependencies(fusion_node
  rclcpp
  std_msgs
  sensor_msgs
  visualization_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  vision_msgs
)

target_link_libraries(fusion_node
  ${PCL_LIBRARIES}
)

install(TARGETS
  fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()

