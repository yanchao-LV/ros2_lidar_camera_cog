cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(lidar_cabbage_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation features)

add_executable(cabbage_detection_node src/cabbage_detection_node.cpp)

ament_target_dependencies(cabbage_detection_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
  visualization_msgs
)

target_include_directories(cabbage_detection_node PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(cabbage_detection_node ${PCL_LIBRARIES})

install(TARGETS cabbage_detection_node DESTINATION lib/${PROJECT_NAME})

ament_package()

