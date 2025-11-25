cmake_minimum_required(VERSION 3.8)
project(lidar_cabbage_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------------------- 原有依赖（保留不动）----------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

# ---------------------- 原有检测节点编译（保留不动）----------------------
add_executable(seedling_detection_node src/cabbage_detection_node.cpp)
ament_target_dependencies(seedling_detection_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)
target_link_libraries(seedling_detection_node ${PCL_LIBRARIES})

# ---------------------- 新增：测试节点编译（只加这部分）----------------------
add_executable(test_node src/test_node.cpp)  # 编译test_node
ament_target_dependencies(test_node          # 测试节点只需要基础依赖
  rclcpp
  sensor_msgs
)

# ---------------------- 安装两个节点（保留不动，新增test_node）----------------------
install(TARGETS
  seedling_detection_node  # 原有检测节点
  test_node                # 新增测试节点
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

