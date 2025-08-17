cmake_minimum_required(VERSION 3.8)
project(sensor_sync)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 手动指定livox_ros2_driver的CMake路径（根据实际安装路径修改）
set(livox_ros2_driver_DIR "/home/abc/ws_livox/install/livox_ros2_driver/share/livox_ros2_driver/cmake")

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(livox_ros2_driver REQUIRED)
find_package(livox_interfaces REQUIRED)

# 编译时间戳转换节点
add_executable(lidar_time_converter src/lidar_time_converter.cpp)
# 时间戳转换节点（修改后）
ament_target_dependencies(lidar_time_converter
  rclcpp 
  sensor_msgs  # 新增：添加sensor_msgs依赖，用于解析PointCloud2
  livox_ros2_driver 
  livox_interfaces
)


# 编译同步节点
add_executable(sync_node src/sync_node.cpp)
ament_target_dependencies(sync_node
  rclcpp
  sensor_msgs
  message_filters
  livox_interfaces  # 依赖雷达消息类型
)

# 安装节点到系统路径
install(TARGETS
  lidar_time_converter
  sync_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

