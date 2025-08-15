cmake_minimum_required(VERSION 3.8)
project(sensor_sync)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 手动指定livox_ros2_driver的CMake配置文件路径（关键！）
set(livox_ros2_driver_DIR "/home/abc/ws_livox/install/livox_ros2_driver/share/livox_ros2_driver/cmake")

# 查找依赖（按顺序）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(livox_ros2_driver REQUIRED)  # 必须在指定路径后查找
find_package(livox_interfaces REQUIRED)  # 新增：查找消息类型包

# 编译同步节点
add_executable(sync_node src/sync_node.cpp)
ament_target_dependencies(sync_node
  rclcpp
  sensor_msgs
  message_filters
  livox_ros2_driver  # 确保包含此依赖
)

# 编译时间戳转换节点
add_executable(lidar_time_converter src/lidar_time_converter.cpp)
# 同步节点
ament_target_dependencies(sync_node
  rclcpp sensor_msgs message_filters livox_ros2_driver livox_interfaces  # 增加 livox_interfaces
)

# 时间戳转换节点
ament_target_dependencies(lidar_time_converter
  rclcpp livox_ros2_driver livox_interfaces  # 增加 livox_interfaces
)

# 安装节点
install(TARGETS
  sync_node
  lidar_time_converter
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

