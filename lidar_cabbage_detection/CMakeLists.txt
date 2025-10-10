cmake_minimum_required(VERSION 3.5)
project(lidar_cabbage_detection)

# 消除PCL_ROOT变量警告（可选但推荐）
cmake_policy(SET CMP0074 NEW)

# 编译选项：开启警告、C++17标准
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# 1. 查找ROS2依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)          # 点云消息
find_package(visualization_msgs REQUIRED)   # 可视化Marker消息
find_package(pcl_conversions REQUIRED)      # ROS2-PCL格式转换
find_package(PCL REQUIRED COMPONENTS        # PCL核心功能
  common 
  segmentation 
  filters 
  features 
  search
)

# 2. 包含头文件目录
include_directories(
  include
  ${PCL_INCLUDE_DIRS}  # PCL头文件
)

# 3. PCL库配置
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 4. 编译节点
add_executable(cabbage_detection_node 
  src/cabbage_detection_node.cpp
)

# 5. 链接依赖
ament_target_dependencies(cabbage_detection_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)
target_link_libraries(cabbage_detection_node 
  ${PCL_LIBRARIES}  # 链接PCL库
)

# 6. 安装节点（ROS2可执行文件路径）
install(TARGETS
  cabbage_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

# 7. 生成ROS2包配置
ament_package()

