cmake_minimum_required(VERSION 3.5)
project(lidar_cabbage_detection)

# 编译选项
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ==============================
# 1. 查找依赖包
# ==============================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# 查找 PCL 核心模块（包含 filters、common、segmentation、search）
find_package(PCL REQUIRED COMPONENTS common filters segmentation search)

# ==============================
# 2. 包含头文件目录
# ==============================
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# ==============================
# 3. 编译节点
# ==============================
add_executable(seedling_detection_node
  src/cabbage_detection_node.cpp
)

# ==============================
# 4. 链接依赖库
# ==============================
ament_target_dependencies(seedling_detection_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  tf2
  tf2_ros
)

target_link_libraries(seedling_detection_node
  ${PCL_LIBRARIES}
)

# ==============================
# 5. 安装目标文件
# ==============================
install(TARGETS
  seedling_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

