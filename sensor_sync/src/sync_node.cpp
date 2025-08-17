#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

using namespace message_filters;

// 同步回调函数：处理匹配的图像和点云
void sync_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg
) {
  // 计算时间戳（秒）
  double img_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec / 1e9;
  double cloud_time = cloud_msg->header.stamp.sec + cloud_msg->header.stamp.nanosec / 1e9;
  double time_diff = std::abs(img_time - cloud_time) * 1000;  // 转换为毫秒

  // 打印同步信息
  RCLCPP_INFO(rclcpp::get_logger("sensor_sync_node"), 
    "数据同步成功！\n"
    "  图像时间戳: %.6f 秒\n"
    "  点云时间戳: %.6f 秒\n"
    "  时间差: %.2f 毫秒",
    img_time, cloud_time, time_diff
  );
}

int main(int argc, char**argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("sensor_sync_node");

  RCLCPP_INFO(node->get_logger(), "传感器同步节点启动中...");

  // 配置QoS（队列大小10，传感器数据类型）
  rclcpp::QoS qos(10);  // 队列大小10
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);  // 可靠传输
  qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);    // 易变数据（不缓存）
  auto rmw_qos = qos.get_rmw_qos_profile();  // 转换为rmw类型的QoS（关键！）

  // 订阅相机图像话题（使用rmw类型QoS）
  Subscriber<sensor_msgs::msg::Image> img_sub(
    node, 
    "/image_raw", 
    rmw_qos  // 第三个参数必须是rmw_qos_profile_t类型
  );

  // 订阅转换后的点云话题（同样使用rmw类型QoS）
  Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub(
    node, 
    "/livox/lidar_converted", 
    rmw_qos  // 第三个参数必须是rmw_qos_profile_t类型
  );

  // 定义同步策略：近似时间同步
  typedef sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, 
    sensor_msgs::msg::PointCloud2
  > SyncPolicy;

  // 初始化同步器（队列大小10）
  Synchronizer<SyncPolicy> sync(SyncPolicy(10), img_sub, cloud_sub);
  sync.registerCallback(&sync_callback);

  RCLCPP_INFO(node->get_logger(), "同步节点已就绪，等待图像和点云数据...");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

