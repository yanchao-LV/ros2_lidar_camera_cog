#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"

using namespace message_filters;

// 全局发布器声明（在回调函数中使用）
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr sync_img_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr sync_cloud_pub;

// 同步回调函数：处理匹配的图像和点云并发布
void sync_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg
) {
  // 检查消息有效性
  if (!img_msg || !cloud_msg) {
    RCLCPP_WARN(rclcpp::get_logger("sensor_sync_node"), "收到空消息，跳过发布");
    return;
  }

  // 计算时间戳差（毫秒）
  double img_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec / 1e9;
  double cloud_time = cloud_msg->header.stamp.sec + cloud_msg->header.stamp.nanosec / 1e9;
  double time_diff = std::abs(img_time - cloud_time) * 1000;

  // 打印同步信息
  RCLCPP_INFO(rclcpp::get_logger("sensor_sync_node"), 
    "数据同步成功！\n"
    "  图像时间戳: %.6f 秒\n"
    "  点云时间戳: %.6f 秒\n"
    "  时间差: %.2f 毫秒",
    img_time, cloud_time, time_diff
  );

  // 发布同步后的图像和点云
  sync_img_pub->publish(*img_msg);
  sync_cloud_pub->publish(*cloud_msg);
}

int main(int argc, char**argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("sensor_sync_node");

  RCLCPP_INFO(node->get_logger(), "传感器同步节点启动中...");

  // 配置QoS（与订阅器保持一致）
  rclcpp::QoS qos(10);
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  auto rmw_qos = qos.get_rmw_qos_profile();

  // 创建发布器（同步后的数据话题）
  sync_img_pub = node->create_publisher<sensor_msgs::msg::Image>(
    "/sync/image",  // 同步后图像话题
    qos
  );
  sync_cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/sync/point_cloud",  // 同步后点云话题
    qos
  );

  // 订阅原始图像和点云话题
  Subscriber<sensor_msgs::msg::Image> img_sub(
    node, 
    "/image_raw", 
    rmw_qos
  );
  Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub(
    node, 
    "/livox/lidar_converted", 
    rmw_qos
  );

  // 定义同步策略（近似时间同步，队列大小10）
  typedef sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, 
    sensor_msgs::msg::PointCloud2
  > SyncPolicy;
  Synchronizer<SyncPolicy> sync(SyncPolicy(10), img_sub, cloud_sub);
  sync.registerCallback(&sync_callback);

  RCLCPP_INFO(node->get_logger(), "同步节点已就绪，开始发布同步数据至：\n"
    "  图像话题: /sync/image\n"
    "  点云话题: /sync/point_cloud");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

