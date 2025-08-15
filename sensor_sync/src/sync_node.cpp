#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "livox_interfaces/msg/custom_msg.hpp"
#include "livox_interfaces/msg/custom_msg.hpp"

using namespace message_filters;

// 同步回调函数：打印时间戳并处理数据
void syncCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg
) {
  // 计算时间差（毫秒）
  double img_time = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec / 1e9;
  double cloud_time = cloud_msg->header.stamp.sec + cloud_msg->header.stamp.nanosec / 1e9;
  double time_diff = std::abs(img_time - cloud_time) * 1000;

  // 打印同步信息（新手可直观看到效果）
  RCLCPP_INFO(rclcpp::get_logger("sync_node"), 
    "同步成功！图像时间: %.3f, 点云时间: %.3f, 时间差: %.2f ms",
    img_time, cloud_time, time_diff
  );

  // 这里可以添加你的融合处理代码（例如：识别农作物）
}

int main(int argc, char**argv) {
  // 初始化ROS2节点
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("sensor_sync_node");

  // 订阅话题（★替换为你在第一步记录的实际话题名★）
  Subscriber<sensor_msgs::msg::Image> img_sub(node, "/image_raw");  // 相机话题
  Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub(node, "/livox/lidar");  // 雷达话题

  // 同步策略：允许最大时间差100ms（可根据实际情况调整）
  typedef sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), img_sub, cloud_sub);  // 10是队列大小
  sync.registerCallback(&syncCallback);  // 绑定回调函数

  // 运行节点
  RCLCPP_INFO(node->get_logger(), "同步节点已启动，等待数据...");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

