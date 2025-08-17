#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"  // 匹配实际话题类型

class LidarTimeConverter : public rclcpp::Node {
public:
  LidarTimeConverter() : Node("lidar_time_converter") {
    // 订阅雷达原始话题（类型：sensor_msgs/msg/PointCloud2）
    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar",  // 原始雷达话题
      10,
      std::bind(&LidarTimeConverter::lidar_callback, this, std::placeholders::_1)
    );

    // 发布转换后的话题（保持同一类型，仅更新时间戳）
    converted_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar_converted",  // 转换后话题
      10
    );

    RCLCPP_INFO(this->get_logger(), "时间戳转换节点已启动，订阅/发布类型: sensor_msgs/msg/PointCloud2");
    RCLCPP_INFO(this->get_logger(), "转换后发布话题: /livox/lidar_converted");
  }

private:
  void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto converted_msg = *msg;  // 复制原始点云数据
    converted_msg.header.stamp = this->now();  // 用系统时间覆盖时间戳
    converted_pub_->publish(converted_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr converted_pub_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarTimeConverter>());
  rclcpp::shutdown();
  return 0;
}

