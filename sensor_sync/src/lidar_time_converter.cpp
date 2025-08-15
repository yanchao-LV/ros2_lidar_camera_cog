#include "rclcpp/rclcpp.hpp"
#include "livox_interfaces/msg/custom_msg.hpp"  // 消息类型在 livox_interfaces 中


class LidarTimeConverter : public rclcpp::Node {
public:
  LidarTimeConverter() : Node("lidar_time_converter") {
    // 订阅雷达原始话题（带内部时间戳）
    lidar_sub_ = this->create_subscription<livox_ros2_driver::msg::CustomMsg>(
      "/livox/lidar",  // 原始雷达话题
      10,  // 队列大小
      std::bind(&LidarTimeConverter::lidar_callback, this, std::placeholders::_1)
    );

    // 发布转换后的话题（带系统时间戳）
    converted_pub_ = this->create_publisher<livox_ros2_driver::msg::CustomMsg>(
      "/livox/lidar_converted",  // 新话题名（用于同步）
      10
    );

    RCLCPP_INFO(this->get_logger(), "时间戳转换节点已启动，转发话题: /livox/lidar_converted");
  }

private:
  void lidar_callback(const livox_ros2_driver::msg::CustomMsg::SharedPtr msg) {
    // 创建一个新消息，复制原始数据
    auto converted_msg = *msg;

    // 用当前系统时间覆盖时间戳（核心操作）
    converted_msg.header.stamp = this->now();  // 系统时间

    // 发布转换后的消息
    converted_pub_->publish(converted_msg);
  }

  rclcpp::Subscription<livox_ros2_driver::msg::CustomMsg>::SharedPtr lidar_sub_;
  rclcpp::Publisher<livox_ros2_driver::msg::CustomMsg>::SharedPtr converted_pub_;
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarTimeConverter>());
  rclcpp::shutdown();
  return 0;
}

