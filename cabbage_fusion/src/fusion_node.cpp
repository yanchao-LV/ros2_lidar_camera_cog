#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_msgs/msg/bounding_box2_d_array.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/pose2_d.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
// 只保留TF2核心头文件，不依赖tf2_geometry_msgs
#include "tf2/transform_datatypes.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/filters/crop_box.h"
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <mutex>
#include <chrono>

// 简化命名空间
using namespace std::chrono_literals;
using PointT = pcl::PointXYZ;
using BoundingBox2DArray = vision_msgs::msg::BoundingBox2DArray;
using BoundingBox2D = vision_msgs::msg::BoundingBox2D;
using Pose2D = vision_msgs::msg::Pose2D;

class CabbageFusionNode : public rclcpp::Node
{
public:
  CabbageFusionNode() : Node("cabbage_fusion_node")
  {
    // 声明参数
    this->declare_parameter("camera_fx", 640.0);
    this->declare_parameter("camera_fy", 640.0);
    this->declare_parameter("camera_cx", 320.0);
    this->declare_parameter("camera_cy", 240.0);
    this->declare_parameter("camera_z_min", 0.5);
    this->declare_parameter("camera_z_max", 5.0);
    this->declare_parameter("mask_margin_x", 0.1);
    this->declare_parameter("mask_margin_y", 0.1);
    this->declare_parameter("mask_margin_z", 0.1);
    this->declare_parameter("cluster_tolerance", 0.15);
    this->declare_parameter("min_cluster_size", 50);
    this->declare_parameter("max_cluster_size", 2000);
    this->declare_parameter("min_cabbage_radius", 0.1);
    this->declare_parameter("max_cabbage_radius", 0.5);
    loadParams();

    // 初始化TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 订阅器
    vision_sub_ = this->create_subscription<BoundingBox2DArray>(
      "/cabbage_detections_camera", 10,
      std::bind(&CabbageFusionNode::visionCallback, this, std::placeholders::_1)
    );
    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", 10,
      std::bind(&CabbageFusionNode::lidarCallback, this, std::placeholders::_1)
    );

    // 发布器
    filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/filtered_lidar_cloud", 10
    );
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cabbage_detection_markers", 10
    );

    RCLCPP_INFO(this->get_logger(), "甘蓝融合节点初始化完成（无外部依赖）");
  }

private:
  // 成员变量
  BoundingBox2DArray latest_vision_boxes_;
  std::mutex vision_mutex_;

  float fx_, fy_, cx_, cy_;
  float z_min_, z_max_;
  float margin_x_, margin_y_, margin_z_;
  float cluster_tol_;
  int min_cluster_size_, max_cluster_size_;
  float min_radius_, max_radius_;

  rclcpp::Subscription<BoundingBox2DArray>::SharedPtr vision_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // 加载参数
  void loadParams()
  {
    this->get_parameter("camera_fx", fx_);
    this->get_parameter("camera_fy", fy_);
    this->get_parameter("camera_cx", cx_);
    this->get_parameter("camera_cy", cy_);
    this->get_parameter("camera_z_min", z_min_);
    this->get_parameter("camera_z_max", z_max_);
    this->get_parameter("mask_margin_x", margin_x_);
    this->get_parameter("mask_margin_y", margin_y_);
    this->get_parameter("mask_margin_z", margin_z_);
    this->get_parameter("cluster_tolerance", cluster_tol_);
    this->get_parameter("min_cluster_size", min_cluster_size_);
    this->get_parameter("max_cluster_size", max_cluster_size_);
    this->get_parameter("min_cabbage_radius", min_radius_);
    this->get_parameter("max_cabbage_radius", max_radius_);
  }

  // 视觉回调
  void visionCallback(const BoundingBox2DArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    latest_vision_boxes_ = *msg;
    RCLCPP_DEBUG(this->get_logger(), "接收视觉检测框: %zu 个", msg->boxes.size());
  }

  // 激光回调
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // 点云转相机坐标系（用tf2_sensor_msgs，已安装且无依赖）
    sensor_msgs::msg::PointCloud2 cloud_in_camera;
    try {
      auto transform = tf_buffer_->lookupTransform(
        "hik_camera",
        cloud_msg->header.frame_id,
        tf2::TimePointZero,
        100ms
      );
      tf2::doTransform(*cloud_msg, cloud_in_camera, transform);
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF转换失败: %s，跳过本次点云", ex.what());
      return;
    }

    // ROS转PCL
    pcl::PointCloud<PointT>::Ptr cloud_pcl(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(cloud_in_camera, *cloud_pcl);

    // 视觉掩码过滤
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    if (createVisionMaskAndFilter(cloud_pcl, cloud_filtered)) {
      // 发布过滤点云
      sensor_msgs::msg::PointCloud2 filtered_cloud_ros;
      pcl::toROSMsg(*cloud_filtered, filtered_cloud_ros);
      filtered_cloud_ros.header = cloud_msg->header;
      filtered_cloud_pub_->publish(filtered_cloud_ros);

      // 激光检测
      detectCabbageInLidar(cloud_filtered, cloud_msg->header);
    }

    // 发布视觉Marker（手动平移转换）
    publishVisionMarkers();
  }

  // 视觉掩码过滤
  bool createVisionMaskAndFilter(
    const pcl::PointCloud<PointT>::Ptr& input_cloud,
    pcl::PointCloud<PointT>::Ptr& output_cloud)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (latest_vision_boxes_.boxes.empty()) {
      *output_cloud = *input_cloud;
      return true;
    }

    pcl::CropBox<PointT> crop_box;
    crop_box.setInputCloud(input_cloud);
    crop_box.setNegative(true);

    for (const auto& box : latest_vision_boxes_.boxes) {
      // 正确访问Pose2D的position字段
      float xmin = box.center.position.x - box.size_x / 2.0 - margin_x_;
      float xmax = box.center.position.x + box.size_x / 2.0 + margin_x_;
      float ymin = box.center.position.y - box.size_y / 2.0 - margin_y_;
      float ymax = box.center.position.y + box.size_y / 2.0 + margin_y_;

      Eigen::Vector4f min_pt, max_pt;
      min_pt[0] = (xmin - cx_) * z_min_ / fx_ - margin_x_;
      max_pt[0] = (xmax - cx_) * z_max_ / fx_ + margin_x_;
      min_pt[1] = (ymin - cy_) * z_min_ / fy_ - margin_y_;
      max_pt[1] = (ymax - cy_) * z_max_ / fy_ + margin_y_;
      min_pt[2] = z_min_ - margin_z_;
      max_pt[2] = z_max_ + margin_z_;
      min_pt[3] = 1.0;
      max_pt[3] = 1.0;

      crop_box.setMin(min_pt);
      crop_box.setMax(max_pt);
    }

    crop_box.filter(*output_cloud);
    return true;
  }

  // 激光检测
  void detectCabbageInLidar(
    const pcl::PointCloud<PointT>::Ptr& cloud,
    const std_msgs::msg::Header& header)
  {
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(cluster_tol_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray lidar_markers;
    int marker_id = 0;

    for (const auto& indices : cluster_indices) {
      PointT centroid;
      for (int idx : indices.indices) {
        centroid.x += cloud->points[idx].x;
        centroid.y += cloud->points[idx].y;
        centroid.z += cloud->points[idx].z;
      }
      centroid.x /= indices.indices.size();
      centroid.y /= indices.indices.size();
      centroid.z /= indices.indices.size();

      float radius = 0.0;
      for (int idx : indices.indices) {
        float dx = cloud->points[idx].x - centroid.x;
        float dy = cloud->points[idx].y - centroid.y;
        float dz = cloud->points[idx].z - centroid.z;
        radius += sqrt(dx*dx + dy*dy + dz*dz);
      }
      radius /= indices.indices.size();

      if (radius >= min_radius_ && radius <= max_radius_) {
        visualization_msgs::msg::Marker marker = createMarker(
          header.frame_id, marker_id++, centroid, radius, 1.0, 0.0, 0.0
        );
        lidar_markers.markers.push_back(marker);
      }
    }

    marker_pub_->publish(lidar_markers);
  }

  // 发布视觉Marker（终极修复：仅用平移手动转换，无任何外部依赖）
  void publishVisionMarkers()
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (latest_vision_boxes_.boxes.empty()) return;

    visualization_msgs::msg::MarkerArray vision_markers;
    int marker_id = 1000;
    float tx = 0.0, ty = 0.0, tz = 0.0;  // 相机到激光的平移量

    // 1. 先获取一次TF平移（避免循环内重复查询）
    try {
      auto transform = tf_buffer_->lookupTransform(
        "livox_frame", "hik_camera", tf2::TimePointZero, 100ms
      );
      // 只提取平移量（忽略旋转，对甘蓝检测足够）
      tx = transform.transform.translation.x;
      ty = transform.transform.translation.y;
      tz = transform.transform.translation.z;
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "TF获取失败: %s，跳过视觉框转换", ex.what());
      return;
    }

    // 2. 遍历视觉框，手动转换坐标
    for (const auto& box : latest_vision_boxes_.boxes) {
      // 2.1 计算相机坐标系下的Marker位置和半径
      float z_center = (z_min_ + z_max_) / 2.0;
      PointT centroid_cam;
      centroid_cam.x = (box.center.position.x - cx_) * z_center / fx_;
      centroid_cam.y = (box.center.position.y - cy_) * z_center / fy_;
      centroid_cam.z = z_center;
      float radius = std::max(box.size_x, box.size_y) * z_center / (2 * fx_) + margin_x_;

      // 2.2 手动转换到激光坐标系：相机位置 + TF平移
      visualization_msgs::msg::Marker marker_lidar = createMarker(
        "livox_frame",  // 直接设置为激光坐标系
        marker_id++,
        PointT{centroid_cam.x + tx, centroid_cam.y + ty, centroid_cam.z + tz},  // 平移计算
        radius,
        0.0, 0.0, 1.0  // 蓝色
      );

      // 2.3 添加到数组
      vision_markers.markers.push_back(marker_lidar);
    }

    // 3. 发布Marker
    marker_pub_->publish(vision_markers);
  }

  // 创建Marker工具函数
  visualization_msgs::msg::Marker createMarker(
    const std::string& frame_id, int id,
    const PointT& centroid, float radius,
    float r, float g, float b)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->now();
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = centroid.x;
    marker.pose.position.y = centroid.y;
    marker.pose.position.z = centroid.z;
    marker.scale.x = 2 * radius;
    marker.scale.y = 2 * radius;
    marker.scale.z = 2 * radius;

    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 0.5;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    return marker;
  }
};

// 主函数
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CabbageFusionNode>());
  rclcpp::shutdown();
  return 0;
}

