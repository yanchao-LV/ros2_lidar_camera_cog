#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_msgs/msg/bounding_box2_d_array.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/pose2_d.hpp"
#include "vision_msgs/msg/point2_d.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"
#include "tf2/transform_datatypes.h"
#include "pcl_conversions/pcl_conversions.h"
#include "/usr/local/pcl-1.12/include/pcl-1.12/pcl/point_cloud.h"
#include "/usr/local/pcl-1.12/include/pcl-1.12/pcl/point_types.h"
#include "/usr/local/pcl-1.12/include/pcl-1.12/pcl/filters/crop_box.h"
#include "/usr/local/pcl-1.12/include/pcl-1.12/pcl/search/kdtree.h"
#include "/usr/local/pcl-1.12/include/pcl-1.12/pcl/segmentation/extract_clusters.h"

#include <mutex>
#include <chrono>
#include <cmath>

using namespace std::chrono_literals;
using PointT = pcl::PointXYZ;
using BoundingBox2DArray = vision_msgs::msg::BoundingBox2DArray;
using BoundingBox2D = vision_msgs::msg::BoundingBox2D;
using Pose2D = vision_msgs::msg::Pose2D;
using Point2D = vision_msgs::msg::Point2D;

class CabbageFusionNode : public rclcpp::Node
{
public:
  CabbageFusionNode() : Node("cabbage_fusion_node")
  {
    this->declare_parameter("camera_fx", 640.0);
    this->declare_parameter("camera_fy", 640.0);
    this->declare_parameter("camera_cx", 320.0);
    this->declare_parameter("camera_cy", 240.0);
    this->declare_parameter("camera_z_min", 0.5);
    this->declare_parameter("camera_z_max", 5.0);
    this->declare_parameter("mask_margin_x", 0.1);
    this->declare_parameter("mask_margin_y", 0.1);
    this->declare_parameter("mask_margin_z", 0.1);
    this->declare_parameter("dbscan_eps", 0.1);
    this->declare_parameter("dbscan_min_samples", 15);
    this->declare_parameter("sensor_tilt_angle", 65.0);
    this->declare_parameter("min_cabbage_size", 0.08);
    this->declare_parameter("max_cabbage_size", 0.3);
    loadParams();

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    vision_sub_ = this->create_subscription<BoundingBox2DArray>(
      "/cabbage_detections_camera", 10,
      std::bind(&CabbageFusionNode::visionCallback, this, std::placeholders::_1)
    );
    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/sync/point_cloud", 10,
      std::bind(&CabbageFusionNode::lidarCallback, this, std::placeholders::_1)
    );

    filtered_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/filtered_lidar_cloud", 10
    );
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cabbage_3d_markers", 10
    );

    RCLCPP_INFO(this->get_logger(), "甘蓝融合节点（欧氏聚类版）初始化完成！");
  }

private:
  BoundingBox2DArray latest_vision_boxes_;
  std::mutex vision_mutex_;
  float fx_, fy_, cx_, cy_;
  float z_min_, z_max_;
  float margin_x_, margin_y_, margin_z_;
  float cluster_tolerance_;
  int min_cluster_size_;
  float sensor_tilt_rad_;
  float min_cabbage_size_;
  float max_cabbage_size_;

  rclcpp::Subscription<BoundingBox2DArray>::SharedPtr vision_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

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
    this->get_parameter("dbscan_eps", cluster_tolerance_);
    this->get_parameter("dbscan_min_samples", min_cluster_size_);
    
    float tilt_angle_deg;
    this->get_parameter("sensor_tilt_angle", tilt_angle_deg);
    sensor_tilt_rad_ = tilt_angle_deg * M_PI / 180.0;
    
    this->get_parameter("min_cabbage_size", min_cabbage_size_);
    this->get_parameter("max_cabbage_size", max_cabbage_size_);
  }

  float getDynamicClusterTolerance(const pcl::PointCloud<PointT>::Ptr& cloud)
  {
    if (cloud->empty()) return cluster_tolerance_;
    float avg_distance = 0.0;
    for (const auto& p : *cloud) {
      avg_distance += sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
    }
    avg_distance /= cloud->size();
    return cluster_tolerance_ + 0.02 * avg_distance;
  }

  void computeClusterMinMax(const pcl::PointCloud<PointT>::Ptr& cloud, 
                            const pcl::PointIndices& indices,
                            PointT& min_pt, PointT& max_pt)
  {
    min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
    max_pt.x = max_pt.y = max_pt.z = std::numeric_limits<float>::lowest();
    for (int idx : indices.indices) {
      const auto& p = cloud->points[idx];
      min_pt.x = std::min(min_pt.x, p.x);
      min_pt.y = std::min(min_pt.y, p.y);
      min_pt.z = std::min(min_pt.z, p.z);
      max_pt.x = std::max(max_pt.x, p.x);
      max_pt.y = std::max(max_pt.y, p.y);
      max_pt.z = std::max(max_pt.z, p.z);
    }
  }

  PointT computeClusterCentroid(const pcl::PointCloud<PointT>::Ptr& cloud, 
                                const pcl::PointIndices& indices)
  {
    PointT centroid;
    centroid.x = centroid.y = centroid.z = 0.0;
    for (int idx : indices.indices) {
      centroid.x += cloud->points[idx].x;
      centroid.y += cloud->points[idx].y;
      centroid.z += cloud->points[idx].z;
    }
    centroid.x /= indices.indices.size();
    centroid.y /= indices.indices.size();
    centroid.z /= indices.indices.size();
    return centroid;
  }

  visualization_msgs::msg::Marker createBoxMarker(const std::string& frame_id, 
                                                  int id,
                                                  const PointT& min_pt, 
                                                  const PointT& max_pt,
                                                  float r, float g, float b)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
    marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
    marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = max_pt.x - min_pt.x;
    marker.scale.y = max_pt.y - min_pt.y;
    marker.scale.z = max_pt.z - min_pt.z;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 0.5;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    return marker;
  }

  visualization_msgs::msg::Marker createTextMarker(const std::string& frame_id, 
                                                   int id,
                                                   const PointT& centroid, 
                                                   float distance,
                                                   float r, float g, float b)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = centroid.x;
    marker.pose.position.y = centroid.y;
    marker.pose.position.z = centroid.z + 0.1;
    marker.pose.orientation.w = 1.0;
    marker.text = "距离: " + std::to_string(distance).substr(0, 4) + "m";
    marker.scale.z = 0.1;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 1.0;
    marker.lifetime = rclcpp::Duration::from_seconds(0.5);
    return marker;
  }

  void visionCallback(const BoundingBox2DArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    latest_vision_boxes_ = *msg;
    RCLCPP_DEBUG(this->get_logger(), "接收视觉框: %zu 个", msg->boxes.size());
  }

  bool filterCloudByVisionBox(const pcl::PointCloud<PointT>::Ptr& input_cloud,
                              pcl::PointCloud<PointT>::Ptr& output_cloud)
  {
    std::lock_guard<std::mutex> lock(vision_mutex_);
    if (latest_vision_boxes_.boxes.empty()) {
      output_cloud->clear();
      return true;
    }

    output_cloud->clear();
    pcl::CropBox<PointT> crop_box;
    crop_box.setInputCloud(input_cloud);
    crop_box.setNegative(false);

    for (const auto& box : latest_vision_boxes_.boxes) {
      // 视觉框中心：通过 position 访问 x/y
      Point2D box_center = box.center.position;
      float box_center_x = static_cast<float>(box_center.x);
      float box_center_y = static_cast<float>(box_center.y);

      // 视觉框尺寸：直接访问 size_x/size_y（根据报错修正）
      float box_width = static_cast<float>(box.size_x);
      float box_height = static_cast<float>(box.size_y);

      // 计算带冗余的2D边界
      float x_min_2d = box_center_x - box_width/2.0 - margin_x_;
      float x_max_2d = box_center_x + box_width/2.0 + margin_x_;
      float y_min_2d = box_center_y - box_height/2.0 - margin_y_;
      float y_max_2d = box_center_y + box_height/2.0 + margin_y_;

      // 3D裁剪范围计算
      Eigen::Vector4f min_pt_3d, max_pt_3d;
      min_pt_3d[0] = (x_min_2d - cx_) * z_min_ / fx_ - margin_x_;
      max_pt_3d[0] = (x_max_2d - cx_) * z_max_ / fx_ + margin_x_;
      min_pt_3d[1] = (y_min_2d - cy_) * z_min_ / fy_ - margin_y_;
      max_pt_3d[1] = (y_max_2d - cy_) * z_max_ / fy_ + margin_y_;
      min_pt_3d[2] = z_min_ - margin_z_;
      max_pt_3d[2] = z_max_ + margin_z_;
      min_pt_3d[3] = 1.0;
      max_pt_3d[3] = 1.0;

      // 裁剪并合并点云
      crop_box.setMin(min_pt_3d);
      crop_box.setMax(max_pt_3d);
      pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>());
      crop_box.filter(*temp_cloud);
      *output_cloud += *temp_cloud;
    }

    RCLCPP_DEBUG(this->get_logger(), "视觉框过滤后点云数: %zu", output_cloud->size());
    return true;
  }

  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
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

    pcl::PointCloud<PointT>::Ptr cloud_pcl(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(cloud_in_camera, *cloud_pcl);

    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>());
    if (!filterCloudByVisionBox(cloud_pcl, cloud_filtered)) {
      RCLCPP_WARN(this->get_logger(), "视觉框过滤失败，跳过本次点云");
      return;
    }

    if (!cloud_filtered->empty()) {
      sensor_msgs::msg::PointCloud2 filtered_cloud_ros;
      pcl::toROSMsg(*cloud_filtered, filtered_cloud_ros);
      filtered_cloud_ros.header = cloud_msg->header;
      filtered_cloud_pub_->publish(filtered_cloud_ros);
    }

    detectCabbageWithEuclideanCluster(cloud_filtered, cloud_msg->header);
  }

  void detectCabbageWithEuclideanCluster(const pcl::PointCloud<PointT>::Ptr& cloud,
                                         const std_msgs::msg::Header& header)
  {
    if (cloud->empty()) return;

    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    tree->setInputCloud(cloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(getDynamicClusterTolerance(cloud));
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(20000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray markers;
    int marker_id = 0;
    for (const auto& indices : cluster_indices) {
      if (indices.indices.size() < (size_t)min_cluster_size_) continue;

      PointT min_pt, max_pt;
      computeClusterMinMax(cloud, indices, min_pt, max_pt);
      PointT centroid = computeClusterCentroid(cloud, indices);
      float distance = sqrt(centroid.x*centroid.x + centroid.y*centroid.y + centroid.z*centroid.z);

      float cluster_width = max_pt.x - min_pt.x;
      float cluster_depth = max_pt.y - min_pt.y;
      float cluster_height = max_pt.z - min_pt.z;
      float max_dim = std::max({cluster_width, cluster_depth, cluster_height});
      if (max_dim < min_cabbage_size_ || max_dim > max_cabbage_size_) continue;

      auto box_marker = createBoxMarker(header.frame_id, marker_id, min_pt, max_pt, 0.0, 1.0, 0.0);
      auto text_marker = createTextMarker(header.frame_id, marker_id + 1000, centroid, distance, 1.0, 1.0, 0.0);
      markers.markers.push_back(box_marker);
      markers.markers.push_back(text_marker);
      marker_id++;
    }

    if (!markers.markers.empty()) {
      marker_pub_->publish(markers);
      RCLCPP_INFO(this->get_logger(), "发布甘蓝检测结果: %d 个", marker_id);
    }
  }
};

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CabbageFusionNode>());
  rclcpp::shutdown();
  return 0;
}

