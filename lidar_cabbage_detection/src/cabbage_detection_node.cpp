#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/segmentation/extract_clusters.h"
#include "pcl/features/normal_3d.h"
#include "pcl/features/principal_curvatures.h"
#include <unordered_map>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

class CabbageDetectionNode : public rclcpp::Node
{
public:
  CabbageDetectionNode() : Node("cabbage_detection_node")
  {
    // 订阅激光雷达点云
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", 10, 
      std::bind(&CabbageDetectionNode::pointCloudCallback, this, std::placeholders::_1));

    // 发布可视化消息
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/cabbage_bboxes", 10);
    text_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/cabbage_labels", 10);
    no_ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/no_ground_cloud", 10);

    // 声明可配置参数（支持动态调整）
    this->declare_parameter("min_relative_height", 0.03);
    this->declare_parameter("max_relative_height", 0.20);
    this->declare_parameter("cluster_tolerance", 0.025);
    this->declare_parameter("min_cluster_size", 30);
    this->declare_parameter("lidar_tilt_angle", 60.0);  // 默认倾斜角增大到60度
    this->declare_parameter("max_real_diameter", 0.25); // 放宽尺寸上限
    this->declare_parameter("min_real_size", 0.07);     // 放宽尺寸下限
    this->declare_parameter("max_aspect_ratio", 2.5);   // 放宽长宽比限制
    this->declare_parameter("max_density", 50000.0);    // 浮点型默认值

    // 加载参数
    this->get_parameters();
    RCLCPP_INFO(this->get_logger(), "Cabbage detection node (optimized) started.");
  }

private:
  // 甘蓝检测结果结构体
  struct CabbageCheckResult {
    bool is_cabbage;
    float dx_real;
    float dy_real;
  };

  // 参数存储
  float min_relative_height_;
  float max_relative_height_;
  float cluster_tolerance_;
  std::size_t min_cluster_size_;
  float lidar_tilt_rad_;
  float max_real_diameter_;
  float min_real_size_;
  float max_aspect_ratio_;
  float max_density_;

  // 通信相关
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr text_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr no_ground_pub_;

  // 其他变量
  std::unordered_map<std::string, int> stable_cabbages_;
  int next_marker_id = 0;

  // 加载参数
  void get_parameters() {
    this->get_parameter("min_relative_height", min_relative_height_);
    this->get_parameter("max_relative_height", max_relative_height_);
    this->get_parameter("cluster_tolerance", cluster_tolerance_);
    this->get_parameter("max_real_diameter", max_real_diameter_);
    this->get_parameter("min_real_size", min_real_size_);
    this->get_parameter("max_aspect_ratio", max_aspect_ratio_);
    this->get_parameter("max_density", max_density_);

    int min_cluster_size_int;
    this->get_parameter("min_cluster_size", min_cluster_size_int);
    min_cluster_size_ = static_cast<std::size_t>(min_cluster_size_int);

    float tilt_angle_deg;
    this->get_parameter("lidar_tilt_angle", tilt_angle_deg);
    lidar_tilt_rad_ = tilt_angle_deg * M_PI / 180.0;

    // 打印参数确认
    RCLCPP_INFO(this->get_logger(), "Params loaded:");
    RCLCPP_INFO(this->get_logger(), "  LiDAR tilt: %.1f° (%.3f rad) | Size range: %.2f~%.2f m",
                tilt_angle_deg, lidar_tilt_rad_, min_real_size_, max_real_diameter_);
    RCLCPP_INFO(this->get_logger(), "  Aspect ratio limit: %.1f | Density limit: %.0f points/m²",
                max_aspect_ratio_, max_density_);
  }

  // 计算点云包围盒
  void computeMinMax3D(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                       pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt)
  {
    min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
    max_pt.x = max_pt.y = max_pt.z = std::numeric_limits<float>::lowest();
    
    for (const auto& p : *cloud) {
      min_pt.x = std::min(min_pt.x, p.x);
      min_pt.y = std::min(min_pt.y, p.y);
      min_pt.z = std::min(min_pt.z, p.z);
      max_pt.x = std::max(max_pt.x, p.x);
      max_pt.y = std::max(max_pt.y, p.y);
      max_pt.z = std::max(max_pt.z, p.z);
    }
  }

  // 计算点到地面的相对高度
  float getRelativeHeight(const pcl::PointXYZ& p, const pcl::ModelCoefficients::Ptr& ground_coeffs) {
    if (ground_coeffs->values.size() < 4) {
      RCLCPP_WARN(this->get_logger(), "Invalid ground coefficients, return height 0.");
      return 0.0f;
    }

    float a = ground_coeffs->values[0];
    float b = ground_coeffs->values[1];
    float c = ground_coeffs->values[2];
    float d = ground_coeffs->values[3];
    
    float numerator = std::abs(a * p.x + b * p.y + c * p.z + d);
    float denominator = std::sqrt(a*a + b*b + c*c);
    return denominator > 1e-6 ? (numerator / denominator) : 0.0f;
  }

  // 地面滤波
  void removeGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                    pcl::ModelCoefficients::Ptr& coefficients) {
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setMaxIterations(1500);
    seg.setInputCloud(input);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
      RCLCPP_WARN(this->get_logger(), "Ground fitting failed! Using original cloud.");
      *output = *input;
      return;
    }

    // 提取非地面点
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*output);

    RCLCPP_INFO(this->get_logger(), "Ground removed: %zu → %zu points", input->size(), output->size());
  }

  // 欧式聚类
  void performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                         std::vector<pcl::PointIndices>& cluster_indices) {
    if (input->size() < min_cluster_size_) {
      RCLCPP_WARN(this->get_logger(), "Too few points for clustering: %zu < %zu", input->size(), min_cluster_size_);
      return;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(static_cast<int>(min_cluster_size_));
    ec.setMaxClusterSize(2000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(input);
    ec.extract(cluster_indices);
  }

  // 核心：甘蓝判断
  CabbageCheckResult isCabbage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                               const pcl::ModelCoefficients::Ptr& ground_coeffs) {
    CabbageCheckResult result;
    result.is_cabbage = false;
    result.dx_real = 0.0f;
    result.dy_real = 0.0f;

    // 计算斜向尺寸与真实尺寸（倾斜斜角校准）
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    float dx_oblique = max_pt.x - min_pt.x;
    float dy_oblique = max_pt.y - min_pt.y;

    // 倾斜角校准：真实尺寸 = 斜向尺寸 × cos(倾斜角)
    float cos_tilt = std::cos(lidar_tilt_rad_);
    result.dx_real = dx_oblique * cos_tilt;
    result.dy_real = dy_oblique * cos_tilt;
    float max_real_size = std::max(result.dx_real, result.dy_real);

    // 调试日志：显示斜向尺寸与校准准后的后的真实尺寸
    RCLCPP_DEBUG(this->get_logger(), "Tilt calibration: oblique(%.3f,%.3f) → real(%.3f,%.3f) m",
                 dx_oblique, dy_oblique, result.dx_real, result.dy_real);

    // 1. 尺寸过滤
    if (!(max_real_size > min_real_size_ && max_real_size < max_real_diameter_)) {
      RCLCPP_INFO(this->get_logger(), "Size invalid: real max diameter=%.3f m (limit=%.3f~%.3f m)",
                  max_real_size, min_real_size_, max_real_diameter_);
      return result;
    }

    // 2. 形状过滤
    float aspect_ratio = (result.dx_real > result.dy_real) ? 
                        (result.dx_real / result.dy_real) : (result.dy_real / result.dx_real);
    if (aspect_ratio > max_aspect_ratio_) {
      RCLCPP_INFO(this->get_logger(), "Shape invalid: aspect ratio=%.2f (limit=%.1f)",
                  aspect_ratio, max_aspect_ratio_);
      return result;
    }

    // 3. 自身高度过滤
    float min_rel_h = std::numeric_limits<float>::max();
    float max_rel_h = std::numeric_limits<float>::lowest();
    for (const auto& p : *cluster) {
      float rel_h = getRelativeHeight(p, ground_coeffs);
      min_rel_h = std::min(min_rel_h, rel_h);
      max_rel_h = std::max(max_rel_h, rel_h);
    }
    float cluster_height = max_rel_h - min_rel_h;
    if (cluster_height < 0.05 || cluster_height > 0.18) {
      RCLCPP_INFO(this->get_logger(), "Height invalid: self-height=%.3f m (5-18cm)", cluster_height);
      return result;
    }

    // 4. 密度过滤
    float real_area = result.dx_real * result.dy_real;
    float density = cluster->size() / real_area;
    if (density < 200 || density > max_density_) {
      RCLCPP_INFO(this->get_logger(), "Density invalid: %.0f points/m² (200-%.0f)",
                  density, max_density_);
      return result;
    }

    // 5. 曲率过滤（优化后）
    if (!analyzeCurvature(cluster)) {
      RCLCPP_INFO(this->get_logger(), "Curvature invalid: not enough spherical points");
      return result;
    }

    // 所有条件满足，判定为甘蓝
    result.is_cabbage = true;
    return result;
  }

  // 曲率分析（核心优化：放宽球面特征判断条件）
  bool analyzeCurvature(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cluster);
    ne.setInputCloud(cluster);
    ne.setSearchMethod(tree);
    
    // 优化1：缩小搜索半径至0.04m（适配13cm直径的小甘蓝）
    ne.setRadiusSearch(0.04);  
    ne.compute(*normals);

    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> pc;
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    pc.setInputCloud(cluster);
    pc.setInputNormals(normals);
    pc.setSearchMethod(tree);
    pc.setRadiusSearch(0.04);  // 同步缩小搜索半径
    pc.compute(*curvatures);

    size_t spherical_points = 0;
    size_t total_points = curvatures->size();
    for (size_t i = 0; i < total_points; i++) {
      // 优化2：放宽曲率差阈值至0.12（允许更大的曲率差异）
      float curvature_diff = std::abs(curvatures->points[i].pc1 - curvatures->points[i].pc2);
      // 优化3：放宽平均曲率范围至0.03~0.35（包容更多表面形态）
      float avg_curvature = (curvatures->points[i].pc1 + curvatures->points[i].pc2) / 2.0f;
      
      if (curvature_diff < 0.12 && avg_curvature > 0.03 && avg_curvature < 0.35) {
        spherical_points++;
      }
    }

    // 优化4：降低球面点比例要求至30%（允许更多非球面点）
    float ratio = total_points > 0 ? (static_cast<float>(spherical_points) / total_points) : 0.0f;
    RCLCPP_DEBUG(this->get_logger(), "Curvature ratio: %zu/%zu (%.1f%%, need ≥30%%)",
                 spherical_points, total_points, ratio * 100);
    return ratio > 0.2;
  }

  // 生成聚类唯一标识
  std::string getClusterKey(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    float cx = (min_pt.x + max_pt.x) / 2.0f;
    float cy = (min_pt.y + max_pt.y) / 2.0f;

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << cx << "," << cy;
    return ss.str();
  }

  // 发布甘蓝包围框
  void publishBoundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                          const std::string& frame_id) {
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "cabbage_bboxes";
    marker.id = next_marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.015;
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    marker.lifetime = rclcpp::Duration::from_seconds(0.8);

    geometry_msgs::msg::Point p[8];
    p[0].x = min_pt.x; p[0].y = min_pt.y; p[0].z = min_pt.z;
    p[1].x = max_pt.x; p[1].y = min_pt.y; p[1].z = min_pt.z;
    p[2].x = max_pt.x; p[2].y = max_pt.y; p[2].z = min_pt.z;
    p[3].x = min_pt.x; p[3].y = max_pt.y; p[3].z = min_pt.z;
    p[4].x = min_pt.x; p[4].y = min_pt.y; p[4].z = max_pt.z;
    p[5].x = max_pt.x; p[5].y = min_pt.y; p[5].z = max_pt.z;
    p[6].x = max_pt.x; p[6].y = max_pt.y; p[6].z = max_pt.z;
    p[7].x = min_pt.x; p[7].y = max_pt.y; p[7].z = max_pt.z;

    std::vector<std::pair<int, int>> edges = {
      {0,1}, {1,2}, {2,3}, {3,0},
      {4,5}, {5,6}, {6,7}, {7,4},
      {0,4}, {1,5}, {2,6}, {3,7}
    };
    for (auto& e : edges) {
      marker.points.push_back(p[e.first]);
      marker.points.push_back(p[e.second]);
    }

    marker_pub_->publish(marker);
  }

  // 发布甘蓝标签
  void publishTextLabel(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                        const std::string& frame_id) {
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    geometry_msgs::msg::Point center;
    center.x = (min_pt.x + max_pt.x) / 2.0f;
    center.y = (min_pt.y + max_pt.y) / 2.0f;
    center.z = max_pt.z + 0.08;

    visualization_msgs::msg::Marker text;
    text.header.frame_id = frame_id;
    text.header.stamp = this->get_clock()->now();
    text.ns = "cabbage_labels";
    text.id = next_marker_id++;
    text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action = visualization_msgs::msg::Marker::ADD;
    text.pose.position = center;
    text.scale.z = 0.08;
    text.color.r = 0.0f;
    text.color.g = 1.0f;
    text.color.b = 0.0f;
    text.color.a = 1.0f;
    text.lifetime = rclcpp::Duration::from_seconds(0.8);
    text.text = "Cabbage";

    text_pub_->publish(text);
  }

  // 点云回调
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    next_marker_id = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty point cloud, skip processing.");
      return;
    }

    // 步骤1：下采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud_filtered);
    RCLCPP_INFO(this->get_logger(), "Downsampled: %zu → %zu points", cloud->size(), cloud_filtered->size());

    if (cloud_filtered->size() < 50) {
      RCLCPP_WARN(this->get_logger(), "Too few points after downsampling: %zu < 50", cloud_filtered->size());
      return;
    }

    // 步骤2：地面滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ModelCoefficients::Ptr ground_coeffs(new pcl::ModelCoefficients);
    removeGround(cloud_filtered, cloud_no_ground, ground_coeffs);

    // 发布地面滤波后点云（调试用）
    sensor_msgs::msg::PointCloud2 no_ground_msg;
    pcl::toROSMsg(*cloud_no_ground, no_ground_msg);
    no_ground_msg.header = msg->header;
    no_ground_pub_->publish(no_ground_msg);

    if (cloud_no_ground->empty()) {
      RCLCPP_WARN(this->get_logger(), "No points left after ground removal, skip clustering.");
      return;
    }

    // 步骤3：高度过滤
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_height_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& p : *cloud_no_ground) {
      float rel_h = getRelativeHeight(p, ground_coeffs);
      if (rel_h > min_relative_height_ && rel_h < max_relative_height_) {
        cloud_height_filtered->points.push_back(p);
      }
    }
    cloud_height_filtered->width = cloud_height_filtered->size();
    cloud_height_filtered->height = 1;
    RCLCPP_INFO(this->get_logger(), "Height-filtered: %zu → %zu points (rel_h: %.2f~%.2f m)",
                cloud_no_ground->size(), cloud_height_filtered->size(), min_relative_height_, max_relative_height_);

    if (cloud_height_filtered->size() < min_cluster_size_) {
      RCLCPP_WARN(this->get_logger(), "Too few points after height filter: %zu < %zu",
                  cloud_height_filtered->size(), min_cluster_size_);
      return;
    }

    // 步骤4：欧式聚类
    std::vector<pcl::PointIndices> cluster_indices;
    performClustering(cloud_height_filtered, cluster_indices);
    RCLCPP_INFO(this->get_logger(), "Found %zu clusters after clustering", cluster_indices.size());

    // 步骤5：聚类识别
    int cabbage_count = 0;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
      // 提取单个聚类
      const auto& indices = cluster_indices[i];
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& idx : indices.indices) {
        cluster->points.push_back(cloud_height_filtered->points[idx]);
      }
      cluster->width = cluster->size();
      cluster->height = 1;
      RCLCPP_DEBUG(this->get_logger(), "Cluster %zu: %zu points", i, cluster->size());

      // 调用isCabbage判断
      CabbageCheckResult check_result = isCabbage(cluster, ground_coeffs);
      if (check_result.is_cabbage) {
        // 连续帧确认
        std::string key = getClusterKey(cluster);
        stable_cabbages_[key]++;
        if (stable_cabbages_[key] >= 2) {
          cabbage_count++;
          float max_real_size = std::max(check_result.dx_real, check_result.dy_real);
          RCLCPP_INFO(this->get_logger(), "✅ Cluster %zu confirmed as cabbage (real size=%.3f m)",
                      i, max_real_size);
          publishBoundingBox(cluster, msg->header.frame_id);
          publishTextLabel(cluster, msg->header.frame_id);
        } else {
          RCLCPP_INFO(this->get_logger(), "⌛ Cluster %zu pending (frame count: %d)", i, stable_cabbages_[key]);
        }
      }
    }

    // 步骤6：清理过期缓存
    for (auto it = stable_cabbages_.begin(); it != stable_cabbages_.end(); ) {
      if (it->second < 1) {
        it = stable_cabbages_.erase(it);
      } else {
        it->second--;
        ++it;
      }
    }

    RCLCPP_INFO(this->get_logger(), "📊 Final detected cabbages: %d", cabbage_count);
  }
};

// 主函数
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CabbageDetectionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

