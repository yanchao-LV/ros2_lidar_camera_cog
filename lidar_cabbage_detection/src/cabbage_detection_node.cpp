#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/common/geometry.h>  
#include <limits>
#include <string>

class CabbageDetectionNode : public rclcpp::Node
{
public:
  CabbageDetectionNode() : Node("cabbage_detection_node")
  {
    // 订阅激光雷达点云
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", 10, 
      std::bind(&CabbageDetectionNode::pointCloudCallback, this, std::placeholders::_1));

    // 发布甘蓝包围框
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/cabbage_bboxes", 10);
    
    // 发布文本标签
    text_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/cabbage_labels", 10);

    RCLCPP_INFO(this->get_logger(), "Cabbage detection node (field version) started.");
  }

private:
  // 自定义计算点云包围盒的函数
  void computeMinMax3D(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                       pcl::PointXYZ& min_pt, pcl::PointXYZ& max_pt)
  {
    // 初始化最小值和最大值
    min_pt.x = min_pt.y = min_pt.z = std::numeric_limits<float>::max();
    max_pt.x = max_pt.y = max_pt.z = std::numeric_limits<float>::lowest();
    
    // 遍历点云更新极值
    for (const auto& p : *cloud) {
      if (p.x < min_pt.x) min_pt.x = p.x;
      if (p.y < min_pt.y) min_pt.y = p.y;
      if (p.z < min_pt.z) min_pt.z = p.z;
      if (p.x > max_pt.x) max_pt.x = p.x;
      if (p.y > max_pt.y) max_pt.y = p.y;
      if (p.z > max_pt.z) max_pt.z = p.z;
    }
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 重置标记ID
    next_marker_id = 0;
    
    // 1. 转换ROS点云到PCL格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(), "Empty point cloud received.");
      return;
    }

    // 2. 下采样（平衡效率和精度）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);  // 降低到1cm分辨率，保留更多细节
    vg.filter(*cloud_filtered);
    
    RCLCPP_INFO(this->get_logger(), "Downsampled cloud size: %zu", cloud_filtered->size());

    // 3. 地面滤波（保留甘蓝点云）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    removeGround(cloud_filtered, cloud_no_ground);
    
    RCLCPP_INFO(this->get_logger(), "Non-ground points: %zu", cloud_no_ground->size());

    // 4. 欧式聚类分割（找到可能的甘蓝点云簇）
    std::vector<pcl::PointIndices> cluster_indices;
    performClustering(cloud_no_ground, cluster_indices);

    RCLCPP_INFO(this->get_logger(), "Found %zu clusters", cluster_indices.size());

    // 5. 对每个聚类进行甘蓝识别
    int cabbage_count = 0;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
      const auto& indices = cluster_indices[i];
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& idx : indices.indices) {
        cluster->points.push_back(cloud_no_ground->points[idx]);
      }
      cluster->width = cluster->size();
      cluster->height = 1;
      
      RCLCPP_INFO(this->get_logger(), "Cluster %zu size: %zu", i, cluster->size());

      // 6. 特征提取与判断
      bool is_cabbage = isCabbage(cluster);
      if (is_cabbage) {
        cabbage_count++;
        RCLCPP_INFO(this->get_logger(), "Cluster %zu identified as cabbage", i);
        publishBoundingBox(cluster, msg->header.frame_id);
        publishTextLabel(cluster, msg->header.frame_id);
      } else {
        RCLCPP_INFO(this->get_logger(), "Cluster %zu rejected as cabbage", i);
      }
    }
    
    RCLCPP_INFO(this->get_logger(), "Detected %d cabbages", cabbage_count);
  }

  // 地面滤波（提取非地面点云）
  void removeGround(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& output) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);  // 降低到2cm阈值（适应更精确的地面分离）
    seg.setInputCloud(input);
    seg.segment(*inliers, *coefficients);
    
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(input);
    extract.setIndices(inliers);
    extract.setNegative(true);  // 保留非地面点
    extract.filter(*output);
    
    RCLCPP_INFO(this->get_logger(), "Ground plane segmentation: %zu inliers", inliers->indices.size());
  }

  // 欧式聚类（找到潜在甘蓝点云簇）
  void performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input,
                         std::vector<pcl::PointIndices>& cluster_indices) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(input);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.03);  // 降低到3cm容差（适应更紧密的聚类）
    ec.setMinClusterSize(30);      // 增加到30个点（排除小干扰）
    ec.setMaxClusterSize(5000);    // 增加最大点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(input);
    ec.extract(cluster_indices);
  }

  // 核心：判断是否为甘蓝
  bool isCabbage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
    // 计算包围盒尺寸（用于长宽比分析）
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    float dx = max_pt.x - min_pt.x;
    float dy = max_pt.y - min_pt.y;
    float dz = max_pt.z - min_pt.z;

    // 1. 尺寸过滤 - 确保在合理范围内（甘蓝直径约10-20cm）
    bool size_valid = (dx > 0.05 && dx < 0.3) && 
                     (dy > 0.05 && dy < 0.3) && 
                     (dz > 0.05 && dz < 0.3);
    
    if (!size_valid) {
      RCLCPP_INFO(this->get_logger(), "Size invalid: dx=%.3f, dy=%.3f, dz=%.3f", dx, dy, dz);
      return false;
    }

    // 2. 长宽比判断（投影近似圆形）
    float aspect_ratio = (dx > dy) ? (dx / dy) : (dy / dx);
    bool is_round = (aspect_ratio < 1.8);  // 放宽阈值
    
    RCLCPP_INFO(this->get_logger(), "Aspect ratio: %.2f (valid: %d)", aspect_ratio, is_round);

    // 3. 高度下限（确保是立体物体）
    bool is_high_enough = (dz > 0.05);  // 增加到5cm高
    
    RCLCPP_INFO(this->get_logger(), "Height: %.3f (valid: %d)", dz, is_high_enough);

    // 4. 曲率分析（重点！判断是否有球面特征）
    bool has_spherical_curvature = analyzeCurvature(cluster);

    // 最终判断：满足尺寸、形状和曲率特征
    return size_valid && is_round && is_high_enough && has_spherical_curvature;
  }

  // 分析点云的曲率特征（判断是否包含球面部分）
bool analyzeCurvature(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster) {
  // 计算法线
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cluster);
  ne.setInputCloud(cluster);
  ne.setSearchMethod(tree);
  ne.setRadiusSearch(0.05);  // 搜索半径
  ne.compute(*normals);
  
  RCLCPP_INFO(this->get_logger(), "Normals computed: %zu", normals->size());

  // 计算主曲率
  pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> pc;
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
  pc.setInputCloud(cluster);
  pc.setInputNormals(normals);
  pc.setSearchMethod(tree);
  pc.setRadiusSearch(0.05);
  pc.compute(*curvatures);
  
  RCLCPP_INFO(this->get_logger(), "Curvatures computed: %zu", curvatures->size());

  // 关键：在使用前声明变量（确保作用域覆盖日志输出行）
  size_t spherical_points = 0;  // 声明为size_t
  size_t total_points = curvatures->size();  // 声明为size_t
  
  for (size_t i = 0; i < total_points; i++) {
    float curvature_diff = std::abs(curvatures->points[i].pc1 - curvatures->points[i].pc2);
    float avg_curvature = (curvatures->points[i].pc1 + curvatures->points[i].pc2) / 2.0;
    
    if (curvature_diff < 0.1 && avg_curvature > 0.05 && avg_curvature < 0.3) {
      spherical_points++;
    }
  }

  // 计算比例（确保ratio在日志前定义）
  float ratio = (total_points > 0) ? static_cast<float>(spherical_points) / total_points : 0.0f;
  
  // 修复日志占位符：%d改为%zu（匹配size_t）
  RCLCPP_INFO(this->get_logger(), "Spherical points: %zu/%zu (%.2f%%)", 
              spherical_points, total_points, ratio * 100);

  return ratio > 0.3;
}


  // 发布甘蓝包围框
  void publishBoundingBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                          const std::string& frame_id) {
    // 计算包围盒
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    
    // 创建线框标记
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "cabbage_bboxes";
    marker.id = next_marker_id++;
    marker.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 设置线宽
    marker.scale.x = 0.02;  // 增加线宽
    
    // 设置颜色（红色）
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;  // 不透明
    
    // 设置生命周期
    marker.lifetime = rclcpp::Duration::from_seconds(1.0);
    
    // 定义立方体的8个顶点
    geometry_msgs::msg::Point p[8];
    p[0].x = min_pt.x; p[0].y = min_pt.y; p[0].z = min_pt.z;
    p[1].x = max_pt.x; p[1].y = min_pt.y; p[1].z = min_pt.z;
    p[2].x = max_pt.x; p[2].y = max_pt.y; p[2].z = min_pt.z;
    p[3].x = min_pt.x; p[3].y = max_pt.y; p[3].z = min_pt.z;
    p[4].x = min_pt.x; p[4].y = min_pt.y; p[4].z = max_pt.z;
    p[5].x = max_pt.x; p[5].y = min_pt.y; p[5].z = max_pt.z;
    p[6].x = max_pt.x; p[6].y = max_pt.y; p[6].z = max_pt.z;
    p[7].x = min_pt.x; p[7].y = max_pt.y; p[7].z = max_pt.z;
    
    // 定义立方体的12条边
    // 底部
    marker.points.push_back(p[0]); marker.points.push_back(p[1]);
    marker.points.push_back(p[1]); marker.points.push_back(p[2]);
    marker.points.push_back(p[2]); marker.points.push_back(p[3]);
    marker.points.push_back(p[3]); marker.points.push_back(p[0]);
    
    // 顶部
    marker.points.push_back(p[4]); marker.points.push_back(p[5]);
    marker.points.push_back(p[5]); marker.points.push_back(p[6]);
    marker.points.push_back(p[6]); marker.points.push_back(p[7]);
    marker.points.push_back(p[7]); marker.points.push_back(p[4]);
    
    // 侧面
    marker.points.push_back(p[0]); marker.points.push_back(p[4]);
    marker.points.push_back(p[1]); marker.points.push_back(p[5]);
    marker.points.push_back(p[2]); marker.points.push_back(p[6]);
    marker.points.push_back(p[3]); marker.points.push_back(p[7]);
    
    marker_pub_->publish(marker);
  }
  
  // 发布文本标签
  void publishTextLabel(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster,
                        const std::string& frame_id) {
    // 计算包围盒
    pcl::PointXYZ min_pt, max_pt;
    computeMinMax3D(cluster, min_pt, max_pt);
    
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "cabbage_labels";
    marker.id = next_marker_id++;
    marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 设置位置（在甘蓝上方）
    marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
    marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
    marker.pose.position.z = max_pt.z + 0.05;  // 高出5cm
    
    // 设置文本
    marker.text = "CABBAGE";
    
    // 设置尺寸
    marker.scale.z = 0.1;  // 文本大小
    
    // 设置颜色（黄色）
    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    
    // 设置生命周期
    marker.lifetime = rclcpp::Duration::from_seconds(1.0);
    
    text_pub_->publish(marker);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr text_pub_;
  int next_marker_id = 0;  // 用于生成唯一的Marker ID
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CabbageDetectionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}