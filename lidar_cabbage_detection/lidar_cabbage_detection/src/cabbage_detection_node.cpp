#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <vector>
#include <memory>
#include <limits>

class CabbageDetectionNode : public rclcpp::Node
{
public:
  CabbageDetectionNode() : Node("cabbage_detection_node"), global_marker_id_(0)
  {
    // 1. 订阅点云
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar", 5,
      std::bind(&CabbageDetectionNode::pointCloudCallback, this, std::placeholders::_1));

    // 2. 发布三维包围框
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "/cabbage_clusters", 5);
    if (!marker_pub_)
    {
      RCLCPP_FATAL(this->get_logger(), "❌ 包围框发布者初始化失败！");
      rclcpp::shutdown();
    }

    // 3. 声明默认参数（确保参数存在，避免读取异常）
    this->declare_parameter("soil_height_offset", 0.07);    // 土壤高度偏移（米）
    this->declare_parameter("min_cluster_size", 10);         // 最小聚类点数
    this->declare_parameter("max_cluster_size", 5000);      // 最大聚类点数
    this->declare_parameter("cluster_tolerance", 0.04);     // 聚类距离（米）
    this->declare_parameter("default_frame_id", "livox_frame");// 默认坐标系
    this->declare_parameter("bbox_alpha", 0.5);             // 包围框透明度
    this->declare_parameter("marker_lifetime_sec", 1.0);    // 标记生命周期（秒）

    // 4. 预读取关键参数（提前检查，避免回调中重复读取）
    this->get_parameter("marker_lifetime_sec", marker_lifetime_sec_);
    // 计算生命周期的秒和纳秒（显式转换为 builtin_interfaces::msg::Duration 格式）
    lifetime_sec_ = static_cast<int>(marker_lifetime_sec_);
    lifetime_nsec_ = static_cast<int>((marker_lifetime_sec_ - lifetime_sec_) * 1e9);

    RCLCPP_INFO(this->get_logger(), "✅ 甘蓝检测节点初始化完成（三维包围框版）");
    RCLCPP_INFO(this->get_logger(), "📌 标记生命周期：%d秒+%d纳秒 | 包围框透明度：%.1f",
                lifetime_sec_, lifetime_nsec_, this->get_parameter("bbox_alpha").as_double());
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  int global_marker_id_;          // 全局唯一标记ID
  double marker_lifetime_sec_;    // 标记生命周期（秒，从参数读取）
  int lifetime_sec_;              // 生命周期-秒（用于 lifetime 赋值）
  int lifetime_nsec_;             // 生命周期-纳秒（用于 lifetime 赋值）

  // 计算土壤最低点x值
  float getSoilLowestX(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
  {
    if (cloud->empty()) return 0.0f;
    float x_max = std::numeric_limits<float>::lowest();
    for (const auto& p : *cloud)
      if (p.x > x_max) x_max = p.x;
    return x_max;
  }

  // 计算聚类的三维边界（中心+边长）
  void getCluster3DBounds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                        const pcl::PointIndices& indices,
                        float& center_x, float& center_y, float& center_z,
                        float& size_x, float& size_y, float& size_z)
  {
    float min_x = 1e9, max_x = -1e9;
    float min_y = 1e9, max_y = -1e9;
    float min_z = 1e9, max_z = -1e9;

    for (int idx : indices.indices)
    {
      const auto& p = cloud->points[idx];
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
      min_z = std::min(min_z, p.z);
      max_z = std::max(max_z, p.z);
    }

    // 计算中心和边长
    center_x = (min_x + max_x) / 2.0f;
    center_y = (min_y + max_y) / 2.0f;
    center_z = (min_z + max_z) / 2.0f;
    size_x = (max_x - min_x) < 0.01f ? 0.01f : (max_x - min_x);  // 最小1cm，避免异常
    size_y = (max_y - min_y) < 0.01f ? 0.01f : (max_y - min_y);
    size_z = (max_z - min_z) < 0.01f ? 0.01f : (max_z - min_z);
  }

  // 获取有效坐标系（稳健版：参数不存在时用默认值）
  std::string getValidFrameId()
  {
    std::string frame_id;
    // 用 get_parameter 安全读取，不存在则赋值默认值
    if (!this->get_parameter("default_frame_id", frame_id) || frame_id.empty())
    {
      frame_id = "livox_frame";
      RCLCPP_WARN(this->get_logger(), "⚠️ 默认坐标系未设置，使用 fallback：%s", frame_id.c_str());
    }
    return frame_id;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 1. 基础检查
    if (!msg)
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ 收到空的点云消息");
      return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    try
    {
      pcl::fromROSMsg(*msg, *cloud);
    }
    catch (const std::exception& e)
    {
      RCLCPP_ERROR(this->get_logger(), "❌ 点云转换失败：%s", e.what());
      return;
    }
    if (cloud->empty())
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ 点云为空");
      return;
    }

    // 2. 读取参数（稳健版：用 get_parameter 避免异常）
    double soil_offset, cluster_tol, bbox_alpha;
    int min_cluster_size, max_cluster_size;
    this->get_parameter("soil_height_offset", soil_offset);
    this->get_parameter("cluster_tolerance", cluster_tol);
    this->get_parameter("min_cluster_size", min_cluster_size);
    this->get_parameter("max_cluster_size", max_cluster_size);
    this->get_parameter("bbox_alpha", bbox_alpha);
    // 显式转换为 PCL 所需的 float 类型
    float soil_offset_f = static_cast<float>(soil_offset);
    float cluster_tol_f = static_cast<float>(cluster_tol);

    // 3. 动态土壤过滤
    float soil_x_max = getSoilLowestX(cloud);
    float filter_upper_x = soil_x_max - soil_offset_f;
    float filter_lower_x = 0.0f;
    if (filter_upper_x <= filter_lower_x)
    {
      RCLCPP_ERROR(this->get_logger(), "❌ 土壤偏移量过大（%.2f米），建议减小", soil_offset);
      return;
    }

    // 4. 过滤x轴（保留幼苗）
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(filter_lower_x, filter_upper_x);
    pass.filter(*cloud_filtered);
    if (cloud_filtered->empty())
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ 过滤后无点，建议减小 soil_height_offset");
      return;
    }
    size_t filtered_size = cloud_filtered->size();
    std::string valid_frame = getValidFrameId();
    RCLCPP_INFO(this->get_logger(), "✅ 过滤后保留点：%zu个（frame_id：%s）",
                filtered_size, valid_frame.c_str());

    // 5. yoz平面聚类（参数显式转换，避免类型不匹配）
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud_filtered);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tol_f);                  // 显式float类型
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);
    if (cluster_indices.empty())
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ 未检测到聚类");
      return;
    }

    // 6. 生成三维包围框（核心：修复 lifetime 赋值）
    visualization_msgs::msg::MarkerArray markers;
    for (const auto& indices : cluster_indices)
    {
      // 安全检查1：聚类空值
      if (indices.indices.empty())
      {
        RCLCPP_WARN(this->get_logger(), "⚠️ 跳过空聚类");
        continue;
      }

      // 安全检查2：索引越界
      bool valid_idx = true;
      for (int idx : indices.indices)
      {
        if (idx < 0 || (size_t)idx >= filtered_size)
        {
          RCLCPP_ERROR(this->get_logger(), "❌ 索引越界：%d（点云大小：%zu）", idx, filtered_size);
          valid_idx = false;
          break;
        }
      }
      if (!valid_idx) continue;

      // 7. 计算三维边界
      float cx, cy, cz, sx, sy, sz;
      getCluster3DBounds(cloud_filtered, indices, cx, cy, cz, sx, sy, sz);

      // 8. 构造包围框（关键：修复 lifetime）
      visualization_msgs::msg::Marker bbox_marker;
      bbox_marker.header.frame_id = valid_frame;
      bbox_marker.header.stamp = this->get_clock()->now();
      bbox_marker.ns = "cabbage_bboxes";
      bbox_marker.id = global_marker_id_++;
      bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
      bbox_marker.action = visualization_msgs::msg::Marker::ADD;

      // 位置与姿态
      bbox_marker.pose.position.x = cx;
      bbox_marker.pose.position.y = cy;
      bbox_marker.pose.position.z = cz;
      bbox_marker.pose.orientation.x = 0.0f;
      bbox_marker.pose.orientation.y = 0.0f;
      bbox_marker.pose.orientation.z = 0.0f;
      bbox_marker.pose.orientation.w = 1.0f;

      // 尺寸
      bbox_marker.scale.x = sx;
      bbox_marker.scale.y = sy;
      bbox_marker.scale.z = sz;

      // 颜色与透明度（显式转换为 float，避免 double→float 隐性问题）
      bbox_marker.color.a = static_cast<float>(bbox_alpha);
      bbox_marker.color.r = 0.0f;
      bbox_marker.color.g = 1.0f;
      bbox_marker.color.b = 0.0f;

      // 【核心修复】显式赋值 lifetime（builtin_interfaces::msg::Duration 类型）
      bbox_marker.lifetime.sec = lifetime_sec_;
      bbox_marker.lifetime.nanosec = lifetime_nsec_;

      markers.markers.push_back(bbox_marker);
    }

    // 9. 发布包围框（异常捕获）
    if (!markers.markers.empty())
    {
      try
      {
        marker_pub_->publish(markers);
        RCLCPP_INFO(this->get_logger(), "✅ 发布%d个甘蓝苗三维包围框（ID：%d~%d）",
                    (int)markers.markers.size(),
                    global_marker_id_ - (int)markers.markers.size(),
                    global_marker_id_ - 1);
      }
      catch (const std::exception& e)
      {
        RCLCPP_ERROR(this->get_logger(), "❌ 包围框发布失败：%s", e.what());
      }
    }
    else
    {
      RCLCPP_WARN(this->get_logger(), "⚠️ 无有效包围框可发布");
    }
  }
};

int main(int argc, char** argv)
{
  try
  {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CabbageDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
  }
  catch (const std::exception& e)
  {
    std::cerr << "❌ 节点启动失败：" << e.what() << std::endl;
    return 1;
  }
  return 0;
}

