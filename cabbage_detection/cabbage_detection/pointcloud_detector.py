import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D
from geometry_msgs.msg import Point, Pose, Vector3
from std_msgs.msg import Header
import pcl
import numpy as np
import open3d as o3d
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class PointCloudDetector(Node):
    def __init__(self):
        super().__init__('pointcloud_detector')
        
        # 参数配置
        self.declare_parameters(
            namespace='',
            parameters=[
                ('voxel_size', 0.02),
                ('cluster_tolerance', 0.05),
                ('min_cluster_size', 50),
                ('max_cluster_size', 5000),
                ('min_radius', 0.07),
                ('max_radius', 0.12),
                ('roughness_threshold', 0.04),
                ('input_topic', '/livox/lidar'),
                ('output_topic', '/detections/pointcloud')
            ]
        )
        
        # 获取参数
        self.voxel_size = self.get_parameter('voxel_size').value
        self.cluster_tolerance = self.get_parameter('cluster_tolerance').value
        self.min_cluster_size = self.get_parameter('min_cluster_size').value
        self.max_cluster_size = self.get_parameter('max_cluster_size').value
        self.min_radius = self.get_parameter('min_radius').value
        self.max_radius = self.get_parameter('max_radius').value
        self.roughness_threshold = self.get_parameter('roughness_threshold').value
        
        # QoS配置 - 确保点云传输可靠性
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # 订阅点云
        self.subscription = self.create_subscription(
            PointCloud2,
            self.get_parameter('input_topic').value,
            self.pointcloud_callback,
            qos_profile=qos_profile
        )
        
        # 发布检测结果
        self.publisher = self.create_publisher(
            Detection3DArray,
            self.get_parameter('output_topic').value,
            10
        )
        
        self.get_logger().info("PointCloud Detector Initialized")

    def pointcloud_callback(self, msg):
        # 将ROS PointCloud2转换为PCL点云
        cloud = pcl.PointCloud()
        cloud.from_list(self.ros2_to_pcl(msg))
        
        if cloud.size() == 0:
            return
        
        # 1. 降采样
        voxel = cloud.make_voxel_grid_filter()
        voxel.set_leaf_size(self.voxel_size, self.voxel_size, self.voxel_size)
        downsampled = voxel.filter()
        
        # 2. 移除地面 (改进的RANSAC)
        seg = downsampled.make_segmenter()
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.05)
        inliers, coefficients = seg.segment()
        objects = downsampled.extract(inliers, negative=True)
        
        # 3. 欧氏聚类
        tree = objects.make_kdtree()
        ec = objects.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(self.cluster_tolerance)
        ec.set_MinClusterSize(self.min_cluster_size)
        ec.set_MaxClusterSize(self.max_cluster_size)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()
        
        # 4. 甘蓝球检测
        detections = Detection3DArray()
        detections.header = msg.header
        
        for j, indices in enumerate(cluster_indices):
            cluster = objects.extract(indices)
            
            # 计算最小包围球
            min_sphere = cluster.make_MinimalEnclosingSphere()
            center = min_sphere[0]
            radius = min_sphere[1]
            
            # 跳过不符合尺寸的对象
            if radius < self.min_radius or radius > self.max_radius:
                continue
            
            # 计算表面粗糙度
            points = cluster.to_array()
            centroid = np.mean(points, axis=0)
            dists = np.linalg.norm(points - centroid, axis=1)
            roughness = np.std(dists - radius)
            
            # 跳过粗糙度高的对象
            if roughness > self.roughness_threshold:
                continue
            
            # 创建检测结果
            detection = Detection3D()
            detection.header = msg.header
            detection.id = f"cabbage_{j}"
            
            # 设置位置
            pose = Pose()
            pose.position.x = center[0]
            pose.position.y = center[1]
            pose.position.z = center[2]
            detection.bbox.center = pose
            
            # 设置尺寸
            detection.bbox.size.x = radius * 2
            detection.bbox.size.y = radius * 2
            detection.bbox.size.z = radius * 2
            
            detections.detections.append(detection)
        
        # 发布检测结果
        self.publisher.publish(detections)
        self.get_logger().info(f'Detected {len(detections.detections)} cabbages')

    def ros2_to_pcl(self, cloud_msg):
        """将ROS2 PointCloud2转换为PCL兼容的点列表"""
        points = []
        for data in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([data[0], data[1], data[2]])
        return points

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()