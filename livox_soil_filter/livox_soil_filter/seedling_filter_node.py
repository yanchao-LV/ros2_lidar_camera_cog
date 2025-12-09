import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN

# 点云字段定义
OUTPUT_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
OUTPUT_POINT_STEP = 12

class SeedlingClusterNode(Node):
    def __init__(self):
        super().__init__('seedling_cluster_node')
        
        # 1. 基础过滤参数（沿用你之前的有效参数）
        self.declare_parameter('input_topic', '/livox/lidar')
        self.declare_parameter('output_pointcloud_topic', '/livox/seedling_pointcloud')
        self.declare_parameter('output_marker_topic', '/livox/seedling_bbox')  # 包围框话题
        self.declare_parameter('soil_seedling_gap', 0.09)  # 你验证的有效阈值
        self.declare_parameter('x_noise_threshold', 0.0)  # 关闭空气过滤
        
        # 2. 聚类参数（可调整）
        self.declare_parameter('dbscan_eps', 0.05)  # 簇内点最大距离（5cm，根据菜苗密度调整）
        self.declare_parameter('dbscan_min_samples', 3)  # 最小簇点数（少于则视为噪声）
        self.declare_parameter('bbox_type', '3d')  # 包围框类型：3d / 2d（y-z平面）
        
        # 3. 获取参数
        self.input_topic = self.get_parameter('input_topic').value
        self.output_pc_topic = self.get_parameter('output_pointcloud_topic').value
        self.output_bbox_topic = self.get_parameter('output_marker_topic').value
        self.soil_seedling_gap = self.get_parameter('soil_seedling_gap').value
        self.x_noise_thr = self.get_parameter('x_noise_threshold').value
        self.dbscan_eps = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').value
        self.bbox_type = self.get_parameter('bbox_type').value.lower()
        
        # 4. 订阅+发布（点云+包围框）
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.callback, 10)
        self.pub_pc = self.create_publisher(PointCloud2, self.output_pc_topic, 10)
        self.pub_bbox = self.create_publisher(MarkerArray, self.output_bbox_topic, 10)
        
        # 5. 日志提示
        self.get_logger().info('✅ 菜苗聚类+包围框节点启动成功！')
        self.get_logger().info(f'🌱 聚类参数：eps={self.dbscan_eps}m, min_samples={self.dbscan_min_samples}')
        self.get_logger().info(f'📦 包围框类型：{self.bbox_type.upper()}')
        self.get_logger().info(f'📍 已启用：包围框嵌入中心三维坐标（pose.position + text字段）')

    def callback(self, msg):
        try:
            # -------------------------- 第一步：原有菜苗过滤逻辑 --------------------------
            # 解析点云x/y/z
            total_bytes = len(msg.data)
            point_step = msg.point_step
            num_points = total_bytes // point_step
            cloud_np = []
            for i in range(num_points):
                start_idx = i * point_step
                x = np.frombuffer(msg.data[start_idx:start_idx+4], dtype=np.float32)[0]
                y = np.frombuffer(msg.data[start_idx+4:start_idx+8], dtype=np.float32)[0]
                z = np.frombuffer(msg.data[start_idx+8:start_idx+12], dtype=np.float32)[0]
                cloud_np.append([x, y, z])
            cloud_np = np.array(cloud_np)
            
            # 基础过滤（无效点）
            valid_mask = ~(np.isnan(cloud_np).any(axis=1) | np.isinf(cloud_np).any(axis=1))
            valid_np = cloud_np[valid_mask]
            if self.x_noise_thr > 0:
                valid_np = valid_np[valid_np[:, 0] > self.x_noise_thr]
            if len(valid_np) < 5:
                return
            
            # 土壤过滤（保留菜苗）
            x_max_soil = np.max(valid_np[:, 0])
            seedling_mask = valid_np[:, 0] <= (x_max_soil - self.soil_seedling_gap)
            seedling_np = valid_np[seedling_mask]
            if len(seedling_np) < self.dbscan_min_samples:
                self.get_logger().debug(f'菜苗点过少：{len(seedling_np)} < {self.dbscan_min_samples}')
                return
            
            # -------------------------- 第二步：DBSCAN聚类（过滤噪声+分菜苗） --------------------------
            # 聚类（用x/y/z三维特征，确保区分空间不同菜苗）
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric='euclidean').fit(seedling_np)
            labels = db.labels_  # 每个点的簇标签（-1=噪声）
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 有效簇数量
            noise_num = list(labels).count(-1)  # 噪声点数量
            
            self.get_logger().info(f'🌱 聚类结果：{num_clusters}株菜苗，噪声点{noise_num}个')
            
            # 过滤噪声点后的菜苗点云（可选发布，纯净无噪声）
            clean_seedling_np = seedling_np[labels != -1]
            if len(clean_seedling_np) == 0:
                return
            
            # -------------------------- 第三步：生成包围框（3D/2D）【核心修改：嵌入三维坐标】 --------------------------
            marker_array = MarkerArray()
            colors = self.get_distinct_colors(num_clusters)  # 不同菜苗不同颜色
            
            for cluster_id in range(num_clusters):
                # 提取当前簇的点云
                cluster_points = seedling_np[labels == cluster_id]
                
                # 【关键修改1：计算聚类簇的平均三维坐标（中心坐标）】
                avg_x = np.mean(cluster_points[:, 0])  # 簇内所有点x的平均值
                avg_y = np.mean(cluster_points[:, 1])  # 簇内所有点y的平均值
                avg_z = np.mean(cluster_points[:, 2])  # 簇内所有点z的平均值
                
                # 计算包围框参数（原有逻辑不变）
                if self.bbox_type == '3d':
                    # 三维包围框：min/max x/y/z
                    bbox_min = np.min(cluster_points, axis=0)  # [x_min, y_min, z_min]
                    bbox_max = np.max(cluster_points, axis=0)  # [x_max, y_max, z_max]
                    bbox_center = (bbox_min + bbox_max) / 2  # 包围框几何中心（仅用于框的位置）
                    scale = bbox_max - bbox_min  # 包围框尺寸（长×宽×高）
                else:
                    # 二维包围框（y-z平面）：忽略x，只算y/z范围
                    bbox_min_y = np.min(cluster_points[:, 1])
                    bbox_max_y = np.max(cluster_points[:, 1])
                    bbox_min_z = np.min(cluster_points[:, 2])
                    bbox_max_z = np.max(cluster_points[:, 2])
                    # 三维Marker适配2D：x取菜苗平均x，厚度设为0.01m
                    bbox_center = [avg_x, (bbox_min_y + bbox_max_y)/2, (bbox_min_z + bbox_max_z)/2]
                    scale = [0.01, bbox_max_y - bbox_min_y, bbox_max_z - bbox_min_z]
                
                # 创建包围框Marker（RViz可显示）
                marker = Marker()
                marker.header = msg.header  # 同点云坐标系（雷达坐标系）
                marker.id = cluster_id  # 每个簇唯一ID（用于后续object_association匹配）
                marker.type = Marker.CUBE  # 立方体包围框
                marker.action = Marker.ADD
                
                # 【关键修改2：将平均三维坐标存入pose.position（核心供后续读取）】
                marker.pose.position.x = float(avg_x)  # 簇中心x坐标
                marker.pose.position.y = float(avg_y)  # 簇中心y坐标
                marker.pose.position.z = float(avg_z)  # 簇中心z坐标
                
                # 姿态：默认无旋转（Quaternion(0,0,0,1)）（原有逻辑不变）
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # 尺寸（原有逻辑不变，加float()避免潜在问题）
                marker.scale.x = float(scale[0])
                marker.scale.y = float(scale[1])
                marker.scale.z = float(scale[2])
                
                # 颜色（半透明，不遮挡点云）（原有逻辑不变）
                marker.color.r = float(colors[cluster_id][0])
                marker.color.g = float(colors[cluster_id][1])
                marker.color.b = float(colors[cluster_id][2])
                marker.color.a = 0.5  # 透明度0.5
                
                # 【关键修改3：在text字段显式存储XYZ（方便调试，可选但推荐）】
                marker.text = f"XYZ:({avg_x:.2f},{avg_y:.2f},{avg_z:.2f})"
                
                # 生命周期（0=永久，直到节点停止）（原有逻辑不变）
                marker.lifetime.sec = 0
                
                marker_array.markers.append(marker)
            
            # -------------------------- 第四步：发布数据（原有逻辑不变） --------------------------
            # 发布过滤+去噪后的菜苗点云
            pc_msg = PointCloud2()
            pc_msg.header = msg.header
            pc_msg.fields = OUTPUT_FIELDS
            pc_msg.point_step = OUTPUT_POINT_STEP
            pc_msg.width = len(clean_seedling_np)
            pc_msg.height = 1
            pc_msg.row_step = pc_msg.point_step * pc_msg.width
            pc_msg.data = clean_seedling_np.astype(np.float32).tobytes()
            pc_msg.is_dense = True
            self.pub_pc.publish(pc_msg)
            
            # 发布包围框（含三维坐标信息）
            self.pub_bbox.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f'❌ 处理失败：{str(e)}')

    def get_distinct_colors(self, num_colors):
        """生成不同颜色（用于区分不同菜苗）"""
        colors = []
        for i in range(num_colors):
            # HSV颜色空间，色调均匀分布，饱和度和明度固定
            hue = i / num_colors
            rgb = self.hsv_to_rgb(hue, 0.7, 0.9)
            colors.append(rgb)
        return colors

    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB（0-1范围）"""
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        if i == 0:
            return (v, t, p)
        elif i == 1:
            return (q, v, p)
        elif i == 2:
            return (p, v, t)
        elif i == 3:
            return (p, q, v)
        elif i == 4:
            return (t, p, v)
        else:
            return (v, p, q)

def main(args=None):
    rclpy.init(args=args)
    node = SeedlingClusterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

