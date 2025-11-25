import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct
from scipy.spatial.transform import Rotation as R

class CameraLidarFusionNode(Node):
    def __init__(self):
        super().__init__('camera_lidar_fusion_node')
        
        # 相机内参（不变）
        self.fx = 1819.69314
        self.fy = 1820.23443
        self.cx = 797.44357
        self.cy = 603.31308
        
        # 外参（不变）
        T_lidar_camera = [
            -0.004561495455514472, -0.06369779020816331, -0.022297855575373768,
            -0.48614771747711283, 0.5067882233718053, -0.46423578059208637, 0.5397325573480806
        ]
        self.t_cam2lidar = np.array(T_lidar_camera[:3])
        qx, qy, qz, qw = T_lidar_camera[3:]
        self.R_cam2lidar = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # 筛选参数（不变）
        self.angle_thresh = 0.035  # ≈2度
        self.min_points = 3
        
        # 话题配置（不变）
        self.sub_bbox = self.create_subscription(String, "/cabbage_sprouts_bbox_data", self.bbox_callback, 10)
        self.sub_lidar = self.create_subscription(PointCloud2, "/livox/lidar", self.lidar_callback, 10)
        self.pub_result = self.create_publisher(String, "/cabbage_sprouts_with_distance", 10)
        
        self.latest_lidar = None
        self.get_logger().info("✅ 融合节点（支持距离+三维坐标）启动成功")

    def bbox_callback(self, msg):
        if self.latest_lidar is None:
            return
        bboxes = self.parse_bbox(msg.data)
        if not bboxes:
            return
        results = []
        for bbox in bboxes:
            cx, cy = bbox["cx"], bbox["cy"]
            # 转换为雷达射线方向
            cam_ray = self.image_to_cam_ray(cx, cy)
            lidar_ray = self.cam_ray_to_lidar_ray(cam_ray)
            # 筛选目标点云
            target_points = self.filter_points_by_direction(lidar_ray)
            # 【核心修改1：同时计算距离和三维坐标】
            distance, lidar_xyz = self.calc_distance_and_xyz(target_points)
            if distance is not None and lidar_xyz is not None:
                # 【核心修改2：追加三维坐标（X,Y,Z）字段】
                results.append(
                    f"{cx:.2f},{cy:.2f},{bbox['width']:.2f},{bbox['height']:.2f},"
                    f"{distance:.2f},{lidar_xyz[0]:.3f},{lidar_xyz[1]:.3f},{lidar_xyz[2]:.3f}"
                )
        if results:
            self.pub_result.publish(String(data=";".join(results)))

    def lidar_callback(self, msg):
        self.latest_lidar = self.pointcloud2_to_numpy(msg)

    # 【核心新增函数：同时计算距离和三维坐标】
    def calc_distance_and_xyz(self, points):
        if len(points) < self.min_points:
            self.get_logger().debug(f"有效点不足（{len(points)} < {self.min_points}）")
            return None, None
        # 计算每个点的三维距离
        distances = np.linalg.norm(points, axis=1)
        # 过滤离群点（和之前一致）
        mean, std = np.mean(distances), np.std(distances)
        valid_mask = (distances >= mean - 1.0*std) & (distances <= mean + 1.0*std)
        valid_points = points[valid_mask]
        valid_distances = distances[valid_mask]
        
        if len(valid_points) < self.min_points // 2:
            return None, None
        # 平均距离 + 平均三维坐标（X,Y,Z）
        avg_distance = np.mean(valid_distances)
        avg_xyz = np.mean(valid_points, axis=0)  # 对X,Y,Z分别取平均
        return avg_distance, avg_xyz

    # 以下函数均不变（仅复用之前的逻辑）
    def parse_bbox(self, bbox_str):
        bboxes = []
        for bbox in bbox_str.split(";"):
            try:
                cx, cy, w, h = map(float, bbox.split(","))
                bboxes.append({"cx": cx, "cy": cy, "width": w, "height": h})
            except:
                continue
        return bboxes

    def image_to_cam_ray(self, cx, cy):
        x = (cx - self.cx) / self.fx
        y = (cy - self.cy) / self.fy
        return np.array([x, y, 1.0])

    def cam_ray_to_lidar_ray(self, cam_ray):
        lidar_ray = self.R_cam2lidar @ cam_ray
        return lidar_ray / np.linalg.norm(lidar_ray)

    def filter_points_by_direction(self, target_ray):
        if self.latest_lidar is None or len(self.latest_lidar) == 0:
            return np.array([])
        points = self.latest_lidar
        point_norms = np.linalg.norm(points, axis=1, keepdims=True)
        point_dirs = points / point_norms
        cos_theta = np.dot(point_dirs, target_ray)
        mask = cos_theta > np.cos(self.angle_thresh)
        return points[mask]

    def pointcloud2_to_numpy(self, cloud_msg):
        fields = {f.name: f for f in cloud_msg.fields}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            return None
        buffer = np.frombuffer(cloud_msg.data, dtype=np.uint8)
        points = []
        for i in range(cloud_msg.width * cloud_msg.height):
            offset = i * cloud_msg.point_step
            x = struct.unpack_from("f", buffer, offset + fields["x"].offset)[0]
            y = struct.unpack_from("f", buffer, offset + fields["y"].offset)[0]
            z = struct.unpack_from("f", buffer, offset + fields["z"].offset)[0]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
        return np.array(points, dtype=np.float32)

def main(args=None):
    rclpy.init(args=args)
    node = CameraLidarFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

