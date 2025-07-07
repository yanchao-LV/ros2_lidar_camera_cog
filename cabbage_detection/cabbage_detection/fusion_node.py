import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection2DArray
from geometry_msgs.msg import PointStamped
from cabbage_detection.fusion_utils import project_3d_to_2d
from sensor_msgs.msg import CameraInfo
import numpy as np
from std_msgs.msg import Float32MultiArray

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        
        # 参数
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('visual_detections_topic', '/detections/visual')
        self.declare_parameter('pointcloud_detections_topic', '/detections/pointcloud')
        
        # 订阅相机内参
        self.camera_matrix = None
        self.dist_coeffs = None
        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10
        )
        
        # 订阅视觉检测
        self.visual_detections = []
        self.create_subscription(
            Detection2DArray,
            self.get_parameter('visual_detections_topic').value,
            self.visual_callback,
            10
        )
        
        # 订阅点云检测
        self.pointcloud_detections = []
        self.create_subscription(
            Detection3DArray,
            self.get_parameter('pointcloud_detections_topic').value,
            self.pointcloud_callback,
            10
        )
        
        # 发布融合结果
        self.fused_pub = self.create_publisher(Float32MultiArray, '/fused_detections', 10)
        
        self.get_logger().info("Fusion Node Initialized")
    
    def camera_info_callback(self, msg):
        """存储相机内参"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d)
    
    def visual_callback(self, msg):
        """存储视觉检测结果"""
        self.visual_detections = msg.detections
    
    def pointcloud_callback(self, msg):
        """处理点云检测结果并执行融合"""
        if self.camera_matrix is None:
            self.get_logger().warn("Camera info not received yet")
            return
            
        self.pointcloud_detections = msg.detections
        fused_results = []
        
        # 对每个点云检测进行融合
        for pc_det in self.pointcloud_detections:
            center = pc_det.bbox.center.position
            
            # 将3D中心点投影到2D图像
            img_point = project_3d_to_2d(
                [center.x, center.y, center.z],
                self.camera_matrix,
                self.dist_coeffs
            )
            
            # 查找匹配的视觉检测
            matched = False
            for vis_det in self.visual_detections:
                bbox = vis_det.bbox
                x_min = bbox.center.x - bbox.size_x/2
                x_max = bbox.center.x + bbox.size_x/2
                y_min = bbox.center.y - bbox.size_y/2
                y_max = bbox.center.y + bbox.size_y/2
                
                # 检查投影点是否在检测框内
                if (x_min <= img_point[0] <= x_max and 
                    y_min <= img_point[1] <= y_max):
                    
                    # 尺寸一致性检查
                    estimated_diameter = (bbox.size_x * 0.002) * center.z
                    actual_diameter = 2 * pc_det.bbox.size.x
                    
                    if 0.8 < estimated_diameter/actual_diameter < 1.2:
                        matched = True
                        confidence = (vis_det.results[0].hypothesis.confidence + 0.9) / 2
                        break
            
            # 创建融合结果
            if matched:
                fused_results.extend([center.x, center.y, center.z, confidence])
            else:
                fused_results.extend([center.x, center.y, center.z, 0.7])  # 降低置信度
        
        # 发布融合结果
        result_msg = Float32MultiArray()
        result_msg.data = fused_results
        self.fused_pub.publish(result_msg)
        self.get_logger().info(f"Published {len(fused_results)//4} fused detections")

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()