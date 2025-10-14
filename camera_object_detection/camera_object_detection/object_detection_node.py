import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# 1. 替换消息类型：使用 BoundingBox2DArray 而非 BoundingBoxes
from vision_msgs.msg import BoundingBox2DArray, BoundingBox2D, Pose2D
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        # 订阅同步后的图像（已修改为 /sync/image）
        self.image_subscription = self.create_subscription(
            Image,
            '/sync/image',
            self.image_callback,
            10
        )
        # 2. 发布器类型改为 BoundingBox2DArray
        self.bbox_publisher = self.create_publisher(
            BoundingBox2DArray,  # 新消息类型：二维边界框数组
            '/cabbage_detections_camera',
            10
        )
        self.bridge = CvBridge()
        # 加载YOLO模型（确保模型路径正确）
        self.model = YOLO("yolov8n.pt")  # 替换为你的模型路径
        self.get_logger().info("目标检测节点已启动，订阅图像话题: /sync/image")

    def image_callback(self, msg):
        try:
            # 将ROS图像转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {str(e)}")
            return

        # 运行YOLO检测（只关注"cabbage"类别，需确保模型训练时包含该类别）
        results = self.model(cv_image)

        # 3. 构建 BoundingBox2DArray 消息（替代原 BoundingBoxes）
        bbox_array = BoundingBox2DArray()
        bbox_array.header = msg.header  # 复用图像时间戳和坐标系
        bbox_array.header.frame_id = 'hik_camera'

        # 遍历检测结果，填充边界框信息
        for result in results:
            for box in result.boxes:
                # 假设模型输出的类别名包含"cabbage"，根据实际情况调整
                if result.names[int(box.cls)] == 'cabbage':
                    # 4. 创建 BoundingBox2D 消息（单个边界框）
                    bbox_2d = BoundingBox2D()
                    
                    # 计算中心点（BoundingBox2D 用中心点+尺寸表示）
                    center = Pose2D()
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    center.x = (xmin + xmax) / 2.0  # 中心点x坐标
                    center.y = (ymin + ymax) / 2.0  # 中心点y坐标
                    bbox_2d.center = center
                    
                    # 计算尺寸
                    bbox_2d.size_x = xmax - xmin  # 宽度
                    bbox_2d.size_y = ymax - ymin  # 高度
                    
                    # 添加到边界框数组
                    bbox_array.boxes.append(bbox_2d)

        # 发布边界框消息
        self.bbox_publisher.publish(bbox_array)
        self.get_logger().debug(f"发布 {len(bbox_array.boxes)} 个甘蓝边界框")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

