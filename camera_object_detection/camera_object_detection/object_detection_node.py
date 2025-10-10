import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBoxes, BoundingBox  # 新增：边界框消息类型
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        self.bridge = CvBridge()
        
        # 模型路径参数
        self.declare_parameter('model_path', '~/runs/train/yolov8n_cabbage/weights/best.pt')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        model_path = os.path.expanduser(model_path)
        self.get_logger().info(f"加载模型路径: {model_path}")
        self.model = YOLO(model_path)
        
        # 订阅相机图像
        self.image_subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10
        )
        
        # 新增：发布边界框话题（供融合节点订阅）
        self.bbox_publisher = self.create_publisher(
            BoundingBoxes,  # 消息类型：多个边界框的集合
            '/cabbage_detections_camera',  # 话题名（需与融合节点订阅的一致）
            10
        )

    def image_callback(self, msg):
        self.get_logger().info("收到相机图像，开始检测...")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(cv_image)  # YOLO检测结果
            
            # 新增：构建边界框消息
            bboxes_msg = BoundingBoxes()
            bboxes_msg.header = msg.header  # 复用图像的时间戳和帧ID（关键：用于时间同步）
            bboxes_msg.header.frame_id = 'hik_camera'  # 相机坐标系ID（需与雷达坐标系校准）
            
            # 遍历检测结果，提取甘蓝的边界框
            for result in results:
                for box in result.boxes:
                    # 假设YOLO模型中甘蓝的类别名为'cabbage'（需与训练时一致）
                    if result.names[int(box.cls)] == 'cabbage':
                        # 构建单个边界框消息
                        bbox = BoundingBox()
                        bbox.class_id = 'cabbage'  # 类别名
                        bbox.confidence = float(box.conf)  # 置信度
                        
                        # 边界框坐标（左上角x, y，右下角x, y）
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        bbox.xmin = int(x1)
                        bbox.ymin = int(y1)
                        bbox.xmax = int(x2)
                        bbox.ymax = int(y2)
                        
                        bboxes_msg.bounding_boxes.append(bbox)  # 添加到集合中
            
            # 发布边界框消息
            self.bbox_publisher.publish(bboxes_msg)
            self.get_logger().info(f"发布{len(bboxes_msg.bounding_boxes)}个甘蓝边界框")
            
            # 可视化检测结果（可选）
            annotated_frame = results[0].plot()
            cv2.imshow('Object Detection', annotated_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"处理图像出错: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断，退出节点...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

