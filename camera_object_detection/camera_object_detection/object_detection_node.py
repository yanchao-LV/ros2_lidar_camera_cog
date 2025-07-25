import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os  # 用于路径处理
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # 初始化CvBridge（用于ROS图像和OpenCV图像转换）
        self.bridge = CvBridge()
        
        # 处理模型路径（展开~为用户主目录）
        self.declare_parameter('model_path', '~/runs/train/yolov8n_cabbage/weights/best.pt')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        model_path = os.path.expanduser(model_path)  # 关键：将~转换为实际路径
        self.get_logger().info(f"加载模型路径: {model_path}")
        
        # 加载YOLOv8模型
        self.model = YOLO(model_path)
        
        # 订阅相机图像话题（核心：必须有这部分才能接收图像）
        self.image_subscription = self.create_subscription(
            Image,
            'image_raw',  # 与你的相机发布的话题名一致（从ros2 topic list确认）
            self.image_callback,  # 收到图像后调用的函数
            10  # 队列长度
        )
        self.image_subscription  # 防止未使用变量警告

    def image_callback(self, msg):
        """收到相机图像时的回调函数：处理图像并显示检测结果"""
        self.get_logger().info("收到相机图像，开始检测...")  # 调试用：确认收到图像
        try:
            # 将ROS的Image消息转换为OpenCV格式（bgr8是常见的彩色图像格式）
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 使用YOLOv8模型检测物体
            results = self.model(cv_image)
            
            # 可视化检测结果（在图像上画框和标签）
            annotated_frame = results[0].plot()
            
            # 显示检测后的图像
            cv2.imshow('Object Detection', annotated_frame)
            cv2.waitKey(1)  # 必须调用，否则窗口无法刷新（1ms延迟）
            
        except Exception as e:
            self.get_logger().error(f"处理图像出错: {e}")

def main(args=None):
    rclpy.init(args=args)  # 初始化ROS 2
    node = ObjectDetectionNode()  # 创建节点实例
    try:
        rclpy.spin(node)  # 循环运行节点，等待回调
    except KeyboardInterrupt:
        # 按下Ctrl+C时退出
        node.get_logger().info("用户中断，退出节点...")
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

if __name__ == '__main__':
    main()
