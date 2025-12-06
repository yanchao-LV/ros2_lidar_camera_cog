import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from std_msgs.msg import String
from ultralytics import YOLO

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_cb, 10)
        self.bbox_data_pub = self.create_publisher(String, '/cabbage_sprouts_bbox_data', 10)
        self.overlay_pub = self.create_publisher(Image, '/image_with_overlay', 10)
        
        # 【新增：中心线坐标发布器（仅用于点云过滤）】
        self.center_line_pub = self.create_publisher(String, '/cabbage_center_line', 10)
        
        self.distance_sub = self.create_subscription(
            String, 
            '/cabbage_sprouts_with_distance', 
            self.distance_cb, 
            10
        )
        
        self.bridge = CvBridge()
        self.model = YOLO("/home/abc/runs/train/yolov8n_cabbage_sprouts2/weights/best.pt")
        self.get_logger().info("✅ 视觉节点（支持距离+三维坐标显示）启动成功")
        
        self.latest_dist_xyz = {}
        self.match_threshold = 10
        self.print_interval = 5
        self.frame_count = 0

    def distance_cb(self, msg):
        try:
            dist_xyz_dict = {}
            target_list = msg.data.split(';')
            for target in target_list:
                target = target.strip()
                if not target:
                    continue
                parts = list(map(float, target.split(',')))
                if len(parts) != 8:
                    self.get_logger().warn(f"⚠️  数据格式错误（应为8个字段）：{target}")
                    continue
                cx_pixel, cy_pixel, _, _, distance, x, y, z = parts
                key = (round(cx_pixel), round(cy_pixel))
                dist_xyz_dict[key] = (distance, x, y, z)
            self.latest_dist_xyz = dist_xyz_dict
            
            self.frame_count += 1
            if self.frame_count % self.print_interval == 0:
                self.get_logger().info(f"📥 距离+坐标数据：{list(dist_xyz_dict.items())}")
        except Exception as e:
            self.get_logger().error(f"❌ 数据解析失败：{str(e)}")
            self.latest_dist_xyz = {}

    def find_matching_dist_xyz(self, bbox_cx, bbox_cy):
        min_dist_pixel = float('inf')
        matched_data = None
        for (dist_cx, dist_cy), (dist, x, y, z) in self.latest_dist_xyz.items():
            pixel_dist = np.sqrt( (bbox_cx - dist_cx)**2 + (bbox_cy - dist_cy)** 2 )
            if pixel_dist < self.match_threshold and pixel_dist < min_dist_pixel:
                min_dist_pixel = pixel_dist
                matched_data = (dist, x, y, z)
                if self.frame_count % self.print_interval == 0:
                    self.get_logger().info(
                        f"🔗 匹配成功：识别框({bbox_cx},{bbox_cy}) | 距离数据({dist_cx},{dist_cy}) "
                        f"| 像素距离{pixel_dist:.1f} | 三维坐标({x:.3f},{y:.3f},{z:.3f})"
                    )
        return matched_data

    def image_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img_h, img_w = cv_img.shape[:2]
        except Exception as e:
            self.get_logger().error(f"❌ 图像转换失败：{str(e)}")
            return

        results = self.model(cv_img)
        centers = []
        bbox_data_list = []
        matched_count = 0

        for r in results:
            for box in r.boxes:
                x1 = float(box.xyxy[0][0])
                y1 = float(box.xyxy[0][1])
                x2 = float(box.xyxy[0][2])
                y2 = float(box.xyxy[0][3])
                
                bbox_cx = round( (x1 + x2) / 2.0 )
                bbox_cy = round( (y1 + y2) / 2.0 )
                centers.append([bbox_cx, bbox_cy])

                cv2.rectangle(cv_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.circle(cv_img, (bbox_cx, bbox_cy), 5, (0, 0, 255), -1)

                matched_data = self.find_matching_dist_xyz(bbox_cx, bbox_cy)
                if matched_data is not None:
                    matched_count += 1
                    distance, x, y, z = matched_data
                    text_dist = f"Dist: {distance:.2f}m"
                    text_xyz = f"XYZ: ({x:.3f}, {y:.3f}, {z:.3f})"
                    text_x = int(x1) + 10
                    text_y_dist = int(y1) - 20 if int(y1) - 20 > 20 else int(y2) + 30
                    text_y_xyz = text_y_dist + 30

                    (w1, h1), _ = cv2.getTextSize(text_dist, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                    cv2.rectangle(
                        cv_img, (text_x - 5, text_y_dist - h1 - 5),
                        (text_x + w1 + 5, text_y_dist + 5), (0, 0, 0), -1
                    )
                    (w2, h2), _ = cv2.getTextSize(text_xyz, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(
                        cv_img, (text_x - 5, text_y_xyz - h2 - 5),
                        (text_x + w2 + 5, text_y_xyz + 5), (0, 0, 0), -1
                    )

                    cv2.putText(
                        cv_img, text_dist, (text_x, text_y_dist),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3
                    )
                    cv2.putText(
                        cv_img, text_xyz, (text_x, text_y_xyz),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
                    )

                size_x = x2 - x1
                size_y = y2 - y1
                bbox_data = f"{(x1+x2)/2.0:.2f},{(y1+y2)/2.0:.2f},{size_x:.2f},{size_y:.2f}"
                bbox_data_list.append(bbox_data)

        if bbox_data_list:
            self.bbox_data_pub.publish(String(data=";".join(bbox_data_list)))

        # 【核心优化：拟合线优先竖直（仅修改这部分，其他不变）】
        if len(centers) >= 2:
            centers_np = np.array(centers, dtype=np.float32)
            x_coords = centers_np[:, 0]  # 所有中心点的x坐标
            # 关键：取x坐标的平均值作为竖直直线的固定x值（保证尽量竖直）
            cx_mean = np.mean(x_coords)
            # 限制x值在图像范围内（避免极端情况）
            cx_mean = np.clip(cx_mean, 50, img_w - 50)
            
            # 绘制竖直拟合线（贯穿全图，和原版视觉一致）
            cv2.line(cv_img, (int(cx_mean), 0), (int(cx_mean), img_h), (0, 0, 255), 3)

            # 【新增：发布中心区域的两个竖直点（供点云使用）】
            margin_ratio = 0.4  # 40%边缘留白，取y轴中心区域
            y_pub_start = img_h * margin_ratio  # 中心区域起点y（图像40%处）
            y_pub_end = img_h * (1 - margin_ratio)  # 中心区域终点y（图像60%处）
            # 竖直直线x坐标固定为cx_mean，y取中心区域
            line_data = f"{cx_mean:.2f},{y_pub_start:.2f},{cx_mean:.2f},{y_pub_end:.2f}"
            self.center_line_pub.publish(String(data=line_data))
            self.get_logger().debug(f"📤 发布竖直中心线点：{line_data}")
        else:
            # 目标不足时发布空消息
            self.center_line_pub.publish(String(data=""))

        self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))
        self.get_logger().info(
            f"📊 检测到 {len(centers)} 个目标 | 匹配到 {matched_count} 个（距离+坐标）数据", 
            throttle_duration_sec=1
        )

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("👋 节点关闭")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

