import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.linalg import inv

# ROS 消息导入（仅用基础消息，无自定义）
from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
from cv_bridge import CvBridge

class ObjectAssociationNode(Node):
    def __init__(self):
        super().__init__("object_association_node")
        
        # ---------------------- 1. 固定参数配置（完全不变！）----------------------
        # 相机内参（fusion_node.py 硬编码值）
        self.fx = 1819.69314
        self.fy = 1820.23443
        self.cx = 797.44357
        self.cy = 603.31308
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

        # 相机到雷达的外参（平移+四元数，完全不变）
        t_cam2lidar = np.array([0.028415790107361963, 0.02703309740070694, -0.11518642647520541])
        q_cam2lidar = np.array([0.48645550966963247, -0.5001898248096832, 0.5210762792559493, -0.49157978748692716])
        
        # 计算雷达到相机的外参（完全不变）
        self.T_lidar2cam = self._get_lidar2cam_transform(t_cam2lidar, q_cam2lidar)

        # 可配置参数（完全不变）
        self.declare_parameter("match_iou_threshold", 0.05)
        self.declare_parameter("image_topic", "/image_raw")
        self.iou_thres = self.get_parameter("match_iou_threshold").value
        self.image_topic = self.get_parameter("image_topic").value

        # ---------------------- 2. 订阅/发布话题（完全不变）----------------------
        self.sub_image = self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.sub_visual_bbox = self.create_subscription(String, "/cabbage_sprouts_bbox_data", self.visual_bbox_cb, 10)
        self.sub_lidar_bbox = self.create_subscription(MarkerArray, "/livox/seedling_bbox", self.lidar_bbox_cb, 10)

        self.pub_fused_image = self.create_publisher(Image, "/fused_image_with_supplement", 10)
        self.pub_association = self.create_publisher(String, "/bbox_association_result", 10)

        # ---------------------- 3. 数据缓存（新增：雷达框ID→三维坐标缓存）----------------------
        self.cv_image = None
        self.visual_bboxes = []  # [(cx, cy, w, h), ...]（像素）
        self.lidar_bboxes = []  # [(id, x, y, z, l, w, h), ...]（雷达系，米）
        self.lidar_id_xyz_cache = {}  # 新增：key=lidar_id, value=(x,y,z) 三维坐标缓存
        self.bridge = CvBridge()
        self.get_logger().info("Python association node started!")  # 去掉中文

    # ---------------------- 工具函数（完全不变）----------------------
    def _get_lidar2cam_transform(self, t_cam2lidar, q_cam2lidar):
        R_cam2lidar = R.from_quat(q_cam2lidar).as_matrix()
        T_cam2lidar = np.eye(4)
        T_cam2lidar[:3, :3] = R_cam2lidar
        T_cam2lidar[:3, 3] = t_cam2lidar
        return inv(T_cam2lidar)

    def lidar_bbox_to_pixel_box(self, x_lidar, y_lidar, z_lidar, l, w, h):
        lidar_point = np.array([x_lidar, y_lidar, z_lidar, 1.0]).reshape(4, 1)
        cam_point = self.T_lidar2cam @ lidar_point
        x_cam, y_cam, z_cam = cam_point[:3].flatten()
        if z_cam <= 0.1:
            return None
        u = (self.fx * x_cam / z_cam) + self.cx
        v = (self.fy * y_cam / z_cam) + self.cy
        w_pixel = int(round((w * self.fx) / z_cam))
        h_pixel = int(round((h * self.fy) / z_cam))
        return (int(round(u)), int(round(v)), w_pixel, h_pixel) if w_pixel >=5 and h_pixel >=5 else None

    def calculate_iou(self, box1, box2):
        x1_1 = box1[0] - box1[2]//2
        y1_1 = box1[1] - box1[3]//2
        x2_1 = box1[0] + box1[2]//2
        y2_1 = box1[1] + box1[3]//2

        x1_2 = box2[0] - box2[2]//2
        y1_2 = box2[1] - box2[3]//2
        x2_2 = box2[0] + box2[2]//2
        y2_2 = box2[1] + box2[3]//2

        inter_area = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        return inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0

    def visual_bbox_cb(self, msg):
        self.visual_bboxes.clear()
        if not msg.data:
            return
        for bbox_str in msg.data.split(';'):
            if not bbox_str.strip():
                continue
            try:
                cx, cy, w, h = map(float, bbox_str.split(','))
                if w <= 0 or h <= 0:
                    self.get_logger().warn(f"Invalid visual bbox size: {bbox_str}")
                    continue
                self.visual_bboxes.append( (cx, cy, w, h) )
            except:
                self.get_logger().warn(f"Invalid visual bbox format: {bbox_str}")

    # ---------------------- 核心修改1：雷达框回调中缓存ID→三维坐标 ----------------------
    def lidar_bbox_cb(self, msg):
        self.lidar_bboxes.clear()
        self.lidar_id_xyz_cache.clear()  # 清空旧缓存，同步更新新数据
        for marker in msg.markers:
            if marker.type == 1 and marker.action == 0:
                if marker.scale.x <= 0 or marker.scale.y <=0 or marker.scale.z <=0:
                    self.get_logger().warn(f"Invalid lidar bbox size: ID={marker.id}")
                    continue
                # 解析雷达框基础信息（原有逻辑不变）
                lidar_id = marker.id
                x = marker.pose.position.x
                y = marker.pose.position.y
                z = marker.pose.position.z
                self.lidar_bboxes.append( (lidar_id, x, y, z, marker.scale.x, marker.scale.y, marker.scale.z) )
                # 新增：缓存当前雷达框的三维坐标（ID映射）
                self.lidar_id_xyz_cache[lidar_id] = (x, y, z)

    # ---------------------- 核心修改2：加大字体+优化行间距 ----------------------
    def image_cb(self, msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {str(e)}")
            return

        # 无框时显示英文提示（字体加大）
        if not self.visual_bboxes and not self.lidar_bboxes:
            fused_img = self.cv_image.copy()
            # 字体加大：fontScale=1.2，thickness=3
            cv2.putText(fused_img, "No Detection", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
            self.pub_fused_image.publish(self.bridge.cv2_to_imgmsg(fused_img, "bgr8"))
            return

        # 点云框→像素框（完全不变）
        lidar_pixel_boxes = []
        for (lidar_id, x, y, z, l, w, h) in self.lidar_bboxes:
            pixel_box = self.lidar_bbox_to_pixel_box(x, y, z, l, w, h)
            if pixel_box:
                lidar_pixel_boxes.append( (lidar_id, *pixel_box, x, y, z) )

        # IOU匹配（完全不变）
        matched_vis_idx = set()
        matched_lidar_ids = set()
        assoc_results = []

        for vis_idx, vis_box in enumerate(self.visual_bboxes):
            max_iou = 0.0
            best_lidar_id = -1
            for (lidar_id, lidar_cx, lidar_cy, lidar_w, lidar_h, x_l, y_l, z_l) in lidar_pixel_boxes:
                iou = self.calculate_iou(vis_box, (lidar_cx, lidar_cy, lidar_w, lidar_h))
                if iou > max_iou and iou > self.iou_thres:
                    max_iou = iou
                    best_lidar_id = lidar_id
            if best_lidar_id != -1:
                assoc_results.append( (vis_idx, best_lidar_id, round(max_iou, 2)) )
                matched_vis_idx.add(vis_idx)
                matched_lidar_ids.add(best_lidar_id)

        # 绘制结果（核心调整：字体加大+行间距优化）
        fused_img = self.cv_image.copy()
        img_h, img_w = fused_img.shape[:2]

        # 1. 匹配成功的视觉框（绿色，字体加大到0.7，粗细2）
        for (vis_idx, lidar_id, iou) in assoc_results:
            try:
                cx, cy, w, h = self.visual_bboxes[vis_idx]
                x1 = int(max(0, cx - w/2))
                y1 = int(max(0, cy - h/2))
                x2 = int(min(img_w-1, cx + w/2))
                y2 = int(min(img_h-1, cy + h/2))
                if x1 >= x2 or y1 >= y2:
                    continue
                # 边框加粗（从2→3）
                cv2.rectangle(fused_img, (x1,y1), (x2,y2), (0,255,0), 3)
                # 从缓存获取三维坐标
                xyz = self.lidar_id_xyz_cache.get(lidar_id, (-1.0, -1.0, -1.0))
                x, y, z = xyz
                # 字体调整：fontScale=0.7，thickness=2；行间距加大到30像素
                label = f"Matched(Vis-{vis_idx}, Lidar-{lidar_id})"
                xyz_text = f"XYZ: ({x:.2f}, {y:.2f}, {z:.2f})" if x != -1.0 else "XYZ: No Data"
                # 第一行：匹配标识
                cv2.putText(fused_img, label, (x1, max(20, y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                # 第二行：三维坐标（行间距30）
                cv2.putText(fused_img, xyz_text, (x1, max(20, y1-40)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            except Exception as e:
                self.get_logger().warn(f"Draw matched bbox failed: {str(e)}")

        # 2. 点云漏检框（红色，字体加大到0.7，粗细2）
        for (lidar_id, lidar_cx, lidar_cy, lidar_w, lidar_h, x_l, y_l, z_l) in lidar_pixel_boxes:
            if lidar_id not in matched_lidar_ids:
                try:
                    x1 = int(max(0, lidar_cx - lidar_w/2))
                    y1 = int(max(0, lidar_cy - lidar_h/2))
                    x2 = int(min(img_w-1, lidar_cx + lidar_w/2))
                    y2 = int(min(img_h-1, lidar_cy + lidar_h/2))
                    if x1 >= x2 or y1 >= y2:
                        continue
                    # 边框加粗（从2→3）
                    cv2.rectangle(fused_img, (x1,y1), (x2,y2), (0,0,255), 3)
                    # 从缓存获取三维坐标
                    xyz = self.lidar_id_xyz_cache.get(lidar_id, (-1.0, -1.0, -1.0))
                    x, y, z = xyz
                    # 计算距离
                    distance = np.sqrt(x**2 + y**2 + z**2) if x != -1.0 else np.nan
                    distance_text = f"Dist: {distance:.2f}m" if not np.isnan(distance) else "Dist: Unknown"
                    xyz_text = f"XYZ: ({x:.2f}, {y:.2f}, {z:.2f})" if x != -1.0 else "XYZ: No Data"
                    # 字体调整：fontScale=0.7，thickness=2；行间距加大到30像素
                    label = f"Lidar-Supplement(ID-{lidar_id})"
                    # 第一行：漏检标识
                    cv2.putText(fused_img, label, (x1, max(20, y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    # 第二行：距离（行间距30）
                    cv2.putText(fused_img, distance_text, (x1, max(20, y1-40)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    # 第三行：三维坐标（行间距30）
                    cv2.putText(fused_img, xyz_text, (x1, max(20, y1-70)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                except Exception as e:
                    self.get_logger().warn(f"Draw lidar bbox failed: {str(e)}")

        # 发布结果（完全不变）
        assoc_str = ";".join([f"{vis_id},{lidar_id},{iou}" for (vis_id, lidar_id, iou) in assoc_results])
        self.pub_association.publish(String(data=assoc_str))
        self.pub_fused_image.publish(self.bridge.cv2_to_imgmsg(fused_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = ObjectAssociationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

