import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
import numpy as np
from scipy.spatial.transform import Rotation as R

class PointCloudCropBandNode(Node):
    def __init__(self):
        super().__init__('crop_band_node')
        
        # 1. 相机内参（用户提供，固定不变）
        self.fx = 1819.69314  # x轴焦距
        self.cx = 797.44357   # x轴光学中心
        
        # 2. 相机→激光雷达外参（用户提供，固定不变）
        T_lidar_camera = [
            -0.004561495455514472, -0.06369779020816331, -0.022297855575373768,
            -0.48614771747711283, 0.5067882233718053, -0.46423578059208637, 0.5397325573480806
        ]
        self.tx, self.ty, self.tz = T_lidar_camera[:3]  # 平移向量
        self.qx, self.qy, self.qz, self.qw = T_lidar_camera[3:]  # 旋转四元数
        
        # 3. 核心作物带配置（极简：仅Y轴±10cm，无多余参数）
        self.band_width = 0.2  # 总宽度20cm（固定，后续完善可开放）
        self.half_width = self.band_width / 2  # 单侧10cm
        
        # 4. 数据存储（实时更新Y轴过滤中心）
        self.lidar_y_center = None
        
        # 5. 话题订阅与发布（固定话题名，适配现有系统）
        # 订阅：作物中心线（来自目标检测节点）
        self.center_line_sub = self.create_subscription(
            String, '/cabbage_center_line', self.center_line_cb, 10
        )
        # 订阅：原始点云（Livox雷达，垂直向下照射）
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/livox/lidar', self.pointcloud_cb, 5
        )
        # 发布：过滤后点云（20cm宽Y轴条带）
        self.cropped_pub = self.create_publisher(
            PointCloud2, '/cropped_livox_lidar', 5
        )
        
        # 启动日志（简洁明了）
        self.get_logger().info("✅ 点云Y轴过滤节点启动成功！")
        self.get_logger().info(f"📌 运行配置：仅保留Y轴±{self.half_width*100:.0f}cm条带（X/Z全保留）")

    def center_line_cb(self, msg):
        """回调：解析中心线u1，映射为雷达Y轴过滤中心"""
        if not msg.data:
            self.lidar_y_center = None
            self.get_logger().warn("⚠️  未收到有效中心线数据")
            return
        
        try:
            # 极简处理：只取u1（竖直中心线u1=u2，无需平均，减少计算）
            u1, _, _, _ = map(float, msg.data.split(','))
            u_center = u1  # 直接用u1作为中心线X像素坐标
            
            # 核心映射：像素u → 相机X偏移 → 雷达Y轴中心（无额外计算）
            xc_offset_ratio = (u_center - self.cx) / self.fx  # 像素相对偏移比例
            rotation = R.from_quat([self.qx, self.qy, self.qz, self.qw])  # 外参旋转
            R_matrix = rotation.as_matrix()
            rotated_offset = R_matrix @ np.array([xc_offset_ratio, 0.0, 0.0])  # 仅映射X轴偏移
            self.lidar_y_center = self.ty + rotated_offset[1]  # 最终Y轴过滤中心
            
            self.get_logger().debug(f"📥 中心线映射完成：u像素={u_center:.2f} → 雷达Y中心={self.lidar_y_center:.3f}m")
        except Exception as e:
            self.get_logger().error(f"❌ 中心线映射失败：{str(e)}（请检查中心线数据格式）")
            self.lidar_y_center = None

    def pointcloud_cb(self, msg: PointCloud2):
        """回调：过滤点云，仅保留Y轴±10cm条带"""
        # 无有效过滤中心时，直接发布原始点云
        if self.lidar_y_center is None:
            self.cropped_pub.publish(msg)
            return
        
        try:
            # 解析点云XYZ坐标（极简解析，无多余处理）
            point_step = msg.point_step  # 每个点的字节数（Livox默认12字节）
            # 二进制数据 → numpy数组（N×3，对应XYZ）
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, point_step // 4)
            pointcloud_xyz = data[:, :3]  # 仅取前3列（XYZ），忽略其他字段
            
        except Exception as e:
            self.get_logger().error(f"❌ 点云解析失败：{str(e)}")
            self.cropped_pub.publish(msg)
            return
        
        # 核心过滤：仅保留Y轴±10cm范围内的点（X/Z全保留，计算量最小）
        filter_mask = (
            (pointcloud_xyz[:, 1] >= self.lidar_y_center - self.half_width) &  # Y轴下限
            (pointcloud_xyz[:, 1] <= self.lidar_y_center + self.half_width)    # Y轴上限
        )
        cropped_xyz = pointcloud_xyz[filter_mask]  # 过滤后的点云
        
        # 打印过滤统计日志（每帧输出，直观查看效果）
        self.get_logger().info(
            f"🌾 点云过滤完成：原始{len(pointcloud_xyz):,}个点 → 保留{len(cropped_xyz):,}个点 "
            f"（Y轴过滤范围：[{self.lidar_y_center - self.half_width:.3f}, {self.lidar_y_center + self.half_width:.3f}]m）"
        )
        
        # 构造过滤后的PointCloud2消息（格式与原始点云一致）
        cropped_msg = PointCloud2()
        cropped_msg.header = msg.header  # 沿用原始时间戳和坐标系（确保对齐）
        cropped_msg.height = 1  # 无序点云（1行）
        cropped_msg.width = len(cropped_xyz)  # 过滤后的点数量
        cropped_msg.is_dense = False  # 允许无效点（实际无）
        cropped_msg.point_step = 12  # 每个点12字节（XYZ各4字节float32）
        cropped_msg.row_step = cropped_msg.point_step * cropped_msg.width  # 每行总字节数
        cropped_msg.data = cropped_xyz.tobytes()  # 转换为二进制数据
        # 点云字段定义（与Livox雷达一致，确保RViz正常显示）
        cropped_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        # 发布过滤后点云
        self.cropped_pub.publish(cropped_msg)

def main(args=None):
    """节点入口函数（极简实现）"""
    rclpy.init(args=args)  # 初始化ROS 2上下文
    node = PointCloudCropBandNode()  # 创建节点实例
    try:
        rclpy.spin(node)  # 自旋节点（持续运行）
    except KeyboardInterrupt:
        node.get_logger().info("👋 收到关闭信号，节点正在退出...")
    finally:
        node.destroy_node()  # 销毁节点
        rclpy.shutdown()    # 关闭ROS 2上下文

if __name__ == '__main__':
    main()

