import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('cabbage_detection'),
        'config'
    )
    
    return LaunchDescription([
        # 点云检测节点
        Node(
            package='cabbage_detection',
            executable='pointcloud_detector',
            name='pointcloud_detector',
            parameters=[os.path.join(config_dir, 'params.yaml')],
            output='screen'
        ),
        
        # 融合节点
        Node(
            package='cabbage_detection',
            executable='fusion_node',
            name='fusion_node',
            output='screen'
        ),
        
        # YOLOv8节点 (假设已安装)
        Node(
            package='yolov8_ros',
            executable='yolov8_node',
            name='yolov8_detector',
            parameters=[{
                'model': 'cabbage_yolov8n.pt',
                'input_topic': '/camera/image_raw',
                'output_topic': '/detections/visual'
            }],
            output='screen'
        )
    ])