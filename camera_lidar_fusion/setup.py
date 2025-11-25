from setuptools import setup

package_name = 'camera_lidar_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'rclpy', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='abc',
    maintainer_email='2771402115@qq.com',
    description='视觉+激光雷达融合测距（Python版）',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_node = camera_lidar_fusion.fusion_node:main',
        ],
    },
)

