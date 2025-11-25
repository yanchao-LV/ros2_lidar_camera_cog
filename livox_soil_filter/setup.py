from setuptools import setup
import os

package_name = 'livox_soil_filter'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'numpy',
        'scipy',
        'scikit-learn'
    ],
    zip_safe=True,
    maintainer='abc',
    maintainer_email='abc@example.com',
    description='菜苗点云过滤+DBSCAN聚类+包围框生成',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seedling_cluster_node = livox_soil_filter.seedling_filter_node:main',
        ],
    },
)

