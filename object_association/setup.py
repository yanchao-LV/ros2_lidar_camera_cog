from setuptools import setup

package_name = 'object_association'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_dir={'': '.'},  # 关键：指定包路径在当前目录
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Visual-Lidar bbox association (pure Python, no custom msg)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'association_node = object_association.association_node:main',
        ],
    },
)

