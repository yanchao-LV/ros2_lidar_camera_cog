from setuptools import setup

package_name = 'pointcloud_crop_band'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    entry_points={
        'console_scripts': [
            'crop_band_node = pointcloud_crop_band.crop_band_node:main',
        ],
    },
)

