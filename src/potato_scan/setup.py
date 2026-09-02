import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'potato_scan'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wodndqke',
    maintainer_email='wodndqke@gmail.com',
    description='Eye-in-hand orbit/NBV scanning of a fixed potato with a UR5e',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_accumulator = potato_scan.pointcloud_accumulator:main',
            'scan_controller = potato_scan.scan_controller:main',
            'eye_detector = potato_scan.eye_detector:main',
            'drill_controller = potato_scan.drill_controller:main',
            'handeye_calibration = potato_scan.handeye_calibration:main',
            'force_drill_tuner = potato_scan.force_drill_tuner:main',
        ],
    },
)
