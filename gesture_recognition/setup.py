from setuptools import setup
import os
from glob import glob

package_name = 'gesture_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vishakh',
    maintainer_email='vishakhcheruparambath22@gmail.com',
    description='Gesture Recognition ROS2 Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_node = gesture_recognition.gesture_node:main',
            'gesture_robot_controller = gesture_recognition.gesture_robot_controller:main',
        ],
    },
    data_files=[
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
)