#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the path to the ZED wrapper launch file
    zed_wrapper_launch = os.path.join(get_package_share_directory('zed_wrapper'), 'launch', 'zed_camera.launch.py')

    return LaunchDescription([
        # **Launch ZED ROS2 Wrapper using IncludeLaunchDescription**
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(zed_wrapper_launch),
            launch_arguments={
                'camera_model': 'zed2i',
                'image_transport_compressed': 'true'  # Enable compressed images
            }.items()
        ),

        # **Start Gesture Recognition Node** (Node-based)
        Node(
            package='gesture_recognition',
            executable='gesture_node',
            name='gesture_recognition',
            output='screen'
        ),

        # **Start Gesture-Based Robot Controller** (Node-based)
        Node(
            package='gesture_recognition',
            executable='gesture_robot_controller',
            name='gesture_robot_controller',
            output='screen'
        ),
    ])