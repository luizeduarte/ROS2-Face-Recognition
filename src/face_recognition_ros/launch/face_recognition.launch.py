"""Launch the full pipeline: camera -> detection -> embedding -> recognition -> visualizer.

    ros2 launch face_recognition_ros face_recognition.launch.py
    ros2 launch face_recognition_ros face_recognition.launch.py camera:=false image_topic:=/my/cam
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PKG = 'face_recognition_ros'


def generate_launch_description():
    params = str(Path(get_package_share_directory(PKG)) / 'config' / 'params.yaml')
    image_topic = LaunchConfiguration('image_topic')
    camera = LaunchConfiguration('camera')
    viewer = LaunchConfiguration('viewer')

    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='true',
                              description='Start a webcam publisher (image_tools/cam2image)'),
        DeclareLaunchArgument('device_id', default_value='0', description='Webcam index'),
        DeclareLaunchArgument('image_topic', default_value='/image',
                              description='Input image topic'),
        DeclareLaunchArgument('viewer', default_value='true',
                              description='Open a window showing the annotated stream'),

        Node(package='image_tools', executable='cam2image', name='camera',
             condition=IfCondition(camera),
             parameters=[{'device_id': LaunchConfiguration('device_id'),
                          'frequency': 30.0, 'width': 640, 'height': 480}],
             remappings=[('image', image_topic)]),

        Node(package=PKG, executable='face_detection', name='face_detection',
             parameters=[params], remappings=[('image', image_topic)]),
        Node(package=PKG, executable='face_embedding', name='face_embedding',
             parameters=[params]),
        Node(package=PKG, executable='face_recognition', name='face_recognition',
             parameters=[params]),
        Node(package=PKG, executable='face_visualizer', name='face_visualizer'),

        Node(package='image_tools', executable='showimage', name='viewer',
             condition=IfCondition(viewer),
             remappings=[('image', 'faces/annotated')]),
    ])
