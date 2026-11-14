from glob import glob

from setuptools import find_packages, setup

package_name = 'face_recognition_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (f'share/{package_name}/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Luize Duarte',
    description='Real-time face detection, embedding and open-set recognition for ROS 2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'face_detection = face_recognition_ros.face_detection_node:main',
            'face_embedding = face_recognition_ros.face_embedding_node:main',
            'face_recognition = face_recognition_ros.face_recognition_node:main',
            'face_visualizer = face_recognition_ros.visualizer_node:main',
            'enroll_from_images = face_recognition_ros.enroll_from_images:main',
        ],
    },
)
