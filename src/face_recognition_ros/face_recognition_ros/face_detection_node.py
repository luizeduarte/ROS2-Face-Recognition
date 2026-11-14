"""Detects faces in camera frames with YuNet.

Subscribes:  image                (sensor_msgs/Image)
Publishes:   faces/detections     (face_recognition_interfaces/FaceDetections)
"""

import rclpy
from face_recognition_interfaces.msg import FaceDetections
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from face_recognition_ros.conversions import detection_to_msg, image_to_bgr
from face_recognition_ros.models import FaceDetector


class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection')
        model_path = self.declare_parameter('model_path', '').value
        score_threshold = self.declare_parameter('score_threshold', 0.8).value

        self.detector = FaceDetector(model_path or None, score_threshold=score_threshold)

        # Sensor-data QoS (best effort, depth 1): if inference is slower than the camera,
        # stale frames are dropped instead of queueing up and adding latency.
        self.create_subscription(Image, 'image', self.on_image, qos_profile_sensor_data)
        self.pub = self.create_publisher(FaceDetections, 'faces/detections', 10)
        self.get_logger().info('Face detection ready')

    def on_image(self, msg: Image):
        frame = image_to_bgr(msg)
        detections = self.detector.detect(frame)

        # Published even when empty, so downstream nodes (and the visualizer) keep
        # advancing frame by frame.
        out = FaceDetections()
        out.header = msg.header
        # Forward the exact frame the boxes refer to, so the embedding node never crops
        # boxes from frame N out of frame N+1.
        out.image = msg
        out.faces = [detection_to_msg(d) for d in detections]
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RCLError:
        if rclpy.ok():  # a shutdown in progress is expected; anything else is a real error
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
