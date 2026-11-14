"""Computes a 128-d SFace identity embedding for every detected face.

Subscribes:  faces/detections     (face_recognition_interfaces/FaceDetections)
Publishes:   faces/embeddings     (face_recognition_interfaces/FaceEmbeddings)
"""

import rclpy
from face_recognition_interfaces.msg import FaceDetections, FaceEmbedding, FaceEmbeddings
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from face_recognition_ros.conversions import image_to_bgr, msg_to_detection
from face_recognition_ros.models import FaceEmbedder


class FaceEmbeddingNode(Node):
    def __init__(self):
        super().__init__('face_embedding')
        model_path = self.declare_parameter('model_path', '').value

        self.embedder = FaceEmbedder(model_path or None)

        self.create_subscription(FaceDetections, 'faces/detections', self.on_detections, 1)
        self.pub = self.create_publisher(FaceEmbeddings, 'faces/embeddings', 10)
        self.get_logger().info('Face embedding ready')

    def on_detections(self, msg: FaceDetections):
        frame = image_to_bgr(msg.image)
        out = FaceEmbeddings()
        out.header = msg.header
        for box in msg.faces:
            face = FaceEmbedding()
            face.box = box
            face.embedding = self.embedder.embed(frame, msg_to_detection(box)).tolist()
            out.faces.append(face)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = FaceEmbeddingNode()
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
