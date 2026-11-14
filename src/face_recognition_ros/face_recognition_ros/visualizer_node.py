"""Draws boxes, names and similarity scores on the camera frame.

Subscribes:  faces/detections     (face_recognition_interfaces/FaceDetections)
             faces/recognitions   (face_recognition_interfaces/RecognitionResult)
Publishes:   faces/annotated      (sensor_msgs/Image)

Both inputs carry the original camera header, so they are paired by exact timestamp.
"""

import cv2
import rclpy
from face_recognition_interfaces.msg import FaceDetections, RecognitionResult
from message_filters import Subscriber, TimeSynchronizer
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from face_recognition_ros.conversions import bgr_to_image, image_to_bgr
from face_recognition_ros.face_library import UNKNOWN

KNOWN_COLOR = (80, 200, 60)
UNKNOWN_COLOR = (60, 60, 230)


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('face_visualizer')
        self.sync = TimeSynchronizer(
            [Subscriber(self, FaceDetections, 'faces/detections'),
             Subscriber(self, RecognitionResult, 'faces/recognitions')],
            queue_size=10)
        self.sync.registerCallback(self.on_frame)
        self.pub = self.create_publisher(Image, 'faces/annotated', 10)

    def on_frame(self, detections: FaceDetections, result: RecognitionResult):
        frame = image_to_bgr(detections.image)
        for face in result.faces:
            b = face.box
            color = UNKNOWN_COLOR if face.label == UNKNOWN else KNOWN_COLOR
            x, y, w, h = int(b.x), int(b.y), int(b.width), int(b.height)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f'{face.label} {face.score:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
        out = bgr_to_image(frame, detections.header)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizerNode()
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
