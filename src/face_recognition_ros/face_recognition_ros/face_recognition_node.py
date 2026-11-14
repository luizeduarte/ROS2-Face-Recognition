"""Matches face embeddings against the enrolled face library.

Subscribes:  faces/embeddings     (face_recognition_interfaces/FaceEmbeddings)
Publishes:   faces/recognitions   (face_recognition_interfaces/RecognitionResult)
Services:    enroll_face          (face_recognition_interfaces/EnrollFace)

The library is persisted to ``library_path`` after every enrollment.
"""

import numpy as np
import rclpy
from face_recognition_interfaces.msg import FaceEmbeddings, RecognitionResult, RecognizedFace
from face_recognition_interfaces.srv import EnrollFace
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from face_recognition_ros.face_library import FaceLibrary


class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__('face_recognition')
        self.library_path = self.declare_parameter(
            'library_path', '~/.local/share/face_recognition_ros/library.npz').value
        threshold = self.declare_parameter('similarity_threshold', 0.363).value
        self.max_enroll_age = self.declare_parameter('max_enroll_age_sec', 1.0).value

        self.library = FaceLibrary.load_or_create(self.library_path, threshold=threshold)
        self.last_frame: FaceEmbeddings | None = None
        self.last_frame_time: Time | None = None

        self.create_subscription(FaceEmbeddings, 'faces/embeddings', self.on_embeddings, 10)
        self.pub = self.create_publisher(RecognitionResult, 'faces/recognitions', 10)
        self.create_service(EnrollFace, 'enroll_face', self.on_enroll)

        self.get_logger().info(
            f'Loaded {len(self.library)} embeddings for {len(self.library.identities)} '
            f'identities from {self.library_path}')

    def on_embeddings(self, msg: FaceEmbeddings):
        self.last_frame = msg
        self.last_frame_time = self.get_clock().now()

        out = RecognitionResult()
        out.header = msg.header
        if msg.faces:
            vectors = np.array([f.embedding for f in msg.faces], dtype=np.float32)
            for face, match in zip(msg.faces, self.library.identify_batch(vectors)):
                out.faces.append(RecognizedFace(box=face.box, label=match.label,
                                                score=match.score))
        self.pub.publish(out)

    def on_enroll(self, request: EnrollFace.Request, response: EnrollFace.Response):
        response.success = False
        label = request.label.strip()
        if self.last_frame is None:
            response.message = 'No frames received yet.'
            return response
        age = (self.get_clock().now() - self.last_frame_time).nanoseconds / 1e9
        if age > self.max_enroll_age:
            response.message = f'Last frame is {age:.1f}s old; is the camera running?'
            return response
        if len(self.last_frame.faces) != 1:
            response.message = (f'Expected exactly one face in view, '
                                f'found {len(self.last_frame.faces)}.')
            return response

        try:
            self.library.add(label, np.array(self.last_frame.faces[0].embedding))
        except ValueError as exc:
            response.message = str(exc)
            return response
        self.library.save(self.library_path)

        response.success = True
        response.message = f'Enrolled "{label}" ({len(self.library)} embeddings in library).'
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
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
