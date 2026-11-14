"""Conversions between the ROS-free model types and face_recognition_interfaces messages."""

from face_recognition_interfaces.msg import FaceBox

from face_recognition_ros.models import Detection


def detection_to_msg(det: Detection) -> FaceBox:
    msg = FaceBox()
    msg.x, msg.y, msg.width, msg.height = det.box
    msg.confidence = det.confidence
    msg.landmarks = list(det.landmarks)
    return msg


def msg_to_detection(msg: FaceBox) -> Detection:
    return Detection(
        box=(msg.x, msg.y, msg.width, msg.height),
        confidence=msg.confidence,
        landmarks=tuple(float(v) for v in msg.landmarks),
    )


# --- sensor_msgs/Image <-> OpenCV BGR -------------------------------------------------
# A tiny numpy-only replacement for cv_bridge. The packaged cv_bridge is compiled
# against one OpenCV major version and breaks when the Python cv2 is another one.

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

_TO_BGR = {
    'bgr8': None,
    'rgb8': cv2.COLOR_RGB2BGR,
    'mono8': cv2.COLOR_GRAY2BGR,
    'bgra8': cv2.COLOR_BGRA2BGR,
    'rgba8': cv2.COLOR_RGBA2BGR,
}
_CHANNELS = {'bgr8': 3, 'rgb8': 3, 'mono8': 1, 'bgra8': 4, 'rgba8': 4}


def image_to_bgr(msg: Image) -> np.ndarray:
    """Convert a sensor_msgs/Image (8-bit bgr/rgb/mono/bgra/rgba) to a BGR ndarray."""
    if msg.encoding not in _CHANNELS:
        raise ValueError(f'unsupported image encoding: {msg.encoding!r}')
    channels = _CHANNELS[msg.encoding]
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    frame = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    code = _TO_BGR[msg.encoding]
    # Copy so the result is writable and contiguous (OpenCV drawing needs that).
    return np.ascontiguousarray(frame if code is None else cv2.cvtColor(frame, code))


def bgr_to_image(frame: np.ndarray, header=None) -> Image:
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = 'bgr8'
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
    return msg
