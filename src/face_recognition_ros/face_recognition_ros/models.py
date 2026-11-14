"""Face detection (YuNet) and embedding (SFace) models from the OpenCV Model Zoo.

Both are small ONNX models run through OpenCV's DNN module, so the pipeline runs in
real time on a CPU with no PyTorch/TensorFlow dependency. This module has no ROS
dependency.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_ZOO = 'https://github.com/opencv/opencv_zoo/raw/main/models'
MODELS = {
    'yunet': (
        f'{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx',
        '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
    ),
    'sface': (
        f'{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx',
        '0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79',
    ),
}


def default_model_dir() -> Path:
    cache = os.environ.get('XDG_CACHE_HOME', '~/.cache')
    return Path(cache).expanduser() / 'face_recognition_ros'


def ensure_model(name: str, model_dir: str | Path | None = None) -> Path:
    """Return the local path of a model, downloading and verifying it if needed."""
    url, sha256 = MODELS[name]
    path = Path(model_dir or default_model_dir()).expanduser() / Path(url).name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.part')
    urllib.request.urlretrieve(url, tmp)
    digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if digest != sha256:
        tmp.unlink()
        raise RuntimeError(f'checksum mismatch for {url}: got {digest}')
    tmp.replace(path)
    return path


@dataclass(frozen=True)
class Detection:
    """One face: box (x, y, w, h), confidence and 5 landmarks, in pixel coordinates."""

    box: tuple[float, float, float, float]
    confidence: float
    landmarks: tuple[float, ...]  # x0, y0, ..., x4, y4

    def to_yunet_row(self) -> np.ndarray:
        """The 15-value layout that cv2.FaceRecognizerSF.alignCrop expects."""
        return np.array([*self.box, *self.landmarks, self.confidence], dtype=np.float32)


class FaceDetector:
    def __init__(self, model_path: str | Path | None = None, score_threshold: float = 0.8,
                 nms_threshold: float = 0.3, top_k: int = 5000):
        model_path = str(model_path or ensure_model('yunet'))
        self._net = cv2.FaceDetectorYN.create(
            model_path, '', (320, 320), score_threshold, nms_threshold, top_k)
        self._size: tuple[int, int] | None = None

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, faces = self._net.detect(image_bgr)
        if faces is None:
            return []
        return [
            Detection(box=tuple(map(float, f[0:4])), confidence=float(f[14]),
                      landmarks=tuple(map(float, f[4:14])))
            for f in faces
        ]


class FaceEmbedder:
    """Aligns a face with its landmarks and maps it to a 128-d identity embedding."""

    dim = 128

    def __init__(self, model_path: str | Path | None = None):
        model_path = str(model_path or ensure_model('sface'))
        self._net = cv2.FaceRecognizerSF.create(model_path, '')

    def embed(self, image_bgr: np.ndarray, detection: Detection) -> np.ndarray:
        aligned = self._net.alignCrop(image_bgr, detection.to_yunet_row())
        return self._net.feature(aligned).flatten().astype(np.float32)
