"""Integration tests for the YuNet + SFace models on real photos.

These need network access the first time (models and sample photos are downloaded and
cached), so they are skipped when offline.
"""

import urllib.request

import cv2
import numpy as np
import pytest

from face_recognition_ros.face_library import UNKNOWN, FaceLibrary
from face_recognition_ros.models import FaceDetector, FaceEmbedder, default_model_dir

SAMPLES = 'https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data'


def fetch_image(name):
    path = default_model_dir() / 'samples' / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f'{SAMPLES}/{name}', path)
    return cv2.imread(str(path))


@pytest.fixture(scope='module')
def models():
    try:
        return FaceDetector(), FaceEmbedder()
    except OSError as exc:
        pytest.skip(f'models unavailable: {exc}')


@pytest.fixture(scope='module')
def photos():
    try:
        return {n: fetch_image(f) for n, f in (('messi5', 'messi5.jpg'), ('basketball1', 'basketball1.png'))}
    except OSError as exc:
        pytest.skip(f'sample photos unavailable: {exc}')


def test_detects_one_face_with_landmarks(models, photos):
    detector, _ = models
    faces = detector.detect(photos['messi5'])
    assert len(faces) == 1
    x, y, w, h = faces[0].box
    assert w > 20 and h > 20
    assert faces[0].confidence > 0.9
    assert len(faces[0].landmarks) == 10


def test_no_face_in_blank_image(models):
    detector, _ = models
    assert detector.detect(np.full((480, 640, 3), 127, np.uint8)) == []


def test_embedding_shape(models, photos):
    detector, embedder = models
    image = photos['messi5']
    emb = embedder.embed(image, detector.detect(image)[0])
    assert emb.shape == (FaceEmbedder.dim,)
    assert emb.dtype == np.float32


def test_same_person_matches_and_stranger_does_not(models, photos):
    detector, embedder = models

    def embed(image):
        return embedder.embed(image, detector.detect(image)[0])

    person = photos['messi5']
    # A mirrored, darker, rescaled copy is a different image of the same person.
    variant = cv2.resize(cv2.flip(person, 1), None, fx=0.8, fy=0.8)
    variant = cv2.convertScaleAbs(variant, alpha=0.8, beta=-10)

    library = FaceLibrary()
    library.add('person', embed(person))
    assert library.identify(embed(variant)).label == 'person'
    assert library.identify(embed(photos['basketball1'])).label == UNKNOWN
