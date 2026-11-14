import numpy as np
import pytest

pytest.importorskip('sensor_msgs')

from face_recognition_ros.conversions import bgr_to_image, image_to_bgr  # noqa: E402


def test_bgr_roundtrip():
    frame = np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    assert np.array_equal(image_to_bgr(bgr_to_image(frame)), frame)


def test_rgb_is_swapped_to_bgr_and_row_padding_ignored():
    msg = bgr_to_image(np.zeros((2, 3, 3), np.uint8))
    msg.encoding = 'rgb8'
    msg.step = 12  # 3 px * 3 bytes + 3 bytes of padding per row
    rows = np.zeros((2, 12), np.uint8)
    rows[:, 0] = 255  # red channel of the first pixel in each row
    msg.data = rows.tobytes()
    out = image_to_bgr(msg)
    assert out.shape == (2, 3, 3)
    assert tuple(out[0, 0]) == (0, 0, 255)


def test_unsupported_encoding():
    msg = bgr_to_image(np.zeros((2, 2, 3), np.uint8))
    msg.encoding = '16UC1'
    with pytest.raises(ValueError):
        image_to_bgr(msg)
