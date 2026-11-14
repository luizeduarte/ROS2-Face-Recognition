# ROS 2 Face Recognition

Real-time face detection and **open-set** recognition as a modular ROS 2 node pipeline.
It runs on a plain CPU (≈ 8 ms per face per frame), recognises people it has been shown
once, and answers **"unknown"** for everybody else.

```
camera ──▶ face_detection ──▶ face_embedding ──▶ face_recognition ──▶ faces/recognitions
 /image      YuNet              SFace (128-d)      cosine match           (name + score)
                │                                       ▲
                └──────────▶ face_visualizer ◀──────────┘──▶ faces/annotated
                                                        enroll_face (service)
```

## Highlights

- **Open-set identification.** Embeddings are compared by cosine similarity against an
  enrolled library with a threshold, so strangers are reported as `unknown`. A closed-set
  classifier (such as an SVM over the known people) cannot do this: it always names someone.
- **Enrol with one call, no retraining.** `ros2 service call /enroll_face …` adds the face
  currently in view; adding a person is a single row appended to the library.
- **Frame-consistent messages.** The detector forwards the exact frame its boxes refer to, so
  embeddings are never cropped from a different frame than the one that was detected.
- **Landmark alignment.** Faces are aligned with the five detected landmarks before embedding.
- **Low latency under load.** The camera subscription uses best-effort, sensor-data QoS, so a
  slow consumer drops stale frames instead of building a queue.
- **Light dependencies.** Models are ONNX files run through OpenCV (no PyTorch / TensorFlow),
  downloaded on first use and checked against a SHA-256.
- **Tested.** Unit tests for the matching logic, image conversion and the models on real
  photos, plus an end-to-end check through the launched ROS graph.

## Quick start

Requires ROS 2 and OpenCV ≥ 4.8 with NumPy. Developed and tested on ROS 2 Lyrical /
Ubuntu; the CI workflow targets Jazzy.

```bash
git clone https://github.com/luizeduarte/ROS2-Face-Recognition.git
cd ROS2-Face-Recognition
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
source install/setup.bash

ros2 launch face_recognition_ros face_recognition.launch.py
```

This starts the webcam, the pipeline and a window with the annotated video. Use
`camera:=false image_topic:=/my/camera/image` for your own camera driver, or
`viewer:=false` for headless use.

### Enrol people

With exactly one person in front of the camera:

```bash
ros2 service call /enroll_face face_recognition_interfaces/srv/EnrollFace "{label: ada}"
```

Enrol a person a few times (different angles and lighting) for more robust matching. Or
build a library offline from a folder with one sub-folder of photos per person:

```bash
ros2 run face_recognition_ros enroll_from_images people/
```

The library is stored at `~/.local/share/face_recognition_ros/library.npz`.

## Interfaces

| Topic / service | Type | Notes |
| --- | --- | --- |
| `image` (in) | `sensor_msgs/Image` | 8-bit `bgr8`, `rgb8`, `mono8`, `bgra8` or `rgba8` |
| `faces/detections` | `FaceDetections` | header, source image, boxes with confidence and 5 landmarks |
| `faces/embeddings` | `FaceEmbeddings` | one 128-d embedding per face |
| `faces/recognitions` | `RecognitionResult` | label (or `unknown`) and cosine score per face |
| `faces/annotated` | `sensor_msgs/Image` | frame with boxes, names and scores |
| `enroll_face` | `EnrollFace` | enrol the single face in view under a label |

Parameters live in [`config/params.yaml`](src/face_recognition_ros/config/params.yaml);
the main one is `similarity_threshold` (default `0.363`, the value recommended for SFace;
raise it to trade missed matches for fewer false ones).

## Design notes

- **Models.** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
  detects faces and landmarks; [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
  produces the embeddings. Both come from the OpenCV Model Zoo.
- **Package split.** `face_recognition_interfaces` holds only messages and services;
  `face_recognition_ros` holds the nodes. The detection, embedding and matching logic
  (`models.py`, `face_library.py`) has no ROS dependency, so it is unit-testable and reusable.
- **No `cv_bridge`.** The packaged `cv_bridge` is built against a specific OpenCV major
  version and fails when the Python `cv2` differs (this happened while developing this
  project), so image conversion is a small NumPy function in `conversions.py`.

## Performance

Measured on an Intel Core 7 240H (CPU only, 640×480, one face):

| Stage | Latency |
| --- | --- |
| Detection (YuNet) | 5.0 ms |
| Alignment + embedding (SFace) | 3.2 ms |

## Tests

```bash
python3 -m pytest src/face_recognition_ros/test -v
```

The model tests download two sample photos and the ONNX models on first run and are
skipped when offline.

## Limitations

- Face recognition is not a security mechanism: there is no liveness / anti-spoofing check,
  so a printed photo can match.
- Accuracy depends on the enrolment photos and on the threshold, which has not been tuned on
  your camera or population. Face recognition models can also perform unevenly across
  demographic groups; evaluate before relying on this for anything that matters.
- The face library is biometric data. It is git-ignored and stored unencrypted.

## History

This project began as a PyTorch (MTCNN + FaceNet + SVM) prototype that passed faces between
nodes as strings and was never fully wired together. It was rewritten around the design above.

## License

[MIT](LICENSE)
