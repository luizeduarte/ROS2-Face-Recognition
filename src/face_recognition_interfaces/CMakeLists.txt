cmake_minimum_required(VERSION 3.8)
project(face_recognition_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FaceBox.msg"
  "msg/FaceDetections.msg"
  "msg/FaceEmbedding.msg"
  "msg/FaceEmbeddings.msg"
  "msg/RecognizedFace.msg"
  "msg/RecognitionResult.msg"
  "srv/EnrollFace.srv"
  DEPENDENCIES std_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
