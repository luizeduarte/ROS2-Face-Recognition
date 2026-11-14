# Faces detected in a single frame. The source image is carried along so that
# downstream nodes always crop from the exact frame the boxes refer to.
std_msgs/Header header
sensor_msgs/Image image
FaceBox[] faces
