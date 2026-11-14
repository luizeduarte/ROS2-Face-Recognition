# Axis-aligned face bounding box in pixel coordinates, plus 5 facial landmarks
# (right eye, left eye, nose tip, right mouth corner, left mouth corner) as
# [x0, y0, x1, y1, ..., x4, y4]. Landmarks are used to align the face before embedding.
float32 x
float32 y
float32 width
float32 height
float32 confidence
float32[10] landmarks
