#!/usr/bin/env python3


import cv2
import numpy as np
import os
import sys

from tracker.hand_tracker import HandTracker
from tensorflow.keras.models import load_model

# Add the directory where the module is located to sys.path
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module_dir)

print(module_dir)

# ------------
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = module_dir + "/models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = module_dir + "/models/hand_landmark.tflite"
ANCHORS_PATH = module_dir + "/models/anchors.csv"

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)
# ------------
sign_classifier = load_model(module_dir + '/models/model2.h5')

SIGNS = ['one', 'two', 'three', 'four', 'five', 'ok', 'rock', 'thumbs_up']
SIGNS_dict = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'ok': 6,
    'rock': 7,
    'thumbs_up': 8
}

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (125, 125, 0)
THICKNESS = 2
#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]


from collections import deque
running_size = 5
collected_gesture = deque([0])
for i in range(running_size - 1):
    collected_gesture.append(i)
previous_gesture_right = 0


def get_mid_point(points):
    x = (points[0][0]+points[5][0]+points[17][0])/3
    y = (points[0][1]+points[5][1]+points[17][1])/3
    return x, y

def gesture_points_detector(image_detector, images):
    shape = image_detector.shape

    image_vis = images.copy()

    points, _ = detector(image_detector)
    gesture_ml = None

    if points is not None:
        sign_coords = points.flatten() / float(image_vis.shape[0]) - 0.5
        sign_class = sign_classifier.predict(np.expand_dims(sign_coords, axis=0))
        sign_text = SIGNS[sign_class.argmax()]

        for point in points:
            x, y = point
            print(y, shape[0])
            if x > 0 and y > 0 or x < shape[1] and y < shape[0]:
                cv2.circle(image_vis, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)

        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]

            if (x0 > 0 and y0 > 0 or x0 < shape[1] and y0 < shape[0]) or (x1 > 0 and y1 > 0 or x1 < shape[1] and y1 < shape[0]):
                cv2.line(image_vis, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

        # Plot Mid point
        x_mid, y_mid = get_mid_point(points)
        if x_mid > shape[1] and y_mid > shape[0] or x_mid < shape[1] and y_mid < shape[0]:
            cv2.circle(image_vis, (int(x_mid), int(y_mid)), THICKNESS * 2, POINT_COLOR, THICKNESS*2)
        
        cv2.putText(image_vis, sign_text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        gesture_ml = int(SIGNS_dict[sign_text])

        collected_gesture.rotate(1)
        collected_gesture[0] = gesture_ml
        average = sum(collected_gesture) / running_size
        rounding = round(average)
        identical = 1

        previous_gesture_right = 0

        for i in range(len(collected_gesture)):
            for j in range(len(collected_gesture)):
                if collected_gesture[i] == collected_gesture[j]:
                    identical = identical * 1
                else:
                    identical = identical * 0

        current_gesture_right = gesture_ml

        if (current_gesture_right != previous_gesture_right) and identical:
            previous_gesture_right = current_gesture_right
            # print('I get command ', current_gesture_right, 'for drones!')

    return points, image_vis, gesture_ml