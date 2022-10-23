# coding=utf-8
# Created by Rock-hyun Choi at 22. 10. 23.
# https://github.com/elyha7/yoloface
from face_detector import YoloDetector
import numpy as np
from PIL import Image
import random
import cv2

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def draw_boxes(detections, image, colors):
    """
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    """
    # for bbox in detections:
    #     left, top, right, bottom = bbox2points(bbox)
    print("detection in draw box=", detections)
    left, top, right, bottom = detections
    cv2.rectangle(image, (left, top), (right, bottom), colors[0], thickness=5)
    return image


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)) for name in names}


model = YoloDetector(target_size=720,gpu=0,min_face=90)
orgimg = np.array(Image.open('test_image.jpg'))
bboxes, points = model.predict(orgimg)
print(bboxes, points)
im = Image.open('test_image3.jpg')
orgimg = np.array(im)
bboxes,points = model.predict(orgimg)

# show original pillow image
# im.show()
# if cv2.waitKey() & 0xFF == ord('q'):
#     pass

# get class color
color_result = class_colors("test")
color_result = (126, 252, 114)
print("color_result=", color_result)

# print(bboxes, points, oriimg)
print(bboxes, points)

print(bboxes[0][0])
print(points[0][0])

# convert to a openCV2 image and convert from RGB to BGR format
opencv_image=cv2.cvtColor(orgimg, cv2.COLOR_RGB2BGR)
image = draw_boxes(bboxes[0][0], opencv_image, colors=color_result)

# draw eye etc
point1 = points[0][0][0]
image = cv2.circle(image, point1, radius=0, color=(0, 0, 255), thickness=5)
image = cv2.circle(image, points[0][0][1], radius=0, color=(0, 0, 255), thickness=5)
image = cv2.circle(image, points[0][0][2], radius=0, color=(0, 0, 255), thickness=5)
image = cv2.circle(image, points[0][0][3], radius=0, color=(0, 0, 255), thickness=5)
image = cv2.circle(image, points[0][0][4], radius=0, color=(0, 0, 255), thickness=5)
cv2.imshow("result", image)
if cv2.waitKey() & 0xFF == ord('q'):
    pass