# coding=utf-8
# Created by Rock-hyun Choi at 22. 10. 23.
from face_detector import YoloDetector
import numpy as np
from PIL import Image

model = YoloDetector(target_size=720,gpu=0,min_face=90)
orgimg = np.array(Image.open('test_image.jpg'))
bboxes,points = model.predict(orgimg)
print(bboxes, points)
orgimg = np.array(Image.open('test_image2.jpg'))
bboxes,points = model.predict(orgimg)
print(bboxes, points)