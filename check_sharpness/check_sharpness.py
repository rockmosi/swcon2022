'''
functions taken from http://stackoverflow.com/questions/9965810/is-there-a-way-to-detect-if-an-image-is-blurry
is reference of
http://www.sayonics.com/publications/pertuz_PR2013.pdf
Pertuz 2012: Analysis of focus measure operators for shape-from-focus
code transformed from C++.openCV -> python.cv2
RETURN: focusMeasure - parameter describing sharpness of an image
'''
from __future__ import division

import cv2
import numpy as np
import os
import sys
import importlib.util
# file search module
spec = importlib.util.spec_from_file_location("FileManager", "C:/python/swcon2022/util/FileManager.py")
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)

size = [640, 480]
dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = dir_path + '/data2/'
# img_path = "D:/data/mask_face/220530_tarin_data4yolo_after_sharp/train/images/"
# label_path = "D:/data/mask_face/220530_tarin_data4yolo_after_sharp/train/labels/"
img_path = "./images/"
label_path = "./labels/"
# print(dir_path)
# print(img_path)


def modifiedLaplacian(img):
    ''''LAPM' algorithm (Nayar89)'''
    #error solution https: // github.com / opencv / opencv / issues / 16809
    M = np.array([-1, 2, -1], dtype="float64")
    G = cv2.getGaussianKernel(ksize=3, sigma=-1)
    Lx = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=M, kernelY=G)
    Ly = cv2.sepFilter2D(src=img, ddepth=cv2.CV_64F, kernelX=G, kernelY=M)
    FM = np.abs(Lx) + np.abs(Ly)
    return cv2.mean(FM)[0]


def varianceOfLaplacian(img):
    ''''LAPV' algorithm (Pech2000)'''
    lap = cv2.Laplacian(img, ddepth=-1)  # cv2.cv.CV_64F)
    stdev = cv2.meanStdDev(lap)[1]
    s = stdev[0] ** 2
    return s[0]


def tenengrad(img, ksize=3):
    ''''TENG' algorithm (Krotkov86)'''
    Gx = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    Gy = cv2.Sobel(img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    FM = Gx * Gx + Gy * Gy
    mn = cv2.mean(FM)[0]
    if np.isnan(mn):
        return np.nanmean(FM)
    return mn


def normalizedGraylevelVariance(img):
    ''''GLVN' algorithm (Santos99)'''
    mean, stdev = cv2.meanStdDev(img)
    s = stdev[0] ** 2 / mean[0]
    return s[0]


def box_position_yolo(file_txt, dw, dh):
    crop_images = list()
    with open(file_txt) as f:
        # f = open(file_txt, 'r')

        # read all lines
        lines = f.readlines()
        # remove \n
        lines = [line.rstrip() for line in lines]
        num_labels = len(lines)

        for i in range(num_labels):
            line = lines[i].split()
            print("txt resut = ", line)
            """
            # str to float
            centric_x = float(line[1]) * size[0]
            centric_y = float(line[2]) * size[1]
            width = int(float(line[3]) * size[0])
            height = int(float(line[4]) * size[1])

            xmin = int(centric_x - width / 2)
            ymin = int(centric_y - height / 2)
            xmax = int(centric_x + width / 2)
            ymax = int(centric_y + height / 2)

            print("line=", line[0], type(line[0]), line)
            line = [float(i) for i in line]
            print("line=", line[0], type(line[0]), line)
            # float to int
            line[0] = int(line[0])
            print("line=", line[0], type(line[0]), line)
            if line[1] < 0:
                line[1] = 0
            if line[2] < 0:
                line[2] = 0
            if line[3] < 0:
                line[3] = 0
            if line[4] < 0:
                line[4] = 0
            cropped_img = img[line[2]:line[4], line[1]:line[3]]

            

            # draw box for check
            # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            # crop_image = img[ymin:ymin+height, xmin:xmin+width]
            # crop_image = img[ymin:ymin + ymax, xmin:xmin + xmax]
            """
            # Split string to float
            print("line=", line[0], type(line[0]), line)
            line = [float(i) for i in line]
            print("line=", line[0], type(line[0]), line)
            # float to int
            line[0] = int(line[0])
            print("line=", line[0], type(line[0]), line)
            label = line[0]
            x = line[1]
            y = line[2]
            w = line[3]
            h = line[4]
            # _, x, y, w, h = map(float, line.split(' '))
            print(label, x, y, w, h)
            # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
            # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)

            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            print("l, r, t, b=", l, r, t, b)
            # crop_image = img[ymin:ymin + ymax, xmin:xmin + xmax]
            crop_image = img[t:b, l:r]
            crop_images.append(crop_image)
    # print("crop_images, num_labels=")
    # print(crop_images[0], num_labels)
    return crop_images, num_labels


if __name__ == '__main__':
    file_list = list()
    label_list = list()
    img_list = list()
    txt_list = list()
    fm.search(img_path, file_list)
    fm.search(label_path, label_list)
    # must do it
    file_list.sort()
    label_list.sort()
    # print(file_list)

    # file separation check the file
    for fl in file_list:
        tmp = fl[-3:]
        if tmp == 'png':
            img_list.append(fl)
    for ll in label_list:
        tmp = ll[-3:]
        if tmp == 'txt':
            txt_list.append(ll)
    # print(file_list)
    print(img_list)
    print(txt_list)
    # test focus measure function
    for img_tmp, txt_tmp in zip(img_list, txt_list):
        # print("img_tmp, txt_tmp=", img_tmp, txt_tmp)
        img = cv2.imread(img_tmp)
        cv2.imshow('original', img)
        dw = img.shape[1]
        dh = img.shape[0]
        crop_imgs, num_labels= box_position_yolo(txt_tmp, dw, dh)
        print("crop_imgs, num_labels=", len(crop_imgs), num_labels)
        print("img_tmp)", img_tmp)
        for i in range(num_labels):
            cv2.imshow('label_image', crop_imgs[i])

            LAMP = modifiedLaplacian(crop_imgs[i])
            LAPV = varianceOfLaplacian(crop_imgs[i])
            TENG = tenengrad(crop_imgs[i])
            GLVN = normalizedGraylevelVariance(crop_imgs[i])
            width, height, channel = crop_imgs[i].shape

            result = f"width, height, channel= {width} {height} {channel} " \
                     f"LAMP, LAPV, TENG, GLVN= {round(LAMP, 3):<9} {round(LAPV, 3):<9} {round(TENG, 3):<9} {round(GLVN, 3):<9}"
            print(result)

            cv2.waitKey(0)
    cv2.destroyAllWindows()
