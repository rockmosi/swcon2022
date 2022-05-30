"""
ref: https://gist.github.com/vdalv/321876a7076caaa771d47216f382cba5
This script reads PascalVOC xml files, and converts them to YOLO txt files.

Note: This script was written and tested on Ubuntu. YMMV on other OS's.

Disclaimer: This code is a modified version of Joseph Redmon's voc_label.py

Instructions:
Place the convert_voc_to_yolo.py file into your data folder.
Edit the dirs array (line 8) to contain the folders where your images and xmls are located. Note: this script assumes
all of your images are .jpg's (line 13).
Edit the classes array (line 9) to contain all of your classes.
Run the script. Upon running the script, each of the given directories will contain a 'yolo' folder that contains all
of the YOLO txt files. A text file containing all of the image paths will be created in the cwd,
for each given directory.
"""

import glob
import os
import xml.etree.ElementTree as ET
from os import getcwd
from os.path import join

dirs = ['train', 'val']
classes = ['without_mask', 'with_mask']


def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '\*.png'):
        # print("filename=", filename)
        image_list.append(filename)

    return image_list


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def convert_annotation(dir_path, output_path, image_path):
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(dir_path + '/' + basename_no_ext + '.xml')
    out_file = open(output_path + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


cwd = getcwd()
print("cwd=", cwd)
for dir_path in dirs:
    full_dir_path = cwd + '\\' + dir_path
    print("full_dir_path=", full_dir_path)
    output_path = full_dir_path + '/yolo/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    image_paths = getImagesInDir(full_dir_path)
    # print("image_paths=", image_paths)
    list_file = open(full_dir_path + '.txt', 'w')

    for image_path in image_paths:
        list_file.write(image_path + '\n')
        convert_annotation(full_dir_path, output_path, image_path)
    list_file.close()

    print("Finished processing: " + dir_path)
