import pickle as pkl
import os
import numpy as np


class FileManager:
    def __init__(self):
        self.data = None


# return folder dir+file name
def search(ori_position, store_list):
    """
    file search from directory and store in list
    :param ori_position:
    :param store_list: store destination
    """
    filenames = os.listdir(ori_position)
    for filename in filenames:
        full_filename = os.path.join(ori_position, filename)
        # print (full_filename)
        store_list.append(full_filename)

    return filenames


def change_name(dir_path:str, change_name:str, start_num:int, image_type='jpg', label_type='txt'):
    """
    change both image file and label txt file
    :param dir_path:
    :param change_name:
    :param start_num:
    """
    # dir_path = "/media/rock/data_disk/data/drone/modified_ir_image by rock" + "/ir1"
    changed_filename = change_name
    start_num = start_num
    store_list = list()
    img_list = list()
    txt_list = list()

    search(dir_path, store_list)
    # must do it
    store_list.sort()
    # print(file_list)
    # file separation
    for fl in store_list:
        tmp = fl[-3:]
        if tmp == image_type:
            img_list.append(fl)
        elif tmp == label_type:
            txt_list.append(fl)

    for img_tmp, txt_tmp in zip(img_list, txt_list):
        print(img_tmp, txt_tmp)
        filename_tmp = changed_filename+str(start_num)
        print(filename_tmp)
        os.rename(img_tmp, dir_path+filename_tmp+'.'+image_type)
        os.rename(txt_tmp, dir_path + filename_tmp + '.' + label_type)
        start_num = start_num+1

# if __name__ == '__main__':

