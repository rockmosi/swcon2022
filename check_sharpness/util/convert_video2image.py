import cv2
import os
movie_folder = "D:/data/mask_face/test_data/"
file_name = "20221021.mp4"
movie_path = movie_folder+file_name


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error:Creating directory. ' + directory)


# dir = movie_folder + file_name[:-4]
dir = file_name[:-4]
create_folder(dir)

vidcap = cv2.VideoCapture(movie_path)
success, image = vidcap.read()
count = 0

while success:
    # write_name = dir + "/" + file_name[:-4] + "_frame%d.jpg" % count
    # write_dir = dir + "/"
    write_name = dir + "/" + file_name[:-4] + "_frame%d.jpg" % count
    # write_name = "frame%d.jpg" % count
    cv2.imwrite(write_name, image)     # save frame as JPEG file
    success, image = vidcap.read()
    print(write_name, success)
    count += 1
  
