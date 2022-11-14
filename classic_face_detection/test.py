import cv2
from util import FileManager

# haarcascade 불러오기
face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('./xml/haarcascade_frontalface_alt_tree.xml')
# eye_cascade = cv2.CascadeClassifier('./xml/haarcascade_eye.xml')

print(type(face_cascade))

file_path = "../images/"
file_list = list()

FileManager.search(file_path, file_list)
print(file_list)

# img = cv2.imread('image/selfie.jpg')
# file_path="../images/20221021_frame311.jpg"
for i in range(len(file_list)):

    img = cv2.imread(file_list[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 찾기
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.4, minNeighbors=3)
    # faces = face_cascade.detectMultiScale(image=gray, scaleFactor=10, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 눈 찾기
    # roi_color = img[y:y + h, x:x + w]
    # roi_gray = gray[y:y + h, x:x + w]
    # eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.4, minNeighbors=3)
    # for (ex, ey, ew, eh) in eyes:
    #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("original", img)
    cv2.waitKey(0)


cv2.destroyAllWindows()