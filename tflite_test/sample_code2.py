import tensorflow as tf
import tflite_runtime.interpreter as tflite


Model_Path = "D:/data/mask_face/02_train_model/221102_exp29/weights/best-int8.tflite"

# Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter = tf.lite.Interpreter(model_path=Model_Path)
#allocate the tensors
interpreter.allocate_tensors()

import cv2
# Read the image and decode to a tensor
image_path='Data\\dogs-vs-cats\\test1\\151.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
#Preprocess the image to required size and cast
input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(img,0), dtype=np.float16)