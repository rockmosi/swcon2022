import tflite_runtime.interpreter as tflite
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
#allocate the tensors
interpreter.allocate_tensors()

