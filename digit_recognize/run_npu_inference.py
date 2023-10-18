#
# Digit recognization model accelerated with vx_delegate
#

import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Set path to the  VX_delegate Libraries
#
# Delegate path:
DELEGATE_PATH = "./libvx_delegate.so"

# Set path to the TFLite model
#
# Model path:
MODEL_PATH = "./digit_recognize_28.tflite"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./digit7.png"

img = Image.open(IMAGE_PATH).resize((28, 28))
img = img.convert("L")

vx_delegate = tflite.load_delegate(
    library = DELEGATE_PATH,
    options={"logging-severity":"debug"}
)

interpreter = tflite.Interpreter(
    model_path = MODEL_PATH,
    experimental_delegates = [vx_delegate]
)


interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print("Input details\n", input_details)
# print("Output details\n", output_details)

input_type = input_details[0]['dtype']
output_type = output_details[0]['dtype']

np_features = np.array(img)
np_features = np_features.astype(input_type)
np_features = np.expand_dims(np_features, axis=0)
np_features = np.expand_dims(np_features, axis=-1)

interpreter.set_tensor(input_details[0]['index'], np_features)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

prediction = np.argmax(output.astype(output_type)[0])
print('Predicted digit: ', prediction)
