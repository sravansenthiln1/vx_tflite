#
# Mobilenet_v1 inference accelerated with vx_delegate
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
MODEL_PATH = "./mobilenet_v1_1.0_224_quant.tflite"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./sample.png"

img = Image.open(IMAGE_PATH).resize((224, 224))
img = np.expand_dims(img, 0)

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

interpreter.set_tensor(input_details[0]["index"], img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])

print(np.argmax(output_data))
