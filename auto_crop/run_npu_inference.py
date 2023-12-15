#
# Automatic document crop model accelerated with vx_delegate
#

import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

# Set path to the  VX_delegate Libraries
#
# Delegate path:
DELEGATE_PATH = "./libvx_delegate.so"

# Set path to the TFLite model
#
# Model path:
MODEL_PATH = "./auto_crop.tflite"

# Set path to the input image (for this example)
#
# Image path:
IMAGE_PATH = "./sample.png"

def find_closest_coordinate(point, coordinates):
    distances = np.linalg.norm(coordinates - point, axis=1)
    closest_index = np.argmin(distances)
    return coordinates[closest_index][::-1]

img = cv2.imread(IMAGE_PATH)

y_scale = img.shape[0] / 256
x_scale = img.shape[1] / 192

img = cv2.resize(img, (192, 256))
img = np.expand_dims(img, 0).astype(np.float32) / 255

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

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

seg = output[0] > 0.9
out = np.argwhere(seg)[:,:2]

# calculate anchors
tl = find_closest_coordinate([0, 0], out) * (x_scale, y_scale)
tr = find_closest_coordinate([0, 192 - 1], out) * (x_scale, y_scale)
bl = find_closest_coordinate([256 - 1, 0], out) * (x_scale, y_scale)
br = find_closest_coordinate([256 - 1, 192 - 1], out) * (x_scale, y_scale)

anchor = np.array([tl, tr, br, bl], dtype=np.float32)

x1 = np.linalg.norm(br - bl)
x2 = np.linalg.norm(tr - tl)

y1 = np.linalg.norm(tr - br)
y2 = np.linalg.norm(tl - bl)

w = max(int(x1), int(x2))
h = max(int(y1), int(y2))

dst = np.array([
	[0, 0],
	[w - 1, 0],
	[w - 1, h - 1],
	[0, h - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(anchor, dst)

img = cv2.imread(IMAGE_PATH)
warp = cv2.warpPerspective(img, M, (w, h))

cv2.imwrite('output.png', warp)
