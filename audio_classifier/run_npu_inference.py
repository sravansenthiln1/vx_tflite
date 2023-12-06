#
# Audio classification inference accelerated with vx_delegate
#

import numpy as np
import tflite_runtime.interpreter as tflite
import librosa

# Set path to the  VX_delegate Libraries
#
# Delegate path:
DELEGATE_PATH = "./libvx_delegate.so"

# Set path to the TFLite model
#
# Model path:
MODEL_PATH = "./audio_classifier.tflite"

# Set path to the input audio (for this example)
#
# Audio path:
AUDIO_PATH = "./sample.wav"

# Map the tag output to the appropriate string
TAGS = {
        0:'none',
        1:'hello',
        2:'khadas',
        3:'vim',
        4:'edge',
        5:'tone',
        6:'mind'
}

scale, sr = librosa.load(AUDIO_PATH)
mel_spectrogram = librosa.feature.melspectrogram(y=scale, sr=sr, n_fft=4096, hop_length=512, n_mels=256, fmax=8000)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)

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

interpreter.set_tensor(input_details[0]["index"], [log_mel_spectrogram])
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])

prediction = np.argmax(output_data[0])

print(prediction, TAGS[prediction])
