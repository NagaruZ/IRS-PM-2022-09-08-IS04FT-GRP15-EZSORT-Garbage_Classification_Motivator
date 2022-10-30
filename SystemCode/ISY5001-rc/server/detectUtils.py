import io

import noisereduce as nr
import numpy as np
import pyrubberband as pyrb
from keras.models import load_model
# For HMM model and audio feature extraction
from python_speech_features import mfcc
from scipy.io import wavfile
from tensorflow.keras.utils import load_img, img_to_array

from MyCustomUnpickler import MyCustomUnpickler

# load audio models
model_map = {'1': 'carton',
             '2': 'metal',
             '3': 'plastic'}
reloaded_models = []
for key, value in model_map.items():
    fr = open(f"server/models/model_{key}.pkl", 'rb')
    unpickler = MyCustomUnpickler(fr)
    reloaded_models.append((unpickler.load(), value))

# load visual model
model = load_model('server/models/model_5.h5')


def detect_audio(audio: bytes):
    # 1: Select test audio file
    audio = io.BytesIO(audio)
    sampling_freq, audio = wavfile.read(audio)

    # preprocessing:
    # 1) merge two channels into one
    # audio = np.mean(audio, axis=1)
    # 2) adjust length to 0.6s
    # length = audio.shape[0] / sampling_freq
    # target_length = 0.6
    # audio = pyrb.time_stretch(audio, sampling_freq, length / target_length)
    # 3) reduce noice
    audio = nr.reduce_noise(y=audio, sr=sampling_freq, stationary=True)

    # 2: Extract MFCC features
    mfcc_features = mfcc(audio, sampling_freq)
    max_score = None
    output_label = None

    # 3: Iterate through all HMM models and
    #   pick the one with the highest score
    for item in reloaded_models:
        reloaded_model, label = item
        score = reloaded_model.get_score(mfcc_features)
        if max_score is None or score > max_score:
            max_score = score
            output_label = label
    return output_label


def detect_image(image: bytes):
    image = io.BytesIO(image)
    labels = ["Carton", "Metal", "Plastic", "Glass"]
    image = load_img(image, target_size=(224, 224))
    img_tensor = img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    prediction = model.predict(img_tensor)

    label = labels[np.argmax(prediction)]

    return label
