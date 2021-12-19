from tensorflow.keras.models import load_model
import numpy as np

import cv2
from tensorflow.keras.preprocessing.image import load_img
# we also save images into array format so import img_array library too
from tensorflow.keras.preprocessing.image import img_to_array

class Detector:
    def __init__(self):
        self.model = load_model('VGG16_ears.h5')

    def detect(self, imagepath):
        image = load_img(imagepath, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        preds = self.model.predict(image)[0]
        return preds

