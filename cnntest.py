from tensorflow.keras.preprocessing.image import load_img
# we also save images into array format so import img_array library too
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2

model=load_model('VGG16_ears.h5')

imagepath='data/ears/test/0002.png'

image = load_img(imagepath,
                 target_size=(224,224))
image = img_to_array(image) / 255.0
image = np.expand_dims(image,axis=0)

preds=model.predict(image)[0]
(startX,startY,endX,endY)=preds

import imutils

image=cv2.imread(imagepath)
image=imutils.resize(image,width=600)

(h,w)=image.shape[:2]

print(preds)

x1 = int(w * (preds[0]-preds[2]/2))
y1 = int(h * (preds[1]-preds[3]/2))
x2 = int(w * (preds[0]+preds[2]/2))
y2 = int(h * (preds[1]+preds[2]/2))

print(x1,y1,x2,y2)


cv2.rectangle(image,(x1,y1),(x2, y2),(0,255,0),3)
cv2.imshow('Detection', image)
cv2.waitKey()
cv2.destroyAllWindows()
import matplotlib.pyplot as plt
plt.imshow(image)