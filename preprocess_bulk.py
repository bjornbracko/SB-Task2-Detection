import cv2
import numpy as np
import glob
import os
from pathlib import Path
import json
import sklearn.metrics
from preprocessing.preprocess import Preprocess
from metrics.evaluation import Evaluation
import matplotlib.pyplot as plt

DIR = 'processed/bc+ee/'

preprocess = Preprocess()
im_list = sorted(glob.glob('data/ears/test/*.png', recursive=True))

for im_name in im_list:
    img = cv2.imread(im_name)

# Apply some preprocessing

## Histogram Equalalization
    #img = preprocess.histogram_equlization_rgb(img) # This one makes VJ worse

## Brightness Correction
    img = preprocess.brightness_correction(img)

## Edge Enhancment
    img = preprocess.edge_enhancment(img)
    filename = DIR + im_name.split('/')[-1]


    cv2.imwrite(filename, img)