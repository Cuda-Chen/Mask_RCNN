import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import cv2 as cv
from skimage import io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.fish import fish

config = fish.FishConfig()
FISH_DIR = os.path.join(ROOT_DIR, "datasets/fish")

# Load dataset
dataset = fish.FishDataset()
dataset.load_fish(FISH_DIR, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load random image and mask.
#image_id = random.choice(dataset.image_ids)
image_id = 13
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
base = os.path.basename(dataset.image_reference(image_id))
filename = os.path.splitext(base)[0]
print(filename)
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)

# apply mask and crop image
for index in range(mask.shape[2]):
    mask = mask.astype(np.uint8) * 255
    masked = cv.bitwise_and(image, image, mask=mask[:,:,index])
    cropped = masked[bbox[index][0]:bbox[index][2], bbox[index][1]:bbox[index][3]]
    outputfile = filename + "_" + str(index) + ".bmp"
    io.imsave(outputfile, cropped)
