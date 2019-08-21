#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Fish trained weights
BALLON_WEIGHTS_PATH = "/path/to/mask_rcnn_fish.h5"  # TODO: update this path
#FISH_WEIGHTS_PATH os.path.join(ROOT_DIR, "")

# load dataset
config = fish.FishConfig()
FISH_DIR = os.path.join(ROOT_DIR, "datasets/fish")

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# Load test dataset
dataset = fish.FishDataset()
dataset.load_fish(FISH_DIR, "test")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set path to fish weights file
#weights_path = "/path/to/mask_rcnn_fish.h5"
weights_path =  "../../mask_rcnn_fish_0500.h5"

# Or, load the last model you trained
#weights_path = model.find_last()

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# Run Detection

for image_id in range(len(dataset.image_ids)):
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, dataset.image_reference(image_id)))

    # get filename
    base = os.path.basename(dataset.image_reference(image_id))
    filename = os.path.splitext(base)[0]

    # Run object detection
    results = model.detect([image], verbose=1)

    r = results[0]
    print("log of ground truth")
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # if no any mask
    if r['masks'].shape[2] <= 0:
        continue

    for index in range(r['masks'].shape[2]):
        mask = r['masks'].astype(np.uint8) * 255
        masked = cv.bitwise_and(image, image, mask=mask[:, :, index])
        cropped = masked[r['rois'][index][0]:r['rois'][index][2], r['rois'][index][1]:r['rois'][index][3]]
        outputfile = filename + "_" + str(index) + ".bmp"
        io.imsave(outputfile, cropped)
