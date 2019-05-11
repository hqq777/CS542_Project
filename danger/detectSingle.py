import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import danger



MODEL_DIR = os.path.join(current_dir, "models")
DANGER_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_danger.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(DANGER_MODEL_PATH):
    utils.download_trained_weights(DANGER_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

############################################################
#  Configurations
############################################################

class InferenceConfig(danger.DangerConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

############################################################
#  Create Model and Load Trained Weights
############################################################
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(DANGER_MODEL_PATH, by_name=True)

class_names = ["lighter_iron_shell","lighter_nail_black","knife","power","scissor"]


############################################################
#  Run Object Detection¶
############################################################
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

