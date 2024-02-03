# ** INSTALL GUIDE**
# Pre- download anaconda first
# 1. conda create -n py310 python=3.10
# 2. conda activate py310
# 3. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# 4. python -m pip install "tensorflow=2.10"
#
# ** Test GPU **
#
# import tensorflow as tf
# tf.config.list_physical_devices('GPU')
#
# tf.test.is_gpu_available()
#
# ==================================

from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET

from matplotlib.patches import Polygon

# Train Part
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import json
from ResUnet3_Plus import ResUnet3_Plus
import Dataloader
lbl_size_tracker = []

leaf = 0
stem = 1
Label_classes = [leaf, stem]


if __name__ == '__main__':

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print(
        "GPU is",
        "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
    frameObjTrain = {'img': [],
                     'mask': []
                     }

    ## instantiating model
    inputs = tf.keras.layers.Input((480, 640, 3))
    Net = ResUnet3_Plus(inputs).Model
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # Assuming your model is named 'Net'
    Net.compile(optimizer='adam',
                loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                      'categorical_crossentropy', 'categorical_crossentropy'],
                loss_weights=[0.15, 0.15, 0.15, 0.15, 0.4],
                metrics=['accuracy'])  # Consider using other metrics like 'dice' or 'iou'

    print("Load Data...")
    root_path = r'../Dataset/train/'
    img_idx, anno_idx = Dataloader.gerAddress(root_path)
    frameObjTrain = Dataloader.LoadData(img_idx=img_idx, anno_idx=anno_idx)
    print("Load Data Successfully")

    # Check shape (for debugging)
    img_shapes = [img.shape for img in frameObjTrain['img']]
    mask_shapes = [mask.shape for mask in frameObjTrain['mask']]
    #print(frameObjTrain['mask'][0][0][0])
    # Print the shapes of images
    print("Shapes of 'img' images:", img_shapes[0])
    # Print the shapes of masks
    print("Shapes of 'mask' images:", mask_shapes[0])

    # Train the model
    retVal = Net.fit(np.array(frameObjTrain['img']),
                     [np.array(frameObjTrain['mask']), np.array(frameObjTrain['mask']), np.array(frameObjTrain['mask']),
                      np.array(frameObjTrain['mask']), np.array(frameObjTrain['mask'])], epochs=100, verbose=1,
                     batch_size=8, shuffle=True)

    # Save the model and training history
    Net.save('./ResUnet.h5')
    with open('history.txt', 'w+') as f:
        f.write(str(retVal.history))

    # Net.summary()
