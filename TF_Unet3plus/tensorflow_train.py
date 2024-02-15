# Fix version reminded , This is the last version with GPU!!!!!
# ** INSTALL GUIDE**
# Pre- download anaconda first
# 1. conda create -n py310 python=3.10
# 2. conda activate py310
# 3. conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# 4. python -m pip install "tensorflow==2.10"
#
# ** Test GPU **
# https://blog.csdn.net/weixin_43051346/article/details/113637289#_1
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

# Remove these lines if you want TensorFlow to automatically choose an available GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
import re
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Ensure that you have the necessary GPU dependencies installed
# If you use TensorFlow with GPU support, it should be able to use the GPU automatically
# Make sure to have a compatible GPU and the appropriate CUDA and cuDNN versions installed

import json
from ResUnet3_Plus import ResUnet3_Plus
import Dataloader
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import OneHotMeanIoU
from sklearn.metrics import confusion_matrix

lbl_size_tracker = []

leaf = 0
stem = 1
Label_classes = [leaf, stem]
# Note that : Label is one hot code.
import numpy as np
# Define a custom metric function for F1 score

# Define a custom callback to print the best IoU

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from indicators import PrintBestIoUCallback, f1

if __name__ == '__main__':
    print("Version:", tf.__version__)

    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("GPU is NOT AVAILABLE. Make sure your TensorFlow installation supports GPU.")
    else:
        print(f"GPU is available. Found {len(physical_devices)} GPU(s).")

    frameObjTrain = {'img': [],
                     'mask': []
                     }

    # Instantiate the model on GPU
    with tf.device('/GPU:0'):
        inputs = tf.keras.layers.Input((480, 640, 3))
        Net = ResUnet3_Plus(inputs).Model
        #  OneHotIoU(num_classes=3, target_class_ids=[0])
        Net.compile(optimizer='adam',
                    loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy',
                          'categorical_crossentropy', 'categorical_crossentropy'],
                    loss_weights=[0.15, 0.15, 0.15, 0.15, 0.4],
                    metrics=['accuracy', OneHotMeanIoU(num_classes=3), f1])

        # Define ModelCheckpoint callback
        checkpoint = ModelCheckpoint(
            filepath='./ResUnet.h5',
            monitor='activation_24_one_hot_mean_io_u',  # change this same as terminal if changed IoU method.
            save_best_only=True,
            verbose=1,
            save_weights_only=False  # must False
        )
        # Define custom callback to print best IoU
        print_best_iou_callback = PrintBestIoUCallback()

        print("Load Data...")
        root_path = r'../Dataset/debug/'
        img_idx, anno_idx = Dataloader.gerAddress(root_path)
        frameObjTrain = Dataloader.LoadData(img_idx=img_idx, anno_idx=anno_idx)
        print("Load Data Successfully")

        img_shapes = [img.shape for img in frameObjTrain['img']]
        mask_shapes = [mask.shape for mask in frameObjTrain['mask']]

        print("Shapes of 'img' images:", img_shapes[0])
        print("Shapes of 'mask' images:", mask_shapes[0])

        retVal = Net.fit(np.array(frameObjTrain['img']),
                         [np.array(frameObjTrain['mask']), np.array(frameObjTrain['mask']),
                          np.array(frameObjTrain['mask']), np.array(frameObjTrain['mask']),
                          np.array(frameObjTrain['mask'])], epochs=5, verbose=1,
                         batch_size=1, shuffle=True, callbacks=[checkpoint, print_best_iou_callback])

        # Save the model and training history
    # Net.save('./ResUnet.h5')
    with open('history.txt', 'w+') as f:
        f.write(str(retVal.history))
    print("Best IoU:", print_best_iou_callback.best_iou)
    print("Best IoU:", print_best_iou_callback.best_f1)
    # Net.summary()
