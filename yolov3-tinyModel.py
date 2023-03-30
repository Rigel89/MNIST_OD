#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#================================================================
#================================================================
#
#   File name   : yolov3-tinyModel.py
#   Author      : Rigel89
#   Created date: 24/03/23
#   GitHub      : 
#   Description : assenbling yolov3 tinny
#
#================================================================

"""
Created on Sat Mar 25 04:25:31 2023

@author: javi
"""

#%% Importing libraries
from yoloGenerator.yoloNN import blockGenerator, yoloReshape
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from yoloGenerator.config import *


ANCHORS = np.array(ANCHORS)
STRIDES = np.array(YOLO_STRIDES)

#%% NN basic blocks

def darknet19(input_shape):
    # Creating the blocks
    blocks_dict1 = {'mdarkB1': {'b1': [16, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'mdarkB2': {'b1': [32, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'mdarkB3': {'b1': [64, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'mdarkB4': {'b1': [128, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'darkB5': {'b1': [256, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    blocks_dict2 = {'darkB6': {'b1': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    blocks_dict3 = {'darkB7': {'b1': [1024, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    
    # Generating the NN
    inputs = Input(shape=input_shape)
    darknet19_tinny1 = blockGenerator((416,416,1), blocks_dict1)
    darknet19_tinny2 = blockGenerator((13,13,1), blocks_dict2)
    darknet19_tinny3 = blockGenerator((13,13,1), blocks_dict3)
    out1=darknet19_tinny1(inputs)
    out2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                      padding='same',
                      name='Mpooling_darknetBreak')(out1)
    out2=darknet19_tinny2(out2)
    out2=MaxPooling2D(pool_size=(2, 2), strides=(1, 1),
                      padding='same',
                      name='darknet19_last_Maxpool')(out2)
    out2=darknet19_tinny3(out2)
    out1, out2 = Model(inputs=[inputs], outputs=[out1,out2])
    return out1, out2


def YOLOv3_tiny(input_shape, no_class, no_anchors):
    blocks_dict3 = {'yolotinyB1': {'b1': [256, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    blocks_dict4 = {'yolotinyB2': {'b1': [512, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                                   'l2': [no_anchors*(no_class+5), (3, 3), (1, 1), 'same', None]}}
    blocks_dict5 = {'yolotinyB3': {'b1': [128, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    blocks_dict6 = {'yolotinyB4': {'b1': [256, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                                   'l2': [no_anchors*(no_class+5), (1, 1), (1, 1), 'same', None]}}
    out1, out2 = darknet19(input_shape)
    yolofaseL1 = blockGenerator((13,13,1), blocks_dict3)
    out2 = yolofaseL1(out2)
    yolofaseL2 = blockGenerator((13,13,1), blocks_dict4)
    largeBNB = yolofaseL2(out2)
    yolofaseS1 = blockGenerator((13,13,1), blocks_dict5)
    smallBNB = yolofaseS1(out2)
    smallBNB_shape = tf.shape(smallBNB)
    smallBNB = tf.image.resize(smallBNB, (int(smallBNB_shape[1] * 2), int(smallBNB_shape[2] * 2)),
                               method='nearest')
    smallBNB = tf.concat([smallBNB, out1], axis=-1)
    yolofaseS2 = blockGenerator((26,26,1), blocks_dict6)
    smallBNB = yolofaseS2(smallBNB)
    return [smallBNB, largeBNB]
    
    
def Create_Yolov3(input_size=YOLO_INPUT_SIZE, channels=YOLO_CHANNELS,
                  training=False, NUM_CLASSES=NUMBER_OF_CLASSES):
    NUM_CLASS = int(NUM_CLASSES)
    input_layer  = Input([input_size, input_size, channels])

    conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)
    decoder = yoloReshape(image_size=YOLO_INPUT_SIZE, grid_size=YOLO_STRIDES,
                 anchors=ANCHORS, no_class=NUMBER_OF_CLASSES)
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decoder(conv_tensor)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    YoloV3 = tf.keras.Model(input_layer, output_tensors)
    return YoloV3
