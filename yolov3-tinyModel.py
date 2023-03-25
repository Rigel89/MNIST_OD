#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 04:25:31 2023

@author: javi
"""

from yoloGenerator.yoloNN import blockGenerator, yoloReshape
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU



blocks_dict3 = {'block1': {'b1': [256, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                           'b2': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                           'b3': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
blocks_dict3 = {'block1': {'b1': [256, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                           'b2': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                           'b3': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}




def darknet19(input_shape):
    # Creating the blocks
    blocks_dict1 = {'mdarkB1': {'b1': [16, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'mdarkB2': {'b1': [32, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'mdarkB3': {'b1': [64, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'mdarkB4': {'b1': [128, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'darkB5': {'b1': [256, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    blocks_dict2 = {'mdarkB6': {'b1': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                    'darkB7': {'b1': [1024, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
    
    # Generating the NN
    inputs = Input(shape=input_shape)
    darknet19_tinny1 = blockGenerator((416,416,1), blocks_dict1)
    darknet19_tinny2 = blockGenerator((13,13,1), blocks_dict2)
    out1=darknet19_tinny1(inputs)
    out2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                    padding='same',
                                    name='Mpooling_darknetBreak')(out1)
    out2=darknet19_tinny2(out2)
    out1, out2 = Model(inputs=[inputs], outputs=[out1,out2])
    return out1, out2


def YOLOv3_tiny(input_shape):
    out1, out2 = darknet19(input_shape)
    
