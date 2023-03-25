# -*- coding: utf-8 -*-
"""
Autor: Rigel89
Star day: 24/03/23

This is a temporary script file.
"""
#%% Importing libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers  import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers  import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.regularizers import l2


#%% 
class yoloReshape(tf.keras.layers.Layer):
    def __init__(self, grid_size=7, no_bnb=2, no_class=10):
        super(yoloReshape, self).__init__()
        self.grid_size = grid_size
        self.no_of_bnb = no_bnb
        self.no_class = no_class

    def call(self, inputs):
        # grids 7x7
        S = [self.grid_size, self.grid_size]
        # classes
        C = self.no_class
        # no of bounding boxes per grid
        B = self.no_of_bnb

        idx1 = S[0] * S[1] * C
        idx2 = idx1 + S[0] * S[1] * B

        # class probabilities
        class_probs = tf.reshape(inputs[:, :idx1], (tf.shape(inputs)[0],) + tuple([S[0], S[1], C]))
        class_probs = tf.nn.softmax(class_probs)

        # confidence
        confs = tf.reshape(inputs[:, idx1:idx2], (tf.shape(inputs)[0],) + tuple([S[0], S[1], B]))
        confs = tf.math.sigmoid(confs)

        # boxes
        boxes = tf.reshape(inputs[:, idx2:], (tf.shape(inputs)[0],) + tuple([S[0], S[1], B * 4]))
        boxes = tf.math.sigmoid(boxes)

        outputs = tf.concat([class_probs, confs, boxes], axis=3)
        return outputs

class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class blockGenerator(tf.keras.Model):
    def __init__(self, inpshape, bocks_info):
        super().__init__()
        self.nConvs = 0
        self.nBlocks = 0
        self.bocks_info = bocks_info
        self.inpshape = inpshape
        self.layersList = list()
        for key, val in self.bocks_info.items():
            if 'm' in key:
                self.block_CN_gen(val, key, self.layersList, maxpooling=True)
            else:
                self.block_CN_gen(val, key, self.layersList, maxpooling=False)
                

    def block_CN_gen(self, dictionary_info, block_name, layers_list, maxpooling=True):
        '''generate a block of convolutions
        dictionary_info : dict of list {kernel_list, stride_list, padding}'''
        for key, (filt, ker, stride, pad, act) in dictionary_info.items():
            layer_name = block_name+'subblock{}_conv{}'.format(self.nBlocks, self.nConvs)
            layers_list.append(Conv2D(filt, kernel_size=ker, strides=stride,
                                      padding=pad,
                                      kernel_regularizer=l2(5e-4),
                                      name=layer_name))
            self.nConvs += 1
            if 'b' in key:
                layers_list.append(BatchNormalization())
            if act!=None:
                    layers_list.append(act)
        if maxpooling:
            layers_list.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                                            padding='same',
                                            name=block_name+'Mpooling_block{}'.format(self.nBlocks)))
        self.nBlocks += 1

    def call(self, x):
        for layer in self.layersList:
            x = layer(x)
        return x  # Model(inputs,x)

    def summary(self):
        inp = Input(shape=self.inpshape, name="input_layer")
        model = Model(inputs=[inp], outputs=self.call(inp))
        model.summary()
        del inp, model
        # return model.summary()


blocks_dict1 = {'bmblock1': {'b1': [16, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                'bmblock2': {'b1': [32, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                'bmblock3': {'b1': [64, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                'bmblock4': {'b1': [128, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                'bmblock5': {'b1': [256, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
blocks_dict2 = {'bmblock1': {'b1': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]},
                'block2': {'b1': [1024, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
blocks_dict3 = {'block1': {'b1': [256, (1, 1), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                           'b2': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)],
                           'b3': [512, (3, 3), (1, 1), 'same', LeakyReLU(alpha=0.1)]}}
               
