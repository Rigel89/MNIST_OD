# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : yoloNN.py
#   Author      : Rigel89
#   Created date: 24/03/23
#   GitHub      : 
#   Description : yolo like NN architecture
#
#================================================================

#%% Importing libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers  import Input, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers  import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.regularizers import l2
from yoloGenerator.config import *
#from config import *

ANCHORS = np.array(YOLO_ANCHORS)
print(int(tf.shape(ANCHORS)[1]))
STRIDES = np.array(YOLO_STRIDES)

#%% Customs layers

class yoloReshape(tf.keras.layers.Layer):
    def __init__(self, image_size=YOLO_INPUT_SIZE, strides=10,
                 anchors=ANCHORS, no_class=NUMBER_OF_CLASSES):
        super(yoloReshape, self).__init__()
        self.image_size = image_size
        self.grid_size = strides
        self.no_of_bnb = int(tf.shape(anchors)[0])
        self.no_class = no_class
        self.ANCHORS = tf.constant(anchors, dtype=tf.float32)

    def call(self, inputs):
        '''
        The output of the layer is a tensor with shape:
            [Batch, GridRow, GridCol, anchor, output()]

        Parameters
        ----------
        inputs : tensor [ndarray]
            last layer output.

        Returns
        -------
        [ndarray]
            yolov3 tensor shape.

        '''
        # Batch size
        input_shape = tf.shape(inputs)
        batch = int(input_shape[0])

        #tf.print((batch, self.grid_size, self.grid_size, self.no_of_bnb, 5 + self.no_class))
        # reshaping the output
        conv = tf.reshape(inputs,
                          (batch, self.grid_size, self.grid_size, self.no_of_bnb, 5 + self.no_class))

        conv_raw_dxdy = conv[:, :, :, :, 0:2] # offset of center position     
        conv_raw_dwdh = conv[:, :, :, :, 2:4] # Prediction box heigth and width offset
        conv_raw_conf = conv[:, :, :, :, 4:5] # confidence of the prediction box
        conv_raw_prob = conv[:, :, :, :, 5: ] # class probability of the prediction box 
        '''
        # Generating a grid
        y = tf.range(self.grid_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, self.grid_size])
        x = tf.range(self.grid_size, dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [self.grid_size, 1])
    
        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch, 1, 1, self.no_of_bnb, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        '''

        # Calculate the xy
        pred_xy = tf.sigmoid(conv_raw_dxdy)# + xy_grid) / self.grid_size #en el codigo original habia un multiplicado
        # Calculate the hw
        pred_wh = (tf.exp(conv_raw_dwdh) * self.ANCHORS) / self.image_size #en el codigo original habia un multiplicado
    
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object
    
        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

'''class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)'''

#%% Block generator

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

    # def summary(self):
    #     inp = Input(shape=self.inpshape, name="input_layer")
    #     model = Model(inputs=[inp], outputs=self.call(inp))
    #     model.summary()
    #     del inp, model
        # return model.summary()
