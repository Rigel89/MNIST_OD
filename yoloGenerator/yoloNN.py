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

ANCHORS = np.array(YOLO_ANCHORS)
STRIDES = np.array(YOLO_STRIDES)

#%% Customs layers

class yoloReshape(tf.keras.layers.Layer):
    def __init__(self, image_size=YOLO_INPUT_SIZE, grid_size=YOLO_STRIDES,
                 anchors=ANCHORS, no_class=NUMBER_OF_CLASSES):
        super(yoloReshape, self).__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.no_of_bnb = tf.shape(anchors)[1]
        self.no_class = no_class
        self.ANCHORS = anchors

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
        batch = input_shape[0]

        # reshaping the output
        conv = tf.reshape(inputs,
                          (batch, self.grid_size, self.grid_size, self.no_of_bnb, 5 + self.no_class))

        conv_raw_dxdy = conv[:, :, :, :, 0:2] # offset of center position     
        conv_raw_dwdh = conv[:, :, :, :, 2:4] # Prediction box heigth and width offset
        conv_raw_conf = conv[:, :, :, :, 4:5] # confidence of the prediction box
        conv_raw_prob = conv[:, :, :, :, 5: ] # class probability of the prediction box 

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


        # Calculate the xy
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) / self.grid_size #en el codigo original habia un multiplicado
        # Calculate the hw
        pred_wh = (tf.exp(conv_raw_dwdh) * self.ANCHORS) / self.image_size #en el codigo original habia un multiplicado
    
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object
    
        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

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

               
#%% Loss function

# def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=YOLO_COCO_CLASSES):
#     NUM_CLASS = len(read_class_names(CLASSES))
#     conv_shape  = tf.shape(conv)
#     batch_size  = conv_shape[0] #batch
#     output_size = conv_shape[1] #Grid
#     input_size  = STRIDES[i] * output_size #Grid*Grid??
#     conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

#     conv_raw_conf = conv[:, :, :, :, 4:5]#esto son las predicciones de NN?
#     conv_raw_prob = conv[:, :, :, :, 5:]

#     pred_xywh     = pred[:, :, :, :, 0:4]#esto son las predicciones de NN?
#     pred_conf     = pred[:, :, :, :, 4:5]

#     label_xywh    = label[:, :, :, :, 0:4] #esto son las true labels?
#     respond_bbox  = label[:, :, :, :, 4:5]
#     label_prob    = label[:, :, :, :, 5:]

#     giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
#     input_size = tf.cast(input_size, tf.float32)

#     bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
#     giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

#     iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
#                    bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
#     # Find the value of IoU with the real box The largest prediction box
#     max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

#     # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
#     respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )

#     conf_focal = tf.pow(respond_bbox - pred_conf, 2)

#     # Calculate the loss of confidence
#     # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
#     conf_loss = conf_focal * (
#             respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
#             +
#             respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
#     )

#     prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

#     giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
#     conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
#     prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    # return giou_loss, conf_loss, prob_loss


# class yolov3_loss_function(tf.keras.losses.Loss):
#     '''
#     This class is the loss function for yolo_v1, which is divided in
#     in 4 parts
#     '''

#     def __init__(self, grid_size=YOLO_STRIDES, anchors=ANCHORS, no_class=NUMBER_OF_CLASSES):
#         '''Initialating que variables depending on the accuracy of the NN'''
#         super().__init__()
#         self.grid_size = grid_size
#         self.no_of_bnb = tf.shape(anchors)[1]
#         self.no_class = no_class
#         self.coord_param = 5.0
#         self.noobj_param = 0.5

#     def get_xy_wh_global(self, xy_local_tensor):
#         '''
#         Translating the local coordintes (grid) xy,
#         in global coordinates (picture) -> [0,1]
#         Args:
#         -xy_local tensor: tf tensor with local coorinates
#         return: xy tensor in global coodinates
#         '''
#         # shape = xy_local_tensor.shape[0]
#         # (x -> row, y -> columns) Index
#         # 0,0 is the left-top corner and 6,6 is the rigth-bot corner of the matrix
#         x_global = tf.range(0, self.grid_size, delta=1, dtype=tf.float32, name='range')
#         x_global = tf.expand_dims(x_global, axis=0)
#         x_global = tf.expand_dims(x_global, axis=0)
#         x_global = tf.tile(x_global, [1, self.grid_size, 1])  # 1,1,7 -> 1,7,7
#         x_global = tf.expand_dims(x_global, axis=-1)          # 1,7,7,1
#         y_global = tf.range(0, self.grid_size, delta=1, dtype=tf.float32, name='range')
#         y_global = tf.expand_dims(y_global, axis=0)
#         y_global = tf.expand_dims(y_global, axis=-1)
#         y_global = tf.tile(y_global, [1, 1, self.grid_size])  # 1,7,1 -> 1,7,7
#         y_global = tf.expand_dims(y_global, axis=-1)          # 1,7,7,1
#         xy_global = tf.concat([x_global, y_global], axis=-1)  # 1,7,7,2
#         xy_global = tf.expand_dims(xy_global, axis=3)         # 1,7,7,1,2
#         xy_global = tf.tile(xy_global, [1,1,1,self.no_of_bnb,1]) # 1,7,7,3,2
#         # print(xy_global.shape)
#         # xy_global = tf.tile(xy_global, [shape, 1, 1, 1])
#         xy_global = tf.add(xy_local_tensor, xy_global)
#         xy_global = tf.multiply(xy_global, 1/self.grid_size)  # tranlate to global coords from 0 to 1
#         return xy_global

#     def xy_minmax(self, xy_tensor, wh_tensor):
#         '''
#         Transform bnb xy centered coordinates in
#         xy_minmax corners of the bnb
#         Args:
#         -xy_tensor: tf tensor with xy centered coordinates
#         -wh_tensor: tf tensor wiht wh
#         return xy_min, xy_max corners of the bnb
#         '''
#         xy_min = xy_tensor - wh_tensor/2  # left-top
#         xy_max = xy_tensor + wh_tensor/2  # rigth-bot
#         return xy_min, xy_max

#     # def iou(self, xy_pred_mins, xy_pred_maxes, xy_true_mins, xy_true_maxes):
#     #     intersection_min = tf.math.maximum(xy_pred_mins, xy_true_mins)     # ?*7*7*2
#     #     intersection_max = tf.math.minimum(xy_pred_maxes, xy_true_maxes)   # ?*7*7*2
#     #     # Calculate the intersection distance w, h -> if it is negative,
#     #     # there is not intersection, so use 0
#     #     # Aquí habia un fallo la siguiente operacion era tf.math.minimum
#     #     intersection = tf.math.maximum(tf.subtract(intersection_max, intersection_min), 0.0)  # ?*7*7*2
#     #     intersection = tf.multiply(intersection[:, :, :, 0], intersection[:, :, :, 1])        # ?*7*7

#     #     pred_area = tf.subtract(xy_pred_maxes, xy_pred_mins)                   # ?*7*7*2
#     #     pred_area = tf.multiply(pred_area[:, :, :, 0], pred_area[:, :, :, 1])  # ?*7*7
#     #     true_area = tf.subtract(xy_true_maxes, xy_true_mins)                   # ?*7*7*2
#     #     true_area = tf.multiply(true_area[:, :, :, 0], true_area[:, :, :, 1])  # ?*7*7

#     #     union_areas = pred_area + true_area - intersection  # ?*7*7
#     #     iou_scores = intersection / union_areas             # ?*7*7
#     #     iou_scores = tf.expand_dims(iou_scores, axis=-1)    # ?*7*7*1

#     #     return iou_scores

#     def choose_max(self, list_of_tensor):
#         tensor_shape = tf.shape(list_of_tensor[0])
#         for t, tensor in enumerate(list_of_tensor):
#             if t == 0:
#                 init_tensor = tensor           # ?*7*7*1
#                 # this create a mask, -1 is not better possition found
#                 mask = tf.where(tensor>0.0, 0.0, -1.0)  # ?*7*7*1
#             else:
#                 init_tensor = tf.math.maximum(init_tensor, tensor)  # ?*7*7*1
#                 mask = tf.where(init_tensor == tensor, tf.cast(t, tf.float32), mask)  # ?*7*7*1
#         # mask give back a tensor (?,7,7,1) with the position of the greater
#         # iou score in the bnb
#         # Create a tensor mask for predict
#         # tensor_shape.append(self.no_of_bnb)
#         for i in range(len(list_of_tensor)):
#             if i == 0:
#                 mask_pred = tf.where(mask == float(i), 1.0, 0.0)
#             else:
#                 mask_pred = tf.concat([mask_pred,
#                                        tf.where(mask == float(i), 1.0, 0.0)], axis=-1)

#         # Create a tensor mask for bnb
#         for i in range(len(list_of_tensor)):
#             if i == 0:
#                 mask_bnb = tf.tile(tf.where(mask == i, 1.0, 0.0), [1, 1, 1, 2])
#             else:
#                 mask_bnb = tf.concat([mask_bnb,
#                                       tf.tile(tf.where(mask == i, 1.0, 0.0),
#                                               [1, 1, 1, 2])], axis=-1)
#         return mask_pred, mask_bnb

#     def call(self, y_true, y_pred):
#         # batch
#         batch_size = tf.cast(tf.shape(y_pred)[0], dtype=tf.float32)
#         # True labels
#         label_class = y_true[:, :, :, :self.no_class]               # ?*7*7*10
#         label_box = y_true[:, :, :, self.no_class:self.no_class+4]  # ?*7*7*4
#         response_mask = y_true[:, :, :, -1:]                        # ?*7*7*1
#         # Predicted labels
#         pred_class = y_pred[:, :, :, :self.no_class]                              # ?*7*7*10
#         pred_trust = y_pred[:, :, :, self.no_class:self.no_class+self.no_of_bnb]  # ?*7*7*2
#         pred_box = y_pred[:, :, :, self.no_class+self.no_of_bnb:]                 # ?*7*7*8
#         # Creating a xy label for the corners of the bnb
#         xy_label_box = self.get_xy_wh_global(label_box[:, :, :, :2])              # ?*7*7*2
#         wh_label_box = label_box[:, :, :, 2:]                                     # ?*7*7*2
#         xy_min_label_box, xy_max_label_box = self.xy_minmax(xy_label_box, wh_label_box)
#         xy_pred_box = [self.get_xy_wh_global(pred_box[:, :, :, 4*x:4*x+2]) for x in range(self.no_of_bnb)]  # [?*7*7*2]
#         wh_pred_box = [pred_box[:, :, :, 4*x+2:4*x+4] for x in range(self.no_of_bnb)]                       # [?*7*7*2]
#         xy_minmax_pred_box = [list(self.xy_minmax(xy, wh)) for xy, wh in zip(xy_pred_box, wh_pred_box)]      # [[?*7*7*2, ?*7*7*2]]]]

#         iou_scores = [self.iou(xy_minmax_pred_box[x][0],
#                                xy_minmax_pred_box[x][1],
#                                xy_min_label_box,
#                                xy_max_label_box) for x in range(self.no_of_bnb)]
#         mask_pred, mask_bnb = self.choose_max(iou_scores)  # ?*7*7*2, ?*7*7*4

#         # Calculating the losses of the centers of the objects
#         # coord * SUM(Iobj_i-j((x - xhat)² + (y - yhat)²))
#         loss_xy = tf.subtract(tf.concat(xy_pred_box, axis=-1),
#                               tf.tile(xy_label_box, [1, 1, 1, self.no_of_bnb]))  # ?*7*7*4
#         loss_xy = tf.math.pow(loss_xy, 2)
#         loss_xy = tf.multiply(mask_bnb, loss_xy)
#         # loss_xy = tf.tile(mask_pred, [1, 1, 1, self.no_of_bnb])*loss_xy          # ?*7*7*4
#         loss_xy = tf.tile(response_mask, [1, 1, 1, self.no_of_bnb*2])*loss_xy    # ?*7*7*4
#         loss_xy = tf.reduce_sum(loss_xy)/batch_size  # float number
#         # print(loss_xy)
#         # Calculating the losses of the weith and heigth of the bb
#         # coord * SUM(Iobj_i-j((w^0.5 - what^0.5)² + (h^0.5 - hhat^0.5)²))
#         loss_wh = tf.subtract(tf.math.sqrt(tf.concat(wh_pred_box, axis=-1)),   # ?*7*7*4
#                               tf.math.sqrt(tf.tile(wh_label_box, [1, 1, 1, self.no_of_bnb])))
#         # print(loss_wh)
#         loss_wh = tf.math.pow(loss_wh, 2)
#         loss_wh = tf.multiply(mask_bnb, loss_wh)
#         # loss_wh = tf.tile(mask_pred, [1, 1, 1, self.no_of_bnb])*loss_wh        # ?*7*7*4
#         loss_wh = tf.tile(response_mask, [1, 1, 1, self.no_of_bnb*2])*loss_wh  # ?*7*7*4
#         loss_wh = tf.reduce_sum(loss_wh)/batch_size  # float number
#         # print(loss_wh)
#         # Calculating the losses of the confidence in the predictions
#         # SUM(Iobj_i-j((C - Chat)²)) + no_objt_param*SUM(Inoobj_i-j(h^0.5 - hhat^0.5)²)
#         # Inoobj_i-j es de oposite of Iobj_i-j
#         loss_conf = pred_trust-tf.tile(response_mask,
#                                        [1, 1, 1, self.no_of_bnb])       # ?*7*7*2
#         loss_conf = mask_pred*tf.math.pow(loss_conf, 2)                 # ?*7*7*2
#         loss_conf_noobj = tf.tile(1-response_mask,
#                                   [1, 1, 1, self.no_of_bnb])*loss_conf  # ?*7*7*2
#         loss_conf = tf.tile(response_mask,
#                             [1, 1, 1, self.no_of_bnb])*loss_conf        # ?*7*7*2
#         loss_conf_noobj = tf.reduce_sum(loss_conf_noobj)/batch_size  # float number
#         loss_conf = tf.reduce_sum(loss_conf)/batch_size  # float number
#         # print(mask_pred)
#         # print(tf.tile(response_mask,[1, 1, 1, self.no_of_bnb])*mask_pred)
#         # print(tf.tile(response_mask,[1, 1, 1, self.no_of_bnb]))
#         # print(loss_conf)
#         # print(loss_conf_noobj)
#         # Calculating the losses of the classes
#         # SUM_everycell(SUM_classes((C - Chat)²))
#         classes_loss = tf.subtract(label_class, pred_class)       # ?*7*7*10
#         classes_loss = tf.math.pow(classes_loss, 2)               # ?*7*7*10
#         classes_loss = tf.math.reduce_sum(classes_loss, axis=-1,
#                                           keepdims=True)          # ?*7*7*1
#         classes_loss = tf.multiply(response_mask, classes_loss)   # ?*7*7*1
#         classes_loss = tf.math.reduce_sum(classes_loss)/batch_size           # float number
#         # print(classes_loss)
#         # Calculating the total loss
#         total_loss = self.coord_param*(loss_xy + loss_wh) + loss_conf + self.noobj_param*loss_conf_noobj + classes_loss
#         return total_loss

