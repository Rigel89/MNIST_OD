# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : pascalVOC2yolov3.py
#   Author      : Rigel89
#   Created date: 09/04/23
#   GitHub      : 
#   Description : loss function class for YOLOv3 
#
#================================================================

#%% Importing libraries

import numpy as np
import tensorflow as tf
from yoloGenerator.metricstools import *



#%% Loss function

@tf.function
def compute_loss(y_true, y_pred, no_classes,
                 threshold, lambda_obj_coor, lambda_noobj_coor): # image_size, anchors, strides,
    no_classes = no_classes
    lambda_obj_coor = float(lambda_obj_coor)
    lambda_noobj_coor = float(lambda_noobj_coor)
    #conv_shape  = tf.shape(conv)
    #batch_size  = conv_shape[0] #batch
    #grid_size = conv_shape[1] #Grid
    #no_bnb = conv_shape[3]
    #bboxes = anchors / image_size
    acu_xywh_loss, acu_conf_loss, acu_prob_loss = 0, 0, 0
    #for y_true, y_pred in zip(y_true_tensors, y_pred_tensors):
    pred_xywh     = y_pred[:, :, :, :, 0:4]
    pred_conf     = y_pred[:, :, :, :, 4:5]
    pred_prob     = y_pred[:, :, :, :, 5:]

    label_xywh    = y_true[:, :, :, :, 0:4]
    respond_bbox  = y_true[:, :, :, :, 4:5]
    label_prob    = y_true[:, :, :, :, 5:]

    #giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)

    #bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4]
    #giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)
    # Calculate Interseccion over Union, value from 0 (poor overlap) to 1 (perfect match)
    iou = bbox_iou(pred_xywh, label_xywh) #(:, 26, 26, 3)

    
    # Find the value of IoU with the real box The largest prediction box
    #max_iou =  tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1) #(:, 26, 26, 1)
    iou = tf.expand_dims(iou, axis=-1) #(:, 26, 26, 3, 1)
    '''    print('loss')
    print(iou.shape)
    print(max_iou.shape)
    print(respond_bbox.shape)'''
    # If the largest iou is less than the threshold,
    # it is considered that the prediction box contains no objects,
    # then the background box. Thres assign a 0.0 when the IoU is 
    # smaller than the threshold and a 1.0 when is bigger
    
    #thres = tf.expand_dims(tf.cast( iou > threshold, tf.float32 ), axis=-2)
    thres = tf.cast( iou > threshold, tf.float32 )
    '''    print('thres')
    print(thres.shape)'''
    #thres = tf.tile(thres,[1,1,1,3,1])
    #respond_bgd = tf.multiply((1.0 - respond_bbox), thres)
    # This line penalize double the items with low IoU, ones in the bag of
    # correct Objects and again in the noobjects, adding the item in the
    # existing object and again in the no correct object detections
    respond_bgd = (1.0-tf.multiply(respond_bbox, thres))

    # Calculate the position loss of the bbox
    xy_mse = tf.pow(pred_xywh[..., 0:2] - label_xywh[..., 0:2],2)
    wh_mse = tf.pow(tf.pow(pred_xywh[..., 2:],0.5) - tf.pow(label_xywh[..., 2:],0.5), 2)
    xywh_loss = tf.multiply(respond_bbox, (xy_mse + wh_mse))

    # Calculate the confidence loss
    conf_focal = tf.pow(tf.multiply(respond_bbox - pred_conf, 1.0-iou), 2)
    #conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects,
    # then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = lambda_obj_coor*tf.reduce_sum(respond_bbox*conf_focal, axis=[1,2,3,4])+lambda_noobj_coor*tf.reduce_sum(respond_bgd*conf_focal, axis=[1,2,3,4])

    prob_loss = tf.multiply(respond_bbox, tf.pow(label_prob-pred_prob,2))

    #giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    #acu_giou_loss += lambda_obj_coor * giou_loss
    xywh_loss = lambda_obj_coor*tf.reduce_mean(tf.reduce_sum(xywh_loss, axis=[1,2,3,4]))
    acu_xywh_loss += xywh_loss
    conf_loss = tf.reduce_mean(conf_loss)
    acu_conf_loss += conf_loss
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))
    acu_prob_loss += prob_loss
    return acu_xywh_loss, acu_conf_loss, acu_prob_loss

