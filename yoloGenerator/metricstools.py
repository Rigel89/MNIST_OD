#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : metrics.py
#   Author      : Rigel89
#   Created date: 29/03/23
#   GitHub      : 
#   Description : metrics functions for NN
#
#================================================================

#%% Libraries
import tensorflow as tf



#%% Functions

# IoU score

def iou(xy_pred_mins, xy_pred_maxes, xy_true_mins, xy_true_maxes):
    '''
    This function calculate the IoU score between to boxes with shape:
        (batch, gridRow, gridCol, Anchor, (X Y))

    Parameters
    ----------
    xy_pred_mins : tensor (ndarray)
        top-left corner of the box1.
    xy_pred_maxes : tensor (ndarray)
        bot-rigth corner of the box1.
    xy_true_mins : tensor (ndarray)
        top-left corner of the box2.
    xy_true_maxes : tensor (ndarray)
        bot-rigth corner of the box2.

    Returns
    -------
    iou_scores : tensor (ndarray)
        (batch, gridRow, gridCol, Anchor, IoU score)

    '''
    intersection_min = tf.math.maximum(xy_pred_mins, xy_true_mins)     # ?*7*7*3*2
    intersection_max = tf.math.minimum(xy_pred_maxes, xy_true_maxes)   # ?*7*7*3*2
    # Calculate the intersection distance w, h -> if it is negative,
    # there is not intersection, so use 0
    # Aquí habia un fallo la siguiente operacion era tf.math.minimum
    intersection = tf.math.maximum(tf.subtract(intersection_max, intersection_min), 0.0)  # ?*7*7*3*2
    intersection = tf.multiply(intersection[:, :, :, :, 0], intersection[:, :, :, :, 1])  # ?*7*7*3
    intersection = tf.expand_dims(intersection,axis=-1)                                   # ?*7*7*3*1

    pred_area = tf.subtract(xy_pred_maxes, xy_pred_mins)                   # ?*7*7*3*2
    pred_area = tf.multiply(pred_area[:, :, :, 0], pred_area[:, :, :, 1])  # ?*7*7*3
    pred_area = tf.expand_dims(pred_area,axis=-1)                          # ?*7*7*3*2
    true_area = tf.subtract(xy_true_maxes, xy_true_mins)                   # ?*7*7*3*2
    true_area = tf.multiply(true_area[:, :, :, 0], true_area[:, :, :, 1])  # ?*7*7*3
    true_area = tf.expand_dims(true_area,axis=-1)                          # ?*7*7*3*2

    union_areas = pred_area + true_area - intersection  # ?*7*7*3*1
    iou_scores = intersection / union_areas             # ?*7*7*3*1

    return iou_scores