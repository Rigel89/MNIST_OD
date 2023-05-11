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

def bbox_iou(boxes1, boxes2):
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]
    '''    print('metrics')
    print(boxes1.shape)
    print((boxes1[..., 2:]* 0.5).shape)
    print(tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1).shape)'''
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    # Calculate the iou value between the two bounding boxes
    iou = inter_area / union_area

    # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    # Calculate the area of the smallest closed convex surface C
    enclose_area = enclose[..., 0] * enclose[..., 1]

    # Calculate the GIoU value according to the GioU formula  
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou

# testing (should be better than giou)
def bbox_ciou(boxes1, boxes2):
    boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
    up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
    right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
    down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    iou = bbox_iou(boxes1, boxes2)

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term
