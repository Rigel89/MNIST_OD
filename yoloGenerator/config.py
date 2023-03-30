#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : config.py
#   Author      : Rigel89
#   Created date: 24/03/23
#   GitHub      : 
#   Description : configuration file
#
#================================================================

#%% Variables

YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_CHANNELS               = 1
NUMBER_OF_CLASSES           = 10

YOLO_STRIDES                = [16] #[16, 32]
YOLO_ANCHORS                = [[[10, 14],  [23, 27],   [37, 58]],
                               [[81,  82], [135, 169], [344, 319]]]