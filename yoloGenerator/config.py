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

# Input image
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
YOLO_CHANNELS               = 1

# NN parameters
NUMBER_OF_CLASSES           = 10
LAMBDA_OBJ                  = 5.0
LAMBDA_NOOBJ                = 0.5

YOLO_TYPE                   = 'yolov3_tinny'
YOLO_STRIDES                = [26, 13]
YOLO_ANCHORS                = [[[10, 14],  [23, 27],   [37, 58]],
                               [[81,  82], [135, 169], [344, 319]]]

# Dataset creation configuration

MAIN_PATH                   = 'C:\\Users\\javie\\Python\\MNIST_OD'
BATCH_DATASET               = 5
FETCH_DATASET               = 6
SHUFFLE_DATASET             = 600
NUMBER_OF_TRAINING_IMAGES   = 4000
NUMBER_OF_TEST_IMAGES       = 500
MIN_SCALE                   = 0.4
MAX_SCALE                   = 14
ROTATION_ANG                = 30

#Training parameters

TRAIN_LOGDIR                = "log"
DATASET_DIR                 = 'MNIST_dataset'
TRAIN_DIR                   = 'train'
TEST_DIR                    = 'test'
TRAIN_CHECKPOINTS_FOLDER    = "checkpoints"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_FROM_CHECKPOINT       = True
TRAIN_SAVE_CHECKPOINT       = False
TRAIN_SAVE_BEST_ONLY        = True
TRAIN_WARMUP_EPOCHS         = 1
TRAIN_EPOCHS                = 70
TRAIN_BATCH                 = 40
TRAIN_PREFETCH              = -1
TRAIN_SHUFFLE               = 800
TRAIN_LR_INIT               = 8e-4
TRAIN_LR_END                = 1e-6

