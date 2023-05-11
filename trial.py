# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : trial.py
#   Author      : Rigel89
#   Created date: 08/05/23
#   GitHub      : 
#   Description : test script
#
#================================================================

#%% Importing libraries

import numpy as np
import tensorflow as tf

#This code must be here because has to be set before made other operations (quit ugly solution!!)
print('SETTING UP GPUs')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('Setting up the GPUs done!')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print('Setting up the GPUs Not done')

import os
from shutil import rmtree
from yoloGenerator.config import *
from yoloGenerator.yolov3_tinyModel import *
from yoloGenerator.lossYolov3 import *
from pascalVOC.pascalVOC2yolov3 import *

#%% Main execution process

# Load the dataset
print('Loading training dataset:')
trainset = tf.data.Dataset.list_files(os.path.join(MAIN_PATH, DATASET_DIR, TRAIN_DIR)+'/*.xml')
trainset = trainset.map(lambda x: tf.py_function(pascal_voc_to_dict,inp=[x],Tout=[tf.float32, tf.float32, tf.float32]),
                        num_parallel_calls=tf.data.AUTOTUNE)
trainset = trainset.shuffle(TRAIN_SHUFFLE).batch(TRAIN_BATCH).prefetch(TRAIN_PREFETCH)
print('    Done!')
print('Loading test dataset:')
testset = tf.data.Dataset.list_files(os.path.join(MAIN_PATH, DATASET_DIR, TEST_DIR)+'/*.xml')
testset = testset.map(lambda x: tf.py_function(pascal_voc_to_dict,inp=[x],Tout=[tf.float32, tf.float32, tf.float32]),
                      num_parallel_calls=tf.data.AUTOTUNE)
testset = testset.batch(TRAIN_BATCH).prefetch(TRAIN_PREFETCH)
print('    Done!')

print('Creating neuronal network')
yolo = Create_Yolov3(input_size=YOLO_INPUT_SIZE, channels=YOLO_CHANNELS,
                        anchors=ANCHORS,no_classes=NUMBER_OF_CLASSES)

print('Loading weights:')
print('    Training from checkpoint: ' + str(TRAIN_FROM_CHECKPOINT))
if TRAIN_FROM_CHECKPOINT:
    print("Trying to load weights from check points:")
    try:
        if os.path.exists("./checkpoints"):
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
            print('    Succesfully load!')
        else:
            print('    There is not existing checkpoint path!')
    except ValueError:
        print("    Shapes are incompatible or there is not checkpoints")
        TRAIN_FROM_CHECKPOINT = False

file = open('./mnist.names')
names = file.readlines()
file.close()
for l, line in enumerate(names):
    names[l] = line.strip('\n')

def image_examples(dataset, neuronal_network, no_images, threshold=0.9):
    print('Creating dataset')
    images = dataset.as_numpy_iterator().next()[0]
    print('NN working')
    data = neuronal_network(images)
    tensor0 = data[0][:no_images, ...]
    tensor1 = data[1][:no_images, ...]
    print('data taken correctly!')
    for img in range(no_images):
        image = images[img, ...]
        image = tf.cast(image*255,dtype=tf.uint8).numpy()
        tensor0_img = tensor0[img, ...]
        tensor1_img = tensor1[img, ...]
        image_position = list()
        ten0 = np.zeros((tensor0_img.shape[0],tensor0_img.shape[1],tensor0_img.shape[3]),dtype=np.float32)
        ten1 = np.zeros((tensor1_img.shape[0],tensor1_img.shape[1],tensor1_img.shape[3]),dtype=np.float32)
        index_ten0 =  tf.reduce_max(tensor0_img[...,4:5], axis=2, keepdims=True)
        index_ten0 = tf.where(tensor0_img[...,4:5]==index_ten0)
        for ind in index_ten0:
            ten0[ind[0],ind[1],:] = tensor0_img[ind[0],ind[1],ind[2],:]
        index_ten1 =  tf.reduce_max(tensor1_img[...,4:5], axis=2, keepdims=True)
        index_ten1 = tf.where(tensor1_img[...,4:5]==index_ten1)
        for ind in index_ten1:
            ten1[ind[0],ind[1],:] = tensor1_img[ind[0],ind[1],ind[2],:]
        for row in range(tensor0_img.shape[0]):
            for col in range(tensor0_img.shape[0]):
                if ten0[row,col,4] > threshold:
                    v = np.zeros(ten0[row,col,:].shape)
                    v[0] = col
                    v[1] = row
                    v = ten0[row,col,:]+v
                    v[0] = v[0]/float(tensor0_img.shape[0])
                    v[1] = v[1]/float(tensor0_img.shape[1])
                    image_position.append(v)
        for row in range(tensor1_img.shape[0]):
            for col in range(tensor1_img.shape[0]):
                if ten1[row,col,4] > threshold:
                    v = np.zeros(ten1[row,col,:].shape)
                    v[0] = col
                    v[1] = row
                    v = ten1[row,col,:]+v
                    v[0] = v[0]/float(tensor1_img.shape[0])
                    v[1] = v[1]/float(tensor1_img.shape[1])
                    image_position.append(v)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        for rect in image_position:
            x_min = rect[0] - rect[2]/2
            x_max = rect[0] + rect[2]/2
            y_min = rect[1] - rect[3]/2
            y_max = rect[1] + rect[3]/2
            number = np.argmax(rect[5:])
            image = cv.rectangle(image, (int(416*x_min), int(416*y_min)),
                                 (int(416*x_max), int(416*y_max)), (255, 0, 0), 2)
            image = cv.putText(
                image, #numpy array on which text is written
                '{}: {:1.2f}'.format(names[number], rect[4]), #text
                (int(416*x_min), int(416*y_min-5)), #position at which writing has to start
                cv.FONT_HERSHEY_DUPLEX, #font family
                0.5, #font size
                (0, 0, 255), #font color
                2) #font stroke
        if img == 0:
            image_show = image
        else:
            image_show = np.concatenate([image_show,image], axis=1)
    plt.imshow(image_show)
    plt.show()

image_examples(trainset, yolo, 5, threshold=0.9)
