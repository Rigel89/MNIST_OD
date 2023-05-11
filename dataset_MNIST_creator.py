# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : dataset_MNIST_creator.py
#   Author      : Rigel89
#   Created date: 11/04/23
#   GitHub      : 
#   Description : creating a dataset for OD with MNIST images
#
#================================================================

#%% Importing libraries

import numpy as np
import pandas as pd
import os
from shutil import rmtree
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2 as cv
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xml.dom.minidom
import uuid
from yoloGenerator.config import *
from pascalVOC.pascalVOC2yolov3 import *

#%% Main function
def main():

    #%% Download the mnist images from source

    #donwload dataset mnist
    (x_train_ori, y_train_ori), (x_test_ori, y_test_ori) = tf.keras.datasets.mnist.load_data()
    assert x_train_ori.shape == (60000, 28, 28)
    assert x_test_ori.shape == (10000, 28, 28)
    assert y_train_ori.shape == (60000,)
    assert y_test_ori.shape == (10000,)

    #%% Path to the files

    paths = dict()

    # Extract from the config file
    paths['main'] = os.path.normcase(MAIN_PATH)
    paths['dataset'] = os.path.join(paths['main'], 'MNIST_dataset')
    paths['train_data'] = os.path.join(paths['dataset'], 'train')
    paths['test_data'] = os.path.join(paths['dataset'], 'test')
    print('Checking if it is an available path...')
    if os.path.exists(paths['dataset']):
        rmtree(paths['dataset'])
        print('Deleting existing paths...')
    else:
        os.makedirs(paths['dataset'])
        os.makedirs(paths['train_data'])
        os.makedirs(paths['test_data'])
        print('There is not an existing path, creting one...')
        print('Paths to the dataset created in: '+ paths['dataset'])
    
    train = tf.data.Dataset.from_tensor_slices((x_train_ori,y_train_ori))
    test = tf.data.Dataset.from_tensor_slices((x_test_ori,y_test_ori))

    #genetating the input data format
    train = train.batch(BATCH_DATASET).shuffle(SHUFFLE_DATASET).prefetch(FETCH_DATASET)
    test = test.batch(BATCH_DATASET).prefetch(FETCH_DATASET)

    print('Creating a training dataset with parameters:')
    print('Number of images: {}'.format(NUMBER_OF_TRAINING_IMAGES))
    print('Min sacale size: {}'.format(MIN_SCALE))
    print('Max sacale size: {}'.format(MAX_SCALE))
    print('Rotation angle: {}'.format(ROTATION_ANG))
    image_generator(train, NUMBER_OF_TRAINING_IMAGES, paths['train_data'], MIN_SCALE, MAX_SCALE, YOLO_INPUT_SIZE, ROTATION_ANG)
    print('Creating a test dataset with parameters:')
    print('Number of images: {}'.format(NUMBER_OF_TEST_IMAGES))
    print('Min sacale size: {}'.format(MIN_SCALE))
    print('Max sacale size: {}'.format(MAX_SCALE))
    print('Rotation angle: {}'.format(ROTATION_ANG))
    image_generator(test, NUMBER_OF_TEST_IMAGES, paths['test_data'], MIN_SCALE, MAX_SCALE, YOLO_INPUT_SIZE, ROTATION_ANG)
    


if __name__ == '__main__':
    main()