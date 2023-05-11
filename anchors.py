# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : anchors.py
#   Author      : Rigel89
#   Created date: 11/04/23
#   GitHub      : 
#   Description : search for the optimal anchors size script
#
#================================================================

#%% Importing libraries

import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf
from pascalVOC.pascalVOC2yolov3 import pascal_voc_to_dict
from yoloGenerator.config import *

#%% Variables

iters = int(NUMBER_OF_TRAINING_IMAGES/TRAIN_BATCH)
anchor_list =  list()
n_clusters = 6
show_points = 600


#%% Dataset

# Use the dataset created to the training process to extract meaningfull data and
# reused to create the anchor. Less efficient but faster.

print('Loading tf training dataset:')
trainset = tf.data.Dataset.list_files(os.path.join(MAIN_PATH, DATASET_DIR, TRAIN_DIR)+'/*.xml')
trainset = trainset.map(lambda x: tf.py_function(pascal_voc_to_dict,inp=[x],Tout=[tf.float32, tf.float32, tf.float32]),
                        num_parallel_calls=tf.data.AUTOTUNE)
trainset = trainset.batch(TRAIN_BATCH).prefetch(tf.data.AUTOTUNE)
print('    Done!')
print('Preparing the new dataset for analyze the anchors:')


for i in range(iters):
    image, small, big = trainset.as_numpy_iterator().next()
    small = tf.reduce_sum(small, axis=-2)
    small_anchors = tf.where(small[:,:,:,4]==1.0)
    for ind in small_anchors:
        anchor_list.append(small[ind[0],ind[1],ind[2],2:4].numpy())
    big = tf.reduce_sum(big, axis=-2)
    big_anchors = tf.where(big[:,:,:,4]==1.0)
    for ind in big_anchors:
        anchor_list.append(big[ind[0],ind[1],ind[2],2:4].numpy())
    print('Iteration: {:4.0f}, anchors: {:5.0f}'.format(i,(len(anchor_list))))
    
print('Total anchors: ', len(anchor_list))
print(anchor_list[0])

X = np.array(anchor_list)
print(X.shape)

#%% Clustering

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init='warn', max_iter=300,
                tol=0.0001, verbose=1, random_state=None, copy_x=True, algorithm='lloyd').fit(X)
centers = np.array(kmeans.cluster_centers_)
print(np.array(kmeans.cluster_centers_*YOLO_INPUT_SIZE, dtype=np.int32))

plt.scatter(X[:show_points,0],X[:show_points,1], s=0.5, c=kmeans.labels_[:show_points], cmap='Dark2')
plt.scatter(centers[:,0], centers[:,1], s=30, marker='x', linewidths=3, edgecolors='r', c=range(n_clusters), cmap='Dark2')
plt.colorbar()
plt.show()


