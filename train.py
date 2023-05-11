# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : train.py
#   Author      : Rigel89
#   Created date: 11/04/23
#   GitHub      : 
#   Description : training script
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


def main():
    global TRAIN_FROM_CHECKPOINT
    
    # Delete existing path and write a new log file
    print('Checking the existing path to the log:')
    if os.path.exists(os.path.join(MAIN_PATH,TRAIN_LOGDIR)):
        rmtree(TRAIN_LOGDIR)
        print('    Path existed and was deleted')
    else:
        print('    No existing path')

    print('Creating new log file')
    writer = tf.summary.create_file_writer(os.path.join(MAIN_PATH,TRAIN_LOGDIR))

    # Load the dataset
    print('Loading training dataset:')
    trainset = tf.data.Dataset.list_files(os.path.join(MAIN_PATH, DATASET_DIR, TRAIN_DIR)+'/*.xml')
    trainset = trainset.map(lambda x: tf.py_function(pascal_voc_to_dict,inp=[x],Tout=[tf.float32, tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.AUTOTUNE)
    trainset = trainset.shuffle(TRAIN_SHUFFLE).batch(TRAIN_BATCH).prefetch(tf.data.AUTOTUNE)
    print('    Done!')
    print('Loading test dataset:')
    testset = tf.data.Dataset.list_files(os.path.join(MAIN_PATH, DATASET_DIR, TEST_DIR)+'/*.xml')
    testset = testset.map(lambda x: tf.py_function(pascal_voc_to_dict,inp=[x],Tout=[tf.float32, tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.AUTOTUNE)
    testset = testset.batch(TRAIN_BATCH).prefetch(tf.data.AUTOTUNE)
    print('    Done!')

    # Training variables for steps
    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    total_steps = TRAIN_EPOCHS * steps_per_epoch

    '''
    Este codigo permite cargar los weigths the darknet en una variable,
    hacer un check de si hay un preentrenamiento con checkpoints disponibles
    y cargarlos, en caso de que no haya check points disponibles carga los pesos
    de darknet en lugar de los chekpoints.
    #No me vale si entreno la red neuronal de cero, puedo descargar darknet
    #de tf o de github si es necesario

    if TRAIN_TRANSFER:
        Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    yolo = Create_Yolov3(input_size=YOLO_INPUT_SIZE, channels=YOLO_CHANNELS,
                         anchors=ANCHORS, training=True, no_classes=NUMBER_OF_CLASSES)
    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    '''
    print('Creating neuronal network')
    yolo = Create_Yolov3(input_size=YOLO_INPUT_SIZE, channels=YOLO_CHANNELS,
                         anchors=ANCHORS,no_classes=NUMBER_OF_CLASSES)# training=True,

    print('Training from checkpoint: ' + str(TRAIN_FROM_CHECKPOINT))
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
    print('Setting up optimizer and iniciallicating training')
    optimizer = tf.keras.optimizers.Adam()


    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True) # There is BatchNormalization layers, so training=True to train the parameters
            xywh_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = YOLO_ANCHORS.shape[0]
            for i in range(grid):
                #conv, pred = pred_result[i*2], pred_result[i*2+1]
                pred = pred_result[i]
                loss_items = compute_loss(target[i], pred, NUMBER_OF_CLASSES,
                                          YOLO_IOU_LOSS_THRESH, LAMBDA_OBJ, LAMBDA_NOOBJ)
                xywh_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = xywh_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            
            if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/xywh_loss", xywh_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            global_steps.assign_add(1)
        return global_steps.numpy(), optimizer.lr.numpy(), xywh_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    def validate_step(image_data, target):
        pred_result = yolo(image_data, training=False) # There is BatchNormalization layers, so training=False during prediction
        xywh_loss=conf_loss=prob_loss=0

        # optimizing process
        grid = YOLO_ANCHORS.shape[0]
        for i in range(grid):
            pred = pred_result[i]
            loss_items = compute_loss(target[i], pred, NUMBER_OF_CLASSES,
                                        YOLO_IOU_LOSS_THRESH, LAMBDA_OBJ, LAMBDA_NOOBJ)#(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
            xywh_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = xywh_loss + conf_loss + prob_loss
            
        return xywh_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

#    mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES) # create second model to measure mAP
    print('Starting training process:')
    best_val_loss = 1000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        for train_vars in trainset:
            image_data = train_vars[0]
            target = train_vars[1:]
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, xywh_loss:{:4.4f}, conf_loss:{:4.4f}, prob_loss:{:4.4f}, total_loss:{:4.4f}".format(epoch, cur_step, steps_per_epoch, results[1], results[2],
                          results[3], results[4], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            #yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            #continue
        else:
            count, xywh_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
            for test_vars in testset:
                image_data = test_vars[0]
                target = test_vars[1:]
                results = validate_step(image_data, target)
                count += 1
                xywh_val += results[0]
                conf_val += results[1]
                prob_val += results[2]
                total_val += results[3]
            # writing validate summary data
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
                tf.summary.scalar("validate_loss/xywh_val", xywh_val/count, step=epoch)
                tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
                tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
            validate_writer.flush()
            
            print("\nValidation step-> xywh_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n"
                  .format(xywh_val/count, conf_val/count, prob_val/count, total_val/count))

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
            print('\nWeights saved every epoch\n')
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count
            print('\nThe weights are being saved this epoch!\n')
        '''
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)'''

'''
    # measure mAP of trained custom model
    try:
        mAP_model.load_weights(save_directory) # use keras weights
        get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
    except UnboundLocalError:
        print("You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py")
'''
if __name__ == '__main__':
    main()
# %%
