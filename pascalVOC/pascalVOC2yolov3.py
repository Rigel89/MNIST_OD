# -*- coding: utf-8 -*-

#================================================================
#
#   File name   : pascalVOC2yolov3.py
#   Author      : Rigel89
#   Created date: 08/04/23
#   GitHub      : 
#   Description : transform from/to pascal VOC format
#
#================================================================

#%% Importing libraries

import numpy as np
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2 as cv
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
import xml.dom.minidom
import uuid
from yoloGenerator.metricstools import iou
from yoloGenerator.config import *

YOLO_STRIDES = np.array(YOLO_STRIDES)
YOLO_ANCHORS = np.array(YOLO_ANCHORS)

#%% Functions to transform files in data and data in files

# Create a pascal voc format file from the dictionary data

def dict_to_xml(tag, d, attribute=None):
    '''
    This function create a xml object from scratch
    attributes:
    - tag : str name of the top tag
    - d : dict like to create child elemments in xml
    - attribute : attribute for top tag (default=None)
    return: xml object
    '''
    elem = ET.Element(tag)
    if attribute is not None:
        elem.attrib = attribute
    for key, val in d.items():
        child = ET.Element(key)
        child.text = str(val)
        elem.append(child)
    return elem


def add_dict_to_xml(xml, d):
    '''
    This function modify chilid elemments
    for an xml object
    attributes:
    - xml : xml parent obj
    - d : dict like to create child elemments in xml
    return: None
    '''
    elem = xml
    for key, val in d.items():
        child = ET.Element(key)
        child.text = str(val)
        elem.append(child)

# Create a PascalVOC format file
def toPascalVocFormat(name_of_the_file, directory_name, dictionary_data, image_size):
    '''
    This function create a xml file.
    Attributes:
    - name_of_the_file: name of the file (str)
    - directory_name: directory path to save the file (str)
    - d: list of dict len(d)=number of objects in the image
      d example: [{'name': 'Class_name', 'x_min': 'int',
                   'ymin': 'int', 'xmax': 'int', 'ymax': 'int'}]
    return: None
    '''
    # Baseline data in a dictionary format
    basicXmlFileDict = {'folder': os.path.dirname(directory_name),
                        'filename': name_of_the_file+'.jpg',
                        'path': os.path.join(directory_name,
                                             name_of_the_file+'.jpg')
                       }

    sizeDict = {'width': str(image_size), 'height': str(image_size), 'depth': '1'}

    dummyObject = {'name': 'Class_name', 'pose': 'Unspecified',
                   'truncated': '0', 'difficult': '0'}

    dummyObjectBNB = {'xmin': 'int', 'ymin': 'int',
                      'xmax': 'int', 'ymax': 'int'}

    dummy_xml = dict_to_xml('annotation', basicXmlFileDict,
                            attribute={'verified': 'no'})
    dummy_xml.append(ET.Element('source'))
    dummy_xml.find('source').append(ET.Element('database'))
    dummy_xml.find('source')[0].text = 'MNIST dataset for OD by Javi'
    dummy_xml.append(ET.Element('size'))
    dummy_xml.append(ET.Element('segmented'))
    dummy_xml.find('segmented').text = '0'
    add_dict_to_xml(dummy_xml.find('size'), sizeDict)

    for item in dictionary_data:
        dummyObject['name'] = item['name']
        dummyObjectBNB['xmin'] = item['xmin']
        dummyObjectBNB['ymin'] = item['ymin']
        dummyObjectBNB['xmax'] = item['xmax']
        dummyObjectBNB['ymax'] = item['ymax']
        child = ET.Element('object')
        add_dict_to_xml(child, dummyObject)
        dummy_xml.append(child)
        ET.SubElement(dummy_xml[-1], 'bndbox')
        add_dict_to_xml(dummy_xml[-1].find('bndbox'), dummyObjectBNB)
    # Creating a xml in the directory
    f = open(os.path.join(directory_name, name_of_the_file+'.xml'), "w")
    file = ET.tostring(dummy_xml)
    dom = xml.dom.minidom.parseString(file)
    pretty_xml_as_string = dom.toprettyxml()
    f.write(pretty_xml_as_string)
    f.close()

# Creating a function to increase the accuracy of the bounding box on the image

def clossing_bnb(image, possition_list):
    '''
    This function take an image and the possition of an object in the image
    and try to close the box arround the object til the first pixel not dark
    Attributes:
    - image: ndarry of the image
    - possition_list: list of the initial possition of the object (list of lists)
      example point inside possition list: [inital_row, initial_col, high(high=width)]
    return: new possition of the objects [x_min, y_min, x_max, y_max]
    '''
    opt_bnb = list()
    for p, point in enumerate(possition_list):
        ROI = image[point[0]:point[0]+point[2], point[1]:point[1]+point[2]]
        # Reducing the top
        top = 0
        stopper = True
        threshold = 255
        while stopper:
            if ROI[-top-1, :].sum() < threshold:
                top += 1
            elif top >= point[2]:
                print('The bnb become 0, something is wrong0')
                break
            else:
                stopper = False
        # Reducing the bottom
        bot = 0
        stopper = True
        while stopper:
            if ROI[bot+1, :].sum() < threshold:
                bot += 1
            elif bot >= point[2]:
                print('The bnb become 0, something is wrong1')
                break
            else:
                stopper = False
        # Reducing the left
        left = 0
        stopper = True
        while stopper:
            if ROI[:, left+1].sum() < threshold:
                left += 1
            elif left >= point[2]:
                print('The bnb become 0, something is wrong2')
                break
            else:
                stopper = False
        # Reducing the rigth
        rigth = 0
        stopper = True
        while stopper:
            if ROI[:, -rigth-1].sum() < threshold:
                rigth += 1
            elif rigth >= point[2]:
                print('The bnb become 0, something is wrong3')
                break
            else:
                stopper = False
        opt_bnb.append([point[1]+left, point[0]+bot,
                       point[1]+point[2]-rigth, point[0]+point[2]-top])

    return opt_bnb

# This is the mapping function for the dataset. The dataset read the
# files of the directory and generate a list of paths, it will return
# a tupple with (image, data_info)

def pascal_voc_to_dict(filepath, anchors=YOLO_ANCHORS, strides=YOLO_STRIDES, no_class = 10):
    """
    Function to get all the objects from the annotation XML file.
    Reference axis:
    ->top-left of the image is (0,0) - bot-rigth is (1,1)
    ->y is the heigth and x is the weigth

    Parameters
    ----------
    filepath : str
        file path.
    anchors : array
        numpy array with the anchors, the dimension must be:
            (number_of_levels_of_definition, boxes_per_level, 2).
    strides : array
        numpy array with the strides, the dimension must be:
            (number_of_levels_of_definition).

    Returns
    -------
    img : array
        image.
    y_true_tensor : tf tensor
        tensor with the ouput info.

    """
    # The next 3 lines must be modify if the dataset or nn change:
    #grid_size = 7
    no_anchors_per_level = anchors.shape[1]
    y_true_tensors = list()
    
    # Setting up the output tensor
    for stride in strides:
        y_true_tensors.append(np.zeros([stride, stride,
                                        no_anchors_per_level,
                                        no_class+5]))

    with tf.io.gfile.GFile(filepath.numpy().decode(), "r") as f:
        # print(f.read())
        root = ET.parse(f).getroot()

        size = root.find("size")
        image_w = float(size.find("width").text)
        image_h = float(size.find("height").text)
        filePath = str(root.find("path").text)
        img = tf.io.read_file(filePath)
        img = tf.io.decode_jpeg(img, channels=1).numpy()/255.0
        # Setting up the output tensor
        # y_true_tensor = np.zeros([grid_size, grid_size, no_class+5])

        for obj in root.findall("object"):
            # Get object's label name.
            label = int(obj.find("name").text)
            # Get objects' pose name.
            # pose = obj.find("pose").text.lower()
            # is_truncated = obj.find("truncated").text == "1"
            # is_difficult = obj.find("difficult").text == "1"
            bndbox = obj.find("bndbox")
            xmax = float(bndbox.find("xmax").text)
            xmin = float(bndbox.find("xmin").text)
            ymax = float(bndbox.find("ymax").text)
            ymin = float(bndbox.find("ymin").text)
            # Calculating the middle point of the image and
            # (h,w) adimensional -> [0,1]
            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h
            # Best fit anchor for every object:
            best_anchor = None
            # Calculate the best anchor fit:
            for l, tensor in enumerate(y_true_tensors):
                no_strides = strides[l]
                loc = [no_strides * x, no_strides * y]
                loc_i = int(loc[1])  # i are the rows of the tensor
                loc_j = int(loc[0])  # j are the columns of the tensor
                y_loc = loc[1] - loc_i
                x_loc = loc[0] - loc_j
                for anc, anchor in enumerate(anchors[l]):
                    w_anc, h_anc = anchor
                    w_anc, h_anc = w_anc/image_w, h_anc/image_h
                    # Calculate IoU:
                    xy_anc_mins = np.array([-w_anc/2.0,-h_anc/2.0])
                    xy_anc_maxes = np.array([w_anc/2.0,h_anc/2.0])
                    xy_true_mins = np.array([-w/2.0,-h/2.0])
                    xy_true_maxes = np.array([w/2.0,h/2.0])
                    intersection_min = tf.math.maximum(xy_anc_mins, xy_true_mins)     # 2
                    intersection_max = tf.math.minimum(xy_anc_maxes, xy_true_maxes)   # 2
                    # Calculate the intersection distance w, h -> if it is negative,
                    # there is not intersection, so use 0
                    # Aquí habia un fallo la siguiente operacion era tf.math.minimum
                    intersection = tf.math.maximum(tf.subtract(intersection_max, intersection_min),
                                                   tf.constant([0.0], dtype=tf.float64))  # 2
                    intersection = tf.multiply(intersection[0], intersection[1])                          # 1
                    # intersection = tf.expand_dims(intersection,axis=-1)                                 # 1*1

                    pred_area = tf.subtract(xy_anc_maxes, xy_anc_mins)                     # 2
                    #print(pred_area)
                    pred_area = tf.multiply(pred_area[0], pred_area[1])                    # 1
                    # pred_area = tf.expand_dims(pred_area,axis=-1)                        # ?*7*7*3*2
                    true_area = tf.subtract(xy_true_maxes, xy_true_mins)                   # 2
                    true_area = tf.multiply(true_area[0], true_area[1])                    # 1
                    # true_area = tf.expand_dims(true_area,axis=-1)                        # 1*1

                    union_areas = pred_area + true_area - intersection  # 1
                    IoU = float(intersection / union_areas)             # 1
                    # Call the function
                    #IoU = iou(xy_anc_mins, xy_anc_maxes, xy_true_mins, xy_true_maxes)
                    if best_anchor is None:
                        best_anchor = np.array([loc_i, loc_j, l, anc, x_loc, y_loc, w, h, label, IoU])
                    else:
                        if IoU > best_anchor[-1]:
                            best_anchor = np.array([loc_i, loc_j, l, anc, x_loc, y_loc, w, h, label, IoU])
                        else:
                            pass
            if y_true_tensors[int(best_anchor[2])][int(best_anchor[0]), int(best_anchor[1]), int(best_anchor[3]), 4] == 0:
                y_true_tensors[int(best_anchor[2])][int(best_anchor[0]), int(best_anchor[1]), int(best_anchor[3]), int(5+best_anchor[8])] = 1
                y_true_tensors[int(best_anchor[2])][int(best_anchor[0]), int(best_anchor[1]), int(best_anchor[3]), 0:4] = best_anchor[4:8]
                y_true_tensors[int(best_anchor[2])][int(best_anchor[0]), int(best_anchor[1]), int(best_anchor[3]), 4] = 1
        output = [img]
        for tensor in y_true_tensors:
            output.append(tensor)

        return tuple(output)


#%% Funtions to treat the images

# Rotate the image
def rotate_image(image, angle, not_print = True):
    '''
    This function rotate the image and plot it
    Imput params:
        image = array like,
        angle = counterclockwise rotate angle (degrees),
        not_print = if 'True' do not plot an image
    Return:
        image = image rotated [array like]
    '''
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    if not not_print:
        plt.imshow(result, cmap='gray')
        plt.show()
    return result

# Function to create a dataset images with MNIST dataset

def image_generator(tf_dataset, number_of_images, directory_name, min_scale, max_scale, image_size, rotation_angle):
    # For every number_of_images that it is going to be created:
    images, labels = tf_dataset.as_numpy_iterator().next()
    batch_size = images.shape[0]
    listOfPositions = list(range(batch_size))
    image_size = int(image_size) #Image final size in pixels
    for numb in tqdm(range(number_of_images)):
        # take a banch of the dataset
        images, labels = tf_dataset.as_numpy_iterator().next()
        numberOfImages = random.randint(1, batch_size)
        # take a random numberOfImages from the banch
        indexOfImages = random.sample(listOfPositions, numberOfImages)
        # generate a back image
        img = np.zeros((image_size, image_size), dtype=np.uint8)
        positionList = list()
        labelsList = list()
        # For every image chose:
        for noi in indexOfImages:
            # Rotate the image and scaled 0.5 to 1.5 times
            rotationAngle = random.randint(-rotation_angle, rotation_angle)
            # Scalate the images from min_scale to max_scale ramdomly,
            # using the original size has reference 1.
            scaleFactor = round(((max_scale-min_scale)*random.random()+min_scale)*28) # 28 is the size of MNIST images
            # print(scaleFactor)
            size = (scaleFactor, scaleFactor)
            stopper = 0
            # For every number noi, try to find a possition to do not overlap each other (5 tries):
            while stopper <= 5:
                # Generate random position
                x = random.randint(0, image_size-scaleFactor)
                y = random.randint(0, image_size-scaleFactor)
                # print(x,y)
                # Initilize the possition has available (not overlap)
                positionCompromised = False
                # Compare with the numbers already added to the black image
                for pos in positionList:
                    # print('Scales',pos[2],scaleFactor)
                    # print(x, y, x+scaleFactor, y+scaleFactor)
                    # print(pos[0],pos[1],pos[0]+pos[2],pos[1]+pos[2])
                    # Is the new number size bigger than the old one
                    if scaleFactor > pos[2]:
                        # The points are True if old point inside square ->(x, y), (x+scale, y+scale)
                        point1 = (pos[0] > x and pos[0] < x+scaleFactor) and (pos[1] > y and pos[1] < y+scaleFactor)
                        point2 = (pos[0]+pos[2] > x and pos[0]+pos[2] < x+scaleFactor) and (pos[1] > y and pos[1] < y+scaleFactor)
                        point3 = (pos[0]+pos[2] > x and pos[0]+pos[2] < x+scaleFactor) and (pos[1]+pos[2] > y and pos[1]+pos[2] < y+scaleFactor)
                        point4 = (pos[0] > x and pos[0] < x+scaleFactor) and (pos[1]+pos[2] > y and pos[1]+pos[2] < y+scaleFactor)
                    else:
                        # The points are True if new point inside square -> pos
                        point1 = (x > pos[0] and x < pos[0]+pos[2]) and (y > pos[1] and y < pos[1]+pos[2])
                        point2 = (x+scaleFactor > pos[0] and x+scaleFactor < pos[0]+pos[2]) and (y > pos[1] and y < pos[1]+pos[2])
                        point3 = (x+scaleFactor > pos[0] and x+scaleFactor < pos[0]+pos[2]) and (y+scaleFactor > pos[1] and y+scaleFactor < pos[1]+pos[2])
                        point4 = (x > pos[0] and x < pos[0]+pos[2]) and (y+scaleFactor > pos[1] and y+scaleFactor < pos[1]+pos[2])
                    # If any point is True the image is ocluding other one and is not included
                    if point1 or point2 or point3 or point4:
                        # print('denegado')
                        positionCompromised = True
                    # print(scaleFactor > pos[2],point1,point2,point3,point4)
                if len(positionList) == 0:
                    positionList.append([x, y, scaleFactor])
                    labelsList.append(labels[noi])
                    img[x:x+scaleFactor, y:y+scaleFactor] = rotate_image(cv.resize(images[noi], size), 
                                                                         rotationAngle, not_print = True)
                    stopper = 10
                elif not positionCompromised:
                    # print('Premio')
                    positionList.append([x, y, scaleFactor])
                    labelsList.append(labels[noi])
                    img[x:x+scaleFactor, y:y+scaleFactor] = rotate_image(cv.resize(images[noi], size), 
                                                                         rotationAngle, not_print = True)
                    stopper = 10
                stopper += 1
        newPositionList = clossing_bnb(img, positionList)
        # newPositionList = list()
        # for box in positionList:
        #     newPositionList.append([box[1], box[0], box[1]+box[2], box[0]+box[2]])
        # print(newPositionList)
        data_dic =  list()
        for label, newPosition in zip(labelsList, newPositionList):
            data_dic.append({'name': label, 'xmin': newPosition[0],
                             'ymin': newPosition[1], 'xmax': newPosition[2],
                             'ymax': newPosition[3]})
        fileName = str(uuid.uuid4())
        toPascalVocFormat(fileName, directory_name, data_dic, image_size)
        cv.imwrite(os.path.join(directory_name, fileName+'.jpg'), img)
        # plt.imshow(img)
        # plt.show()

# # Show dataset images and boxes to check the results
# def show_results(images_tensor, result_tensor, grid, bnb, classes, threshold=0.5):
#     for img in range(images_tensor.shape[0]):
#         image_rect = list()
#         for row in range(predictions.shape[1]):
#             for col in range(predictions.shape[2]):
#                 for p, pred in enumerate(predictions[img, row, col, classes: classes+bnb]):
#                     if p == 0:
#                         prob = pred
#                         best = p
#                     elif pred > prob:
#                         prob = pred
#                         best = p
#                     else:
#                         pass
#         if prob > threshold:
#             center_x = (predictions[img, row, col, classes+bnb+best*4]+col)/grid
#             center_y = (predictions[img, row, col, classes+bnb+best*4+1]+row)/grid
#             w = predictions[img, row, col, classes+bnb+best*4+2]
#             h = predictions[img, row, col, classes+bnb+best*4+3]
#             image_rect.append([center_x, center_y, w, h])
#         else:
#             pass
#         print(img)
#         image = images_tensor[img, :, :, 0:]
#         # image = tf.squeeze(image, [0])
#         image = tf.tile(image, [1,1,3])*255
#         image = image.numpy().astype(np.uint8)
#         # print(image)
#         print('we extract the info!')
#         for rect in image_rect:
#             x_min = rect[0] - rect[2]/2
#             x_max = rect[0] + rect[2]/2
#             y_min = rect[1] - rect[3]/2
#             y_max = rect[1] + rect[3]/2
#             image = cv.rectangle(image, (int(144*x_min), int(144*y_min)),
#                                  (int(144*x_max), int(144*y_max)), (255, 0, 0), 2)
#         plt.imshow(image)
#         plt.show()