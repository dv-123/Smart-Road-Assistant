# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 23:45:13 2019

@author: bhaik
"""

#%%
# importing required libraries

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import cv2
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

#%%

# creating yolo filter boxes

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    
    '''
    Filters the classified YOLO boxes by thresholding on the objects classified and on the basis of box_confidence
    
    Returns values of the filter:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because we don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    
    '''
    
    # Computing the box scores
    box_scores = np.multiply(box_confidence,box_class_probs)
    
    # Finding the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)      # gives the index of the maximum number in the obtained box_score matrix/array (1D)
    box_class_scores = K.max(box_scores, axis=-1)    # gives the maximum value from the box_score
    
    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and the boxes with probability >= threshold are classified as True or 1 and remaining False or 0
    filtering_mask = K.greater_equal(box_class_scores, threshold)
    
    # Applying the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores,filtering_mask)       # the scores grater than corresponding threshold are selected
    boxes = tf.boolean_mask(boxes,filtering_mask)                   # the boxes grater than corresponding threshold are selected
    classes = tf.boolean_mask(box_classes,filtering_mask)           # the classes grater than corresponding threshold are selected
    
    return scores, boxes, classes
#%%

# Creating Intersection / Union Function --> IOU

def iou(box1,box2):
    ''' This function implements Intersection Over Union
    
    The corresponding dimenssions of box1 and box2 are:
        box1 --> (x1, y1, x2, y2)
        box2 --> (x1, y1, x2, y2)
    '''
    # calculation the area of intersection
    xi1 = max(box1[0],box2[0])
    yi1 = max(box1[1],box2[1])
    xi2 = min(box1[2],box2[2])
    yi2 = min(box1[3],box2[3])
    inter_area = max((yi2-yi1),0)* max((xi2-xi1),0)
    
    # the max between 0 and the actual area is taken because we will not consider the negative values.
    
    # Calculating the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[3]-box1[1])*(box1[2]-box1[0])
    box2_area = (box2[3]-box2[1])*(box2[2]-box2[0])
    union_area = (box1_area+box2_area)-inter_area
    
    # computing the iou
    iou = inter_area/union_area
    
    return iou
#%%
    
# making the YOLO non-max supression function to remove the multiple uotputs of the same detected object

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.6):
    
    '''
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    these will be the final classified boxes
    
    '''
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # Using predefined function tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes,scores,max_boxes_tensor,iou_threshold=iou_threshold)
    
    # Using K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)
    
    return scores, boxes, classes
#%%
    
# creating the yolo exaluation function

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    '''
    This function converts the output of YOLO encoding (a lot of boxes) to our predicted boxes along with their scores, box coordinates and classes.
    
    One main argument to keep in mind is fron the yolo trained model -->
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
    
    the finction will return following values -->
    cores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    '''
    
    # first we will be retriving the outputs from the yolu outputs of the model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    
    # Converting boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    # this function is imported from yad2k.models.keras_yolo
    
    # filtering the boxes
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    
    # scaling the boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)
    # this is imported from yolo_utils
    
    # using the non-max supression on the filtered bosex
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)
    
    return scores, boxes, classes
#%%
    
# initialising the session
sess = K.get_session()

# gathering the class_names, anchors from model_data
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

# specifying the image shape
image_shape = (720., 1280.)

# loading the yolo pretrained model
print("loading YOLOV2 weights/model")
yolo_model = load_model("model_data/yolo.h5")
print("model loaded")

# model summary
yolo_model.summary()

# classifying the yolo outputs from the model with class names
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# applying the yolo_eval functions on yolo_outputs
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
#%%

# making the predict function
def predict(sess, image_file):
    
    '''
    The function will run the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    '''
    
    # processing the image data
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
    # Runing the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # We will need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    
    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    
    # Generating colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    # saving the image
    image.save(os.path.join("out", image_file), quality=90)
    
    # displaying the image
    img = cv2.imread(os.path.join("out", "test.jpg"),1)
    cv2.imshow('output',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return out_scores, out_boxes, out_classes
    
#%%
# running the predict function
out_scores, out_boxes, out_classes = predict(sess, "test.jpg")


#%%






















    