# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 02:03:30 2019

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
from yolo_utils_vid_Kalman import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes,Predict
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

#%%

# creating yolo filter boxes
#%%


#%%

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    
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
    ''' 
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
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    
    # Runing the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # We will need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    
    # Print predictions info
    #d = 0
    #print('Found {} boxes'.format(len(out_boxes)))
    #for z in out_classes:
    #    d = 1+d
    #    print('box {} is {}'.format(d, class_names[z]))
    
    # Generating colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    
    return image,out_scores, out_boxes, out_classes,colors,image_file
    
#%%
# running the predict function
cap = cv2.VideoCapture("test1.mp4")
#%%

#%%
while True:
    
    ret, frame = cap.read()
    #img_YCC = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    #img_YCC[:,:,0] = cv2.equalizeHist(img_YCC[:,:,0])
    #img_output = cv2.cvtColor(img_YCC, cv2.COLOR_YCrCb2BGR)
    #blur = cv2.medianBlur(img_output,3)
    image,out_scores, out_boxes, out_classes,colors,image_file = predict(sess, frame)
    image = cv2.resize(image,(1280,720),interpolation = cv2.INTER_CUBIC)
    prediction = Predict()
    prediction.Predict_Path(out_boxes, out_classes, image)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
    #cv2.rectangle(image, (10, 500), (0, 500), (0,0,255), 8)
    
    #%%
#    last_prediction = current_prediction
#    last_measurement = cur_measurement
#    for i, c in reversed(list(enumerate(out_classes))):
#        predicted_class = class_names[c]
#        box = out_boxes[i]
#        x = (box[0] + box[2])/2
#        y = (box[1] + box[3])/2
#        cur_measurement = np.array([[np.float32(x)],[np.float32(y)]])
#        kalman.correct(cur_measurement)
#        current_prediction = kalman.predict()
#        lmx, lmy = last_measurement[0], last_measurement[1]
#        cmx, cmy = cur_measurement[0], cur_measurement[1]
#        lpx, lpy = last_prediction[0], last_prediction[1]
#        cpx, cpy = current_prediction[0], current_prediction[1]
#        cv2.line(image, (lmx, lmy), (cmx, cmy), (0,0,225))
#        cv2.line(image, (lpx, lpy), (cpx, cpy), (225,255,255))
        
    #%%
    cv2.imshow("output",frame)
    cv2.imshow("debug",image)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()

#%%
