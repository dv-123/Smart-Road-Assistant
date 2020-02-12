import colorsys
import imghdr
import os
import random
import cv2
import sys
import math
from keras import backend as K

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img_input, model_image_size):
    image = img_input
    resized_image = cv2.resize(image,model_image_size,interpolation = cv2.INTER_CUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):

    thickness = 2 
    
    #d=0
    #a = [0,0,0]
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        #print(box)
        score = out_scores[i]
        
        label = '{} {:.2f}'.format(predicted_class, score)
        
        cv2.rectangle(image,(box[3],box[2]),(box[1],box[0]),colors[c],thickness)
        x_1 = int((box[3]+box[1])/2)
        y_1 = int((box[2]+box[0])/2)
        
        cv2.circle(image,(x_1,y_1),5,(0,255,0),-1)
        cv2.putText(image, label,(box[1],box[0]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(225,0,0))
        
        #area = (box[3]-box[1])*(box[2]-box[0])
        #d=d+1
        #a[d] = area
        #if d == 2:
        #    out = a[d-2] - a[d-1]
        #    print(out)
        #    d=0
        
        
        
        
        
        
        
        #print(area)

#%%
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted
    
class Predict:
    def Predict_Path(self,out_boxes,out_classes,frame):
        kfObj = KalmanFilter()
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            x_1 = int((box[3]+box[1])/2)
            y_1 = int((box[2]+box[0])/2)
            
            predictedCoords = kfObj.Estimate(x_1, y_1)
            cv2.circle(frame, (int(x_1), int(y_1)), 20, [0,0,255], 2, 8)
            cv2.line(frame,(int(x_1), int(y_1 + 20)), (int(x_1 + 50), int(y_1 + 20)), [100,100,255], 2,8)
            cv2.putText(frame, "Actual", (int(x_1 + 50), int(y_1 + 20)), cv2.FONT_HERSHEY_SIMPLEX,0.5, [50,200,250])
            
            cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0,255,255], 2, 8)
            cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15), (predictedCoords[0] + 50, predictedCoords[1] - 30), [100, 10, 255], 2, 8)
            cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
            
            x_p,y_p = predictedCoords[0],predictedCoords[1]
            
            slope = math.atan((y_p-y_1)/(x_p-x_1))
            slope = math.degrees(slope)
            
            cv2.rectangle(frame, (200,470), (1040, 700), (0,0,255), 2)
            
            xi1 = max(box[0],200)
            yi1 = max(box[1],470)
            xi2 = min(box[2],1040)
            yi2 = min(box[3],700)
            
            inter_area = max((yi2-yi1),0)* max((xi2-xi1),0)
            
            detected_box_area = (box[3]-box[1])*(box[2]-box[0])
            
            drawn_box_area = (1040-200)*(700-470)
            
            union_area = (detected_box_area+drawn_box_area)-inter_area
            
            i_o_u = inter_area/union_area
            
            #print(i_o_u)
            #print(inter_area)
            
            if i_o_u > 0.05:
                print("Accident can occur")
                #cv2.putText(frame,"There is a possibility of Crash !!!",cv2.FONT_HERSHEY_SIMPLEX,color)
            
            #area of bonut.
            #after setting on the car.
            #
            #if slope < 0:
            #    #print("ariving car")
            #    if x_p > 200 and x_p < 1040:
            #        if y_p > 470 and y_p < 700:
            #            print("Accident can occur")
            #else:
                #print("0")
            
            #print(slope)



