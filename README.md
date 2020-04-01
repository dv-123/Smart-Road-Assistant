# Smart-Road-Assistant

# Abstract-
This device is developed for driver assistance in night in order to reduce the probability of accidents due to low visibility and other night circumstances The assistant will turn on in right time or in lower visibility and will help the driver to improve his safety by providing the alerting signals.

# what we are going to do is
first we are going to take the night images of roads and vehicles on it we are going to apply Equalization technique such as histogram equalizer to enhance contrast level of the input image then the pre-process input image goes to the YOLO V3 - you only look once - algorithm version of neural network CNN predict the obstacle in front of the car if there is any obstacle detected then there will be alert for the driver .

# what is our innovation - -
There is a problem of delay and we have tried to remove it by applying the probability theory and predicting the direction of motion of the next vehicle for next 2 or 3 seconds and on the basis of this judgement we are going to give alert signal to driver in order to increase the response time of the driver.

# Problem Statement
Solution for improving road safety by increasing better visibility for night driving. This problem is related to the improvement of object detection ability for cars and drivers at night when visibility is low. It is critical to solving because many accidents at roads happen at night due to lower visibility and carelessness of drivers, so this will assist the drivers in remaining active and will reduce the chances of accidents.

# SolutionNRA (Night Road Assistant)-
It helps the driver to run the vehicle safely even in the low/dim lighting conditions. It uses the input from the high-quality camera then this input is preprocessed using image equalization techniques(contrast management) and then applied to the YOLO-V3 to predict what kind of object is in the front of vehicle and on the basis of the distance measured the system will check the probability of the object to get in collision with the vehicle and generate an alert signal to the driver to avoid collisions/accident.InnovationThere is a problem of delay and we have tried to remove it by applying the probability theory and predicting the direction of motion of the next vehicle for next 2 or 3 seconds and on the basis of this judgment we are going to give an alert signal to driver in order to increase the response time of the driver. Also, we will be enhancing the night vision of the autonomous system by using various sensors and contrast management techniques and interpolation.

# YOLO for Object Detection
Object detection is a computer vision task that involves both localizing one or more objects within an image and classifying each object in the image. It is a challenging computer vision task that requires both successful object localization in order to locate and draw a bounding box around each object in an image, and object classification to predict the correct class of object that was localized. The “You Only Look Once,” or YOLO, family of models are a series of end-to-end deep learning models designed for fast object detection, developed by Joseph Redmon, et al. and first described in the 2015 paper titled “You Only Look Once: Unified, Real-Time Object Detection.” The approach involves a single deep convolutional neural network (originally a version of GoogLeNet, later updated and called DarkNet based on VGG) that splits the input into a grid of cells and each cell directly predicts a bounding box and object classification. The result is a large number of candidate bounding boxes that are consolidated into a final prediction by a post-processing step. There are three main variations of the approach, at the time of writing; they are YOLOv1, YOLOv2, and YOLOv3. The first version proposed the general architecture, whereas the second version refined the design and made use of predefined anchor boxes to improve bounding box proposal, and version three further refined the model architecture and training process. Although the accuracy of the models is close but not as good as Region-Based Convolutional Neural Networks (R-CNNs), they are popular for object detection because of their detection speed, often demonstrated in real-time on video or with camera feed input. Further YOLO V3 can be run using the steps given on the following link https://pjreddie.com/darknet/yolo/

# Kalman Filter
Class implementing Kalman filter algorithm. This class implements the Kalman filter algorithm. It maintains the current state and output estimates, and updates these when new data is provided. The system model is given in the form of the state space matrices A, B, C, D, and process and sensor noise covariance matrices Q and R: x(k+1) = A x(k) + B u(k) + N(0,Q) y(k) = C x(k) + D u(k) + N(0,R) where N is the standard normal distribution with zero mean and covariance Q or R. This class implements the Observer interface. Methods • initialize(KF,t0,x0,u0) - Initialize filter given initial time, state, and inputs • estimate(KF,t,u,z) - Update the state and outpute estimates given new input and output data. • getStateEstimate(KF) - Return a state estimate structure with mean and covariance. Note that although these methods take time as an input, the state space matrices are time-invariant and so must be defined with respect to the desired sampling rate. The filter only keeps track of the current time given to it and does not use time in any calculations. See also ExtendedKalmanFilter, UnscentedKalmanFilter, https://docs.opencv.org/trunk/dd/d6a/classcv_1_1KalmanFilter.html

# Histogram equalization
This method usually increases the global contrast of many images, especially when the usable data of the image is represented by close contrast values. Through this adjustment, the intensities can be better distributed on the histogram. This allows for areas of lower local contrast to gain a higher contrast. Histogram equalization accomplishes this by effectively spreading out the most frequent intensity values.


# Some Extra Information: 

Model and weight file not uploaded


