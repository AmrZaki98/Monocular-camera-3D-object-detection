import numpy as np
import cv2
import os
import time 
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, BatchNormalization,Input,Flatten, LeakyReLU, Reshape,Softmax
from tensorflow.keras.models import Model
BIN = 16

dims_avg = {'Cyclist': np.array([ 1.73532436,  0.58028152,  1.77413709]), 
            'Van': np.array([ 2.18928571,  1.90979592,  5.07087755]), 
            'Tram': np.array([  3.56092896,   2.39601093,  18.34125683]), 
            'Car': np.array([ 1.52159147,  1.64443089,  3.85813679]), 
            'Pedestrian': np.array([ 1.75554637,  0.66860882,  0.87623049]),
            'Truck': np.array([  3.07392252,   2.63079903,  11.2190799 ])}
file_path="./kitti/data_object_label_2/training/label_2/"
image_path="./kitti/data_object_image_2/training/image_2/"
possible_labels=["car","Pedestrian","van","Cyclist"]

def orientation_loss(y_true, y_pred):
    # Find number of anchors
    anchors = tf.reduce_sum(tf.square(y_true), axis=2)
    anchors = tf.greater(anchors, tf.constant(0.5))
    anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)
    
    # Define the loss
    loss = -(y_true[:,:,0]*y_pred[:,:,0] + y_true[:,:,1]*y_pred[:,:,1])
    loss = tf.reduce_sum(loss, axis=1)
    loss = loss / anchors
    
    return tf.reduce_mean(loss)

def Get_Model(BIN=BIN):
    I=Input(shape=(64,64,3))
    label=Input(shape=(1))
    #label_one_hot=tf.feature_column.indicator_column(label)
    feat=Conv2D(64,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(I)
    feat=Conv2D(64,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=MaxPool2D(2)(feat)
    feat=BatchNormalization()(feat)
    feat=Conv2D(128,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=Conv2D(128,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=MaxPool2D(2)(feat)
    feat=BatchNormalization()(feat)
    feat=Conv2D(256,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=Conv2D(256,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=MaxPool2D(2)(feat)
    feat=BatchNormalization()(feat)
    feat=Conv2D(512,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=Conv2D(512,3,padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(feat)
    feat=MaxPool2D(2)(feat)
    feat=BatchNormalization()(feat)
    feat=Dropout(0.5)(feat)
    feat=Flatten()(feat)
    
    
    confidence=Dense(512)(feat)
    confidence=LeakyReLU(0.1)(confidence)
    confidence=Dropout(0.5)(confidence)
    confidence=Dense(256)(confidence)
    confidence=LeakyReLU(0.1)(confidence)
    confidence=Dropout(0.5)(confidence)
    confidence=Dense(BIN)(confidence)
    confidence=Softmax(name="confidence")(confidence)
    
    
    """
    orientation=Dense(256)(feat)
    orientation=BatchNormalization()(orientation)
    orientation=LeakyReLU(0.1)(orientation)
    orientation=Dropout(0.5)(orientation)
    orientation=Dense(2*BIN,kernel_regularizer=tf.keras.regularizers.L2(l2=0.0005))(orientation)
    orientation=Reshape([-1,BIN,2])(orientation)
    orientation=tf.math.l2_normalize(orientation,dim=2,name="Orientation")
    """

    
    model=Model(inputs=I,outputs=[confidence])
    return model
    


