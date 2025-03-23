import warnings
import os

warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
from xml.etree import ElementTree
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from skimage import measure
import xml.etree.ElementTree as ET
import cairosvg
import plotly.express as px
from PIL import Image
import shutil 
from lxml import etree
import re
import io
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K


def residual_block(inputs, num_filters):
    res = inputs 
    
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if res.shape[-1] != x.shape[-1]:
        res = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same')(res)
    
    x = tf.keras.layers.Add()([x, res])
    x = tf.keras.layers.ReLU()(x)
    
    return x

def encoder_block(inputs, num_filters):
    x = residual_block(inputs, num_filters)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    if skip_features.shape[1] != x.shape[1] or skip_features.shape[2] != x.shape[2]:
        skip_features = tf.keras.layers.UpSampling2D(size=(2, 2))(skip_features)
    
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = residual_block(x, num_filters)  
    return x

# def resunet_model(input_shape=(384, 384, 3), num_classes=1):
#     inputs = tf.keras.layers.Input(input_shape)
    
#     s1 = encoder_block(inputs, 32)
#     s2 = encoder_block(s1, 64)
#     s3 = encoder_block(s2, 128)
#     s4 = encoder_block(s3, 256)
    
#     b1 = residual_block(s4, 512)
    
#     d1 = decoder_block(b1, s4, 256)
#     d2 = decoder_block(d1, s3, 128)
#     d3 = decoder_block(d2, s2, 64)
#     d4 = decoder_block(d3, s1, 32)
    
#     outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)
    
#     model = tf.keras.models.Model(inputs, outputs, name="ResU-Net")
    
#     return model

# model = resunet_model()
# model.summary()


def resunet_model(input_shape=(320, 320, 3), num_classes=1):
    inputs = tf.keras.layers.Input(input_shape)
    
    s1 = encoder_block(inputs, 32)
    s2 = encoder_block(s1, 64)
    s3 = encoder_block(s2, 128)
    # s4 = encoder_block(s3, 256)
    
    b1 = residual_block(s3, 256)
    
    # d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(b1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)
    
    model = tf.keras.models.Model(inputs, outputs, name="ResU-Net")
    
    return model

model = resunet_model()
model.summary()



def calculate_dice_coefficient(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    y_pred = K.cast(y_pred, dtype='float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return K.mean(dice)


def calculate_dice(pred_binary, gt_binary):
    """ 
    Compute dice coefficient for two binary masks.
    
    Parameters:
    -----------
    pred_binary (numpy array): Binary mask for predicted panel.
    gt_binary (numpy array): Binary mask for ground truth panel.
    
    Returns:
    --------
    float: Dice coef value.
    """

    intersection = np.logical_and(pred_binary, gt_binary).sum()
    sum_cardinality = pred_binary.sum() + gt_binary.sum()
    return 2*intersection / sum_cardinality if sum_cardinality > 0 else 0



def calculate_mean_dice(predicted_mask, gt_mask):
    """ 
    Calculate mean dice coefficient panel-wise

    Parameters:
    -----------
    predicted_mask (numpy array): Binary mask from the model.
    gt_mask (numpy array): Binary ground truth mask.

    Returns:
    --------
    float: Mean dice coef value.
    """

    pred_labels = label(predicted_mask)
    gt_labels = label(gt_mask)
    
    pred_objects = np.unique(pred_labels[pred_labels > 0])
    gt_objects = np.unique(gt_labels[gt_labels > 0])
    
    dice_values = []

    for gt_label in gt_objects:
        gt_binary = gt_labels == gt_label
        best_dice = 0
        
        for pred_label in pred_objects:
            pred_binary = pred_labels == pred_label
            dice = calculate_dice(pred_binary, gt_binary)
            best_dice = max(best_dice, dice)
        
        dice_values.append(best_dice)
    
    mean_dice = np.mean(dice_values) if dice_values else 0
    return mean_dice, dice_values
