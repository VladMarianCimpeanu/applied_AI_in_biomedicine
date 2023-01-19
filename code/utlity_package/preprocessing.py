import keras_cv
import tensorflow as tf
import cv2
import warnings
import logging
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

from IPython.display import clear_output
from PIL import Image
from datetime import datetime

tfk = tf.keras
tfkl = tf.keras.layers


image_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string), 
    "class": tf.io.FixedLenFeature([], tf.int64), 
    }

autotune = tf.data.AUTOTUNE

def preprocess_data(img):
  """
  Apply blur and contrast limited adpative histogram equalization to 
  increase the image quality
  """
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img_np = img.numpy()
  img_np = np.squeeze(img_np)
  grayscale_img = img_np.astype('uint8')
  grayscale_img = clahe.apply(grayscale_img)
  grayscale_img = cv2.GaussianBlur(grayscale_img,(5,5),0)
  grayscale_img = np.expand_dims(grayscale_img, axis=2)
  return tf.convert_to_tensor(grayscale_img)


def tf_clahe(image, label):
  im_shape = image.shape
  x = tf.py_function(preprocess_data, [image], Tout=image.dtype)
  # the shape is set explicitly because tensorflow can not ensure
  # that the shape is not modified during the execution of the function
  x.set_shape(im_shape)
  return x, label


def to_dict(image, label):
    image = tf.cast(image, tf.float32)
    return {"images": image, "labels": label}


def mixup(samples):
    samples = keras_cv.layers.MixUp()(samples, training=True)
    return samples


def prep_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32)
    return images, labels


def _parse_data(unparsed_example):
    return tf.io.parse_single_example(unparsed_example, image_feature_description)


def _bytestring_to_pixels(parsed_example):
    byte_string = parsed_example['image']
    image = tf.io.decode_image(byte_string)
    image = tf.reshape(image, [256, 256, 1])
    return image, parsed_example["class"]


def rescale(img, label):
    x = tf.cast(img, tf.float32) / 255.0
    # in the tfrecord format, labels are in sparse representation. Bring them to one-hot.
    y = tf.one_hot(label, depth=3)
    return x, y