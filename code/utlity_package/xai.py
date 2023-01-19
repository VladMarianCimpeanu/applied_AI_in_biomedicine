"""
xai.py
This package contains all the useful functions used for the explainability pipeline of the model.
"""


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


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    This function computes the heatmap from the last convolution layer of the given model.
    This function has been taken from the keras guide to the grad-CAM impementation.
    :param img_array: image to be classified and explained by the model. Since the model takes as input a batch of images, 
    The first dimension should be 1.
    :param model: keras model used for the explaination.
    :param last_conv_layer_name: name of the last convolution layer for the given model.
    :param pred_index: optional parameter. True label index.
    :return numpy array containing the heatmap.
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def gradcam(img, heatmap, alpha=0.4):
    """
    This function return the input image superimposed with the given heatmap.
    :param img: numpy image to be modified.
    :param heatmap: numpy array of the heatmap to superimpose.
    :param alpha: transparency of the heatmap.
    :return superimposed heatmap over the image in numpy array. 
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tfk.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tfk.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tfk.utils.array_to_img(superimposed_img)

    return superimposed_img


def apply_patch(img_to_patch, h=50, w=50, x=0, y=0, color=(230 / 255., 57 / 255., 70 / 255.), alpha= 0.5):
    """
    This function applies a patch over the input image.
    :param img_to_patch: numpy array of the image to patch. It can have 2 or 3
    dimensions.
    :param h: patch's height.
    :param w: patch's width.
    :param x: x coordinate of the pach's top left corner.
    :param y: y coordinate of the pach's top left corner.
    :param color: Optional parameter when using 3 dimensional images. Color of the patch.
    If the image is bidimensional, the patch will be black.
    :param alpha: transparency of the patch.
    :return numpy array of the patched image.
    """
    img_to_patch = img_to_patch.copy()
    h_limit = np.clip([x + w], 0, img_to_patch.shape[0])[0]
    v_limit = np.clip([y + h], 0, img_to_patch.shape[1])[0]
    if len(img_to_patch.shape) == 2:
        mask = np.zeros_like(img_to_patch[y: v_limit, x: h_limit,])
        img_to_patch[y: v_limit, x: h_limit] = mask
    else:
        for indx, rgb_element in enumerate(color):
            mask = np.full_like(img_to_patch[y: v_limit, x: h_limit, indx], rgb_element)
            original_component = img_to_patch[y: v_limit, x: h_limit, indx]
            img_to_patch[y: v_limit, x: h_limit, indx] = original_component * (1 - alpha) + mask * alpha
    return img_to_patch


def print_distribution(distribution, precision=2):
    dist_str = 'Distribution: '
    for item, state in zip(distribution, ['n', 'p', 't']):
        dist_str += f'{state}- {np.format_float_positional(item, precision=precision)} --'
    return dist_str
        

def get_occlusion_interpretation(model, img, label, h=50, w=50, alpha=0.5, animation=True, h_step=None, w_step=None, preprocessing=None):
    """
    This function applies occlusion method to explain a black_box model.
    It returns an image with superimpressed patches over the significant parts
    of the image.
    :param model: model to explain.
    :param img: image to use for the prediction and to explain. The image must be
    a numpy array with 2 dimensions and values take place in the interval [0, 1].
    :param label: true label of the image.(sparse representation)
    :param h: height of the patch to apply.
    :param w: width of the patch to apply.
    :param alpha: transparency of the explainable patch.
    :param preprocessing: preprocessing function for the model.
    :return numpy.array with 3 channels.
    """
    important_patches = []
    if w_step == None:
        w_step = w // 8
    if h_step ==None:
        h_step = h // 8
    for x in range(0, img.shape[0], w_step):
        for y in range(0, img.shape[1], h_step):
            # transform image to the right shape for the predicttion.
            
            _ = np.expand_dims(apply_patch(img, h, w, x, y), axis=[0, -1])
            if preprocessing is not None:
                _ *= 255
                _ = preprocessing(_)
            prediction = model.predict(_, verbose=0)
            l = tf.argmax(prediction, axis=-1).numpy()[0]
            color=(0.32, 0.717, 0.53)
            if l != label:
                important_patches.append((x, y))
                color = (230 / 255., 57 / 255., 70 / 255.)  
                
            # animated scan
            if animation:
                clear_output(wait=True)
                img_to_plot = np.stack((img,) * 3, axis=-1)
                img_to_plot = apply_patch(img_to_plot, h, w, x, y, color=color)
                plt.title(print_distribution(prediction[0], precision=2))
                plt.grid()
                plt.axis('off')
                plt.imshow(img_to_plot)
                plt.show()
                #time.sleep(0.01)

    #highlight important_patches
    highlighted_img = img.copy()
    highlighted_img = np.stack((highlighted_img,) * 3, axis=-1)
    for x, y in important_patches:
        highlighted_img = apply_patch(highlighted_img, h, w, x, y, alpha=alpha)
    return highlighted_img