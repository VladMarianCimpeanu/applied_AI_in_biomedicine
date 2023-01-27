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

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

from IPython.display import clear_output
from PIL import Image
from datetime import datetime

tfk = tf.keras
tfkl = tf.keras.layers


def build_heatmap(model, validation_generator):
    """
    This method generates an heatmap used by seaborn to create a confusion matrix.
    :param model: model for which the method computes the confusion matrix.
    :param validation_generator: dataset object over which the model computes its predictions.
    :return pd.DataFrame object containing the confusion matrix.
    """
    y_predicted = model.predict(validation_generator)
    y_predicted = tf.argmax(y_predicted, axis=1)
    y_test_labels = np.concatenate([y for x, y in validation_generator], axis=0)
    y_test_labels = tf.argmax(y_test_labels, axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        y_test_labels, 
        y_predicted,
        num_classes=3
    )
    c = []
    for item in confusion_matrix:
        c.append(np.around(item / np.sum(item), decimals=3))
    df_heatmap = pd.DataFrame(c)
    return df_heatmap


def compute_report(model, test):
    """
    Compute recall, precision and f1-score for each class, for a model predicting
    on a given set.
    :param model: model to evaluate
    :param test: test set over which this method evaluate the input model.
    :return a pandas dataframe containing precision, recall and f1-score of the model.
    """

    # compute predictions
    y_pred = model.predict(test)
    y_pred = tf.argmax(y_pred, axis=1)

    # get true labels
    y_true = np.concatenate([y for x, y in test], axis=0)
    y_true = tf.argmax(y_true, axis=1)

    # return report  
    report = classification_report(y_true,
                                    y_pred,
                                    target_names=['n', 'p', 't'],
                                    output_dict=True)

    report = pd.DataFrame(report)
    report = report.drop(['accuracy',	'macro avg',	'weighted avg'], axis=1)
    return report


def plot_accuracy(history):
    """
    This function, given the history dictionary of a model, plots the accuracy plot.
    :param history: dictionary with keys 'accuracy' and 'val_accuracy'.
    """
    x_axis = len(history.history["accuracy"])
    print(max(history.history["val_accuracy"]), max(history.history["accuracy"]))
    history.history.keys()
    plt.figure(figsize=(12, 6))
    plt.plot([i for i in range(x_axis)], history.history["accuracy"], label="train accuracy")
    plt.plot([i for i in range(x_axis)], history.history["val_accuracy"], label="validation accuracy")
    plt.legend()

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.show()


def find_nearest(array, value):
    """
    This function compute the index of the closest element for a given vale.
    :param array: numpy array where to search.
    :param value: value to search.
    :return integer representing the index of the closest element.
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def compute_best_point(precision, recall):
    """
    Compute the closest point of the precision vs recall curve to the point (1, 1).
    Return an integer representing the index of the closest point to (1, 1).
    """
    distances = np.sqrt((1 - precision) ** 2 + (1 - recall) ** 2)
    return np.argmin(distances)


def plot_precision_recall_t(true_y, predictions):
    """
    This function plots the precision vs recall curve for the tubercolosis as positive class.
    :param true_y: array of one hot encoding for the true labels.
    :param predictions: array of predictions. Predictions must be distributions.
    :return best threshold (the one closest to the point with recall=precision=1).
    """
    labels = tf.argmax(true_y, axis=-1).numpy()

    mask_n = labels == 0
    mask_t = labels == 2
    mask = np.logical_or(mask_n, mask_t)

    precision, recall, thresholds_t = metrics.precision_recall_curve(mask_t[mask], predictions[mask, 2])
    
    color_t = "#7209b7"

    plt.figure(figsize=(7, 7))

    idx_t_m = compute_best_point(precision=precision, recall=recall)
    idx_t = find_nearest(thresholds_t, 0.5)

    plt.plot(
        recall,
        precision,
        label='Tubercoloss as positive class',
        color=color_t
    )
    plt.plot(recall[idx_t_m],
        precision[idx_t_m],
        marker="o",
        markersize=10,
        markeredgecolor=color_t,
        markerfacecolor=color_t,
        label=f'best threshold ({thresholds_t[idx_t_m]})'
    )
    plt.plot(recall[idx_t],
        precision[idx_t],
        marker="o",
        markersize=10,
        markeredgecolor="#f72585",
        markerfacecolor="#f72585",
        label=f'default threshold (0.5)'
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-vs-Recall curve")

    plt.legend()
    plt.show()
    return thresholds_t[idx_t_m]


def build_heatmap_custom(true_y, predictions, threshold, policy=None):
    """
    Compute pd dataframe containing the confusion matrix.
    :param true_y: array of one hot encoding for the true labels.
    :param predictions: array of predictions. Predictions must be distributions.
    :param policy: function policy to compute the label. If none(default), tf.argmax() is used.
    :param threshold: parameter of policy
    :return a pandas dataframe containing the confusion matrix.
    """
    if policy is None:
        y_pred = tf.argmax(predictions, axis=-1)
    else:
        y_pred = policy(predictions, threshold)

    # get true labels
    y_true = tf.argmax(true_y, axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        y_true, 
        y_pred,
        num_classes=3
    )
    c = []
    for item in confusion_matrix:
        c.append(np.around(item / np.sum(item), decimals=3))
    df_heatmap = pd.DataFrame(c)
    return df_heatmap


def compute_report_custom(true_y, predictions, threshold, policy=None):
    """
    Compute recall, precision and f1-score for each class, for a model predicting
    on a given set.
    :param true_y: array of one hot encoding for the true labels.
    :param predictions: array of predictions. Predictions must be distributions.
    :param threshold: parameter of policy
    :param policy: function policy to compute the label. If none(default), tf.argmax() is used. 
    :return a pandas dataframe containing precision, recall and f1-score of the model.
    """

    # compute predictions
    if policy is None:
        y_pred = tf.argmax(predictions, axis=-1)
    else:
        y_pred = policy(predictions, threshold)

    # get true labels
    y_true = tf.argmax(true_y, axis=1)

    # return report  
    report = classification_report(y_true,
                                 y_pred,
                                 target_names=['n', 'p', 't'],
                                 output_dict=True)

    report = pd.DataFrame(report)
    report = report.drop(['accuracy', 'macro avg', 'weighted avg'], axis=1)
    return report


def label_policy(y, threshold):
    """
    This function determines the decision policy to determine the labels of a prediction.
    :param y: is an array of predictions one hot encoded.
    :param threshold: threshold above which, tuberculosis is selected.
    :return decoded predictions.
    """
    tubercolosis_sensitivity = threshold
    
    default_labels = tf.argmax(y, axis=-1).numpy()
    pneumonia_mask = default_labels == 1
    n_and_t_mask = default_labels != 1

    tubercolosis_mask = y[:, 2] >= tubercolosis_sensitivity
    tubercolosis_mask = np.logical_and(tubercolosis_mask, n_and_t_mask)

    healthy_mask = y[:, 2] < tubercolosis_sensitivity
    healthy_mask = np.logical_and(healthy_mask, n_and_t_mask)
    
    default_labels[tubercolosis_mask] = 2
    default_labels[healthy_mask] = 0
    
    return default_labels