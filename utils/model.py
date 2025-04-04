import sys
import os
sys.path.append(os.path.join(os.getcwd(), '..'))
import numpy as np
import tensorflow as tf

from config import *

def huber_loss(y_true, y_pred):
    """Loss function for value"""
    e = tf.subtract(y_pred, y_true)
    return tf.reduce_mean(tf.where(tf.abs(e) < 1, tf.square(e) * 0.5, tf.abs(e) - 0.5))


def mse_mrse_loss(y_true, y_pred): # predict vs target
    """Mean squared error loss function"""
    ae = tf.square(tf.subtract(y_true, y_pred)) # absolute error
    loss = tf.where(tf.abs(y_true) < 1, ae, tf.square(tf.subtract(1.0, tf.divide(y_pred, y_true))))
    return tf.reduce_mean(loss)


def measure_ratio_good(y_true, y_pred):
    """Measure the ratio of good predictions(absolute relative error)"""
    re = tf.abs(tf.subtract(1.0, tf.divide(y_pred, y_true)))
    return tf.reduce_sum(tf.cast(re <= tolerant_ratio_error, tf.float32)) / tf.cast(tf.size(y_true), tf.float32)


def sync(net, tar_net):
    """Synchronize target network with main network"""
    for var, tar_var in zip(net.trainable_variables, tar_net.trainable_variables):
        tar_var.assign(var)
