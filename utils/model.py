import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))

from config import *

def huber_loss(x, x_):
    """Loss function for value"""
    e = tf.subtract(x, x_)
    return tf.reduce_mean(tf.where(tf.abs(e) < 1, tf.square(e) * 0.5, tf.abs(e) - 0.5))


def mse_mrse_loss(x, x_): # predict vs target
    """Mean squared error loss function"""
    ae = tf.square(tf.subtract(x, x_)) # absolute error
    loss = tf.where(tf.abs(x_) < 1, ae, tf.square(tf.subtract(1.0, tf.divide(x, x_))))
    return tf.reduce_mean(loss)


def measure_ratio_good(x, x_):
    """Measure the ratio of good predictions(absolute relative error)"""
    re = tf.abs(tf.subtract(1.0, tf.divide(x, x_)))
    return tf.reduce_sum(tf.cast(re <= tolerant_ratio, tf.float32)) / tf.cast(tf.size(x), tf.float32)


def sync(net, tar_net):
    """Synchronize target network with main network"""
    for var, tar_var in zip(net.trainable_variables, tar_net.trainable_variables):
        tar_var.assign(var)
