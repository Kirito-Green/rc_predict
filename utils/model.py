import sys
import os
import time
sys.path.append(os.path.join(os.getcwd(), '..'))

from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
import numpy as np

from spektral.data import BatchLoader
from utils.analysis import error_analysis, scatter_plot, scatter_plot_all
from config import *


def huber_loss(y_true, y_pred):
    """Loss function for value"""
    e = tf.subtract(y_pred, y_true)
    return tf.reduce_mean(tf.where(tf.abs(e) < 1, tf.square(e) * 0.5, tf.abs(e) - 0.5))


def msre_loss(y_true, y_pred):
    loss = tf.square(tf.subtract(1.0, tf.divide(y_pred, y_true)))
    return tf.reduce_mean(loss)


def mse_msre_loss(y_true, y_pred):  # predict vs target
    """Mean squared error loss function"""
    ae = tf.square(tf.subtract(y_true, y_pred))  # absolute error
    loss = tf.where(tf.abs(y_true) < 1, ae, tf.square(
        tf.subtract(1.0, tf.divide(y_pred, y_true))))
    return tf.reduce_mean(loss)


def measure_ratio_good(y_true, y_pred):
    """Measure the ratio of good predictions(absolute relative error)"""
    re = tf.abs(tf.subtract(1.0, tf.divide(y_pred, y_true)))
    return tf.reduce_sum(tf.cast(re <= tolerant_ratio_error, tf.float32)) / tf.cast(tf.size(y_true), tf.float32)


def measure_ratio_bad(y_true, y_pred):
    """Measure the ratio of bad predictions(absolute relative error)"""
    re = tf.abs(tf.subtract(1.0, tf.divide(y_pred, y_true)))
    return tf.reduce_sum(tf.cast(re > tolerant_ratio_error, tf.float32)) / tf.cast(tf.size(y_true), tf.float32)


def sync(net, tar_net):
    """Synchronize target network with main network"""
    for var, tar_var in zip(net.trainable_variables, tar_net.trainable_variables):
        tar_var.assign(var)


# analysis
def gnn_analysis(model, batch_size, x_train, y_train, x_valid, y_valid, x_test, y_test, name):
    x_train_loader = BatchLoader(x_train, batch_size=batch_size, shuffle=False)
    x_valid_loader = BatchLoader(x_valid, batch_size=batch_size, shuffle=False)
    x_test_loader = BatchLoader(x_test, batch_size=batch_size, shuffle=False)
    y_train_predict = model.predict(x_train_loader.load(
    ), steps=x_train_loader.steps_per_epoch).reshape(-1, 1)
    y_valid_predict = model.predict(x_valid_loader.load(
    ), steps=x_valid_loader.steps_per_epoch).reshape(-1, 1)
    y_test_predict = model.predict(x_test_loader.load(
    ), steps=x_test_loader.steps_per_epoch).reshape(-1, 1)

    dict_train = error_analysis(
        y_train, y_train_predict, title=f'{name} train analysis')
    dict_valid = error_analysis(
        y_valid, y_valid_predict, title=f'{name} valid analysis')
    dict_test = error_analysis(
        y_test, y_test_predict, title=f'{name} test analysis')

    return {
        'model': name,
        'train avg err': dict_train['mean error'],
        'train max err': dict_train['max error'],
        f'train ratio(err>{tolerant_ratio_error})': dict_train['bad ratio'],
        'valid avg err': dict_valid['mean error'],
        'valid max err': dict_valid['max error'],
        f'valid ratio(err>{tolerant_ratio_error})': dict_valid['bad ratio'],
        'test avg err': dict_test['mean error'],
        'test max err': dict_test['max error'],
        f'test ratio(err>{tolerant_ratio_error})': dict_test['bad ratio'],
    }


def gnn_plot(model, batch_size, x_train, y_train, x_valid, y_valid, x_test, y_test, dir, name):
    x_train_loader = BatchLoader(x_train, batch_size=batch_size, shuffle=False)
    x_valid_loader = BatchLoader(x_valid, batch_size=batch_size, shuffle=False)
    x_test_loader = BatchLoader(x_test, batch_size=batch_size, shuffle=False)
    y_train_pred = model.predict(x_train_loader.load(
    ), steps=x_train_loader.steps_per_epoch).reshape(-1, 1)
    y_valid_pred = model.predict(x_valid_loader.load(
    ), steps=x_valid_loader.steps_per_epoch).reshape(-1, 1)
    y_test_pred = model.predict(x_test_loader.load(
    ), steps=x_test_loader.steps_per_epoch).reshape(-1, 1)

    # 1 pictures
    y_true = np.concatenate([y_train, y_valid, y_test], axis=0)
    y_pred = np.concatenate([y_train_pred, y_valid_pred, y_test_pred], axis=0)
    scatter_plot(y_true, y_pred, dir, f'{name}_all')

    # train valid test
    scatter_plot_all(
        y_train, y_train_pred, y_valid, y_valid_pred, y_test, y_test_pred, dir, f'{name}_train_valid_test')


def test_runtime(model, batch_size, x_test, y_test):
    x_test_loader = BatchLoader(x_test, batch_size=batch_size, shuffle=False)

    start = time.time()
    y_test_pred = model.predict(x_test_loader.load(
    ), steps=x_test_loader.steps_per_epoch).reshape(-1, 1)
    end = time.time()
    avg_time = (end - start) / len(y_test)
    print(f'Test time: {avg_time:.1f}s')

    return avg_time
