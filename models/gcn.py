import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))

from tensorflow.keras.initializers import GlorotUniform, HeUniform
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

from config import *
from spektral.layers import GCNConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool


class GCN(Model):
    '''
    Graph Convolutional Network (GCN) model for graph prediction.
    '''

    def __init__(self, **kwargs):
        '''
        激活函数有: sigmoid, tanh, softmax, relu, leaky_relu, prelu, elu, selu, swish, mish
        Xavier(glorot)初始化适用于tanh、sigmoid
        he initializer适用于relu

        '''
        super().__init__()
        self.conv_layers_num = 3
        self.hidden_dim = 128
        self.droupout_rate = 0.5
        self.activation = 'relu'  # tanh
        self.pool_func = 'sum'
        self.readout_mode = 'single'  # single concat
        # droupout
        if 'training' in kwargs:
            self.training = kwargs['training']
        else:
            self.training = True

        if self.activation == 'relu':
            self.kernel_initializer = HeUniform()
        else:
            self.kernel_initializer = GlorotUniform()

        self.conv_layers = []
        for i in range(self.conv_layers_num):
            self.conv_layers.append(GCNConv(
                self.hidden_dim, activation=self.activation, kernel_initializer=self.kernel_initializer))

        if self.pool_func == 'avg':
            self.pool = GlobalAvgPool()
        elif self.pool_func == 'max':
            self.pool = GlobalMaxPool()
        elif self.pool_func == 'sum':
            self.pool = GlobalSumPool()
        else:
            raise ValueError('pool_func must be one of [avg, max, sum]')

        self.fc1 = Dense(2 * self.hidden_dim, activation=self.activation)
        self.fc2 = Dense(1, activation='linear')

    def add_self_loop(self, a):
        diag_values = tf.linalg.diag_part(a)
        mask_zero = abs(diag_values) < tolerant_zero_error
        diag_values = tf.where(mask_zero, 1.0, 0.0)
        diag_matrix = tf.linalg.diag(diag_values)
        a_hat = a + diag_matrix

        return a_hat

    def norm_a(self, a):
        # calculate the degree matrix
        d_values = tf.reduce_sum(a, axis=-1)
        d_minus_sqrt_values = tf.pow(d_values, -0.5)
        d_minus_sqrt = tf.linalg.diag(d_minus_sqrt_values)
        # calculate the normalized adjacency matrix
        a_norm = tf.matmul(tf.matmul(d_minus_sqrt, a), d_minus_sqrt)

        return a_norm

    def call(self, inputs):
        x, a = inputs
        x = tf.cast(x, tf.float32)
        a = tf.cast(a, tf.float32)
        a = self.add_self_loop(a)
        a = self.norm_a(a)

        if self.readout_mode == 'single':
            # GCN layer
            for conv in self.conv_layers:
                x = conv([x, a])

            # Readout
            res = self.pool(x)
        elif self.readout_mode == 'concat':
            # GCN layer
            h = []
            h.append(self.pool(x))
            for conv in self.conv_layers:
                x = conv([x, a])
                h.append(self.pool(x))

            # Readout
            # res = concat(h1, h2, h3, ...)
            res = h[0]
            for i in range(1, len(h)):
                res = tf.concat([res, h[i]], axis=-1)
        else:
            raise ValueError('readout_mode must be one of [single, concat]')

        # FC
        x = self.fc1(res)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # model build
    model_total = GCN()
    model_couple = GCN()
    print('model build done')
