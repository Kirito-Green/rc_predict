from config import *
from tensorflow.keras.initializers import GlorotUniform, HeUniform
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from spektral.layers import GINConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))


class GIN(Model):
    '''
    Graph Isomorphism Network model for graph prediction.
    '''

    def __init__(self, **kwargs):
        '''
        激活函数有: sigmoid, tanh, softmax, relu, leaky_relu, prelu, elu, selu, swish, mish
        Xavier(glorot)初始化适用于tanh、sigmoid
        he initializer适用于relu
        '''
        super().__init__()
        self.conv_layers_num = 3
        self.hidden_dim = 32
        self.droupout_rate = 0.5
        self.activation = 'relu'  # tanh
        self.pool_func = 'sum'
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
            self.conv_layers.append(GINConv(self.hidden_dim, mlp_hidden=[64, 64],
                                            activation=self.activation, mlp_activation=self.activation,
                                            aggregate='sum', kernel_initializer=self.kernel_initializer))

        if self.pool_func == 'avg':
            self.pool = GlobalAvgPool()
        elif self.pool_func == 'max':
            self.pool = GlobalMaxPool()
        elif self.pool_func == 'sum':
            self.pool = GlobalSumPool()
        else:
            raise ValueError('pool_func must be one of [avg, max, sum]')

        self.fc1 = Dense(2 * self.hidden_dim *
                         (self.conv_layers_num + 1), activation=self.activation)
        self.fc2 = Dense(1, activation='linear')

    def call(self, inputs):
        x, a = inputs
        x = tf.cast(x, tf.float32)
        a = tf.cast(a, tf.float32)
        a = tf.sparse.from_dense(a)

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

        # FC
        x = self.fc1(res)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # model build
    model_total = GIN()
    model_couple = GIN()
    print('model build done')
