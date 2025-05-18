from tensorflow.keras.initializers import GlorotUniform, HeUniform
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from spektral.layers import GATConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))


class GAT(Model):
    '''
    Graph Attention Network (GAT) model for graph prediction.
    '''

    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_dim = 128
        self.num_heads = 3
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
        self.conv1 = GATConv(self.hidden_dim, attn_heads=self.num_heads, concat_heads=False,
                             activation=self.activation, kernel_initializer=self.kernel_initializer)  # [batch, n, 16]
        self.conv2 = GATConv(self.hidden_dim, attn_heads=self.num_heads, concat_heads=False,
                             activation=self.activation, kernel_initializer=self.kernel_initializer)  # [batch 256, 16]
        self.conv3 = GATConv(self.hidden_dim, attn_heads=self.num_heads, concat_heads=False,
                             activation=self.activation, kernel_initializer=self.kernel_initializer)  # [batch 64, 16]

        if self.pool_func == 'avg':
            self.pool = GlobalAvgPool()
        elif self.pool_func == 'max':
            self.pool = GlobalMaxPool()
        elif self.pool_func == 'sum':
            self.pool = GlobalSumPool()
        else:
            raise ValueError('pool_func must be one of [avg, max, sum]')

        self.fc1 = Dense(self.hidden_dim * 2, activation=self.activation)
        self.fc2 = Dense(1, activation='linear')

    def call(self, inputs):
        x, a = inputs
        x = tf.cast(x, tf.float32)
        a = tf.cast(a, tf.float32)

        # GAT
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])

        # readout layer
        x = self.pool(x)

        # FC
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    # model build
    model_total = GAT()
    model_couple = GAT()
