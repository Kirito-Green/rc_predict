import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))

from tensorflow.keras.initializers import GlorotUniform, HeUniform
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf

from config import *
from spektral.layers import GCNConv, GraphSageConv, ChebConv, TAGConv, GATConv, GINConv, \
    GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool


class GNN(Model):
    '''
    Graph Convolutional Network (GCN) model for graph prediction.'
    Using hierarchical pooling (DiffPool) to reduce the number of nodes in the graph.
    '''

    def __init__(self, **kwargs):
        '''
        激活函数有: sigmoid, tanh, softmax, relu, leaky_relu, prelu, elu, selu, swish, mish
        Xavier(glorot)初始化适用于tanh、sigmoid
        he initializer适用于relu

        '''
        super().__init__()
        self.block_num = 4
        self.conv_layers_num = 3
        self.k = self.conv_layers_num
        self.hidden_dim = 128
        self.droupout_rate = 0.5
        self.num_heads = 3
        self.activation = 'relu'  # tanh

        # model_name
        if 'model_name' in kwargs:
            self.model_name = kwargs['model_name']
        else:
            self.model_name = 'cheb'
        if self.model_name not in ['gcn', 'cheb', 'tag', 'gat', 'gin']:
            raise ValueError(
                'model_name must be one of [gcn, cheb, tag, gat, gin]')

        # training or inference
        if 'training' in kwargs:
            self.training = kwargs['training']
        else:
            self.training = True

        # kernel initializer
        if self.activation == 'relu':
            self.kernel_initializer = HeUniform()
        else:
            self.kernel_initializer = GlorotUniform()

        # pre layers
        self.pre_layers = Sequential()
        self.pre_layers.add(Dense(self.hidden_dim, activation=self.activation,
                                  kernel_initializer=self.kernel_initializer))
        self.pre_layers.add(Dense(self.hidden_dim, activation=self.activation,
                                  kernel_initializer=self.kernel_initializer))

        # conv layers and pool layers
        self.gnn_layers = []
        self.pool_nodes = [8**b for b in range(self.block_num-1, -1, -1)] # [512, 64, 8, 1]
        # self.pool_features = [self.hidden_dim for b in range(self.block_num)]
        self.pool_features = [self.hidden_dim*(2**b) for b in range(1, self.block_num+1)] # [2, 4, 8, 16]
        if self.model_name in ['gcn', 'gat', 'gin']:  # k = 1
            for b in range(self.block_num):
                for i in range(self.conv_layers_num):
                    if self.model_name == 'gcn':
                        self.gnn_layers.append(GCNConv(
                            self.hidden_dim, activation=self.activation, kernel_initializer=self.kernel_initializer))
                    elif self.model_name == 'graphsage':
                        self.gnn_layers.append(GraphSageConv(self.hidden_dim, aggregate='mean', activation=self.activation,
                                                             kernel_initializer=self.kernel_initializer))
                    elif self.model_name == 'gat':
                        self.gnn_layers.append(GATConv(self.hidden_dim, attn_heads=self.num_heads, concat_heads=False,
                                                       activation=self.activation, kernel_initializer=self.kernel_initializer))
                    elif self.model_name == 'gin':
                        self.gnn_layers.append(GINConv(self.hidden_dim, mlp_hidden=[self.hidden_dim],
                                                       activation=self.activation, mlp_activation=self.activation,
                                                       aggregate='sum', kernel_initializer=self.kernel_initializer))
                self.gnn_layers.append(DiffPool(self.pool_nodes[b],
                                                channels=self.pool_features[b],
                                                activation=self.activation,
                                                kernel_initializer=self.kernel_initializer))
        else:  # k = 4 cheb tag
            for b in range(self.block_num):
                if self.model_name == 'cheb':
                    self.gnn_layers.append(ChebConv(self.hidden_dim, K=self.k, activation=self.activation,
                                                    kernel_initializer=self.kernel_initializer))
                elif self.model_name == 'tag':
                    self.gnn_layers.append(TAGConv(self.hidden_dim, K=self.k, activation=self.activation,
                                                   kernel_initializer=self.kernel_initializer))
                self.gnn_layers.append(DiffPool(self.pool_nodes[b],
                                                channels=self.pool_features[b],
                                                activation=self.activation,
                                                kernel_initializer=self.kernel_initializer))

        # post layers
        self.post_layers = Sequential()
        self.post_layers.add(Dense(self.hidden_dim, activation=self.activation,
                                   kernel_initializer=self.kernel_initializer))
        self.post_layers.add(Dense(1, activation='linear',
                                   kernel_initializer=self.kernel_initializer))

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

        # pre layers
        x = self.pre_layers(x)

        # conv layers and pool layers
        if self.model_name in ['gcn', 'cheb']:
            a = self.add_self_loop(a)
            a = self.norm_a(a)

        if self.model_name in ['graphsage', 'gin', 'tag']:
            a = tf.sparse.from_dense(a)
        
        for layer in self.gnn_layers:
            if isinstance(layer, DiffPool):
                if self.model_name in ['graphsage', 'gin', 'tag']:
                    a = tf.sparse.to_dense(a)
                    x, a = layer([x, a])
                    a = tf.sparse.from_dense(a)
                else:
                    x, a = layer([x, a])

                if self.model_name in ['gcn', 'cheb']:
                    a = self.norm_a(a)
            else:
                x = layer([x, a])

        # post layers
        x = self.post_layers(x)

        return x


if __name__ == "__main__":
    # model build
    model_total = GNN()
    model_couple = GNN()
    print('model build done')
