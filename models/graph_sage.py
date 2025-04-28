import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))
from spektral.layers import GraphSageConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.initializers import GlorotUniform, HeUniform

from config import *

class GraphSage(Model):
	'''
	Graph aggregated samples (GraphSAGE) model for graph prediction.
	'''

	def __init__(self, **kwargs):
		super().__init__()
		self.hidden_dim = 32
		self.aggregate = 'mean'
		self.droupout_rate = 0.5
		self.activation = 'tanh' # tanh
		if 'training' in kwargs:
			self.training = kwargs['training']
		else:
			self.training = True

		if self.activation == 'relu':
			self.kernel_initializer = HeUniform()
		else:
			self.kernel_initializer = GlorotUniform()
		self.conv1 = GraphSageConv(self.hidden_dim, aggregate=self.aggregate,
							 	activation=self.activation, kernel_initializer=self.kernel_initializer) # [batch, n, 16]
		self.d1 = Dropout(self.droupout_rate)
		self.conv2 = GraphSageConv(self.hidden_dim, aggregate=self.aggregate,
							 	activation=self.activation, kernel_initializer=self.kernel_initializer) # [batch 256, 16]
		self.d2 = Dropout(self.droupout_rate)
		self.conv3 = GraphSageConv(self.hidden_dim, aggregate=self.aggregate,
							 	activation=self.activation, kernel_initializer=self.kernel_initializer) # [batch 64 16]
		self.d3 = Dropout(self.droupout_rate)
		# self.pool = GlobalAvgPool()
		# self.pool = GlobalMaxPool()
		self.pool = GlobalSumPool()
		self.fc1 = Dense(self.hidden_dim * 2, activation=self.activation)
		self.fc2 = Dense(1, activation='linear')

	def add_self_loop(self, a):
		diag_values = tf.linalg.diag_part(a)
		mask_zero = abs(diag_values) < tolerant_zero_error
		diag_values = tf.where(mask_zero, 1.0, 0.0)
		diag_matrix = tf.linalg.diag(diag_values)
		a_hat = a + diag_matrix

		return a_hat

	def call(self, inputs):
		x, a = inputs
		x = tf.cast(x, tf.float32)
		a = tf.cast(a, tf.float32)
		a = self.add_self_loop(a)
		
		# GraphSage
		a = tf.sparse.from_dense(a)
		x = self.conv1([x, a])
		x = self.d1(x, training=self.training)
		x = self.conv2([x, a])
		x = self.d2(x, training=self.training)
		x = self.conv3([x, a])
		x = self.d3(x, training=self.training)
		
        # readout layer
		x = self.pool(x)
		
		# FC
		x = self.fc1(x)
		x = self.fc2(x)

		return x


if __name__ == "__main__":
	# model build
	model_total = GraphSage()
	model_couple = GraphSage()