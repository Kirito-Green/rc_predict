import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))
from spektral.layers import GCNConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten


from config import *

class GCN(Model):
	'''
	Graph Convolutional Network (GCN) model for graph prediction.
	'''

	def __init__(self):
		super().__init__()
		self.conv1 = GCNConv(16, activation='relu') # [batch, n, 16]
		self.pool1 = DiffPool(k=256, activation='relu') #[batch 256, 16]
		self.conv2 = GCNConv(16, activation='relu') # [batch 256, 16]
		self.pool2 = DiffPool(k=64, activation='relu') # [batch 64 16]
		self.conv3 = GCNConv(16, activation='relu') # [batch 64 16]
		self.pool3 = GlobalAvgPool()
		self.fc1 = Dense(16, activation='relu')
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
		# GCN 1
		x = self.conv1([x, a])
		x, a = self.pool1([x, a])
		a = self.norm_a(a)
		# GCN 2
		x = self.conv2([x, a])
		x, a = self.pool2([x, a])
		a = self.norm_a(a)
		# GCN 3
		x = self.conv3([x, a])
		x = self.pool3(x)
		# FC
		x = self.fc1(x)
		x = self.fc2(x)

		return x


if __name__ == "__main__":
	# model build
	model_total = GCN()
	model_couple = GCN()
	print('model build done')