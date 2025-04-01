from spektral.layers import GraphSageConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '../'))

from config import *

class GraphSage(Model):
	'''
	Graph aggregated samples (GraphSAGE) model for graph prediction.
	'''

	def __init__(self):
		super(GraphSage, self).__init__()
		self.conv1 = GraphSageConv(16, aggregate='mean', activation='relu') # [batch, n, 16]
		self.pool1 = DiffPool(k=256, activation='relu') #[batch 256, 16]
		self.conv2 = GraphSageConv(16, aggregate='mean', activation='relu') # [batch 256, 16]
		self.pool2 = DiffPool(k=64, activation='relu') # [batch 64 16]
		self.conv3 = GraphSageConv(16, aggregate='mean', activation='relu') # [batch 64 16]
		self.pool3 = GlobalAvgPool()
		self.fc1 = Dense(16, activation='relu')
		self.fc2 = Dense(1, activation='linear')

	def add_self_loop(self, a):
		diag_values = tf.linalg.diag_part(a)
		mask_zero = abs(diag_values) < tolerant_error
		diag_values = tf.where(mask_zero, 1.0, 0.0)
		diag_matrix = tf.linalg.diag(diag_values)
		a_hat = a + diag_matrix

		return a_hat

	def call(self, inputs):
		x, a = inputs
		x = tf.cast(x, tf.float32)
		a = tf.cast(a, tf.float32)
		a = self.add_self_loop(a)
		# GraphSage 1
		a = tf.sparse.from_dense(a)
		x = self.conv1([x, a])
		a = tf.sparse.to_dense(a)
		x, a = self.pool1([x, a])
		# GraphSage 2
		a = tf.sparse.from_dense(a)
		x = self.conv2([x, a])
		a = tf.sparse.to_dense(a)
		x, a = self.pool2([x, a])
		# GraphSage 3
		a = tf.sparse.from_dense(a)
		x = self.conv3([x, a])
		x = self.pool3(x)
		# FC
		x = self.fc1(x)
		x = self.fc2(x)

		return x


if __name__ == "__main__":
	# model build
	model_total = GraphSage()
	model_couple = GraphSage()