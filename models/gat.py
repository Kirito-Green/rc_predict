import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))
from spektral.layers import GATConv, GlobalAvgPool, GlobalMaxPool, GlobalSumPool, DiffPool
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten


class GAT(Model):
	'''
	Graph Attention Network (GAT) model for graph prediction.
	'''

	def __init__(self):
		super(GAT, self).__init__()
		self.conv1 = GATConv(16, attn_heads=1, concat_heads=True, activation='relu') # [batch, n, 16]
		self.pool1 = DiffPool(k=256, activation='relu') #[batch 256, 16]
		self.conv2 = GATConv(16, attn_heads=1, concat_heads=True, activation='relu') # [batch 256, 16]
		self.pool2 = DiffPool(k=64, activation='relu') # [batch 64 16]
		self.conv3 = GATConv(16, attn_heads=1, concat_heads=True, activation='relu') # [batch 64 16]
		self.pool3 = GlobalAvgPool()
		self.fc1 = Dense(16, activation='relu')
		self.fc2 = Dense(1, activation='linear')

	def call(self, inputs):
		x, a = inputs
		x = tf.cast(x, tf.float32)
		a = tf.cast(a, tf.float32)
		# GAT 1
		x = self.conv1([x, a])
		x, a = self.pool1([x, a])
		# GAT 2
		x = self.conv2([x, a])
		x, a = self.pool2([x, a])
		# GAT 3
		x = self.conv3([x, a])
		x = self.pool3(x)
		# FC
		x = self.fc1(x)
		x = self.fc2(x)

		return x


if __name__ == "__main__":
	# model build
	model_total = GAT()
	model_couple = GAT()