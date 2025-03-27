import os
import sys
import numpy as np
import scipy.sparse as sp
from spektral.data import Dataset, Graph
from tqdm import tqdm, trange
import multiprocessing
import time

sys.path.append(os.path.join(os.getcwd(), '..'))

from data.layout import convert_data
from config import *


class MyDataset(Dataset):
	'''
	a dataset of pattern results
	'''

	def __init__(self, pattern_num, x_name, y_name, g_name, update=False, **kwargs):
		self.x_name = x_name
		self.y_name = y_name
		self.g_name = g_name
		self.update = update
		self.pattern_num = pattern_num
		self.vertical_space = 0.025  # um
		self.neighbor_dist_max = 1.2  # um
		self.path_read = os.path.join(dir_prj, "data/convert_data/pattern{}".format(self.pattern_num))
		self.path_save = os.path.join(dir_prj, "data/graph_data/pattern{}".format(self.pattern_num))
		super().__init__(**kwargs)

	def cal_graph(self, args):
		x_, y, id = args
		n = len(x_)
		x = np.array(x_, dtype=np.float32)
		x[:, 2] = self.vertical_space * x[:, 2]

		# 直接求解稀疏矩阵
		row_index = []
		col_index = []
		data = []
		a_ = np.zeros((n, n), dtype=np.int32)

		row_index.append(0)
		for j in range(n):
			row_index.append(row_index[-1])
			for k in range(n):
				if j == k:
					continue
				elif k < j:
					a_[j][k] = a_[k][j]
					if a_[j][k] == 1:
						row_index[-1] = row_index[-1] + 1
						col_index.append(k)
						data.append(1)
				else:
					p1 = np.array([x[j][0], x[j][1], x[j][2]])  # x y z
					p2 = np.array([x[k][0], x[k][1], x[k][2]])
					if np.linalg.norm(p1 - p2) <= self.neighbor_dist_max:
						a_[j][k] = 1
						row_index[-1] = row_index[-1] + 1
						col_index.append(k)
						data.append(1)
		a = sp.csr_matrix((data, col_index, row_index), shape=(n, n))

		if not os.path.exists(self.path_save):
			os.mkdir(self.path_save)
		np.savez(os.path.join(self.path_save, f'{self.g_name}_{id}.npz'),
		         x=x, data=data, col_index=col_index, row_index=row_index, y=y)
		print(f'graph_{self.g_name}_{id} saved')

	def download(self):
		if not os.path.exists(self.path_read):
			convert_data(self.pattern_num)

		if not os.path.exists(os.path.join(self.path_save, f'{self.g_name}_0.npz'))\
			or self.update == True:
			start = time.time()
			x = np.load(os.path.join(self.path_read, self.x_name), allow_pickle=True)
			y = np.load(os.path.join(self.path_read, self.y_name))
			ids = np.arange(len(x))
			args = []
			for i in range(len(x)):
				args.append((x[i], y[i], ids[i]))
			with multiprocessing.Pool(processes=9) as pool:
				pool.map(self.cal_graph, args)
			end = time.time()
			print(f'download time: {end - start}s')

	def read(self):  # normal feature
		# return a list of graph list
		output = []

		id = 0
		while True:
			path_graph = os.path.join(self.path_save, f'{self.g_name}_{id}.npz')
			if os.path.exists(path_graph):
				g = np.load(path_graph, allow_pickle=True)
				x = g['x']
				data = g['data']
				col_index = g['col_index']
				row_index = g['row_index']
				n = x.shape[0]
				a = sp.csr_matrix((data, col_index, row_index), shape=(n, n))
				y = g['y']
				output.append(Graph(x=x, a=a, y=y))
				id = id + 1
			else:
				break

		return output


if __name__ == "__main__":
	dataset_total = MyDataset(pattern_num=4,
							  x_name='x_total.npy',
	                          y_name='y_total.npy',
	                          g_name='total',
	                          update=False)
	dataset_couple = MyDataset(pattern_num=4,
							   x_name='x_couple.npy',
	                           y_name='y_couple.npy',
	                           g_name='couple',
	                           update=False)
