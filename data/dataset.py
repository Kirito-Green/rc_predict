import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import numpy as np
import scipy.sparse as sp
from spektral.data import Dataset, Graph
from tqdm import tqdm, trange
import multiprocessing

from data.layout import convert_data, convert_data_parallel


class MyDataset(Dataset):
	'''
	a dataset of pattern results
	'''

	def __init__(self, 
			     dir_prj,
				 pattern_nums, 
			  	 x_name, 
				 y_name, 
				 g_name, 
				 num_process=8, 
				 update=False, 
				 **kwargs):
		self.dir_prj = dir_prj
		self.pattern_nums = pattern_nums
		self.x_name = x_name
		self.y_name = y_name
		self.g_name = g_name
		self.num_process = num_process
		self.update = update
		self.vertical_space = 0.024  # um
		self.neighbor_dist_max = 5  # um
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

		np.savez(os.path.join(self.dir_save, f'{self.g_name}_{id}.npz'),
		         x=x, data=data, col_index=col_index, row_index=row_index, y=y)

	def download(self):
		if len(self.pattern_nums) == 0:
			print('No pattern numbers provided')
			return
		elif len(self.pattern_nums) == 1 and self.pattern_nums[0] == -1: # list all pattern numbers
			self.pattern_nums = []
			dir_path = os.path.join(self.dir_prj, "data/raw_data")
			files = os.listdir(dir_path)
			for file in files:
				if file.startswith('pattern'):
					pattern_num = file.split('pattern')[1]
					self.pattern_nums.append(pattern_num)
			print('pattern numbers:', self.pattern_nums)

		for pattern_num in self.pattern_nums:
			self.pattern_num = pattern_num
			self.dir_read = os.path.join(self.dir_prj, "data/convert_data/pattern{}".format(self.pattern_num))
			self.dir_save = os.path.join(self.dir_prj, "data/graph_data/pattern{}".format(self.pattern_num))
			if not os.path.exists(os.path.join(self.dir_read, self.x_name)):
				print('pattern{} convert data not exist'.format(self.pattern_num))
				# convert_data(self.dir_prj, self.pattern_num)
				convert_data_parallel(self.dir_prj, self.pattern_num, self.num_process)
			if not os.path.exists(self.dir_save):
				os.mkdir(self.dir_save)

			if not os.path.exists(os.path.join(self.dir_save, f'{self.g_name}_0.npz'))\
				or self.update == True:
				x = np.load(os.path.join(self.dir_read, self.x_name), allow_pickle=True)
				y = np.load(os.path.join(self.dir_read, self.y_name))
				ids = np.arange(len(x))
				args = [(x[i], y[i], ids[i]) for i in range(len(x))]

				# work bar
				pbar = tqdm(total=len(args), desc=f'Downloading data from pattern{self.pattern_num}', unit='file')
				update = lambda *args: pbar.update()

				pool = multiprocessing.Pool(processes=self.num_process)
				# map the function to the arguments
				# pool.map(self.cal_graph, args)

				# async update
				for arg in args:
					pool.apply_async(self.cal_graph, (arg,), callback=update)
				pool.close()
				pool.join()
			print(f'pattern{self.pattern_num} data downloaded')
		print('all data downloaded')

	def read(self):  # normal feature
		# return a list of graph list
		output = []

		for pattern_num in self.pattern_nums:
			print('reading pattern{}'.format(pattern_num))
			self.pattern_num = pattern_num
			self.dir_read = os.path.join(self.dir_prj, "data/convert_data/pattern{}".format(self.pattern_num))
			self.dir_save = os.path.join(self.dir_prj, "data/graph_data/pattern{}".format(self.pattern_num))
			
			graphs = os.listdir(self.dir_save)
			cnt = 0
			cnt_max = 1500
			for graph in graphs:
				if graph.startswith(f'{self.g_name}'):
					cnt += 1
					path_graph = os.path.join(self.dir_save, graph)
					g = np.load(path_graph, allow_pickle=True)
					x = g['x']
					data = g['data']
					col_index = g['col_index']
					row_index = g['row_index']
					n = x.shape[0]
					a = sp.csr_matrix((data, col_index, row_index), shape=(n, n))
					y = g['y']
					output.append(Graph(x=x, a=a, y=y))

					if cnt > cnt_max:
						break
			print(f'pattern{self.pattern_num} data readed')
		print('all data readed')

		return output


if __name__ == "__main__":
	dataset_total = MyDataset(dir_prj='D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/',
							  pattern_nums=[28],
							  x_name='x_total.npy',
	                          y_name='y_total.npy',
	                          g_name='total',
							  num_process=8,
	                          update=False)
	dataset_couple = MyDataset(dir_prj='D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/',
							   pattern_nums=[28],
							   x_name='x_couple.npy',
	                           y_name='y_couple.npy',
	                           g_name='couple',
							   num_process=8,
	                           update=False)
	print(dataset_total)
	print(dataset_couple)
