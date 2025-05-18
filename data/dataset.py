import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import multiprocessing
from tqdm import tqdm
from spektral.data import Dataset, Graph
import scipy.sparse as sp
import numpy as np

from config import *
from data.preprocess import data_preprocess
from data.layout import convert_data_parallel


class MyDataset(Dataset):
    '''
    a dataset of pattern results
    '''

    def __init__(self,
                 dir_prj,
                 ndm,
                 k,
                 pattern_nums,
                 x_name,
                 y_name,
                 g_name,
                 num_process=8,
                 update=False,
                 **kwargs):

        self.dir_prj = dir_prj
        self.ndm = ndm
        self.k = k
        self.pattern_nums = pattern_nums
        self.x_name = x_name
        self.y_name = y_name
        self.g_name = g_name
        self.num_process = num_process
        self.update = update
        super().__init__(**kwargs)

    def cal_graph_dist(self, args):  # 按最大距离确定邻居
        x_, y, id = args
        n = len(x_)
        x = np.array(x_, dtype=np.float32)

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
                    if np.linalg.norm(p1 - p2) <= self.ndm / window_size:
                        a_[j][k] = 1
                        row_index[-1] = row_index[-1] + 1
                        col_index.append(k)
                        data.append(1)
        a = sp.csr_matrix((data, col_index, row_index), shape=(n, n))

        np.savez(os.path.join(self.dir_save, f'{self.g_name}_{id}.npz'),
                 x=x, data=data, col_index=col_index, row_index=row_index, y=y)

    def cal_graph_k(self, args):  # 按数量确定邻居
        x_, y, id = args
        n = len(x_)
        x = np.array(x_, dtype=np.float32)

        # 直接求解稀疏矩阵
        row_index = []
        col_index = []
        data = []

        row_index.append(0)
        for j in range(n):
            row_index.append(row_index[-1])

            # find the k nearest neighbors
            dist = []
            p1 = np.array([x[j][0], x[j][1], x[j][2]])  # x y z
            for k in range(n):
                p2 = np.array([x[k][0], x[k][1], x[k][2]])
                dist.append(np.linalg.norm(p1 - p2))
            dist = np.array(dist)
            k_index = np.argsort(dist)[1:K+1]  # 取出k个最近邻

            for k in k_index:
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
        # list all pattern numbers
        elif len(self.pattern_nums) == 1 and self.pattern_nums[0] == -1:
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
            self.dir_read = os.path.join(
                self.dir_prj, "data/convert_data/pattern{}".format(self.pattern_num))
            self.dir_graph = os.path.join(
                # self.dir_prj, "data/graph_data_ndm{}/".format(self.ndm))
                self.dir_prj, "data/graph_data_k{}/".format(self.k))
            self.dir_save = os.path.join(
                self.dir_graph, "pattern{}".format(self.pattern_num))
            if not os.path.exists(os.path.join(self.dir_read, self.x_name)):
                print('pattern{} convert data not exist'.format(self.pattern_num))
                # convert_data(self.dir_prj, self.pattern_num)
                convert_data_parallel(
                    self.dir_prj, self.pattern_num, self.num_process)

            if not os.path.exists(self.dir_graph):
                os.mkdir(self.dir_graph)

            if not os.path.exists(self.dir_save):
                os.mkdir(self.dir_save)

            if not os.path.exists(os.path.join(self.dir_save, f'{self.g_name}_0.npz'))\
                    or self.update == True:
                x = np.load(os.path.join(self.dir_read,
                            self.x_name), allow_pickle=True)
                y = np.load(os.path.join(self.dir_read, self.y_name))

                # data preprocess
                n1 = len(x)
                x, y = data_preprocess(x, y, special=False)
                n2 = len(x)
                print(f'pattern{self.pattern_num} data filtered, {n1-n2} data removed')

                ids = np.arange(len(x))
                args = [(x[i], y[i], ids[i]) for i in range(len(x))]

                # work bar
                pbar = tqdm(total=len(
                    args), desc=f'Downloading data from pattern{self.pattern_num}', unit='file')
                update = lambda *args: pbar.update()

                pool = multiprocessing.Pool(processes=self.num_process)
                # map the function to the arguments
                # pool.map(self.cal_graph_dist, args)

                # async update
                for arg in args:
                    # dist
                    # pool.apply_async(self.cal_graph_dist,
                    #                  (arg,), callback=update)

                    # k
                    pool.apply_async(self.cal_graph_k,
                                     (arg,), callback=update)
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
            self.dir_read = os.path.join(
                self.dir_prj, "data/convert_data/pattern{}".format(self.pattern_num))
            self.dir_save = os.path.join(
                # self.dir_prj, "data/graph_data_ndm{}/pattern{}".format(self.ndm, self.pattern_num))
                self.dir_prj, "data/graph_data_k{}/pattern{}".format(self.k, self.pattern_num))

            cnt = 0
            graphs = os.listdir(self.dir_save)
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
                    a = sp.csr_matrix(
                        (data, col_index, row_index), shape=(n, n))
                    y = g['y']
                    output.append(Graph(x=x, a=a, y=y))

            print(f'pattern{self.pattern_num} {cnt} data readed')
        print('all data readed')

        return output


if __name__ == "__main__":
    dataset_total = MyDataset(dir_prj,
                            #  dir_prj='/home/prj/rc_predict/',
                              ndm=25,
                              k=K,
                              pattern_nums=[26],
                              x_name='x_total.npy',
                              y_name='y_total.npy',
                              g_name='total',
                              num_process=8,
                              update=False)
    dataset_couple = MyDataset(dir_prj,
                            #    dir_prj='/home/prj/rc_predict/',
                               ndm=25,
                               k=K,
                               pattern_nums=[26],
                               x_name='x_couple.npy',
                               y_name='y_couple.npy',
                               g_name='couple',
                               num_process=8,
                               update=False)
    print(dataset_total)
    print(dataset_couple)
