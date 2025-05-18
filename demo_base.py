# %% [markdown]
# # 版图电容预测基本模型

# %% [markdown]
# ## 参数解析

# %%
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../"))

import argparse

from config import *


# sys.argv = ['run.py']

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dir_prj', type=str, default=dir_prj,
										help='project directory')
parser.add_argument('--seed', type=int, default=seed,
										help='random seed')
parser.add_argument('--pattern_nums', type=int, nargs='+', default=pattern_nums,
										help='pattern nums')
parser.add_argument('--nodes_range', type=int, nargs='+', default=nodes_range,
										help='the range of numbers of polygons')
parser.add_argument('--num_process', type=int, default=num_process,
										help='multiprocessing number')
parser.add_argument('--n_components', type=int, default=n_components,
										help='number of components for DDR')
parser.add_argument('--disable_norm', action='store_true',
										help='disable normalization')
parser.add_argument('--disable_ddr', action='store_true',
										help='disable dimensionality reduction')
parser.add_argument('--use_ddr_pca', action='store_true',
										help='use DDR PCA')
parser.add_argument('--use_ddr_kpca', action='store_true',
										help='use DDR KPCA')
parser.add_argument('--use_ddr_var', action='store_true',
										help='use DDR VAR')
parser.add_argument('--use_ddr_ae', action='store_true',	
										help='use DDR AE')

args = parser.parse_args()
dir_prj = args.dir_prj
seed = args.seed
pattern_nums = args.pattern_nums
nodes_range = args.nodes_range
num_process = args.num_process
n_components = args.n_components
DISABLE_NORM = args.disable_norm
DISABLE_DDR = args.disable_ddr
USE_DDR_PCA = args.use_ddr_pca
USE_DDR_KPCA = args.use_ddr_kpca
USE_DDR_VAR = args.use_ddr_var
USE_DDR_AE = args.use_ddr_ae
USE_DDR = not DISABLE_DDR

# %% [markdown]
# ## 环境设置

# %%
os.environ['NUMEXPR_MAX_THREADS'] = '32'


# %% [markdown]
# ## 路径定义

# %%
import os


# log save path
dir_logs = os.path.join(os.getcwd(), '../logs')
if not os.path.exists(dir_logs):
	os.mkdir(dir_logs)

# results save path
dir_results = os.path.join(os.getcwd(), '../results')
if not os.path.exists(dir_results):
	os.mkdir(dir_results)


# %% [markdown]
# ## log 设置

# %%
import logging


console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(dir_logs, f'base_seed{seed}.log'), mode='w', encoding='utf-8')

# 设置日志格式
logging.basicConfig(
    format="%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[console_handler, file_handler],
    level=logging.INFO
)

# loging args
logging.info('------------------------args start----------------------------')
for k, v in vars(args).items():
		logging.info(f'{k} = {v}')
logging.info('-------------------------args end-----------------------------')

# %% [markdown]
# ## 库导入

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.getcwd(), '../'))

from data.layout import convert_data_parallel
from utils.analysis import ratio_bad


# %% [markdown]
# ## 数据导入与数据清洗

# %%
def data_process_sample(x, y, cnt_max): # 单类样本采样最大限制
	if len(x) > cnt_max:
		new_x = x[:cnt_max]
		new_y = y[:cnt_max]
	else:
		new_x = x
		new_y = y

	return new_x, new_y

def data_process_truncate(x, y, nodes_range, reserve=False): # 双边 [0 3000] # 矩阵大小截断 筛选可选
	nodes_range = np.array(nodes_range, dtype=np.int32)
	raw_num = np.array([len(i) for i in x]).reshape(-1, 1)
	valid_num = np.array([nodes_range[1] if len(i) > nodes_range[1] else len(i) for i in x], dtype=np.int32).reshape(-1, 1)
	if reserve:
		mask_reserve = np.ones(shape=(len(x),), dtype=np.bool_)
	else:
		mask_reserve = (nodes_range[0] <= raw_num) & (raw_num <= nodes_range[1])
	sum_reserve = np.sum(mask_reserve, dtype=np.int32)
	new_x = np.zeros(shape=(sum_reserve, nodes_range[1], 5), dtype=np.float32)
	new_y = np.zeros(shape=(sum_reserve, 1), dtype=np.float32)
	index = 0
	for i, num in enumerate(valid_num):
		num = num[0]
		if mask_reserve[i]:
			new_x[index][:num] = x[i][:num]
			new_y[index][0] = y[i][0]
			index += 1
	valid_num = valid_num[mask_reserve].reshape(-1, 1)
	new_y = np.concatenate([valid_num, new_y], axis=1)

	return new_x, new_y, raw_num


x_total = []
x_couple = []
y_total = []
y_couple = []
raw_nums = []
if len(pattern_nums) == 0:
	logging.info("pattern_nums is empty, please check the config.py")
elif pattern_nums[0] == -1:
	pattern_nums = []
	dir_path = os.path.join(dir_prj, "data/raw_data")
	files = os.listdir(dir_path)
	for file in files:
		if file.startswith('pattern'):
			pattern_num = file.split('pattern')[1]
			pattern_nums.append(pattern_num)
	logging.info(f'pattern numbers: {pattern_nums}')

for pattern_num in pattern_nums:
	dir_load = os.path.join(dir_prj, "data/convert_data/pattern{}".format(pattern_num))
	if not os.path.exists(dir_load):
		# convert_data(pattern_num)
		convert_data_parallel(dir_prj, pattern_num, num_process=8)

	x_total_ = np.load(os.path.join(dir_load, "x_total.npy"), allow_pickle=True)
	y_total_ = np.load(os.path.join(dir_load, "y_total.npy")).reshape(-1, 1)
	x_couple_ = np.load(os.path.join(dir_load, "x_couple.npy"), allow_pickle=True)
	y_couple_ = np.load(os.path.join(dir_load, "y_couple.npy")).reshape(-1, 1)
	# data sample
	x_total_, y_total_ = data_process_sample(x_total_, y_total_, cnt_max=cnt_max)
	x_couple_, y_couple_ = data_process_sample(x_couple_, y_couple_, cnt_max=cnt_max)
	# data truncate
	x_total_, y_total_, total_raw_nums = data_process_truncate(x_total_, y_total_, nodes_range, reserve=False)
	x_couple_, y_couple_, _ = data_process_truncate(x_couple_, y_couple_, nodes_range, reserve=False)

	# concatenate data
	if len(x_total) == 0:
		x_total = x_total_.copy()
		x_couple = x_couple_.copy()
		y_total = y_total_.copy()
		y_couple = y_couple_.copy()
		raw_nums = total_raw_nums.copy()
	else:
		x_total = np.concatenate([x_total, x_total_], axis=0)
		x_couple = np.concatenate([x_couple, x_couple_], axis=0)
		y_total = np.concatenate([y_total, y_total_], axis=0)
		y_couple = np.concatenate([y_couple, y_couple_], axis=0)
		raw_nums = np.concatenate([raw_nums, total_raw_nums],  axis=0)
	logging.info("load data from {}".format(dir_load))

valid_num = y_total[:, 0]
logging.info(f'raw nums shape: {raw_nums.shape} mean: {np.mean(raw_nums)} max: {np.max(raw_nums)} min: {np.min(raw_nums)}')
logging.info(f'valid nums shape: {valid_num.shape} mean: {np.mean(valid_num)} max: {np.max(valid_num)} min: {np.min(valid_num)}')

logging.info(f'x total shape: {x_total.shape} x couple shape: {x_couple.shape}')
logging.info(f'x total first 10 samples:\n {x_total[0][:10]}')
logging.info(f'x couple first 10 samples:\n {x_couple[0][:10]}')

logging.info(f'y total shape: {y_total.shape} y couple shape: {y_couple.shape}')
logging.info(f'y total first 10 samples:\n {y_total[:10]}')
logging.info(f'y couple first 10 samples:\n {y_couple[:10]}')

# %% [markdown]
# ## 数据分割

# %%
# data split 6:2:2
from sklearn.model_selection import train_test_split


x_total_train, x_total_valid_test, y_total_train, y_total_valid_test = train_test_split(x_total, y_total, test_size=0.4, random_state=seed, shuffle=True)
x_total_valid, x_total_test, y_total_valid, y_total_test = train_test_split(x_total_valid_test, y_total_valid_test, test_size=0.5, random_state=seed, shuffle=True)
logging.info(f'total x len: {len(x_total_train)} : {len(x_total_valid)} : {len(x_total_test)}')
logging.info(f'x total train first 10 samples\n {x_total_train[0][:10]}')
logging.info(f'y total train first 10 samples\n {y_total_train[:10]}')

x_couple_train, x_couple_valid_test, y_couple_train, y_couple_valid_test = train_test_split(x_couple, y_couple, test_size=0.4, random_state=seed, shuffle=True)
x_couple_valid, x_couple_test, y_couple_valid, y_couple_test = train_test_split(x_couple_valid_test, y_couple_valid_test, test_size=0.5, random_state=seed, shuffle=True)
logging.info(f'couple x len: {len(x_couple_train)} : {len(x_couple_valid)} : {len(x_couple_test)}')
logging.info(f'x couple train first 10 samples\n {x_couple_train[0][:10]}')
logging.info(f'y couple train first 10 samples\n {y_couple_train[:10]}')

# %% [markdown]
# ## 数据预处理(归一化)

# %%
# min max 
from data.preprocess import data_norm, data_norm


if DISABLE_NORM:
	x_total_train_norm_flat = x_total_train.reshape(len(x_total_train), -1)
	x_total_valid_norm_flat = x_total_valid.reshape(len(x_total_valid), -1)
	x_total_test_norm_flat = x_total_test.reshape(len(x_total_test), -1)
	x_couple_train_norm_flat = x_couple_train.reshape(len(x_couple_train), -1)
	x_couple_valid_norm_flat = x_couple_valid.reshape(len(x_couple_valid), -1)
	x_couple_test_norm_flat = x_couple_test.reshape(len(x_couple_test), -1)
else:
	# 数据归一化
	x_total_train_norm = data_norm(x_total_train, y_total_train)
	x_total_valid_norm = data_norm(x_total_valid, y_total_valid)
	x_total_test_norm = data_norm(x_total_test, y_total_test)
	x_couple_train_norm = data_norm(x_couple_train, y_couple_train)
	x_couple_valid_norm = data_norm(x_couple_valid, y_couple_valid)
	x_couple_test_norm = data_norm(x_couple_test, y_couple_test)

	## 数据扁平化
	x_total_train_norm_flat = x_total_train_norm.reshape(len(x_total_train_norm), -1)
	x_total_valid_norm_flat = x_total_valid_norm.reshape(len(x_total_valid_norm), -1)
	x_total_test_norm_flat = x_total_test_norm.reshape(len(x_total_test_norm), -1)
	x_couple_train_norm_flat = x_couple_train_norm.reshape(len(x_couple_train_norm), -1)
	x_couple_valid_norm_flat = x_couple_valid_norm.reshape(len(x_couple_valid_norm), -1)
	x_couple_test_norm_flat = x_couple_test_norm.reshape(len(x_couple_test_norm), -1)


# %% [markdown]
# ## 降维处理

# %% [markdown]
# #### 传统降维方法

# %%
# 初始维度 10000维
from sklearn import decomposition
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


if USE_DDR:	
	if USE_DDR_PCA:
		pca_total = decomposition.PCA(n_components=n_components, copy=True, whiten=True)
		pca_total.fit(x_total_train_norm_flat)
		pca_couple = decomposition.PCA(n_components=n_components, copy=True, whiten=True)
		pca_couple.fit(x_couple_train_norm_flat)
		ddr_total = pca_total
		ddr_couple = pca_couple
	elif USE_DDR_KPCA:
		kpca_total = decomposition.KernelPCA(n_components=n_components, kernel='rbf')
		kpca_total.fit(x_total_train_norm_flat)
		kpca_couple = decomposition.KernelPCA(n_components=n_components, kernel='rbf')
		kpca_couple.fit(x_couple_train_norm_flat)
		ddr_total = kpca_total
		ddr_couple = kpca_couple
	elif USE_DDR_VAR:
		var_total = VarianceThreshold(threshold=0.5)
		var_total.fit(x_total_train_norm_flat)
		var_couple = VarianceThreshold(threshold=0.5)
		var_couple.fit(x_couple_train_norm_flat)
		ddr_total = var_total
		ddr_couple = var_couple
	else:
		ddr_total = None
		ddr_couple = None
else:
	ddr_total = None
	ddr_couple = None

# transform
if ddr_total is not None and ddr_couple is not None:
	x_total_train_norm_flat_ddr = ddr_total.transform(x_total_train_norm_flat)
	x_total_valid_norm_flat_ddr = ddr_total.transform(x_total_valid_norm_flat)
	x_total_test_norm_flat_ddr = ddr_total.transform(x_total_test_norm_flat)
	x_couple_train_norm_flat_ddr = ddr_couple.transform(x_couple_train_norm_flat)
	x_couple_valid_norm_flat_ddr = ddr_couple.transform(x_couple_valid_norm_flat)
	x_couple_test_norm_flat_ddr = ddr_couple.transform(x_couple_test_norm_flat)

# plot
# ratio = ddr_total.explained_variance_ratio_
# cum_ratio = np.cumsum(ratio)
# logging.info(f'total cum ratio {cum_ratio}')
# plt.plot(range(n_components), cum_ratio)
# plt.show()
# logging.info(f'total sum ratio {np.sum(ratio)}')

# ratio = ddr_couple.explained_variance_ratio_
# cum_ratio = np.cumsum(ratio)
# logging.info(f'couple cum ratio {cum_ratio}')
# plt.plot(range(n_components), cum_ratio)
# plt.show()
# logging.info(f'couple sum ratio {np.sum(ratio)}')

# %% [markdown]
# #### 新型降维方法

# %% [markdown]
# ##### 自编码器

# %%
if USE_DDR and USE_DDR_AE:
	import tensorflow as tf
	from tensorflow import keras

	# encoding dim 100
	encoding_dim = n_components

	# input dim 10000
	input_dim = x_total_train_norm_flat.shape[1]
	input_layer = keras.layers.Input(shape=(input_dim,))
	# encoding layer
	encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
	# decoding layer
	decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

	# autoencoder
	autoencoder = keras.models.Model(input_layer, decoded)
	# encoder
	encoder = keras.models.Model(input_layer, encoded)
	# decoder
	encoded_input = keras.layers.Input(shape=(encoding_dim,))
	decoder_layer = autoencoder.layers[-1]
	decoder = keras.models.Model(encoded_input, decoder_layer(encoded_input))

	# compile
	autoencoder.compile(optimizer='adam', loss='mean_squared_error')

	# fit
	autoencoder.fit(x_total_train_norm_flat, x_total_train_norm_flat,
									epochs=100,
									batch_size=32,
									shuffle=True,
									validation_data=(x_total_valid_norm_flat, x_total_valid_norm_flat))

	# history
	history = autoencoder.history.history
	plt.plot(history['loss'], label='train')
	plt.plot(history['val_loss'], label='valid')
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()

	# encode data
	x_total_train_norm_flat_ddr = encoder.predict(x_total_train_norm_flat)
	x_total_valid_norm_flat_ddr = encoder.predict(x_total_valid_norm_flat)
	x_total_test_norm_flat_ddr = encoder.predict(x_total_test_norm_flat)
	x_couple_train_norm_flat_ddr = encoder.predict(x_couple_train_norm_flat)
	x_couple_valid_norm_flat_ddr = encoder.predict(x_couple_valid_norm_flat)
	x_couple_test_norm_flat_ddr = encoder.predict(x_couple_test_norm_flat)

# %% [markdown]
# ### 禁用降维方法

# %%
if DISABLE_DDR:
	x_total_train_norm_flat_ddr = x_total_train_norm_flat
	x_total_valid_norm_flat_ddr = x_total_valid_norm_flat
	x_total_test_norm_flat_ddr = x_total_test_norm_flat
	x_couple_train_norm_flat_ddr = x_couple_train_norm_flat
	x_couple_valid_norm_flat_ddr = x_couple_valid_norm_flat
	x_couple_test_norm_flat_ddr = x_couple_test_norm_flat


# %% [markdown]
# ## 命名简化

# %%
# total 
xt_train = x_total_train_norm_flat_ddr
xt_valid = x_total_valid_norm_flat_ddr
xt_test = x_total_test_norm_flat_ddr
yt_train = y_total_train[:, 1].reshape(-1, 1)
yt_valid = y_total_valid[:, 1].reshape(-1, 1)
yt_test = y_total_test[:, 1].reshape(-1, 1)

# couple
xc_train = x_couple_train_norm_flat_ddr
xc_valid = x_couple_valid_norm_flat_ddr
xc_test = x_couple_test_norm_flat_ddr
yc_train = y_couple_train[:, 1].reshape(-1, 1)
yc_valid = y_couple_valid[:, 1].reshape(-1, 1)
yc_test = y_couple_test[:, 1].reshape(-1, 1)

# %% [markdown]
# ## 模型预测

# %% [markdown]
# ### 线性回归

# %%
from sklearn.linear_model import LinearRegression
from utils.analysis import model_analysis


# total
logging.info('--------------------------------total------------------------------')
lr_t = LinearRegression()
lr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(lr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'linear reg')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.Series(dict_total).to_frame().T

# couple
logging.info('--------------------------------couple------------------------------')
lr_c = LinearRegression()
lr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(lr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'linear reg')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.Series(dict_couple).to_frame().T

# %% [markdown]
# ### 支持向量机回归

# %%
from sklearn.svm import SVR


# total
logging.info('--------------------------------total------------------------------')
# linear svr
lr_svf_t = SVR(kernel='linear', max_iter=1000)
lr_svf_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(lr_svf_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'linear svr')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# poly svr
poly_svf_t = SVR(kernel='poly', max_iter=1000)
poly_svf_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(poly_svf_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'poly svr')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# rbf svr
rbf_svf_t = SVR(kernel='rbf', max_iter=1000)
rbf_svf_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(rbf_svf_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'rbf svr')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
logging.info('--------------------------------couple------------------------------')
# linear svr
lr_svf_c = SVR(kernel='linear', max_iter=1000)
lr_svf_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(lr_svf_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'linear svr')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# poly svr
poly_svf_c = SVR(kernel='poly', max_iter=1000)
poly_svf_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(poly_svf_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'poly svr')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# rbf svr
rbf_svf_c = SVR(kernel='rbf', max_iter=1000)
rbf_svf_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(rbf_svf_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'rbf svr')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### K近邻回归

# %%
from sklearn.neighbors import KNeighborsRegressor

# total 
logging.info('--------------------------------total------------------------------')
# uniform knn
uni_knn_t = KNeighborsRegressor(n_neighbors=5, weights='uniform')
uni_knn_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(uni_knn_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'uniform knn')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# distance knn
dis_knn_t = KNeighborsRegressor(n_neighbors=5, weights='distance')
dis_knn_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(dis_knn_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'distance knn')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
logging.info('--------------------------------couple------------------------------')
# uniform knn
uni_knn_c = KNeighborsRegressor(n_neighbors=5, weights='uniform')
uni_knn_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(uni_knn_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'uniform knn')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# distance knn
dis_knn_c = KNeighborsRegressor(n_neighbors=5, weights='distance')
dis_knn_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(dis_knn_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'distance knn')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### 回归树

# %%
from sklearn.tree import DecisionTreeRegressor

# total
logging.info('--------------------------------total------------------------------')
dtr_t = DecisionTreeRegressor()
dtr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(dtr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'decision tree total')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
logging.info('--------------------------------couple------------------------------')
dtr_c = DecisionTreeRegressor()
dtr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(dtr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'decision tree couple')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### 集成模型

# %% [markdown]
# #### 基础模型

# %%
from sklearn.ensemble import RandomForestRegressor

# total
logging.info('--------------------------------total------------------------------')
# random forest
rfr_t = RandomForestRegressor()
rfr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(rfr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'random forest')
logging.info(f'total model analysis:\n {dict_total}')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
logging.info('--------------------------------couple------------------------------')
# random forest
rfr_c = RandomForestRegressor()
rfr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(rfr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'random forest')
logging.info(f'couple model analysis:\n {dict_couple}')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)


# %% [markdown]
# ## 结果存储

# %%
# save results_total and results_couple to csv
results_total.to_csv(os.path.join(dir_results, "base_total.csv"), index=False)
results_couple.to_csv(os.path.join(dir_results, "base_couple.csv"), index=False)



