# %% [markdown]
# # 版图电容预测基本模型

# %% [markdown]
# ## 基本库导入

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.getcwd(), '..'))

from config import *
from data.layout import convert_data_parallel
from utils.analysis import ratio_good

# %% [markdown]
# ## 数据导入与数据清洗

# %%
def data_process2(x, y, thresh): # [400 2500]
	thresh = np.array(thresh, dtype=np.int32)
	raw_num = np.array([len(i) for i in x]).reshape(-1, 1)
	valid_num = np.array([thresh[1] if len(i) > thresh[1] else len(i) for i in x], dtype=np.int32).reshape(-1, 1)
	sum_reserve = np.sum(valid_num >= thresh[0], dtype=np.int32)
	new_x = np.zeros(shape=(sum_reserve, thresh[1], 5), dtype=np.float32)
	new_y = np.zeros(shape=(sum_reserve, 1), dtype=np.float32)
	index = 0
	for i, num in enumerate(valid_num):
		num = num[0]
		if num >= thresh[0]:
			new_x[index][:num] = x[i][:num]
			new_y[index][0] = y[i][0]
			index += 1
	valid_num = valid_num[valid_num>=thresh[0]].reshape(-1, 1)
	new_y = np.concatenate([valid_num, new_y], axis=1)

	return new_x, new_y, raw_num

x_total = []
x_couple = []
y_total = []
y_couple = []
raw_nums = []
if len(pattern_nums) == 0:
	print("pattern_nums is empty, please check the config.py")
for pattern_num in pattern_nums:
	dir_load = os.path.join(dir_prj, "data/convert_data/pattern{}".format(pattern_num))
	if not os.path.exists(dir_load):
		# convert_data(pattern_num)
		convert_data_parallel(pattern_num, num_process=8)

# data process 2
x_total_ = np.load(os.path.join(dir_load, "x_total.npy"), allow_pickle=True)
y_total_ = np.load(os.path.join(dir_load, "y_total.npy")).reshape(-1, 1)
x_total_, y_total_, total_raw_nums = data_process2(x_total_, y_total_, thresh)
x_couple_ = np.load(os.path.join(dir_load, "x_couple.npy"), allow_pickle=True)
y_couple_ = np.load(os.path.join(dir_load, "y_couple.npy")).reshape(-1, 1)
x_couple_, y_couple_, _ = data_process2(x_couple_, y_couple_, thresh)

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
print("load data from {}".format(dir_load))

print('raw nums shape:', raw_nums.shape)
print('raw nums mean:', np.mean(raw_nums))
print('raw nums max:', np.max(raw_nums))
print('raw nums min:', np.min(raw_nums))

valid_num = y_total[:, 0]
print('valid nums shape:', valid_num.shape)
print('valid nums mean:', np.mean(valid_num))
print('valid nums max:', np.max(valid_num))
print('valid nums min:', np.min(valid_num))

print(x_total.shape)
print('x total first 10 samples:')
print(x_total[0][:10])
print('x couple first 10 samples:')
print(x_couple[0][:10])

print(y_total.shape)
print('y total first 10 samples:')
print(y_total[:10])
print('y couple first 10 samples:')
print(y_couple[:10])

# %% [markdown]
# ## 数据分割

# %%
# data split 6:2:2
from sklearn.model_selection import train_test_split

x_total_train, x_total_valid_test, y_total_train, y_total_valid_test = train_test_split(x_total, y_total, test_size=0.4, random_state=seed, shuffle=True)
x_total_valid, x_total_test, y_total_valid, y_total_test = train_test_split(x_total_valid_test, y_total_valid_test, test_size=0.5, random_state=seed, shuffle=True)
print('total x len:', len(x_total_train), len(x_total_valid), len(x_total_test))
print('x total train first 10\n', x_total_train[0][:10])
print('x total train last 10\n', x_total_train[0][-10:])
print('y total train first 10\n', y_total_train[:10])

x_couple_train, x_couple_valid_test, y_couple_train, y_couple_valid_test = train_test_split(x_couple, y_couple, test_size=0.4, random_state=seed, shuffle=True)
x_couple_valid, x_couple_test, y_couple_valid, y_couple_test = train_test_split(x_couple_valid_test, y_couple_valid_test, test_size=0.5, random_state=seed, shuffle=True)
print('couple x len', len(x_couple_train), len(x_couple_valid), len(x_couple_test))
print('x couple train first 10\n', x_couple_train[0][:10])
print('x couple train last 10\n', x_couple_train[0][-10:])
print('y couple train first 10\n', y_couple_train[:10])

# %% [markdown]
# ## 数据预处理

# %%
from sklearn.preprocessing import StandardScaler


def cal_mean_std(x, y):
	num_s = 0
	num_sum = round(np.sum(y[:, 0]))
	x_valid = np.zeros((num_sum, 5), dtype=np.float32)
	for i in range(len(x)):
		num = round(y[i][0])
		x_valid[num_s:num_s+num, :] = x[i][:num]
		num_s += num
	mean = np.mean(x_valid, axis=0)
	std = np.std(x_valid, axis=0)
	std[std == 0] = 1

	return mean, std

def get_mask(x, y):
	mask = np.zeros(shape=x.shape, dtype=np.int32)
	for i in range(len(x)):
		num = round(y[i][0])
		mask[i][:num] = 1

	return mask

def data_process(x, y, mean, std):
	mask = get_mask(x, y)

	return np.multiply((x - mean) / std, mask)

# x
# total
mean_total, std_total = cal_mean_std(x_total_train, y_total_train)
print('mean total:', mean_total)
print('std total:', std_total)
x_total_train_std = data_process(x_total_train, y_total_train, mean_total, std_total)
x_total_train_std_flat = x_total_train_std.reshape(len(x_total_train_std), -1)
x_total_valid_std = data_process(x_total_valid, y_total_valid, mean_total, std_total)
x_total_valid_std_flat = x_total_valid_std.reshape(len(x_total_valid_std), -1)
x_total_test_std = data_process(x_total_test, y_total_test, mean_total, std_total)
x_total_test_std_flat = x_total_test_std.reshape(len(x_total_test_std), -1)
print('x total train std flat first 20\n', x_total_train_std_flat[0][:20])
print('x total train std flat last 20\n', x_total_train_std_flat[0][-20:])

# couple
mean_couple, std_couple = cal_mean_std(x_couple_train, y_couple_train)
print('mean couple:', mean_couple)
print('std couple:', std_couple)
x_couple_train_std = data_process(x_couple_train, y_couple_train, mean_couple, std_couple)
x_couple_train_std_flat = x_couple_train_std.reshape(len(x_couple_train_std), -1)
x_couple_valid_std = data_process(x_couple_valid, y_couple_valid, mean_couple, std_couple)
x_couple_valid_std_flat = x_couple_valid_std.reshape(len(x_couple_valid_std), -1)
x_couple_test_std = data_process(x_couple_test, y_couple_test, mean_couple, std_couple)
x_couple_test_std_flat = x_couple_test_std.reshape(len(x_couple_test_std), -1)
print('x couple train std flat first 20\n', x_couple_train_std_flat[0][:20])
print('x couple train std flat last 20\n', x_couple_train_std_flat[0][-20:])

# y
# total
scaler_yt = StandardScaler()
y_total_train_std = y_total_train[:, 1].copy().reshape(-1, 1)
# scaler_yt.fit(y_total_train_std)
# y_total_train_std = scaler_yt.transform(y_total_train_std)
# print('y total train std first 10\n', y_total_train_std[:10].reshape(-1))

# # couple
scaler_yc = StandardScaler()
y_couple_train_std = y_couple_train[:, 1].copy().reshape(-1, 1)
# scaler_yc.fit(y_couple_train_std)
# y_couple_train_std = scaler_yc.transform(y_couple_train_std)
# print('y couple train std first 10\n', y_couple_train_std[:10].reshape(-1))

if DISABLE_STD:
	x_total_train_std_flat = x_total_train.reshape(len(x_total_train), -1)
	x_total_valid_std_flat = x_total_valid.reshape(len(x_total_valid), -1)
	x_total_test_std_flat = x_total_test.reshape(len(x_total_test), -1)
	x_couple_train_std_flat = x_couple_train.reshape(len(x_couple_train), -1)
	x_couple_valid_std_flat = x_couple_valid.reshape(len(x_couple_valid), -1)
	x_couple_test_std_flat = x_couple_test.reshape(len(x_couple_test), -1)


# %% [markdown]
# ## 降维处理

# %% [markdown]
# #### 传统降维方法

# %%
# 初始维度 10000维
from sklearn import decomposition
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2


if USE_DDR_PCA:
	pca_total = decomposition.PCA(n_components=n_components, copy=True, whiten=True)
	pca_total.fit(x_total_train_std_flat)
	pca_couple = decomposition.PCA(n_components=n_components, copy=True, whiten=True)
	pca_couple.fit(x_couple_train_std_flat)
	ddr_total = pca_total
	ddr_couple = pca_couple
elif USE_DDR_KPCA:
	kpca_total = decomposition.KernelPCA(n_components=n_components, kernel='rbf')
	kpca_total.fit(x_total_train_std_flat)
	kpca_couple = decomposition.KernelPCA(n_components=n_components, kernel='rbf')
	kpca_couple.fit(x_couple_train_std_flat)
	ddr_total = kpca_total
	ddr_couple = kpca_couple
elif USE_DDR_VAR:
	var_total = VarianceThreshold(threshold=0.5)
	var_total.fit(x_total_train_std_flat)
	var_couple = VarianceThreshold(threshold=0.5)
	var_couple.fit(x_couple_train_std_flat)
	ddr_total = var_total
	ddr_couple = var_couple
else:
	ddr_total = None
	ddr_couple = None

# transform
if ddr_total is not None and ddr_couple is not None:
	x_total_train_std_flat_ddr = ddr_total.transform(x_total_train_std_flat)
	x_total_valid_std_flat_ddr = ddr_total.transform(x_total_valid_std_flat)
	x_total_test_std_flat_ddr = ddr_total.transform(x_total_test_std_flat)
	x_couple_train_std_flat_ddr = ddr_couple.transform(x_couple_train_std_flat)
	x_couple_valid_std_flat_ddr = ddr_couple.transform(x_couple_valid_std_flat)
	x_couple_test_std_flat_ddr = ddr_couple.transform(x_couple_test_std_flat)

# plot
# ratio = ddr_total.explained_variance_ratio_
# cum_ratio = np.cumsum(ratio)
# print(f'total cum ratio {cum_ratio}')
# plt.plot(range(n_components), cum_ratio)
# plt.show()
# print(f'total sum ratio {np.sum(ratio)}')

# ratio = ddr_couple.explained_variance_ratio_
# cum_ratio = np.cumsum(ratio)
# print(f'couple cum ratio {cum_ratio}')
# plt.plot(range(n_components), cum_ratio)
# plt.show()
# print(f'couple sum ratio {np.sum(ratio)}')

# %% [markdown]
# #### 新型降维方法

# %% [markdown]
# ##### 自编码器

# %%
import tensorflow as tf
from tensorflow import keras

# encoding dim 100
encoding_dim = n_components

# input dim 10000
input_dim = x_total_train_std_flat.shape[1]
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

if USE_DDR_AE:
	# fit
	autoencoder.fit(x_total_train_std_flat, x_total_train_std_flat,
									epochs=100,
									batch_size=32,
									shuffle=True,
									validation_data=(x_total_valid_std_flat, x_total_valid_std_flat))

	# history
	history = autoencoder.history.history
	plt.plot(history['loss'], label='train')
	plt.plot(history['val_loss'], label='valid')
	plt.legend()
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.show()

# %%
# encode data
x_total_train_std_flat_ddr = encoder.predict(x_total_train_std_flat)
x_total_valid_std_flat_ddr = encoder.predict(x_total_valid_std_flat)
x_total_test_std_flat_ddr = encoder.predict(x_total_test_std_flat)
x_couple_train_std_flat_ddr = encoder.predict(x_couple_train_std_flat)
x_couple_valid_std_flat_ddr = encoder.predict(x_couple_valid_std_flat)
x_couple_test_std_flat_ddr = encoder.predict(x_couple_test_std_flat)


if DISABLE_DDR:
	x_total_train_std_flat_ddr = x_total_train_std_flat
	x_total_valid_std_flat_ddr = x_total_valid_std_flat
	x_total_test_std_flat_ddr = x_total_test_std_flat
	x_couple_train_std_flat_ddr = x_couple_train_std_flat
	x_couple_valid_std_flat_ddr = x_couple_valid_std_flat
	x_couple_test_std_flat_ddr = x_couple_test_std_flat

# %% [markdown]
# ## 命名简化

# %%
# total 
xt_train = x_total_train_std_flat_ddr
xt_valid = x_total_valid_std_flat_ddr
xt_test = x_total_test_std_flat_ddr
yt_train = y_total_train[:, 1].reshape(-1, 1)
yt_valid = y_total_valid[:, 1].reshape(-1, 1)
yt_test = y_total_test[:, 1].reshape(-1, 1)

# couple
xc_train = x_couple_train_std_flat_ddr
xc_valid = x_couple_valid_std_flat_ddr
xc_test = x_couple_test_std_flat_ddr
yc_train = y_couple_train[:, 1].reshape(-1, 1)
yc_valid = y_couple_valid[:, 1].reshape(-1, 1)
yc_test = y_couple_test[:, 1].reshape(-1, 1)

# %% [markdown]
# ## 模型预测

# %% [markdown]
# #### 模型评估

# %%
from utils.analysis import model_analysis
import pandas as pd



# %% [markdown]
# ### 线性回归

# %%
from sklearn.linear_model import LinearRegression


# total
print('--------------------------------total------------------------------')
lr_t = LinearRegression()
lr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(lr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'linear reg')
results_total = pd.Series(dict_total).to_frame().T

# couple
print('--------------------------------couple------------------------------')
lr_c = LinearRegression()
lr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(lr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'linear reg')
results_couple = pd.Series(dict_couple).to_frame().T

# %% [markdown]
# ### 支持向量机回归

# %%
from sklearn.svm import SVR

# total
print('--------------------------------total------------------------------')
# linear svr
lr_svf_t = SVR(kernel='linear', max_iter=100000)
lr_svf_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(lr_svf_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'linear svr')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# poly svr
poly_svf_t = SVR(kernel='poly')
poly_svf_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(poly_svf_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'poly svr')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# rbf svr
rbf_svf_t = SVR(kernel='rbf')
rbf_svf_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(rbf_svf_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'rbf svr')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
print('--------------------------------couple------------------------------')
# linear svr
lr_svf_c = SVR(kernel='linear', max_iter=100000)
lr_svf_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(lr_svf_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'linear svr')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# poly svr
poly_svf_c = SVR(kernel='poly')
poly_svf_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(poly_svf_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'poly svr')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# rbf svr
rbf_svf_c = SVR(kernel='rbf')
rbf_svf_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(rbf_svf_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'rbf svr')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### K近邻回归

# %%
from sklearn.neighbors import KNeighborsRegressor

# total 
print('--------------------------------total------------------------------')
# uniform knn
uni_knn_t = KNeighborsRegressor(n_neighbors=5, weights='uniform')
uni_knn_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(uni_knn_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'uniform knn')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# distance knn
dis_knn_t = KNeighborsRegressor(n_neighbors=5, weights='distance')
dis_knn_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(dis_knn_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'distance knn')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
print('--------------------------------couple------------------------------')
# uniform knn
uni_knn_c = KNeighborsRegressor(n_neighbors=5, weights='uniform')
uni_knn_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(uni_knn_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'uniform knn')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# distance knn
dis_knn_c = KNeighborsRegressor(n_neighbors=5, weights='distance')
dis_knn_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(dis_knn_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'distance knn')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### 回归树

# %%
from sklearn.tree import DecisionTreeRegressor

# total
print('--------------------------------total------------------------------')
dtr_t = DecisionTreeRegressor()
dtr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(dtr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'decision tree total')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
print('--------------------------------couple------------------------------')
dtr_c = DecisionTreeRegressor()
dtr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(dtr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'decision tree couple')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### 集成模型

# %% [markdown]
# #### 基础模型

# %%
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# total
print('--------------------------------total------------------------------')
# random forest
rfr_t = RandomForestRegressor()
rfr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(rfr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'random forest')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# extra tree
etr_t = ExtraTreesRegressor()
etr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(etr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'extra tree')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)
# gradient boosting
gbr_t = GradientBoostingRegressor()
gbr_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(gbr_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'gradient boosting')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
print('--------------------------------couple------------------------------')
# random forest
rfr_c = RandomForestRegressor()
rfr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(rfr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'random forest')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)
# extra tree
etr_c = ExtraTreesRegressor()
etr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(etr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'extra tree')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)	
# gradient boosting
gbr_c = GradientBoostingRegressor()
gbr_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(gbr_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'gradient boosting')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# #### XGBoost

# %%
import xgboost as xgb


# total
# XGBoost extreme gradient boosting
xgb_t = xgb.XGBRegressor(booster='gbtree',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=6,
                         min_child_weight=3,
                         seed=42)
xgb_t.fit(xt_train, yt_train.ravel())
dict_total = model_analysis(xgb_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'XGBoost')
results_total = pd.concat([results_total, pd.Series(dict_total).to_frame().T], axis=0)

# couple
# XGBoost extreme gradient boosting
xgb_c = xgb.XGBRegressor(booster='gbtree',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=6,
                         min_child_weight=3,
                         seed=42)
xgb_c.fit(xc_train, yc_train.ravel())
dict_couple = model_analysis(xgb_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'XGBoost')
results_couple = pd.concat([results_couple, pd.Series(dict_couple).to_frame().T], axis=0)

# %% [markdown]
# ### 懒模型

# %%
from lazypredict.Supervised import LazyRegressor
import warnings

warnings.filterwarnings("ignore")

# total
print('--------------------------------total------------------------------')
reg_t = LazyRegressor(verbose=0, custom_metric=ratio_good, predictions=True, ignore_warnings=True)
models_t, predictions_t = reg_t.fit(xt_train, xt_valid, yt_train.ravel(), yt_valid.ravel())
print(models_t)

# couple
print('--------------------------------couple------------------------------')
reg_c = LazyRegressor(verbose=0, custom_metric=ratio_good, predictions=True, ignore_warnings=True)
models_c, predictions_c = reg_c.fit(xc_train, xc_valid, yc_train.ravel(), yc_valid.ravel())
print(models_c)

# %% [markdown]
# ## 结果存储

# %%
# save results_total and results_couple to csv
dir_results = os.path.join(dir_prj, "results")
if not os.path.exists(dir_results):
	os.mkdir(dir_results)

results_total.to_csv(os.path.join(dir_results, "base_total.csv"), index=False)
results_couple.to_csv(os.path.join(dir_results, "base_couple.csv"), index=False)

# save lazyregressor results to csv
models_t.to_csv(os.path.join(dir_results, "lazy_models_total.csv"), index=True)
models_c.to_csv(os.path.join(dir_results, "lazy_models_couple.csv"), index=True)


