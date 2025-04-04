# %% [markdown]
# # 版图电容预测

# %% [markdown]
# ## 基本库导入

# %%
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))
import numpy as np
import matplotlib.pyplot as plt

from config import *
from data.layout import convert_data, convert_data_parallel

# %% [markdown]
# ## 数据导入与数据清洗

# %%
def data_process(x, reserve_num):
	raw_num = np.array([len(i) for i in x]).reshape(-1, 1)
	valid_num = np.array([reserve_num if len(i) > reserve_num else len(i) for i in x]).reshape(-1, 1)
	new_x = np.zeros(shape=(len(x), reserve_num, 5), dtype=np.float32)
	for i in range(len(x)):
		num = min(len(x[i]), reserve_num)
		new_x[i][:num] = x[i][:num]

	return new_x, valid_num, raw_num

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
		convert_data_parallel(dir_prj, pattern_num, num_process=8)

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
# ## 数据标准化

if DISABLE_STD:
    x_total_train_std_flat = x_total_train.reshape(len(x_total_train), -1)
    x_total_valid_std_flat = x_total_valid.reshape(len(x_total_valid), -1)
    x_total_test_std_flat = x_total_test.reshape(len(x_total_test), -1)
    x_couple_train_std_flat = x_couple_train.reshape(len(x_couple_train), -1)
    x_couple_valid_std_flat = x_couple_valid.reshape(len(x_couple_valid), -1)
    x_couple_test_std_flat = x_couple_test.reshape(len(x_couple_test), -1)
# %% [markdown]
# ## 命名简化

# %%
# total 
xt_train = x_total_train_std_flat
xt_valid = x_total_valid_std_flat
xt_test = x_total_test_std_flat
yt_train = y_total_train[:, 1].reshape(-1, 1)
yt_valid = y_total_valid[:, 1].reshape(-1, 1)
yt_test = y_total_test[:, 1].reshape(-1, 1)

# couple
xc_train = x_couple_train_std_flat
xc_valid = x_couple_valid_std_flat
xc_test = x_couple_test_std_flat
yc_train = y_couple_train[:, 1].reshape(-1, 1)
yc_valid = y_couple_valid[:, 1].reshape(-1, 1)
yc_test = y_couple_test[:, 1].reshape(-1, 1)

# %% [markdown]
# ## 模型预测

# %% [markdown]
# #### 模型评估

# %%
from utils.analysis import model_analysis


# %%
import xgboost as xgb
import pandas as pd


# total
print('--------------------------------total------------------------------')
# XGBoost extreme gradient boosting
xgb_t = xgb.XGBRegressor(booster='gbtree',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=6,
                         min_child_weight=3,
                         seed=42)
dict_total = model_analysis(xgb_t, xt_train, yt_train, xt_valid, yt_valid, xt_test, yt_test, 'XGBoost')
results_total = pd.Series(dict_total).to_frame().T

# couple
print('--------------------------------couple------------------------------')
# XGBoost extreme gradient boosting
xgb_c = xgb.XGBRegressor(booster='gbtree',
                         n_estimators=100,
                         learning_rate=0.1,
                         max_depth=6,
                         min_child_weight=3,
                         seed=42)
dict_couple = model_analysis(xgb_c, xc_train, yc_train, xc_valid, yc_valid, xc_test, yc_test, 'XGBoost')
results_couple = pd.Series(dict_couple).to_frame().T

# %% save results
dir_results = os.path.join(dir_prj, "results")
if not os.path.exists(dir_results):
	os.mkdir(dir_results)

results_total.to_csv(os.path.join(dir_results, "xgb_total.csv"), index=False)
results_couple.to_csv(os.path.join(dir_results, "xgb_couple.csv"), index=False)
