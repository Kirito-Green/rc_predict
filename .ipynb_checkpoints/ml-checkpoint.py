import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import *


def data_truncate(x, percentage):  # reserve percentage
	nums = [len(i) for i in x]
	reserve_num = round(max(nums) * percentage)
	new_x = np.zeros(shape=(len(x), reserve_num, 5), dtype=np.float32)
	for i in range(len(x)):
		num = min(len(x[i]), reserve_num)
		new_x[i][0:num] = x[i][0:num]

	return new_x


if __name__ == "__main__":
	pattern_num = 4
	dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"
	dir_load = os.path.join(dir_prj, "data/convert_data/pattern{}".format(pattern_num))
	x_ = np.load(os.path.join(dir_load, "x.npy"), allow_pickle=True)
	x = data_truncate(x_, percentage)
	y = np.load(os.path.join(dir_load, "y.npy"))

	# data split


	# data preprocessing

