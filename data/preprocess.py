import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))

import numpy as np
import pandas as pd

from config import *


def data_enhance():
	pass


def data_multi_sample():
	pass


def data_process():
	pass


if __name__ == "__main__":
	# analysis the data
	pattern_nums = []
	dir_path = os.path.join(dir_prj, "data/raw_data")
	files = os.listdir(dir_path)
	for file in files:
		if file.startswith('pattern'):
			pattern_num = file.split('pattern')[1]
			pattern_nums.append(pattern_num)
	print('pattern numbers:', pattern_nums)

	nums = []
	total_cap_3d_all = []
	couple_cap_3d_all = []
	for pattern_num in pattern_nums:
		dir_pattern = os.path.join(dir_prj, "data/raw_data/pattern{}".format(pattern_num))

		id = 1
		while True:
			if os.path.exists(os.path.join(dir_pattern, "analysis_results")):
				path_csv = os.path.join(dir_prj, "data/raw_data/pattern{}/analysis_results/result_all.csv".format(pattern_num))
				data_file = pd.read_csv(path_csv)
				id = -1
			else:
				path_csv = os.path.join(dir_prj, "data/raw_data/pattern{}/analysis_results_{}/result_all.csv".format(pattern_num, id))
				if not os.path.exists(path_csv):
					break
				data_file = pd.read_csv(path_csv)

			total_cap_3d = data_file['total_cap_3d']
			couple_cap_3d = data_file['couple_cap_3d']

			nums.append(len(total_cap_3d))
			total_cap_3d_all.append(total_cap_3d)
			couple_cap_3d_all.append(couple_cap_3d)

			if id == -1:
				break
			else:
				id += 1

	# analysis the data
	total_cap_3d_all = np.array([element for sub_list in total_cap_3d_all for element in sub_list])
	couple_cap_3d_all = np.array([element for sub_list in couple_cap_3d_all for element in sub_list])
	print('nums:', nums)
	print(np.min(total_cap_3d_all), np.max(total_cap_3d_all), np.mean(total_cap_3d_all), np.std(total_cap_3d_all))
	print(np.min(couple_cap_3d_all), np.max(couple_cap_3d_all), np.mean(couple_cap_3d_all), np.std(couple_cap_3d_all))
	print(np.sum(total_cap_3d_all==0), np.sum(couple_cap_3d_all==0))