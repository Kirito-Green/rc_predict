import gdspy
import os
import numpy as np


def extra_data_from_gds(gds_path):
	gds_file = gdspy.GdsLibrary(infile=gds_path)
	top_cell = gds_file.top_level()[0]
	labels = top_cell.labels
	polygons = top_cell.polygons
	paths = top_cell.paths

	return  labels, polygons, paths


if __name__ == "__main__":
	pattern_num = 5
	dir_path = ("D:/learn_more_from_life/computer/EDA/work/prj/rc_predict"
	             f"/data/raw_data/pattern{pattern_num}/pattern_results/")
	files = os.listdir(dir_path)
	nums = []
	names = []
	for file in files:
		if file.startswith("pattern") or file.startswith('Pattern'):
			dir_gds = os.path.join(dir_path, file)
			gds_path = os.path.join(dir_gds, f'{file}.gds')
			labels, polygons = extra_data_from_gds(gds_path)
			num = len(polygons)
			nums.append(num)
			names.append(file)

	if len(nums) > 0:
		index_min = np.argmin(nums)
		index_max = np.argmax(nums)
		print('name:', names[index_min], 'min num:', nums[index_min])
		print('name:', names[index_max], 'max num:', nums[index_max])
