import numpy as np
import pandas as pd
import os
import sys
from functools import cmp_to_key
from tqdm import trange

sys.path.append(os.path.join(os.getcwd(), '..'))

from utils.read_gds import extra_labels_and_polygons
from config import *


def cmp(p1, p2): # 先主导提即x=0 y=0 z=0再耦合导体即x<0再层数layer=0再layer<0再dist再x（层数值越小越在上层）
	if p2[0] == 0 and p2[1] == 0 and p2[2] == 0: # main polygon
		return 1
	elif p1[0] == 0 and p1[1] == 0 and p1[2] == 0:
		return -1
	else: # environment polygon
		if p1[3] > 0 > p2[3]:
			return 1
		elif p1[3] < 0 < p2[3]:
			return -1
		else: # layer
			if abs(p1[2]) > abs(p2[2]):
				return 1
			elif abs(p1[2]) < abs(p2[2]):
				return -1
			else: # dist
				if p1[0] * p1[0] + p1[1] * p1[1] > p2[0] * p2[0] + p2[1] * p2[1]:  # >
					return 1
				elif p1[0] * p1[0] + p1[1] * p1[1] < p2[0] * p2[0] + p2[1] * p2[1]:  # <
					return -1
				else:
					return 0


def get_polygon_data(polygon):
	endpoint_pos = np.mean(np.array(polygon.polygons), axis=0)
	x, y = np.mean(endpoint_pos, axis=0)
	max_pos = np.max(endpoint_pos, axis=0)
	min_pos = np.min(endpoint_pos, axis=0)
	width = max_pos[0] - min_pos[0]
	height = max_pos[1] - min_pos[1]
	layer = np.mean(polygon.layers)

	return np.array([x, y, layer, width, height])


def is_in_polygon(label, polygon):
	data_polygon = get_polygon_data(polygon)
	label_pos = np.array(label.position)
	data_label = np.append(label_pos, label.layer)  # x y layer

	return  data_label[2] == data_polygon[2] and \
		data_polygon[0]-data_polygon[3]/2 <= data_label[0] <= data_polygon[0]+data_polygon[3]/2 and \
		data_polygon[1]-data_polygon[4]/2 <= data_label[1] <= data_polygon[1]+data_polygon[4]/2


def convert_polygons_total(net_name, labels, polygons):
	main_label = labels[0]
	for label in labels:
		if label.text.lower() == net_name.lower():
			main_label = label
			break
	data_polygons = []
	main_polygon = np.zeros((1, 5))
	for polygon in polygons:
		data_polygon = get_polygon_data(polygon)
		data_polygons.append(data_polygon)

		# get main polygon
		if is_in_polygon(main_label, polygon):
			main_polygon = data_polygon.copy()

	# feature engineering
	for i in range(len(data_polygons)):
		data_polygons[i][0] -= main_polygon[0]
		data_polygons[i][1] -= main_polygon[1]
		data_polygons[i][2] -= main_polygon[2]

	# sort
	data_polygons.sort(key=cmp_to_key(cmp))

	return data_polygons


def convert_polygons_couple(net_name, labels, polygons):
	main_label = labels[0]
	for label in labels:
		if label.text.lower() == net_name.lower():
			main_label = label
			break
	data_polygons = []
	main_polygon = np.zeros((1, 5))
	env_polygons_id = []
	for i, polygon in enumerate(polygons):
		data_polygon = get_polygon_data(polygon)
		data_polygons.append(data_polygon)

		# get main polygon
		if is_in_polygon(main_label, polygon):
			main_polygon = data_polygon.copy()

		# get environment polygon
		for label in labels:
			if label != main_label and is_in_polygon(label, polygon):
				env_polygons_id.append(i)
				break

	# feature engineering
	# main polygon processed
	for i in range(len(data_polygons)):
		data_polygons[i][0] -= main_polygon[0]
		data_polygons[i][1] -= main_polygon[1]
		data_polygons[i][2] -= main_polygon[2]

	# environment polygon processed
	for i in env_polygons_id:
		data_polygons[i][3] = -data_polygons[i][3] # width
		data_polygons[i][4] = -data_polygons[i][4] # height

	# sort
	data_polygons.sort(key=cmp_to_key(cmp))

	return data_polygons


def convert_data(pattern_num):
	path_csv = os.path.join(dir_prj, "data/raw_data/pattern{}/analysis_results/result_all.csv".format(pattern_num))
	dir_pattern = os.path.join(dir_prj, "data/raw_data/pattern{}/pattern_results".format(pattern_num))
	dir_save = os.path.join(dir_prj, "data/convert_data/pattern{}".format(pattern_num))

	data_file = pd.read_csv(path_csv)
	net_names = data_file['net_name']
	case_names = data_file['case_name']
	total_cap_3d = data_file['total_cap_3d']
	couple_cap_3d = data_file['couple_cap_3d']
	data_polygons_total_all = []
	data_polygons_couple_all = []
	data_total_cap_3d = []
	data_couple_cap_3d = []

	for i in trange(len(net_names)):
		gds_name = case_names[i] + ".gds"
		gds_path = os.path.join(dir_pattern, case_names[i], gds_name)
		if not os.path.exists(gds_path):
			continue
		labels, polygons = extra_labels_and_polygons(gds_path)
		data_polygons_total = convert_polygons_total(net_names[i], labels, polygons)
		data_polygons_couple = convert_polygons_couple(net_names[i], labels, polygons)

		# append data
		data_polygons_total_all.append(data_polygons_total) # x
		data_polygons_couple_all.append(data_polygons_couple)
		data_total_cap_3d.append(total_cap_3d[i]) # target
		data_couple_cap_3d.append(couple_cap_3d[i])

	data_polygons_total_all = np.array(data_polygons_total_all, dtype=object)
	data_polygons_couple_all = np.array(data_polygons_couple_all, dtype=object)
	data_total_cap_3d = np.array(data_total_cap_3d, dtype=np.float32)
	data_couple_cap_3d = np.array(data_couple_cap_3d, dtype=np.float32)

	# save data
	if not os.path.exists(dir_save):
		os.mkdir(dir_save)
	np.save(os.path.join(dir_save, "x_total.npy"), data_polygons_total_all)
	np.save(os.path.join(dir_save, "x_couple.npy"), data_polygons_couple_all)
	np.save(os.path.join(dir_save, "y_total.npy"), data_total_cap_3d)
	np.save(os.path.join(dir_save, "y_couple.npy"), data_couple_cap_3d)

	# load data
	# x_total = np.load(os.path.join(dir_save, "x_total.npy"), allow_pickle=True)
	# x_couple = np.load(os.path.join(dir_save, "x_couple.npy"), allow_pickle=True)
	# y_total = np.load(os.path.join(dir_save, "y_total.npy"))
	# y_couple = np.load(os.path.join(dir_save, "y_couple.npy"))
	# print(x_total[0][:5])
	# print(x_couple[0][:5])


if __name__ == "__main__":
	pattern_num = 4
	# convert_data(pattern_num)
