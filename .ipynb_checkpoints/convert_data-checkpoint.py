import numpy as np
import pandas as pd
import yaml
import os
from functools import cmp_to_key
from tqdm import trange

from read_gds import extract_data_from_gds


def cmp(p1, p2): # 先层数再距离再横向距离
    if p1[0] * p1[0] + p1[1] * p1[1] > p2[0] * p2[0] + p2[1] * p2[1]:  # >
        return 1
    elif p1[0] * p1[0] + p1[1] * p1[1] < p2[0] * p2[0] + p2[1] * p2[1]:  # <
        return -1
    else:
        return 0


def convert_polygons(net_name, labels, polygons):
    for label in labels:
        if label.text.lower() == net_name.lower():
            break
    label_pos = np.array(label.position)
    data_label = np.append(label_pos, label.layer)  # x y layer
    data_polygons = []
    tar_pos = [0, 0]
    for polygon in polygons:
        endpoint_pos = np.mean(np.array(polygon.polygons), axis=0)
        x, y = np.mean(endpoint_pos, axis=0)
        max_pos = np.max(endpoint_pos, axis=0)
        min_pos = np.min(endpoint_pos, axis=0)
        width = max_pos[0] - min_pos[0]
        height = max_pos[1] - min_pos[1]
        layer = np.mean(polygon.layers)
        data_polygon = np.array([x, y, width, height, layer])
        data_polygons.append(data_polygon)

        # get target polygon
        if min_pos[0] <= data_label[0] <= max_pos[0] and \
            min_pos[1] <= data_label[1] <= max_pos[1]:
            tar_pos = [x, y]

    # feature extracting
    for i in range(len(data_polygons)):
        data_polygons[i][0] -= tar_pos[0]
        data_polygons[i][1] -= tar_pos[1]

    # sort by distance
    data_polygons.sort(key=cmp_to_key(cmp))

    return data_polygons


if __name__ == "__main__":
    pattern_num = 4
    # ubuntu
    # path_csv = ("/media/kael/台电/learn_more_from_life/computer/EDA"
    #             "/work/prj/rc_predict/raw_data/pattern{}/error_analysiss"
    #             "/result_all.csv".format(pattern_num))
    # dir_pattern = ("/media/kael/台电/learn_more_from_life/computer"
    #                "/EDA/work/prj/rc_predict/raw_data/pattern{}"
    #                "/pattern_results".format(pattern_num))
    # windows
    dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"
    path_csv = os.path.join(dir_prj, "data/raw_data/pattern{}/error_analysiss/result_all.csv".format(pattern_num))
    dir_pattern = os.path.join(dir_prj, "data/raw_data/pattern{}/pattern_results".format(pattern_num))
    dir_save = os.path.join(dir_prj, "data/convert_data/pattern{}".format(pattern_num))

    data_file = pd.read_csv(path_csv)
    net_names = data_file['net_name']
    case_names = data_file['case_name']
    total_cap_3d = data_file['total_cap_3d']
    data_polygons_all = []
    data_total_cap_3d = []
    num = []
    for i in trange(len(net_names)):
        gds_name = case_names[i] + ".gds"
        gds_path = os.path.join(dir_pattern, case_names[i], gds_name)
        if not os.path.exists(gds_path):
            continue
        labels, polygons = extract_data_from_gds(gds_path)
        data_polygons = convert_polygons(net_names[i], labels, polygons)

        # append data
        data_polygons_all.append(data_polygons) # x
        data_total_cap_3d.append(total_cap_3d[i]) # target
        num.append(len(data_polygons))

    data_polygons_all = np.array(data_polygons_all, dtype=object)
    data_total_cap_3d = np.array(data_total_cap_3d, dtype=np.float32)

    # save data
    if not os.path.exists(dir_save):
        os.mkdir(dir_save)
    np.save(os.path.join(dir_save, "x.npy"), data_polygons_all)
    np.save(os.path.join(dir_save, "y.npy"), data_total_cap_3d)

    # load data
    # x = np.load(os.path.join(dir_save, "x.npy"), allow_pickle=True)
    # y = np.load(os.path.join(dir_save, "y.npy"))
    # print(x)
    # print(y)
