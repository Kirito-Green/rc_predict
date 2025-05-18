import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import multiprocessing
from tqdm import tqdm
import pandas as pd
import numpy as np

from config import *
from utils.gds import extract_data_from_gds, paths2polygons


def get_polygon_data(polygon):
    endpoint_pos = np.array(polygon.polygons)[0]
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

    return data_label[2] == data_polygon[2] and \
        data_polygon[0]-data_polygon[3]/2 <= data_label[0] <= data_polygon[0]+data_polygon[3]/2 and \
        data_polygon[1]-data_polygon[4] / \
        2 <= data_label[1] <= data_polygon[1]+data_polygon[4]/2


def split_polygon(polygon):
    # split polygon into small parts
    # data_polygon: x y layer width height flag
    data_polygon = np.copy(polygon)
    data_polygons = []
    width = data_polygon[3]
    height = data_polygon[4]
    if width > max_length:
        data_polygon[3] = width / 2
        data_polygon[4] = height
        # split into two parts
        data_polygon1 = np.copy(data_polygon)
        data_polygon2 = np.copy(data_polygon)
        data_polygon1[0] -= width / 4
        data_polygon2[0] += width / 4
        # extend list
        data_polygons.extend(split_polygon(data_polygon1))
        data_polygons.extend(split_polygon(data_polygon2))
        return data_polygons
    elif height > max_length:
        data_polygon[3] = width
        data_polygon[4] = height / 2
        # split into two parts
        data_polygon1 = np.copy(data_polygon)
        data_polygon2 = np.copy(data_polygon)
        data_polygon1[1] -= height / 4
        data_polygon2[1] += height / 4
        # extend list
        data_polygons.extend(split_polygon(data_polygon1))
        data_polygons.extend(split_polygon(data_polygon2))
        return data_polygons

    # if width and height are small enough, return the polygon
    data_polygons.append(data_polygon)
    return data_polygons


def convert_polygons_total(net_name, labels, polygons, paths):
    main_label = labels[0]
    for label in labels:
        if label.text.lower() == net_name.lower():
            main_label = label
            break

    data_polygons = []

    # get polygons from paths
    path_polygons = paths2polygons(paths)
    # append polygons
    all_polygons = []
    all_polygons.extend(polygons)
    all_polygons.extend(path_polygons)
    for polygon in all_polygons:
        data_polygon = get_polygon_data(polygon)

        # get main polygon
        # 0 main polygon 1 environment polygon
        if is_in_polygon(main_label, polygon):
            data_polygon = np.append(data_polygon, 0)  # main polygon
        else:
            data_polygon = np.append(data_polygon, 1)  # environment polygon

        # append polygon
        # data_polygons.append(data_polygon)
        polygon_parts = split_polygon(data_polygon)
        data_polygons.extend(polygon_parts)

    return data_polygons


def convert_polygons_couple(net_name, labels, polygons, paths):
    main_label = labels[0]
    for label in labels:
        if label.text.lower() == net_name.lower():
            main_label = label
            break

    data_polygons = []

    # get polygons from paths
    path_polygons = paths2polygons(paths)
    # append polygons
    all_polygons = []
    all_polygons.extend(polygons)
    all_polygons.extend(path_polygons)
    for polygon in all_polygons:
        data_polygon = get_polygon_data(polygon)

        # get main polygon
        # 0 main polygon 1 environment polygon 2 couple polygon
        if is_in_polygon(main_label, polygon):
            data_polygon = np.append(data_polygon, 0)  # main polygon
        else:
            couple_flag = 0
            # get environment polygon
            for label in labels:
                if label != main_label and is_in_polygon(label, polygon):
                    couple_flag = 1
                    break

            if couple_flag:
                data_polygon = np.append(data_polygon, 2)  # couple polygon
            else:
                data_polygon = np.append(
                    data_polygon, 1)  # environment polygon

        # append polygon
        # data_polygons.append(data_polygon)
        polygon_parts = split_polygon(data_polygon)
        data_polygons.extend(polygon_parts)

    return data_polygons


def call_convert_data(args):
    id, case_name, net_name, dir_gds, dir_save, total_cap_3d, couple_cap_3d = args
    gds_name = case_name + ".gds"
    gds_path = os.path.join(dir_gds, case_name, gds_name)
    if not os.path.exists(gds_path):
        print('gds path not exist: {}'.format(gds_path))
        return
    labels, polygons, paths = extract_data_from_gds(gds_path)
    data_polygons_total = convert_polygons_total(
        net_name, labels, polygons, paths)
    data_polygons_couple = convert_polygons_couple(
        net_name, labels, polygons, paths)

    # save data
    np.savez(os.path.join(dir_save, f'convert_{id}.npz'),
             x_total=data_polygons_total,
             x_couple=data_polygons_couple,
             y_total=total_cap_3d,
             y_couple=couple_cap_3d)


def convert_data_parallel(dir_prj, pattern_num, num_process=8):
    print('converting data from pattern{}'.format(pattern_num))
    data_polygons_total_all = []
    data_polygons_couple_all = []
    data_total_cap_3d = []
    data_couple_cap_3d = []

    dir_pattern = os.path.join(
        dir_prj, "data/raw_data/pattern{}".format(pattern_num))
    if not os.path.exists(dir_pattern):
        print('pattern{} raw data not exist'.format(pattern_num))
        return

    id = 1
    while True:
        if os.path.exists(os.path.join(dir_pattern, "analysis_results")):
            path_csv = os.path.join(
                dir_prj, "data/raw_data/pattern{}/analysis_results/result_all.csv".format(pattern_num))
            dir_gds = os.path.join(
                dir_prj, "data/raw_data/pattern{}/pattern_results".format(pattern_num))
            dir_convert = os.path.join(
                dir_prj, "data/convert_data")
            dir_save = os.path.join(
                dir_convert, "pattern{}".format(pattern_num))
            data_file = pd.read_csv(path_csv)
            id = -1
        else:
            path_csv = os.path.join(
                dir_prj, "data/raw_data/pattern{}/analysis_results_{}/result_all.csv".format(pattern_num, id))
            dir_gds = os.path.join(
                dir_prj, "data/raw_data/pattern{}/pattern_results_{}".format(pattern_num, id))
            dir_convert = os.path.join(
                dir_prj, "data/convert_data")
            dir_save = os.path.join(
                dir_convert, "pattern{}".format(pattern_num))
            if not os.path.exists(dir_gds):
                break
            data_file = pd.read_csv(path_csv)

        if not os.path.exists(dir_convert):
            os.mkdir(dir_convert)
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)

        net_names = data_file['net_name']
        case_names = data_file['case_name']
        total_cap_3d = data_file['total_cap_3d']
        couple_cap_3d = data_file['couple_cap_3d']

        args = []
        cnt_sum = 0
        for i in range(len(net_names)):
            cnt_sum += 1

            # 限制样本数量
            if cnt_sum > cnt_max:
                break

            args.append((i, case_names[i], net_names[i], dir_gds,
                        dir_save, total_cap_3d[i], couple_cap_3d[i]))

        # work bar
        pbar = tqdm(total=len(
            args), desc=f"Converting data from pattern{pattern_num}_{id}", unit="file")
        update = lambda *args: pbar.update()

        pool = multiprocessing.Pool(processes=num_process)
        # map the function to the arguments
        # pool.map(call_convert_data, args)

        for arg in args:
            pool.apply_async(call_convert_data, (arg, ), callback=update)
        pool.close()
        pool.join()

        # gather data
        for i in range(len(net_names)):
            load_path = os.path.join(dir_save, f'convert_{i}.npz')
            if not os.path.exists(load_path):
                continue
            with np.load(load_path, allow_pickle=True) as data:
                data_polygons_total_all.append(data['x_total'])
                data_polygons_couple_all.append(data['x_couple'])
                data_total_cap_3d.append(data['y_total'])
                data_couple_cap_3d.append(data['y_couple'])

        # delete data
        for i in range(len(net_names)):
            try:
                delete_path = os.path.join(dir_save, f'convert_{i}.npz')
                if os.path.exists(delete_path):
                    os.remove(delete_path)
            except Exception as e:
                print(f'Error: {e}')

        # loop control
        if id == -1:
            break
        else:
            id += 1

    # convert to numpy array
    data_polygons_total_all = np.array(data_polygons_total_all, dtype=object)
    data_polygons_couple_all = np.array(data_polygons_couple_all, dtype=object)
    data_total_cap_3d = np.array(data_total_cap_3d, dtype=np.float32)
    data_couple_cap_3d = np.array(data_couple_cap_3d, dtype=np.float32)

    # save data
    np.save(os.path.join(dir_save, "x_total.npy"), data_polygons_total_all)
    np.save(os.path.join(dir_save, "x_couple.npy"), data_polygons_couple_all)
    np.save(os.path.join(dir_save, "y_total.npy"), data_total_cap_3d)
    np.save(os.path.join(dir_save, "y_couple.npy"), data_couple_cap_3d)
    print(f'convert data from pattern{pattern_num} done!')


if __name__ == "__main__":
    pattern_nums = [3, 4, 26]
    for pattern_num in pattern_nums:
        convert_data_parallel(
                dir_prj, pattern_num)
        # convert_data_parallel(
        #     '/home/prj/rc_predict/', pattern_num)
