import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import gdspy

from config import *

def extract_data_from_gds(gds_path):
    gds_file = gdspy.GdsLibrary(infile=gds_path)
    top_cell = gds_file.top_level()[0]
    labels = top_cell.labels
    polygons = top_cell.polygons
    paths = top_cell.paths

    return labels, polygons, paths


def paths2polygons(paths):
    polygons = []
    for path in paths:
        points = path.points
        widths = path.widths
        layer = path.layers[0]
        n = len(points)
        for i in range(1, n):
            dist = np.array(points[i] - points[i - 1])
            dist_norm = dist / np.sqrt(np.sum(dist ** 2))
            stretch_side = np.where(
                abs(dist) < tolerant_zero_error, widths[i - 1] / 2, 0)
            stretch_front = widths[i - 1] / 2 * dist_norm
            if i == 1:
                pts = [(points[i - 1] + stretch_side), (points[i - 1] - stretch_side),
                       (points[i] + stretch_side + stretch_front), (points[i] - stretch_side + stretch_front)]
            elif i == n - 1:
                pts = [(points[i - 1] + stretch_side + stretch_front), (points[i - 1] - stretch_side + stretch_front),
                       (points[i] + stretch_side), (points[i] - stretch_side)]
            else:
                pts = [(points[i - 1] + stretch_side + stretch_front), (points[i - 1] - stretch_side + stretch_front),
                       (points[i] + stretch_side + stretch_front), (points[i] - stretch_side + stretch_front)]
            polygon = gdspy.Polygon(pts, layer=layer)
            polygons.append(polygon)

    return polygons


if __name__ == "__main__":
    pattern_num = 8
    dir_path = ("D:/learn_more_from_life/computer/EDA/work/prj/rc_predict"
                f"/data/raw_data/pattern{pattern_num}/pattern_results/")
    files = os.listdir(dir_path)
    nums = []
    names = []
    for file in files:
        if file.startswith("pattern") or file.startswith('Pattern'):
            dir_gds = os.path.join(dir_path, file)
            gds_path = os.path.join(dir_gds, f'{file}.gds')
            labels, polygons, paths = extract_data_from_gds(gds_path)
            paths_polygons = paths2polygons(paths)
            polygons.extend(paths_polygons)
            num = len(polygons)
            nums.append(num)
            names.append(file)

    if len(nums) > 0:
        index_min = np.argmin(nums)
        index_max = np.argmax(nums)
        print('name:', names[index_min], 'min num:', nums[index_min])
        print('name:', names[index_max], 'max num:', nums[index_max])
