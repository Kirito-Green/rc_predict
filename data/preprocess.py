import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../"))

import pandas as pd
import numpy as np

from config import *
from utils.gds import extract_data_from_gds, paths2polygons


def data_enhance():
    pass


def data_multi_sample():
    pass


def get_mask(x, y):
    mask = np.zeros(shape=x.shape, dtype=np.int32)
    for i in range(len(x)):
        num = round(y[i][0])
        if x.ndim == 3:
            mask[i, :num, :] = 1
        elif x.ndim == 2:
            mask[i, :num] = 1

    return mask


def data_filter(x, y): # 剔除异常值
    new_x = []
    new_y = []
    for i in range(len(x)):
        if y[i] >= cap_thresh:
            new_x.append(x[i])
            new_y.append(y[i])

    return new_x, new_y


def get_concat(x):
    new_x = x[0]
    for i in range(1, len(x)):
        new_x = np.concatenate((new_x, x[i]), axis=0)

    return np.array(new_x, dtype=np.float32)


def data_norm(x, method='z_score'):
    new_x = []

    # min max normalization
    if method == 'min_max':
        all_x = get_concat(x)
        min_x = np.min(all_x, axis=0)
        max_x = np.max(all_x, axis=0)
        for x_ in x:
            temp_x = x_.copy()
            cols = 5
            temp_x[:, :cols] = (temp_x[:, :cols] - min_x[:cols]) / (max_x[:cols] - min_x[:cols])
            new_x.append(temp_x)
    elif method == 'z_score':
        all_x = get_concat(x)
        mean_x = np.mean(all_x, axis=0)
        std_x = np.std(all_x, axis=0)
        std_x[std_x == 0] = 1e-8
        for x_ in x:
            temp_x = x_.copy()
            cols = 5
            temp_x[:, :cols] = (temp_x[:, :cols] - mean_x[:cols]) / std_x[:cols]
            new_x.append(temp_x)
    else:
        raise ValueError("method must be min_max or z_score")
    
    return new_x


def data_norm_special(x):
    # special min max normalization
    new_x = []
    for x_ in x:
        temp_x = np.array(x_.copy(), dtype=np.float32)
        temp_x[:, 2] = vertical_space * temp_x[:, 2]
        temp_x[:, :3] = temp_x[:, :3] / window_size + 0.5
        temp_x[:, 3:5] = temp_x[:, 3:5] / max_length
        new_x.append(temp_x)

    return new_x


def data_encode(x):
    # one hot encoding
    new_x = []
    for x_ in x:
        temp_x = np.array(x_.copy(), dtype=np.float32)
        cnt = (np.max(temp_x[:, 5]) - np.min(temp_x[:, 5]) + 1).astype(np.int32)
        one_hot = np.eye(cnt)[temp_x[:, 5].astype(np.int32)]
        temp_x = np.delete(temp_x, 5, axis=1)
        temp_x = np.concatenate((temp_x, one_hot), axis=1)
        new_x.append(temp_x)

    return new_x


def data_preprocess(x, y, special=False):
    if special:
        new_x, new_y = data_filter(x, y)
        new_x = data_norm_special(new_x)
        new_x = data_encode(new_x)
    else:
        new_x, new_y = data_filter(x, y)
        new_x = data_norm(new_x)
        new_x = data_encode(new_x)

    return new_x, new_y


if __name__ == "__main__":
    pass