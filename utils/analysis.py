import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import *


def model_analysis(model, x_train, y_train, x_valid, y_valid, x_test, y_test, name):
    y_train_pred = model.predict(x_train).reshape(-1, 1)
    y_valid_pred = model.predict(x_valid).reshape(-1, 1)
    y_test_pred = model.predict(x_test).reshape(-1, 1)

    dict_train = error_analysis(
        y_train, y_train_pred, title=f'{name} train analysis')
    dict_valid = error_analysis(
        y_valid, y_valid_pred, title=f'{name} valid analysis')
    dict_test = error_analysis(
        y_test, y_test_pred, title=f'{name} test analysis')

    return {
        'model': name,
        'train avg err': dict_train['mean error'],
        'train max err': dict_train['max error'],
        f'train ratio(err>{tolerant_ratio_error})': dict_train['bad ratio'],
        'valid avg err': dict_valid['mean error'],
        'valid max err': dict_valid['max error'],
        f'valid ratio(err>{tolerant_ratio_error})': dict_valid['bad ratio'],
        'test avg err': dict_test['mean error'],
        'test max err': dict_test['max error'],
        f'test ratio(err>{tolerant_ratio_error})': dict_test['bad ratio'],
    }


def error_analysis(y_true, y_predict, title):
    # relative_error = np.abs(y_true - y_predict) / y_true
    eval_error = np.where(y_true <= measure_thresh, np.abs(
        y_predict - y_true), np.abs(y_predict - y_true) / y_true)  # 评估误差
    min_error = np.min(eval_error)
    max_error = np.max(eval_error)
    mean_error = np.mean(eval_error)
    std_error = np.std(eval_error)
    ratio_good = np.sum(eval_error <= tolerant_ratio_error) / len(y_true)
    ratio_bad = np.sum(eval_error > tolerant_ratio_error) / len(y_true)

    # keep 2 demicals
    min_error = np.round(min_error * 100, 0)  # %
    max_error = np.round(max_error * 100, 0)
    mean_error = np.round(mean_error * 100, 0)
    std_error = np.round(std_error * 100, 0)
    ratio_good = np.round(ratio_good, 2)
    ratio_bad = np.round(ratio_bad, 2)

    print(title)
    print('mean error:', mean_error)
    print('min error:', min_error)
    print('max error:', max_error)
    print('std error:', std_error)
    print('good ratio:', ratio_good)
    print('bad ratio:', ratio_bad)
    print('')

    return {
        'min error': min_error,
        'max error': max_error,
        'mean error': mean_error,
        'std error': std_error,
        'good ratio': ratio_good,
        'bad ratio': ratio_bad,
    }


def ratio_good(y_true, y_predict):
    # relative_error = np.abs(y_true - y_predict) / y_true
    eval_error = np.where(y_true <= measure_thresh, np.abs(
        y_predict - y_true), np.abs(y_predict - y_true) / y_true)  # 评估误差
    return np.sum(eval_error <= tolerant_ratio_error) / len(y_true)


def ratio_bad(y_true, y_predict):
    # relative_error = np.abs(y_true - y_predict) / y_true
    eval_error = np.where(y_true <= measure_thresh, np.abs(
        y_predict - y_true), np.abs(y_predict - y_true) / y_true)  # 评估误差
    return np.sum(eval_error > tolerant_ratio_error) / len(y_true)


def scatter_plot(y_true, y_pred, dir, name):
    # re = (y_pred - y_true) / y_true * 100 # 相对误差(%)
    eval_error = np.where(y_true <= measure_thresh, y_pred - y_true,
                          (y_pred - y_true) / y_true) * 100  # 评估误差(%)

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(eval_error, y_true, s=1, c='dodgerblue', marker='.')
    plt.yscale('log')
    plt.xlabel('Evaluation error(%)')
    plt.ylabel('Capacitance(fF)')
    plt.savefig(os.path.join(dir, f'{name}_scatter.jpg'))


def scatter_plot_all(y_train_true, y_train_pred, y_valid_true, y_valid_pred, y_test_true, y_test_pred, dir, name):
    plt.figure(figsize=(8, 6), dpi=300)
    train_eval_error = np.where(y_train_true <= measure_thresh, y_train_pred - y_train_true,
                                 (y_train_pred - y_train_true) / y_train_true) * 100  # 评估误差(%)
    valid_eval_error = np.where(y_valid_true <= measure_thresh, y_valid_pred - y_valid_true,
                                 (y_valid_pred - y_valid_true) / y_valid_true) * 100  # 评估误差(%)
    test_eval_error = np.where(y_test_true <= measure_thresh, y_test_pred - y_test_true,
                                    (y_test_pred - y_test_true) / y_test_true) * 100  # 评估误差(%)
    plt.scatter(train_eval_error, y_train_true, s=1, c='dodgerblue', marker='.')
    plt.scatter(valid_eval_error, y_valid_true, s=1, c='orange', marker='.')
    plt.scatter(test_eval_error, y_test_true, s=1, c='green', marker='.')
    plt.legend(['train', 'valid', 'test'])
    plt.yscale('log')
    plt.xlabel('Evaluation error(%)')
    plt.ylabel('Capacitance(fF)')
    plt.savefig(os.path.join(dir, f'{name}_scatter.jpg'))


def plot_count_pattern():
    df = pd.read_csv(os.path.join(
        dir_prj, "results/count.csv"))
    plt.figure()
    sns.barplot(data=df, x='class', y='count', color='lightblue')
    plt.xlabel('Pattern Class')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        dir_prj, "results/count.jpg"))


def plot_count_cap():
    # plot hist
    plt.figure()
    df_total = pd.read_csv(os.path.join(dir_prj, "results/total_cap.csv"))
    sns.histplot(data=df_total, x='total_cap', color='lightblue', kde=True)
    plt.xlabel('Total Capacitance (fF)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(dir_prj, "results/total_cap.jpg"))

    plt.figure()
    df_couple = pd.read_csv(os.path.join(dir_prj, "results/couple_cap.csv"))
    sns.histplot(data=df_couple, x='couple_cap', color='lightblue', kde=True)
    plt.xlabel('Couple Capacitance (fF)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(dir_prj, "results/couple_cap.jpg"))


def plot_count_polygon():
    df = pd.read_csv(os.path.join(
        dir_prj, "results/polygon_num.csv"))
    plt.figure()
    sns.histplot(data=df, x='polygon_num', color='lightblue', kde=True)
    plt.xlabel('Number of polygons')
    plt.ylabel('Count')
    plt.savefig(os.path.join(
        dir_prj, "results/polygon_num.jpg"))


def count_pattern():
    nums = []
    for pattern_num in pattern_nums:
        dir_convert = os.path.join(
            dir_prj, "data/convert_data/pattern{}".format(pattern_num))

        y = np.load(os.path.join(dir_convert, "y_total.npy"))
        nums.append(len(y))

        print(f'pattern{pattern_num} pattern counted: {nums[-1]}')

    # save the count of patterns
    index = [f'class{i}' for i in range(1, len(nums) + 1)]
    data = [[index[i], nums[i]] for i in range(len(nums))]
    df = pd.DataFrame(data=data, columns=['class', "count"])
    df.to_csv(os.path.join(dir_prj, "results/count.csv"), index=False)

    plot_count_pattern()


def count_polygon():
    nums = []
    for pattern_num in pattern_nums:
        dir_convert = os.path.join(
            dir_prj, "data/convert_data/pattern{}".format(pattern_num))

        x = np.load(os.path.join(dir_convert, "x_total.npy"),
                    allow_pickle=True)
        
        nums.extend([len(x[i]) for i in range(len(x))])
        print(f'pattern{pattern_num} polygon counted')

    # print the min and max number of polygons
    print('min num:', min(nums), 'max num:', max(nums))

    # save data
    df = pd.DataFrame(data=nums, columns=['polygon_num'])
    df.to_csv(os.path.join(dir_prj, "results/polygon_num.csv"), index=False)

    # plot the distribution of polygons
    plot_count_polygon()


def count_cap():
    # analysis capacitance
    y_total_all = []
    y_couple_all = []
    for pattern_num in pattern_nums:
        dir_convert = os.path.join(
            dir_prj, "data/convert_data/pattern{}".format(pattern_num))

        y_total = np.load(os.path.join(dir_convert, "y_total.npy"),
                    allow_pickle=True)
        y_couple = np.load(os.path.join(dir_convert, "y_couple.npy"),
                    allow_pickle=True)
        y_total_all.extend(y_total)
        y_couple_all.extend(y_couple)

    # print bad capacitance
    y_total_all = np.array(y_total_all)
    y_couple_all = np.array(y_couple_all)
    print('bad total cap:', np.sum(y_total_all < cap_thresh))
    print('bad couple cap:', np.sum(y_couple_all < cap_thresh))

    # save data
    df_total = pd.DataFrame(data=y_total_all, columns=['total_cap'])
    df_couple = pd.DataFrame(data=y_couple_all, columns=['couple_cap'])
    df_total.to_csv(os.path.join(dir_prj, "results/total_cap.csv"), index=False)
    df_couple.to_csv(os.path.join(dir_prj, "results/couple_cap.csv"), index=False)

    # plot the distribution of capacitance
    plot_count_cap()


if __name__ == "__main__":
    count_pattern()
    count_polygon()
    count_cap()
