import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))
import numpy as np
import matplotlib.pyplot as plt

from config import *


def model_analysis(model, x_train, y_train, x_valid, y_valid, x_test, y_test, name):
	y_train_pred = model.predict(x_train).reshape(-1, 1)
	y_valid_pred = model.predict(x_valid).reshape(-1, 1)
	y_test_pred = model.predict(x_test).reshape(-1, 1)

	dict_train = error_analysis(y_train, y_train_pred, title=f'{name} train analysis')
	dict_valid = error_analysis(y_valid, y_valid_pred, title=f'{name} valid analysis')
	dict_test = error_analysis(y_test, y_test_pred, title=f'{name} test analysis')

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
	eval_error = np.where(y_true <= 1, np.abs(y_predict - y_true), np.abs(y_predict - y_true) / y_true) # 评估误差
	min_error = np.min(eval_error)
	max_error = np.max(eval_error)
	mean_error = np.mean(eval_error)
	std_error = np.std(eval_error)
	ratio_good = np.sum(eval_error <= tolerant_ratio_error) / len(y_true)
	ratio_bad = np.sum(eval_error > tolerant_ratio_error) / len(y_true)

	# keep 2 demicals
	min_error = np.round(min_error, 2) * 100 # %
	max_error = np.round(max_error, 2) * 100
	mean_error = np.round(mean_error, 2) * 100
	std_error = np.round(std_error, 2) * 100
	ratio_good = np.round(ratio_good, 2)
	ratio_bad = np.round(ratio_bad, 2)

	print(title)
	print('min error:', min_error)
	print('max error:', max_error)
	print('mean error:', mean_error)
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
	eval_error = np.where(y_true <= 1, np.abs(y_predict - y_true), np.abs(y_predict - y_true) / y_true) # 评估误差
	return np.sum(eval_error <= tolerant_ratio_error) / len(y_true)


def ratio_bad(y_true, y_predict):
	# relative_error = np.abs(y_true - y_predict) / y_true
	eval_error = np.where(y_true <= 1, np.abs(y_predict - y_true), np.abs(y_predict - y_true) / y_true) # 评估误差
	return np.sum(eval_error > tolerant_ratio_error) / len(y_true)


def scatter_plot(y_true, y_pred, dir, name):
	# re = (y_pred - y_true) / y_true * 100 # 相对误差(%)
	eval_error = np.where(y_true <= 1, y_pred - y_true, (y_pred - y_true) / y_true) * 100 # 评估误差(%)
	y_true = y_true * 1e-15
	
	plt.figure(figsize=(8, 6), dpi=300)
	plt.scatter(eval_error, y_true, s=1, c='dodgerblue', marker='.')
	plt.yscale('log')
	plt.xlabel('Evaluation error(%)')
	plt.ylabel('Capacitance(F)')
	plt.savefig(os.path.join(dir, f'{name}_scatter.jpg'))
