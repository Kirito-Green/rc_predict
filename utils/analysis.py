import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from spektral.data import BatchLoader
from config import *


def model_analysis(model, x_train, y_train, x_valid, y_valid, x_test, y_test, name):
	y_train_predict = model.predict(x_train).reshape(-1, 1)
	y_valid_predict = model.predict(x_valid).reshape(-1, 1)
	y_test_predict = model.predict(x_test).reshape(-1, 1)
	mse_train = mean_squared_error(y_train, y_train_predict)
	mse_valid = mean_squared_error(y_valid, y_valid_predict)
	mse_test = mean_squared_error(y_test, y_test_predict)
	r2_train = r2_score(y_train, y_train_predict)
	r2_valid = r2_score(y_valid, y_valid_predict)
	r2_test = r2_score(y_test, y_test_predict)
	print(f"{name} train mse:", mse_train)
	print(f"{name} train r2:", r2_train)
	print(f"{name} valid mse:", mse_valid)
	print(f"{name} valid r2:", r2_valid)
	print(f"{name} test mse:", mse_test)
	print(f"{name} test r2:", r2_test)
	print('')
	print('')

	dict_train = error_analysis(y_train, y_train_predict, title=f'{name} train analysis')
	dict_valid = error_analysis(y_valid, y_valid_predict, title=f'{name} valid analysis')
	dict_test = error_analysis(y_test, y_test_predict, title=f'{name} test analysis')

	return {
	'model': name,
	'train mse': mse_train,
	'train r2': r2_train,
	'train ratio': dict_train['good ratio'],
	'valid mse': mse_valid,
	'valid r2': r2_valid,
	'valid ratio': dict_valid['good ratio'],
	'test mse': mse_test,
	'test r2': r2_test,
	'test ratio': dict_test['good ratio'],
	}


def gnn_analysis(model, batch_size, x_train, y_train, x_valid, y_valid, x_test, y_test, name):
	x_train_loader = BatchLoader(x_train, batch_size=batch_size, shuffle=False)
	x_valid_loader = BatchLoader(x_valid, batch_size=batch_size, shuffle=False)
	x_test_loader = BatchLoader(x_test, batch_size=batch_size, shuffle=False)
	y_train_predict = model.predict(x_train_loader.load(), steps=x_train_loader.steps_per_epoch).reshape(-1, 1)
	y_valid_predict = model.predict(x_valid_loader.load(), steps=x_valid_loader.steps_per_epoch).reshape(-1, 1)
	y_test_predict = model.predict(x_test_loader.load(), steps=x_test_loader.steps_per_epoch).reshape(-1, 1)
	mse_train = mean_squared_error(y_train, y_train_predict)
	mse_valid = mean_squared_error(y_valid, y_valid_predict)
	mse_test = mean_squared_error(y_test, y_test_predict)
	r2_train = r2_score(y_train, y_train_predict)
	r2_valid = r2_score(y_valid, y_valid_predict)
	r2_test = r2_score(y_test, y_test_predict)
	print(f"{name} train mse:", mse_train)
	print(f"{name} train r2:", r2_train)
	print(f"{name} valid mse:", mse_valid)
	print(f"{name} valid r2:", r2_valid)
	print(f"{name} test mse:", mse_test)
	print(f"{name} test r2:", r2_test)
	print('')
	print('')

	dict_train = error_analysis(y_train, y_train_predict, title=f'{name} train analysis')
	dict_valid = error_analysis(y_valid, y_valid_predict, title=f'{name} valid analysis')
	dict_test = error_analysis(y_test, y_test_predict, title=f'{name} test analysis')

	return {
	'model': name,
	'train mse': mse_train,
	'train r2': r2_train,
	'train ratio': dict_train['good ratio'],
	'valid mse': mse_valid,
	'valid r2': r2_valid,
	'valid ratio': dict_valid['good ratio'],
	'test mse': mse_test,
	'test r2': r2_test,
	'test ratio': dict_test['good ratio'],
	}


def error_analysis(y, y_predict, title):
	relative_error = np.abs(y - y_predict) / y
	min_error = np.min(relative_error)
	max_error = np.max(relative_error)
	mean_error = np.mean(relative_error)
	std_error = np.std(relative_error)
	num_good = np.sum(relative_error <= tolerant_ratio_error) / len(y)
	num_bad = np.sum(relative_error > tolerant_ratio_error) / len(y)
	print(title)
	print('min error:', min_error)
	print('max error:', max_error)
	print('mean error:', mean_error)
	print('std error:', std_error)
	print('good ratio:', num_good)
	print('bad ratio:', num_bad)
	print('')

	return {
		'min error': min_error,
		'max error': max_error,
		'mean error': mean_error,
		'std error': std_error,
		'good ratio': num_good,
		'bad ratio': num_bad,
	}


def ratio_good(y, y_predict):
	relative_error = np.abs(y - y_predict) / y
	return np.sum(relative_error <= tolerant_ratio_error) / len(y)
