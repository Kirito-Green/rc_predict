import argparse
import os
import sys

sys.path.append(os.getcwd())

from config import *


parser = argparse.ArgumentParser()
parser.add_argument('--dir_prj', type=str, default=dir_prj,
										help='project directory')
parser.add_argument('--seed', type=int, default=seed,
										help='random seed')
parser.add_argument('--pattern_nums', type=int, nargs='+', default=pattern_nums,
										help='pattern nums')
parser.add_argument('--thresh', type=int, nargs='+', default=thresh,
										help='threshold')
parser.add_argument('--num_process', type=int, default=num_process,
                    help='multiprocessing number')
parser.add_argument('--model_name', type=str, default=model_name,
                    help='model name [gcn, graph_sage, gat]')
parser.add_argument('--lr', type=float, default=lr,
                    help='adam learning rate')
parser.add_argument('--batch_size', type=int, default=batch_size,
										help='batch size')
parser.add_argument('--epochs', type=int, default=epochs,
										help='number of epochs')
parser.add_argument('--load_params', type=bool, default=LOAD_PARAMS,
										help='load parameters')
parser.add_argument('--set_memory_growth', type=bool, default=SET_MEMORY_GROWTH,
										help='set memory growth')
parser.add_argument('--set_memory_limit', type=int, default=SET_MEMORY_LIMIT,
										help='memory limit, -1 for no limit')
parser.add_argument('--set_multi_gpu_num', type=int, default=SET_MULTI_GPU_NUM,
										help='set multi gpu numbers')


args = parser.parse_args()
dir_prj = args.dir_prj
seed = args.seed
pattern_nums = args.pattern_nums
thresh = args.thresh
num_process = args.num_process
model_name = args.model_name
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
LOAD_PARAMS = args.load_params
SET_MEMORY_GROWTH = args.set_memory_growth
SET_MEMORY_LIMIT = args.set_memory_limit
SET_MULTI_GPU_NUM = args.set_multi_gpu_num


print('dir_prj:', args.dir_prj)
print('pattern_nums', args.pattern_nums)
print(type(args.pattern_nums[0]))