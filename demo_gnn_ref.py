# %% [markdown]
# # GNN模型预测版图寄生电容

# %% [markdown]
# ## Debug 设置

# %%
import faulthandler


faulthandler.enable()

# %% [markdown]
# ## 参数解析

# %%
import sys
import argparse
from config import *


# sys.argv = ['run.py']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dir_prj', type=str, default=dir_prj,
                    help='project directory')
parser.add_argument('--seed', type=int, default=seed,
                    help='random seed')
parser.add_argument('--pattern_nums', type=int, nargs='+', default=pattern_nums,
                    help='pattern nums')
parser.add_argument('--nodes_range', type=int, nargs='+', default=nodes_range,
                    help='the range of number of nodes')
parser.add_argument('--num_process', type=int, default=num_process,
                    help='multiprocessing number')
parser.add_argument('--ndm', type=int, default=ndm,
                    help='number of neighbor distance maximum')
parser.add_argument('--k', type=int, default=K,
                    help='number of neighbors')
parser.add_argument('--model_name', type=str, default=model_name,
                    help='model name [gcn, graph_sage, gat]')
parser.add_argument('--lr', type=float, default=lr,
                    help='adam learning rate')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='number of epochs')
parser.add_argument('--load_params', type=str2bool, default=LOAD_PARAMS,
                    help='load parameters')
parser.add_argument('--set_memory_growth', type=str2bool, default=SET_MEMORY_GROWTH,
                    help='set memory growth')
parser.add_argument('--set_memory_limit', type=int, default=SET_MEMORY_LIMIT,
                    help='memory limit, -1 for no limit')
parser.add_argument('--set_multi_gpu_num', type=int, default=SET_MULTI_GPU_NUM,
                    help='set multi gpu numbers')
parser.add_argument('-nt', '--no_train', action='store_true',
                    help='no train')


args = parser.parse_args()
dir_prj = args.dir_prj
seed = args.seed
pattern_nums = args.pattern_nums
nodes_range = args.nodes_range
num_process = args.num_process
ndm = args.ndm
K = args.k
model_name = args.model_name
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
LOAD_PARAMS = args.load_params
SET_MEMORY_GROWTH = args.set_memory_growth
SET_MEMORY_LIMIT = args.set_memory_limit
SET_MULTI_GPU_NUM = args.set_multi_gpu_num
SET_MULTI_GPU_NUM = min(SET_MULTI_GPU_NUM, 4)
NO_TRAIN = args.no_train
TRAIN = not NO_TRAIN


# %% [markdown]
# ## 路径定义

# %%
import os


# log save path
dir_logs = os.path.join(os.getcwd(), '../logs')
if not os.path.exists(dir_logs):
    os.mkdir(dir_logs)

# params save path
dir_params = os.path.join(os.getcwd(), '../params')
if not os.path.exists(dir_params):
    os.mkdir(dir_params)

# results save path
dir_results = os.path.join(os.getcwd(), '../results')
if not os.path.exists(dir_results):
    os.mkdir(dir_results)

# tensorboard save path
dir_tensorboard = os.path.join(os.getcwd(), '../tensorboard')
if not os.path.exists(dir_tensorboard):
    os.mkdir(dir_tensorboard)

# %% [markdown]
# ## log 设置

# %%
import logging


save_name = f'{model_name}_lr{lr}_batchsize{batch_size}_epochs{epochs}_k{K}_seed{seed}'
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(dir_logs, (f'{save_name}.log')), mode='w', encoding='utf-8')

# 设置日志格式
logging.basicConfig(
    format="%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[console_handler, file_handler],
    level=logging.INFO
)

# loging args
logging.info('------------------------args start----------------------------')
for k, v in vars(args).items():
    logging.info(f'{k} = {v}')
logging.info('-------------------------args end-----------------------------')

# %% [markdown]
# ## tensorflow框架设置

# %% [markdown]
# ### 多卡设置

# %%
import os


gpus = ''
for i in range(SET_MULTI_GPU_NUM):
    if i == 0:
        gpus += str(i)
    else:
        gpus += ',' + str(i)

os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# %% [markdown]
# ### log level 设置

# %%
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# %% [markdown]
# ### 显存限制

# %%
import tensorflow as tf


# 方法一 set memory growth
if SET_MEMORY_GROWTH:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            logging.info(gpus[:SET_MULTI_GPU_NUM])
            for gpu in gpus[:SET_MULTI_GPU_NUM]:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.error(e)

# 方法二 set memory limit
if SET_MEMORY_LIMIT > 0:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            logging.info(gpus[:SET_MULTI_GPU_NUM])
            for gpu in gpus[:SET_MULTI_GPU_NUM]:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=SET_MEMORY_LIMIT)])
            logging.info(f'set memory limit to {SET_MEMORY_LIMIT}MB')
        except RuntimeError as e:
            logging.error(e)

# %% [markdown]
# ## 库导入

# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# 自定义模块导入
from data.dataset import MyDataset
from models.gcn import GCN
from models.graph_sage import GraphSage
from models.gat import GAT
from models.gin import GIN

# fix seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# %% [markdown]
# ## 数据导入与数据清洗

# %%
# load data
logging.info('---------------------loading total data-----------------------')
dataset_total = MyDataset(dir_prj=dir_prj,
                          ndm=ndm,
                          k=K,
                          pattern_nums=pattern_nums,
                          x_name='x_total.npy',
                          y_name='y_total.npy',
                          g_name='total',
                          num_process=num_process,
                          update=False)
logging.info('---------------------load total data done---------------------')

logging.info('---------------------loading couple data----------------------')
dataset_couple = MyDataset(dir_prj=dir_prj,
                           ndm=ndm,
                           k=K, 
                           pattern_nums=pattern_nums,
                           x_name='x_couple.npy',
                           y_name='y_couple.npy',
                           g_name='couple',
                           num_process=num_process,
                           update=False)
logging.info('---------------------load couple data done--------------------')

# clean data
logging.info('------------------------before cleaning-----------------------')
logging.info(f'dataset total: {dataset_total}')
logging.info(f'dataset couple: {dataset_couple}')
logging.info('------------------------after cleaning------------------------')
dataset_total.filter(lambda g: nodes_range[0] <= g.n_nodes <= nodes_range[1])
dataset_couple.filter(lambda g: nodes_range[0] <= g.n_nodes <= nodes_range[1])
logging.info(f'dataset total: {dataset_total}')
logging.info(f'dataset couple: {dataset_couple}')
print(f'total x shape: {dataset_total[0].x.shape} sample: {dataset_total[0].x[:5]}')
print(f'couple x shape: {dataset_couple[0].x.shape} sample: {dataset_couple[0].x[:5]}')

# %% [markdown]
# ## 数据分割

# %%
# split data # 6:2:2
np.random.shuffle(dataset_total)
np.random.shuffle(dataset_couple)
n = len(dataset_total)
train_data_total, valid_data_total, test_data_total = dataset_total[0:int(n*0.6)], \
    dataset_total[int(n*0.6):int(n*0.8)], \
    dataset_total[int(n*0.8):]
n = len(dataset_couple)
train_data_couple, valid_data_couple, test_data_couple = dataset_couple[0:int(n * 0.6)], \
    dataset_couple[int(n * 0.6):int(n * 0.8)], \
    dataset_couple[int(n * 0.8):]

# %% [markdown]
# ## 模型构建

# %%
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from spektral.data import BatchLoader
from utils.model import huber_loss, mse_msre_loss, measure_ratio_bad, measure_ratio_good


# loss and weighted metrics
loss_func = mse_msre_loss
weighted_metrics = [measure_ratio_good]

# set multi gpu
if SET_MULTI_GPU_NUM > 1:
    devices = [f'/gpu:{i}' for i in range(SET_MULTI_GPU_NUM)]
    mirror_strategy = tf.distribute.MirroredStrategy(devices=devices)
    with mirror_strategy.scope():
        logging.info(
            '---------------------set multi gpu done---------------------')
        logging.info(f'num devices: {mirror_strategy.num_replicas_in_sync}')
        logging.info(f'devices: {devices}')
        logging.info(
            '------------------------------------------------------------')
        # GCN model
        if model_name == 'gcn':
            model_total = GCN(training=True)
            model_couple = GCN(training=True)
            model_best_total = GCN(training=True)
            model_best_couple = GCN(training=True)
            logging.info('buld model gcn done')
        # GraphSAGE model
        elif model_name == 'graph_sage':
            model_total = GraphSage(training=True)
            model_couple = GraphSage(training=True)
            model_best_total = GraphSage(training=True)
            model_best_couple = GraphSage(training=True)
            logging.info('buld model graph_sage done')
        # GAT model
        elif model_name == 'gat':
            model_total = GAT(training=True)
            model_couple = GAT(training=True)
            model_best_total = GAT(training=True)
            model_best_couple = GAT(training=True)
            logging.info('buld model gat done')
        # GIN model
        elif model_name == 'gin':
            model_total = GIN(training=True)
            model_couple = GIN(training=True)
            model_best_total = GIN(training=True)
            model_best_couple = GIN(training=True)
            logging.info('buld model gin done')
        else:
            logging.error('model_name error')
            raise ValueError('model_name error')
        model_total.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                            loss=loss_func,
                            weighted_metrics=weighted_metrics)
        model_couple.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                             loss=loss_func,
                             weighted_metrics=weighted_metrics)
        # best model
        model_best_total.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                                 loss=loss_func,
                                 weighted_metrics=weighted_metrics)
        model_best_couple.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                                  loss=loss_func,
                                  weighted_metrics=weighted_metrics)
else:
    # GCN model
    if model_name == 'gcn':
        model_total = GCN(training=True)
        model_couple = GCN(training=True)
        model_best_total = GCN(training=True)
        model_best_couple = GCN(training=True)
        logging.info('buld model gcn done')
    # GraphSAGE model
    elif model_name == 'graph_sage':
        model_total = GraphSage(training=True)
        model_couple = GraphSage(training=True)
        model_best_total = GraphSage(training=True)
        model_best_couple = GraphSage(training=True)
        logging.info('buld model graph_sage done')
    # GAT model
    elif model_name == 'gat':
        model_total = GAT(training=True)
        model_couple = GAT(training=True)
        model_best_total = GAT(training=True)
        model_best_couple = GAT(training=True)
        logging.info('buld model gat done')
    # GIN model
    elif model_name == 'gin':
        model_total = GIN(training=True)
        model_couple = GIN(training=True)
        model_best_total = GIN(training=True)
        model_best_couple = GIN(training=True)
        logging.info('buld model gin done')
    else:
        logging.error('model_name error')
        raise ValueError('model_name error')

    model_total.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                        loss=loss_func,
                        weighted_metrics=weighted_metrics)
    model_couple.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                         loss=loss_func,
                         weighted_metrics=weighted_metrics)
    # best model
    model_best_total.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                             loss=loss_func,
                             weighted_metrics=weighted_metrics)
    model_best_couple.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                              loss=loss_func,
                              weighted_metrics=weighted_metrics)

# %% [markdown]
# ## 模型导入

# %%
model_total_load_path = os.path.join(dir_params, f'total_{save_name}.h5')
model_couple_load_path = os.path.join(dir_params, f'couple_{save_name}.h5')

# load model
if os.path.exists(model_total_load_path) and LOAD_PARAMS:
    loader = BatchLoader(
        train_data_total[:batch_size], batch_size=batch_size, shuffle=True)
    model_total.fit(loader.load(),
                    steps_per_epoch=1,
                    epochs=1)
    try:
        model_total.load_weights(model_total_load_path)
        logging.info('load model total done')
    except:
        logging.error('load model total failed')
if os.path.exists(model_couple_load_path) and LOAD_PARAMS:
    loader = BatchLoader(
        train_data_couple[:batch_size], batch_size=batch_size, shuffle=True)
    model_couple.fit(loader.load(),
                     steps_per_epoch=1,
                     epochs=1)
    try:
        model_couple.load_weights(model_couple_load_path)
        logging.info('load model couple done')
    except:
        logging.error('load model couple failed')

# %% [markdown]
# ## 模型预测

# %% [markdown]
# ### 集总电容模型

# %% [markdown]
# #### 模型训练

# %% [markdown]
# ##### 手动记录

# %%
if not USE_TENSORBOARD and TRAIN:
    from spektral.data import BatchLoader
    from utils.model import sync

    loader_train = BatchLoader(
        train_data_total, batch_size=batch_size, shuffle=True)
    loader_valid = BatchLoader(
        valid_data_total, batch_size=batch_size, shuffle=True)

    best_epoch = 0
    best_val_score = 1e10

    # train model
    loss_all = []
    val_loss_all = []
    start = time.time()
    logging.info('----------------model total train start------------------')
    for epoch in range(epochs):
        history = model_total.fit(loader_train.load(),
                                  validation_data=loader_valid.load(),
                                  steps_per_epoch=loader_train.steps_per_epoch,
                                  validation_steps=loader_valid.steps_per_epoch,
                                  epochs=1,
                                  shuffle=False)
        loss_all.append(history.history['loss'][0])
        val_loss_all.append(history.history['val_loss'][0])

        # log
        logging.info(
            f'total model:{model_name} epoch:{epoch} loss:{loss_all[-1]} val_loss:{val_loss_all[-1]}')

        # save model every model_save_freq epochs
        if (epoch + 1) % model_save_freq == 0:
            model_total_save_path = os.path.join(
                dir_params, f'total_{save_name}.h5')
            model_total.save_weights(model_total_save_path)

        # save best model on validation set
        val_loss, val_score = model_total.evaluate(
            loader_valid.load(), steps=loader_valid.steps_per_epoch)
        if val_score < best_val_score:
            best_epoch = epoch
            best_val_score = val_score
            sync(model_total, model_best_total)
    end = time.time()
    logging.info(
        f'model total train done epoch: {epochs}, time: {end - start}s')
    logging.info('-----------------model total train end-------------------')

    # save best model
    model_total_save_path = os.path.join(
        dir_params, f'total_best_{save_name}.h5')
    model_best_total.save_weights(model_total_save_path)
    model_total.summary()

    # plot
    plt.figure()
    plt.plot(loss_all, label='train loss')
    plt.plot(val_loss_all, label='valid loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # save picture
    plt_save_path = os.path.join(
        dir_results, f"total_{save_name}.jpg")
    plt.savefig(plt_save_path)

# %% [markdown]
# ##### Tensorboard使用

# %%
if USE_TENSORBOARD and TRAIN:
    from spektral.data import BatchLoader

    loader_train = BatchLoader(
        train_data_total, batch_size=batch_size, shuffle=True)
    loader_valid = BatchLoader(
        valid_data_total, batch_size=batch_size, shuffle=True)

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(dir_tensorboard, f'total_{save_name}'))
    log_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logging.info(
            f'total model:{model_name} epoch:{epoch} loss:{logs["loss"]} val_loss:{logs["val_loss"]}')
    )
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            dir_params, f'total_best_{save_name}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1
    )

    # train model
    start = time.time()
    logging.info('----------------model total train start------------------')
    history = model_total.fit(loader_train.load(),
                              validation_data=loader_valid.load(),
                              steps_per_epoch=loader_train.steps_per_epoch,
                              validation_steps=loader_valid.steps_per_epoch,
                              epochs=epochs,
                              shuffle=False,
                              callbacks=[tb_callback, log_callback, save_callback])

    end = time.time()
    logging.info(
        f'model total train done epoch: {epochs}, time: {end - start}s')
    logging.info('-----------------model total train end-------------------')

    # plot
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # save picture
    plt_save_path = os.path.join(
        dir_results, f"total_{save_name}.jpg")
    plt.savefig(plt_save_path)

# %% [markdown]
# #### 模型测试

# %%
from utils.model import gnn_analysis, gnn_plot, test_runtime


# best model load
# build model
loader = BatchLoader(
    train_data_total[:batch_size], batch_size=batch_size, shuffle=True)
model_total = GCN(training=False) if model_name == 'gcn' else GraphSage(
    training=False) if model_name == 'graph_sage' else GAT(
    training=False) if model_name == 'gat' else GIN(
    training=False) if model_name == 'gin' else None
model_total.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                    loss=loss_func,
                    weighted_metrics=weighted_metrics)
model_total.fit(loader.load(),
                steps_per_epoch=1,
                epochs=1)
model_total_load_path = os.path.join(
    dir_params, f'total_best_{save_name}.h5')
if os.path.exists(model_total_load_path):
    model_total.load_weights(model_total_load_path)
else:
    raise ValueError('model total load path does not exist')
model_total.summary()

yt_train = np.array([data.y for data in train_data_total]).reshape(-1, 1)
yt_valid = np.array([data.y for data in valid_data_total]).reshape(-1, 1)
yt_test = np.array([data.y for data in test_data_total]).reshape(-1, 1)

# get table
dict_total = gnn_analysis(model_total, batch_size,
                          train_data_total, yt_train,
                          valid_data_total, yt_valid,
                          test_data_total, yt_test, name=model_name)
data_total = pd.Series(dict_total).to_frame(name='total').T

# scatter plot
gnn_plot(model_total, batch_size,
         train_data_total, yt_train,
         valid_data_total, yt_valid,
         test_data_total, yt_test, dir=dir_results,
         name=f'total_{save_name}')

# runtime test
total_avg_time = test_runtime(model_total, batch_size,
                              test_data_total, yt_test)

# save results
data_total.to_csv(os.path.join(
    dir_results, f"total_{save_name}.csv"), index=False)

# %% [markdown]
# ### 耦合电容模型

# %% [markdown]
# #### 模型训练

# %% [markdown]
# ##### 手动记录

# %%
if (not USE_TENSORBOARD) and TRAIN:
    from spektral.data import BatchLoader
    from utils.model import sync

    loader_train = BatchLoader(
        train_data_couple, batch_size=batch_size, shuffle=True)
    loader_valid = BatchLoader(
        valid_data_couple, batch_size=batch_size, shuffle=True)

    best_epoch = 0
    best_val_score = 1e10

    # train model
    loss_all = []
    val_loss_all = []
    start = time.time()
    logging.info('----------------model couple train start------------------')
    for epoch in range(epochs):
        history = model_couple.fit(loader_train.load(),
                                   validation_data=loader_valid.load(),
                                   steps_per_epoch=loader_train.steps_per_epoch,
                                   validation_steps=loader_valid.steps_per_epoch,
                                   epochs=1,
                                   shuffle=False)
        loss_all.append(history.history['loss'][0])
        val_loss_all.append(history.history['val_loss'][0])

        # log
        logging.info(
            f'couple model:{model_name} epoch:{epoch} loss:{loss_all[-1]} val_loss:{val_loss_all[-1]}')

        # save model every model_save_freq epochs
        if (epoch + 1) % model_save_freq == 0:
            model_couple_save_path = os.path.join(
                dir_params, f'couple_{save_name}.h5')
            model_couple.save_weights(model_couple_save_path)

        # save best model on validation set
        val_loss, val_score = model_couple.evaluate(
            loader_valid.load(), steps=loader_valid.steps_per_epoch)
        if val_score < best_val_score:
            best_epoch = epoch
            best_val_score = val_score
            sync(model_couple, model_best_couple)
    end = time.time()
    logging.info(
        f'model couple train done epoch: {epochs}, time: {end - start}s')
    logging.info('-----------------model couple train end-------------------')

    # save best model
    model_couple_save_path = os.path.join(
        dir_params, f'couple_best_{save_name}.h5')
    model_best_couple.save_weights(model_couple_save_path)
    model_couple.summary()

    # plot
    plt.figure()
    plt.plot(loss_all, label='train loss')
    plt.plot(val_loss_all, label='valid loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # save picture
    plt_save_path = os.path.join(
        dir_results, f"couple_{save_name}.jpg")
    plt.savefig(plt_save_path)

# %% [markdown]
# ##### Tensorboard 使用

# %%
if USE_TENSORBOARD and TRAIN:
    from spektral.data import BatchLoader

    loader_train = BatchLoader(
        train_data_couple, batch_size=batch_size, shuffle=True)
    loader_valid = BatchLoader(
        valid_data_couple, batch_size=batch_size, shuffle=True)

    tf_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(dir_tensorboard, f'couple_{save_name}'))
    log_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: logging.info(
            f'couple model:{model_name} epoch:{epoch} loss:{logs["loss"]} val_loss:{logs["val_loss"]}')
    )
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            dir_params, f'couple_best_{save_name}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1
    )

    start = time.time()
    logging.info('----------------model couple train start------------------')
    history = model_couple.fit(loader_train.load(),
                               validation_data=loader_valid.load(),
                               steps_per_epoch=loader_train.steps_per_epoch,
                               validation_steps=loader_valid.steps_per_epoch,
                               epochs=epochs,
                               shuffle=False,
                               callbacks=[tf_callback, log_callback, save_callback])

    end = time.time()
    logging.info(
        f'model couple train done epoch: {epochs}, time: {end - start}s')
    logging.info('-----------------model couple train end-------------------')

    # plot
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='valid loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # save picture
    plt_save_path = os.path.join(
        dir_results, f"couple_{save_name}.jpg")
    plt.savefig(plt_save_path)

# %% [markdown]
# #### 模型测试

# %%
from utils.model import gnn_analysis


# load best model
# build model
loader = BatchLoader(
    train_data_couple[:batch_size], batch_size=batch_size, shuffle=True)
model_couple = GCN(training=False) if model_name == 'gcn' else GraphSage(
    training=False) if model_name == 'graph_sage' else GAT(
    training=False) if model_name == 'gat' else GIN(
    training=False) if model_name == 'gin' else None
model_couple.compile(optimizer=Adam(learning_rate=lr, epsilon=epsilon),
                     loss=loss_func,
                     weighted_metrics=weighted_metrics)
model_couple.fit(loader.load(),
                 steps_per_epoch=1,
                 epochs=1)
model_couple_save_path = os.path.join(
    dir_params, f'couple_best_{save_name}.h5')
if os.path.exists(model_couple_save_path):
    model_couple.load_weights(model_couple_save_path)
else:
    raise FileNotFoundError(
        f"model_couple_save_path: {model_couple_save_path} not found!")

yc_train = np.array([data.y for data in train_data_couple]).reshape(-1, 1)
yc_valid = np.array([data.y for data in valid_data_couple]).reshape(-1, 1)
yc_test = np.array([data.y for data in test_data_couple]).reshape(-1, 1)

# get table
dict_couple = gnn_analysis(model_couple, batch_size,
                           train_data_couple, yc_train,
                           valid_data_couple, yc_valid,
                           test_data_couple, yc_test, name=model_name)
data_couple = pd.Series(dict_couple).to_frame(name='couple').T

# scatter plot
gnn_plot(model_couple, batch_size,
         train_data_couple, yc_train,
         valid_data_couple, yc_valid,
         test_data_couple, yc_test, dir=dir_results,
         name=f'couple_{save_name}')

# runtime test
couple_avg_time = test_runtime(model_couple, batch_size,
                                test_data_couple, yc_test)
# all time
avg_time = total_avg_time + couple_avg_time
logging.info(f'all avg time: {avg_time}')

# save results
data_couple.to_csv(os.path.join(
    dir_results, f"couple_{save_name}.csv"), index=False)


