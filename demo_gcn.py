# %% [markdown]
# # GNN模型预测版图寄生电容

# %% [markdown]
# ## 基本库导入

# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.getcwd(), '..'))

# 自定义模块导入
from data.dataset import MyDataset
from models.gcn import GCN

# fix seed
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# params
pattern_num = 4

# %% [markdown]
# 

# %% [markdown]
# ## 数据导入与数据分割

# %%
# load data
dataset_total = MyDataset(pattern_num=pattern_num,
                          x_name='x_total.npy',
                          y_name='y_total.npy',
                          g_name='total',
                          update=False)
print('load total data done')
dataset_couple = MyDataset(pattern_num=pattern_num,
                           x_name='x_couple.npy',
                           y_name='y_couple.npy',
                           g_name='couple',
                           update=False)
print('load couple data done')
print('dataset total', dataset_total)
print('dataset couple', dataset_couple)

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
from spektral.data import BatchLoader


lr = 1e-3
model_total = GCN()
model_total.compile(optimizer=Adam(lr=lr),
		            loss='mean_squared_error',
                    weighted_metrics=['MeanSquaredError'])
model_couple = GCN()
model_couple.compile(optimizer=Adam(lr=lr),
		            loss='mean_squared_error',
                    weighted_metrics=['MeanSquaredError'])

model_total_save_path = os.path.abspath(os.path.join(os.getcwd(), '../../params/model_total.h5'))
model_couple_save_path = os.path.abspath(os.path.join(os.getcwd(), '../../params/model_couple.h5'))

# load model
if os.path.exists(model_total_save_path):
    loader = BatchLoader(train_data_total[:16], batch_size=16, shuffle=False)
    model_total.fit(loader.load(),
                    steps_per_epoch=1,
                    epochs=1)
    try:
        model_total.load_weights(model_total_save_path)
        print('load model total done')
        model_total.summary()
    except:
        print('load model total failed')
# if os.path.exists(model_couple_save_path):
    # model_couple.load_weights(model_couple_save_path)
    # print('load model couple done')


# %% [markdown]
# ## 显存限制

# %%
# 显存限制
SET_MEMORY_GROWTH = True
SET_MEMORY_LIMIT = False

# 方法一 set memory growth
if SET_MEMORY_GROWTH:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            print(gpus[0])
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

# 方法二 set memory limit
if SET_MEMORY_LIMIT:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            print(gpus[0])
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3900)])
        except RuntimeError as e:
            print(e)

# %% [markdown]
# ## 集总电容模型

# %% [markdown]
# ### 集总电容模型训练

# %%
from spektral.data import BatchLoader


loader_train = BatchLoader(train_data_total, batch_size=16, shuffle=False)
loader_valid = BatchLoader(valid_data_total, batch_size=16, shuffle=False)
model_total.fit(loader_train.load(),
                validation_data=loader_valid.load(),
                steps_per_epoch=loader_train.steps_per_epoch,
                validation_steps=loader_valid.steps_per_epoch,
                epochs=5,
                shuffle=False)
model_total.summary()

# plot 
history = model_total.history.history
plt.plot(history['loss'], label='train loss')
plt.plot(history['val_loss'], label='valid loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# save model
model_total.save(model_total_save_path)

# %% [markdown]
# ### 集总电容模型测试

# %%
from sklearn.metrics import r2_score


tolerant_rate = 0.05

def analysis_result(y, y_predict, title):
    relative_error = np.abs(y - y_predict) / y
    max_error = np.max(relative_error)
    mean_error = np.mean(relative_error)
    std_error = np.std(relative_error)
    num_good = np.sum(relative_error <= tolerant_rate) / len(y)
    num_bad = np.sum(relative_error > tolerant_rate) / len(y)
    print(title)
    print('max error:', max_error)
    print('mean error:', mean_error)
    print('std error:', std_error)
    print('good ratio:', num_good)
    print('bad ratio:', num_bad)
    print('')


loader = BatchLoader(test_data_total, batch_size=16)
mse = model_total.evaluate(loader.load(), 
                           steps=loader.steps_per_epoch)
y_predict = model_total.predict(loader.load(), 
                           steps=loader.steps_per_epoch).ravel()
y = np.array([x.y for x in test_data_total])
r2 = r2_score(y, y_predict)
print(f"Test evaluate mse: {mse}")
print(f"Test evaluate r2: {r2}")
analysis_result(y, y_predict, 'total')

# %% [markdown]
# ### 模型可视化

# %%


# %% [markdown]
# ## 耦合电容模型

# %% [markdown]
# ### 耦合电容模型训练

# %%
from spektral.data import BatchLoader


loader_train = BatchLoader(train_data_total, batch_size=32)
loader_valid = BatchLoader(valid_data_total, batch_size=32)
model_couple.fit(loader_train.load(),
                validation_data=loader_valid.load(),
                steps_per_epoch=loader_train.steps_per_epoch,
                validation_steps=loader_valid.steps_per_epoch,
                epochs=100,
                shuffle=False)
model_couple.summary()

# plot 
history = model_couple.history.history
plt.plot(history['loss'], label='train loss')
plt.plot(history['val_loss'], label='valid loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# save model
model_couple.save(model_couple_save_path)

# %% [markdown]
# ### 耦合电容模型测试

# %%
dataset_total[0].y

# %% [markdown]
# ### 模型可视化

# %%



