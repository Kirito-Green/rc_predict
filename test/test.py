# %% [markdown]
# # GNN learning

# %% [markdown]
# ## 基本模型学习

# %% [markdown]
# ### 导入数据

# %%
from spektral.datasets import TUDataset

dataset = TUDataset('PROTEINS')

# %% [markdown]
# ### 数据处理

# %%
import numpy as np


dataset.map(lambda g: g.n_nodes < 500)
np.random.shuffle(dataset)
train_data, valid_data, test_data = dataset[0:int(len(dataset)*0.6)], \
                                    dataset[int(len(dataset)*0.6):int(len(dataset)*0.8)], \
                                    dataset[int(len(dataset)*0.8):]


# %% [markdown]
# ### GraphSAGE(图采样与聚合)

# %%
import tensorflow as tf
from spektral.layers import GraphSageConv, GlobalSumPool
from keras.api._v2.keras.models import Model
from keras.api._v2.keras.layers import Input, Dropout, Dense


class MyGraphSage(Model):
    
    def __init__(self, n_labels):
        super(MyGraphSage, self).__init__()
        self.conv1 = GraphSageConv(16, aggregate='mean', activation="relu")
        self.pool1 = GlobalSumPool()
        self.dense = Dense(n_labels, activation="linear")

    def call(self, inputs):  # 定义一个名为call的方法，用于处理输入数据
        x, a = inputs
        x = tf.cast(x, tf.float32)  # 将输入数据转换为float32类型
        a = tf.cast(a, tf.float32)
        a = tf.sparse.from_dense(a)
        x = self.conv1([x, a])
        x = self.pool1(x)
        x = self.dense(x)

        return x  # 返回最终的输出结果
    
model = MyGraphSage(dataset.n_labels)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])

# %%
from spektral.data import BatchLoader, SingleLoader
import matplotlib.pyplot as plt


loader_train = BatchLoader(train_data, batch_size=32)
loader_valid = BatchLoader(valid_data, batch_size=32)
# loader_train = SingleLoader(train_data)
# loader_valid = SingleLoader(valid_data)
model.fit(loader_train.load(),
          validation_data=loader_valid.load(),
          steps_per_epoch=loader_train.steps_per_epoch,
          validation_steps=loader_valid.steps_per_epoch,
          epochs=10,
          shuffle=False)
model.summary()

# plot
history = model.history.history
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %%
from tensorflow.keras import backend as K

a = tf.constant([[1, 2, 3], [4, 5, 6]])
a = tf.sparse.from_dense(a)
a = tf.cast(a, tf.float32)
print(a)

print(K.ndim(a))

# %%
loader = BatchLoader(test_data, batch_size=32)
accuracy = model.evaluate(loader.load(), 
                          steps=loader.steps_per_epoch)[1]
print(f"Test accuracy: {accuracy}")

# %% [markdown]
# ### GAT(图注意力网络)

# %%


# %% [markdown]
# ## 应用

# %% [markdown]
# ### 节点分类

# %%


# %% [markdown]
# ### 节点回归

# %%


# %% [markdown]
# ### 边分类

# %%


# %% [markdown]
# ### 边回归

# %%


# %% [markdown]
# ### 图分类

# %%


# %% [markdown]
# ### 图回归

# %%



