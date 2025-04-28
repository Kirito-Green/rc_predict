# config file

# dir project
dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"

# params
seed = 42
pattern_nums = [4]
tolerant_ratio_error = 0.10 # 允许预测精度误差
tolerant_zero_error = 1e-6 # 判断精度误差的阈值
num_process = 8 # 进程数
cnt_max = 1500 # 单类样本最大数量

# 向量化
thresh = [0, 3000]
# thresh = [1800, 2000]
n_components = 100 # 降维

# 图转化
model_name = "gat"
vertical_space = 0.024 # um
window_size = 150 # um
ndm = 25 # neighbor distance max um
lr = 1e-3
epsilon=1e-7
batch_size = 16
epochs = 100
model_save_freq = 1e5 # none used


# switch
# 向量化
DISABLE_DDR = True
DISABLE_NORM = True
USE_DDR_PCA = False
USE_DDR_KPCA = False
USE_DDR_VAR = False
USE_DDR_AE = False

# 图转化
LOAD_PARAMS = False
USE_TENSORBOARD = True

# 显存限制
SET_MEMORY_GROWTH = True
SET_MEMORY_LIMIT = -1 # 显存限制，单位MB

# 多卡训练
SET_MULTI_GPU_NUM = 1
