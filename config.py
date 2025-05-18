# config file

# dir project
# The line `dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"` in the config file
# is setting the directory path for the project. This variable `dir_prj` is storing the path
# "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/" where the project files related to the
# configuration settings are located. This path can be used throughout the code to refer to this
# specific directory for reading or writing files related to the project.
dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"

# params
seed = 42
pattern_nums = [3, 4, 26]
measure_thresh = 1  # 测量阈值
tolerant_ratio_error = 0.10  # 允许预测精度误差
tolerant_zero_error = 1e-6  # 判断精度误差的阈值
num_process = 8  # 进程数
cnt_max = 1500  # 单类样本最大数量

# 版图处理
n_components = 100  # 降维
cap_thresh = 0.05 # 去除低于阈值的电容

# 图转化
model_name = "gcn" # gcn cheb tag gat gin
nodes_range = [50, 1500] # 节点数范围
vertical_space = 0.024  # um
max_length = 150 # polygon max length um
window_size = 150  # um
ndm = 25  # neighbor distance max um
K = 20 # neighbor number
warmup = 20 # 预热epoch数
lr_warmup = 1e-3 # 预热学习率
lr = 1e-3
epsilon = 1e-7
batch_size = 16
epochs = 20
model_save_freq = 1e5  # none used


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
SET_MEMORY_LIMIT = -1  # 显存限制，单位MB

# 多卡训练
SET_MULTI_GPU_NUM = 1
