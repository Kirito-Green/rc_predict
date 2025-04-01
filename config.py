# config file

# dir project
dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"

# params
seed = 42
pattern_nums = [4]
tolerant_ratio = 0.05 # 允许预测精度误差
tolerant_error = 1e-6 # 判断精度误差的阈值
num_process = 8 # 进程数

# 向量化
thresh = [400, 2500]
# thresh = [1800, 2000]
n_components = 100 # 降维

# 图转化
model_name = "gcn"
# model_name = 'graph_sage'
# model_name = 'gat'
lr = 1e-3
batch_size = 16
epochs = 3
model_save_freq = 100 
# switch
# 导入模型
LOAD_PARAMS = False

# 显存限制
SET_MEMORY_GROWTH = True
SET_MEMORY_LIMIT = -1 # 显存限制，单位MB

# 多卡训练
SET_MULTI_GPU_NUM = 1

# 向量化
DISABLE_DDR = False
DISABLE_STD = True
USE_DDR_PCA = True
USE_DDR_KPCA = False
USE_DDR_VAR = False
USE_DDR_AE = False