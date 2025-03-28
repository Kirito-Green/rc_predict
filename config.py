# config file

# dir project
# windows
dir_prj = "D:/learn_more_from_life/computer/EDA/work/prj/rc_predict/"
# ubuntu

# server
dir_prj = "/home/prj/rc_predict/"

# params
seed = 42
lr = 1e-3
pattern_nums = [1]
tolerant_rate = 0.05 # 允许误差

# 向量化
reserve_num = 2000
thresh = [400, 2500]
n_components = 1000

# 图转化
model_name = "gcn"
# model_name = 'graph_sage'
# model_name = 'gat'

# switch
# 导入模型
LOAD_PARAMS = False

# 显存限制
SET_MEMORY_GROWTH = True
SET_MEMORY_LIMIT = False

# 多卡训练
SET_MULTI_GPU = False

# 向量化
DISABLE_DDR = False
DISABLE_STD = True