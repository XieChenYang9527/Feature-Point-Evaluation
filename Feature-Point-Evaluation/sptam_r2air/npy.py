import numpy as np

# 定义文件路径
file_path = "/home/xcy/Desktop/evo/orb-euroc/result/10ape/distances.npy"

# 加载.npy文件
data = np.load(file_path)

# 打印数据
print(data)