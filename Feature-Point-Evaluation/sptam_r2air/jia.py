# 定义文件路径
path_v202 = "/home/xcy/Desktop/evo/10.txt"
path_tum = "/home/xcy/Desktop/evo/gftt-euroc/10g.tum"

# 读取v202.txt的内容
with open(path_v202, 'r') as f:
    timestamps = f.readlines()

# 对每个时间戳进行转换
converted_timestamps = ["{:.18e}".format(float(ts.strip()) / 1e9) for ts in timestamps]

# 读取v202.tum的内容
with open(path_tum, 'r') as f:
    tum_data = f.readlines()

# 检查两个文件的行数是否匹配
if len(converted_timestamps) != len(tum_data) :
    raise ValueError("The number of lines in the two files do not match!")

# 合并转换后的时间戳和TUM数据
combined_data = []
for ts, tum in zip(converted_timestamps, tum_data):
    combined_data.append(ts + " " + tum)

# 保存合并后的数据到v202.tum文件
with open(path_tum, 'w') as f:
    for line in combined_data:
        f.write(line)