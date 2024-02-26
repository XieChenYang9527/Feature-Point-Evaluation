import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 图片文件的路径
image_path = "/home/xcy/Downloads/001.jpg"

# .npy 文件的路径
npy_path = "/home/xcy/anzhuang/wendang/D2NET/001/keypoints.npy"

# 读取图片
image = mpimg.imread(image_path)

# 读取 .npy 文件
keypoints = np.load(npy_path)

# 打印关键点的形状
print(f"Shape of keypoints: {keypoints.shape}")
print(keypoints)
# 在图片上绘制关键点
plt.imshow(image)
plt.scatter(keypoints[:, 0], keypoints[:, 1], s=2, c='r')
plt.show()