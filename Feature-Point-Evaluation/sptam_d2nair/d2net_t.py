import imageio
from d2ney_ce import extract_keypoints  # 路径可能需要根据你的文件系统调整

# 读取图像文件并转换为NumPy数组
image_data = imageio.v2.imread('/home/xcy/Downloads/001.jpg')

# 使用图像数据提取关键点
keypoints, scores, descriptors = extract_keypoints(image_data)

# 打印或其他操作...
print(f"Keypoints shape: {keypoints.shape}")
print(keypoints)