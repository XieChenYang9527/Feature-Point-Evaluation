from PIL import Image
import torch
import numpy as np
from demo_superpoint import SuperPointFrontend
import cv2
# 请在这里设置SuperPointFrontend的初始化参数
weights_path = "/home/xcy/Downloads/superpoint/superpoint_v1.pth"
cuda = torch.cuda.is_available()


def divide_into_regions(corners, num_regions=(5, 5), points_per_region=40):
    xys = corners[:2, :].T
    scores = corners[2, :]

    max_x = xys[:, 0].max()
    max_y = xys[:, 1].max()
    region_size_x = max_x / num_regions[0]
    region_size_y = max_y / num_regions[1]

    selected_indices = []

    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            indices = ((i * region_size_x <= xys[:, 0]) & (xys[:, 0] < (i + 1) * region_size_x) &
                       (j * region_size_y <= xys[:, 1]) & (xys[:, 1] < (j + 1) * region_size_y))
            indices = np.where(indices)[0]

            if len(indices) <= points_per_region:
                selected_indices.extend(indices)
            else:
                indices = sorted(indices, key=lambda x: scores[x], reverse=True)
                selected_indices.extend(indices[:points_per_region])

    return corners[:, selected_indices]
def process_image_sp(image_data):
    """
    处理单个图像，提取关键点和描述符。
    :param image_data: NumPy数组，单个灰度图像。
    :return: 关键点坐标, 描述符
    """
    if len(image_data.shape) == 3:  # 检查是否是彩色图像
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    # 初始化SuperPoint
    spfe = SuperPointFrontend(weights_path=weights_path,
                              nms_dist=4,
                              conf_thresh=0.010,
                              nn_thresh=0.7,
                              cuda=cuda)

    # 预处理图像
    h, w = image_data.shape
    image = image_data.astype(np.float32) / 255.0

    # 使用SuperPoint前端处理图像
    corners, descriptors, heatmap = spfe.run(image)

    # 调用divide_into_regions函数筛选关键点
    selected_corners = divide_into_regions(corners)

    # 将结果转换为NumPy数组并返回
    xys = selected_corners[:2, :].T
    desc = descriptors[:, selected_corners[2].astype(int)].T

    return xys, desc

