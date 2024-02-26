from R2D2 import extract_keypoints, extract_multiscale, load_network, NonMaxSuppression, norm_RGB
from PIL import Image
import numpy as np
import torch

# 新的 divide_into_regions 函数
def divide_into_regions(xys, scores, num_regions=(5, 5), points_per_region=40):
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
    return np.array(selected_indices)

# Set the parameters
model = "models/r2d2_WASF_N16.pt"
tag = 'r2d2'
top_k = 5000
scale_f = 2 ** 0.25
min_size = 256
max_size = 1024
min_scale = 0
max_scale = 1
reliability_thr = 0.7
repeatability_thr = 0.7

# Check if we have a GPU available
iscuda = torch.cuda.is_available()

# load the network
net = load_network(model)
if iscuda: net = net.cuda()


# create the non-maxima detector
detector = NonMaxSuppression(rel_thr=reliability_thr, rep_thr=repeatability_thr)

def process_image(image_data):
    # 将NumPy数组转换为PIL图像
    img = Image.fromarray(image_data).convert('RGB')
    W, H = img.size
    img = norm_RGB(img)[None]
    if iscuda: img = img.cuda()

    # extract keypoints/descriptors for a single image
    xys, desc, scores = extract_multiscale(net, img, detector,
                                           scale_f=scale_f,
                                           min_scale=min_scale,
                                           max_scale=max_scale,
                                           min_size=min_size,
                                           max_size=max_size,
                                           verbose=False)

    # xys, desc, scores 转换为 numpy 数组
    xys = xys.cpu().numpy()
    desc = desc.cpu().numpy()
    scores = scores.cpu().numpy()

    # Use only the top points from each region
    idxs = divide_into_regions(xys, scores, num_regions=(5, 5), points_per_region=80)
    xys = xys[idxs]
    desc = desc[idxs]
    scores = scores[idxs]

    return xys, desc