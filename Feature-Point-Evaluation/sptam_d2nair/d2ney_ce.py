import numpy as np
import torch
from pyramid import process_multiscale
from PIL import Image
from model_test import D2Net

def divide_into_regions(keypoints, scores, image_shape, num_regions=(5, 5), points_per_region=40):
    max_x = image_shape[1]  # 图片的宽度
    max_y = image_shape[0]  # 图片的高度
    region_size_x = max_x / num_regions[0]
    region_size_y = max_y / num_regions[1]
    selected_indices = []
    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            indices = ((i * region_size_x <= keypoints[:, 0]) & (keypoints[:, 0] < (i + 1) * region_size_x) &
                       (j * region_size_y <= keypoints[:, 1]) & (keypoints[:, 1] < (j + 1) * region_size_y))
            indices = np.where(indices)[0]
            if len(indices) <= points_per_region:
                selected_indices.extend(indices)
            else:
                indices = sorted(indices, key=lambda x: scores[x], reverse=True)
                selected_indices.extend(indices[:points_per_region])
    return np.array(selected_indices)

def extract_keypoints(image_data, model_file='models/d2_ots.pth', use_relu=True, use_cuda=True, scales=[1]):
    # Preprocess function
    def preprocess_image(image, preprocessing='caffe'):
        image = image.astype(np.float32)
        image = np.transpose(image, [2, 0, 1])
        if preprocessing == 'caffe':
            # RGB -> BGR
            image = image[:: -1, :, :]
            # Zero-center by mean pixel
            mean = np.array([103.939, 116.779, 123.68])
            image = image - mean.reshape([3, 1, 1])
        elif preprocessing == 'torch':
            image /= 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean.reshape([3, 1, 1])) / std.reshape([3, 1, 1])
        return image

    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = D2Net(
        model_file=model_file,
        use_relu=use_relu,
        use_cuda=use_cuda
    )

    image = image_data

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    resized_image = image

    if max(resized_image.shape) > 1600:
        scale = 1600 / max(resized_image.shape)
        resized_image = Image.fromarray(resized_image.astype('uint8'))
        resized_image = resized_image.resize((int(resized_image.size[0] * scale), int(resized_image.size[1] * scale)))
        resized_image = np.array(resized_image)

    if sum(resized_image.shape[: 2]) > 2800:
        scale = 2800 / sum(resized_image.shape[: 2])
        resized_image = Image.fromarray(resized_image.astype('uint8'))
        resized_image = resized_image.resize((int(resized_image.size[0] * scale), int(resized_image.size[1] * scale)))
        resized_image = np.array(resized_image)

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image, preprocessing='caffe')

    with torch.no_grad():
        keypoints, scores, descriptors = process_multiscale(
            torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=device),
            model,
            scales=scales
        )

    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    keypoints = keypoints[:, [1, 0, 2]]

    indices = divide_into_regions(keypoints, scores, image.shape, num_regions=(5, 5), points_per_region=40)
    keypoints = keypoints[indices]
    scores = scores[indices]
    descriptors = descriptors[indices]

    return keypoints, scores, descriptors
