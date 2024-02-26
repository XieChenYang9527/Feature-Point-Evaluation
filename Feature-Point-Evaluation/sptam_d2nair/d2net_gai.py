import numpy as np
import imageio
import torch
from pyramid import process_multiscale
from PIL import Image
from model_test import D2Net

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

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Creating CNN model
model = D2Net(
    model_file='models/d2_ots.pth',
    use_relu=True,
    use_cuda=use_cuda
)

# Image path
image_path = '/home/xcy/Downloads/001.jpg'

# Reading and preprocessing image
image = imageio.v2.imread(image_path)

if len(image.shape) == 2:
    image = image[:, :, np.newaxis]
    image = np.repeat(image, 3, -1)

# Resizing
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

# Preprocess image
input_image = preprocess_image(resized_image, preprocessing='caffe')

# Extract features
with torch.no_grad():
    keypoints, scores, descriptors = process_multiscale(
        torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32), device=device),
        model,
        scales=[1]
    )

# Input image coordinates
keypoints[:, 0] *= fact_i
keypoints[:, 1] *= fact_j
# i, j -> u, v
keypoints = keypoints[:, [1, 0, 2]]

# Print keypoints
print(f"Keypoints shape: {keypoints.shape}")
print(keypoints)