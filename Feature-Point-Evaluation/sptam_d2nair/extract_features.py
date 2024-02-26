import argparse
import numpy as np
import imageio
import torch
from tqdm import tqdm
import scipy
import scipy.io
import scipy.misc
from model_test import D2Net
#from lib.utils import preprocess_image
from pyramid import process_multiscale
from PIL import Image
def preprocess_image(image, preprocessing=None):
    image = image.astype(np.float32)
    image = np.transpose(image, [2, 0, 1])
    if preprocessing is None:
        pass
    elif preprocessing == 'caffe':
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
    else:
        raise ValueError('Unknown preprocessing parameter.')
    return image

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--image_list_file', type=str, required=True,
    help='path to a file containing a list of images to process'
)

parser.add_argument(
    '--preprocessing', type=str, default='caffe',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default='models/d2_tf.pth',
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=1600,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=2800,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_extension', type=str, default='.d2-net',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

args = parser.parse_args()

print(args)

# Creating CNN model
model = D2Net(
    model_file=args.model_file,
    use_relu=args.use_relu,
    use_cuda=use_cuda
)

# Process the file
with open(args.image_list_file, 'r') as f:
    lines = f.readlines()
for line in tqdm(lines, total=len(lines)):
    path = line.strip()

    image = imageio.v2.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    from PIL import Image

    # ...
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        scale = args.max_edge / max(resized_image.shape)
        resized_image = Image.fromarray(resized_image.astype('uint8'))
        resized_image = resized_image.resize((int(resized_image.size[0] * scale), int(resized_image.size[1] * scale)))
        resized_image = np.array(resized_image)

    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        scale = args.max_sum_edges / sum(resized_image.shape[: 2])
        resized_image = Image.fromarray(resized_image.astype('uint8'))
        resized_image = resized_image.resize((int(resized_image.size[0] * scale), int(resized_image.size[1] * scale)))
        resized_image = np.array(resized_image)

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    # ...
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    if args.output_type == 'npz':
        with open(path + args.output_extension, 'wb') as output_file:
            np.savez(
                output_file,
                keypoints=keypoints,
                scores=scores,
                descriptors=descriptors
            )
    elif args.output_type == 'mat':
        with open(path + args.output_extension, 'wb') as output_file:
            scipy.io.savemat(
                output_file,
                {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                }
            )
    else:
        raise ValueError('Unknown output type.')
