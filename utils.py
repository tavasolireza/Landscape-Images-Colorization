import os
from skimage.color import lab2rgb
import torch
import matplotlib.pyplot as plt
import numpy as np


def split_dataset():
    os.makedirs('/content/images/train/class/', exist_ok=True)  # 40,000 images
    os.makedirs('/content/images/val/class/', exist_ok=True)  # 1,000 images
    for i, file in enumerate(os.listdir('/content/landscape')):
        if i < 1000:  # first 1000 will be val
            os.rename('/content/landscape/' + file, '/content/images/val/class/' + file)
        else:  # others will be val
            os.rename('/content/landscape/' + file, '/content/images/train/class/' + file)


def convert_to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    plt.clf()
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
