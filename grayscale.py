import numpy as np
from torchvision import datasets
from skimage.color import rgb2lab, rgb2gray
import torch


class Grayscale(datasets.ImageFolder):

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            original_image = np.asarray(self.transform(img))
            LAB_image = (rgb2lab(original_image) + 128) / 255
            AB_image = LAB_image[:, :, 1:3]
            AB_image = torch.from_numpy(AB_image.transpose((2, 0, 1))).float()
            original_image = rgb2gray(original_image)
            original_image = torch.from_numpy(original_image).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return original_image, AB_image, target
