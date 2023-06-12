import random
import numbers
import math

from PIL import Image, ImageOps
import numpy as np

import torch
import torchvision

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    Randomly select the smaller edge from the range of 'size'.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        selected_size = np.random.randint(low=self.size[0], high=self.size[1] + 1, dtype=int)
        scale = GroupScale(selected_size, interpolation=self.interpolation)
        return scale(img_group)

class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images

class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)
    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self):
        pass
    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group

class ToTensorFormat(object):
    def __init__(self):
        pass
    def __call__(self, image_group):
        pic = np.stack(image_group, axis=0)
        img = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()
        img = img.float().div(255)
        return img

class GroupNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.FloatTensor(mean).view(len(mean), 1, 1, 1)
        self.std = torch.FloatTensor(std).view(len(std), 1, 1, 1)
    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

def build_transforms(is_train):
    aug = []
    if is_train:
        aug += [GroupRandomScale(size=(128, 160)),
                GroupRandomCrop(size=128),
                GroupRandomHorizontalFlip()]
    else:
        scaled_size = int(128 / 0.875 + 0.5)
        aug += [GroupScale(size=scaled_size),
                GroupCenterCrop(size=128)]
    aug += [ToTensorFormat(),
            GroupNormalize()]
    aug = torchvision.transforms.Compose(aug)
    return aug
