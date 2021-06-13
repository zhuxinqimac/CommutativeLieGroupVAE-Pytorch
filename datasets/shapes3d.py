#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: shapes3d.py
# --- Creation Date: 16-01-2021
# --- Last Modified: Tue 13 Apr 2021 16:55:42 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Dataset for 3D Shapes
"""

import numpy as np
from torch.utils.data import Dataset
import os
import shutil
import h5py
import zipfile
from PIL import Image
import torch
import random
from datasets.transforms import PairTransform


class shapes3d(Dataset):
    """
    Args:
        root (str): Root directory of dataset containing 3dshapes.h5
        transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, transform=None, fixed_shape=None):
        super(shapes3d, self).__init__()
        self.file = root
        self.transform = transform
        self.fixed_shape = fixed_shape

        self.dataset_zip = self.load_data()
        self.data = self.dataset_zip['images'][:]  # array shape [480000,64,64,3], uint8 in range(256)
        # self.latents_sizes = np.array([3, 6, 40, 32, 32])
        self.latents_sizes = np.array([10, 10, 10, 8, 4, 15])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
        # self.latents_classes = np.load(os.path.join(self.file, "latents_classes.npy"))
        self.latents_classes = self.dataset_zip['labels'][:]  # array shape [480000,6], float64

        # if fixed_shape is not None:
            # self._reduce_data(fixed_shape)

    def generative_factors(self, index):
        return self.latents_classes[index]

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def index_to_latent(self, index):
        return self.latents_classes[index]

    def get_img_by_latent(self, latent_code):
        """
        Returns the image defined by the latent code

        Args:
            latent_code (:obj:`list` of :obj:`int`): Latent code of length 6 defining each generative factor
        Returns:
            Image defined by given code
        """
        idx = self.latent_to_index(latent_code)
        return self.__getitem__(idx)

    def sample_latent(self):
        f = []
        for factor in self.latents_sizes:
            f.append(np.random.randint(0, factor))
        return np.array(f)

    def load_data(self):
        root = os.path.join(self.file, "3dshapes.h5")
        dataset_zip = h5py.File(root, 'r')
        # data = np.load(root)
        return dataset_zip

    def __getitem__(self, index):
        data = self.data[index]
        data = Image.fromarray(data)
        labels = self.latents_classes[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, labels[1:]

    def __len__(self):
        return self.data.shape[0]


class PairShapes3D(shapes3d):
    def __init__(self, root, download=False, transform=None, offset=2, max_varied=1, wrapping=False, noise_name=None, output_targets=True, fixed_shape=None):
        """ dSprites dataset with symmetry sampling included if output_targets is True.

        Args:
            root (str): Root directory of dataset containing '3dshapes.h5' or to download it to
            transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            offset (int, list[int]): Offset of generative factor indices when sampling symmetries
            max_varied (int): Max number of symmetries acting per observation
            wrapping (bool): Wrap at boundaries or invert action
            noise_name (str): Name of noise to add, default None
            output_targets (bool): If True output image pair corresponding to symmetry action. If False, standard dSprites.
        """
        super().__init__(root, transform)
        self.factor = [0, 1, 2, 3, 5]
        self.offset = offset
        self.max_varied = max_varied
        self.wrapping = wrapping
        self.noise_transform = PairTransform(noise_name) if noise_name is not None else None
        self.output_targets = output_targets

    def get_next_img_by_offset(self, label1, img1, factor):
        max_offsets = [10, 10, 10, 8, 1, 15]

        new_latents = np.array(list(label1))
        offset = torch.zeros(label1.shape).to(img1.device)

        for f in factor:
            cur_offset = self.offset if self.offset < max_offsets[f] else max_offsets[f]
            if torch.rand(1) < 0.5:
                cur_offset = cur_offset * -1
            if self.wrapping:
                new_latents[f] = (label1[f] + cur_offset) % (self.latents_sizes[f])
            else:
                new_latents[f] = (label1[f] + cur_offset).clip(min=0, max=self.latents_sizes[f]-1)
            offset[f] = cur_offset

        idx = self.latent_to_index(new_latents)
        return idx, offset

    def get_next_img_by_rand(self, latent1):
        idx = torch.randint(len(self), (1,)).int()
        offset = self.index_to_latent(idx)[1:] - latent1
        return idx, offset

    def __getitem__(self, index):

        factor = self.factor
        img1, label1 = super().__getitem__(index)

        if not self.output_targets:
            return img1, label1

        if not isinstance(factor, list):
            factor = [factor]
        else:
            factor = random.choices(factor, k=self.max_varied)

        # TODO: Always set offset to 1 for val set? So we can eval metrics. Images wouldn't show multi steps though...
        if self.offset != -1:
            idx, offset = self.get_next_img_by_offset(label1, img1, factor)
        else:
            idx, offset = self.get_next_img_by_rand(label1)

        img2, label2 = super().__getitem__(idx)

        if self.noise_transform is not None:
            img1, img2 = self.noise_transform(img1, img2)

        return (img1, offset), img2
