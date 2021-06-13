import numpy as np
from torch.utils.data import Dataset
import os
import shutil
import zipfile
from PIL import Image
import torch
import random
from datasets.transforms import PairTransform


class dSprites(Dataset):
    """
    `dSprites <https://github.com/deepmind/dsprites-dataset>`_ Dataset
    Args:
        root (str): Root directory of dataset containing 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz' or to download it to
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, download=False, transform=None, fixed_shape=None):
        super(dSprites, self).__init__()
        self.file = root
        self.transform = transform
        self.fixed_shape = fixed_shape

        if download:
            self.download()

        self.data = self.load_data()
        self.latents_sizes = np.array([3, 6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
        self.latents_values = np.load(os.path.join(self.file, "latents_values.npy"))
        self.latents_classes = np.load(os.path.join(self.file, "latents_classes.npy"))

        if fixed_shape is not None:
            self._reduce_data(fixed_shape)

    def download(self):
        if not os.path.exists(os.path.join(self.file, "imgs.npy")):
            data_url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
            import sys
            if sys.version_info[0] < 3:
                import urllib2 as request
            else:
                import urllib.request as request
            file = os.path.join(self.file, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

            os.makedirs(self.file, exist_ok=True)
            with request.urlopen(data_url) as response, open(file, 'wb+') as out_file:
                shutil.copyfileobj(response, out_file)

            zip_ref = zipfile.ZipFile(file, 'r')
            zip_ref.extractall(self.file)
            zip_ref.close()

    def _reduce_data(self, shape):
        """
        Reduces the data stored in memory if only a fixed shape is required.
        """
        self.latents_sizes = np.array([1, 6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
        self.latents_bases[0] = 0

        data = []
        values, classes = [], []
        for i, img in enumerate(self.data):
            if self.latents_classes[i, 1] == shape:
                data.append(img)
                values.append(self.latents_values[i])
                classes.append(self.latents_classes[i])
        self.data = np.array(data)
        self.latents_classes = np.array(classes)
        self.latents_values = np.array(values)

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
        root = os.path.join(self.file, "imgs.npy")
        data = np.load(root)
        return data

    def __getitem__(self, index):
        data = self.data[index]
        data = Image.fromarray(data * 255, mode='L')
        labels = self.latents_classes[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, labels[1:]

    def __len__(self):
        return self.data.shape[0]


class PairSprites(dSprites):
    def __init__(self, root, download=False, transform=None, offset=2, max_varied=1, wrapping=False, noise_name=None, output_targets=True, fixed_shape=None):
        """ dSprites dataset with symmetry sampling included if output_targets is True.

        Args:
             root (str): Root directory of dataset containing 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz' or to download it to
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
            transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            offset (int, list[int]): Offset of generative factor indices when sampling symmetries
            max_varied (int): Max number of symmetries acting per observation
            wrapping (bool): Wrap at boundaries or invert action
            noise_name (str): Name of noise to add, default None
            output_targets (bool): If True output image pair corresponding to symmetry action. If False, standard dSprites.
        """
        # super().__init__(root, download, transform, None)
        super().__init__(root, download, transform, fixed_shape)
        self.factor = [1, 2, 3, 4]
        self.offset = offset
        self.max_varied = max_varied
        self.wrapping = wrapping
        self.noise_transform = PairTransform(noise_name) if noise_name is not None else None
        self.output_targets = output_targets

    def get_next_img_by_offset(self, label1, img1, factor):
        max_offsets = [1, 2, 20, 20, 20]
        # true angles: (2,3): 2.09. (4,5): 0.63. (6,7),(8,9): 0.79

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
