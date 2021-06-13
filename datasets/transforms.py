import torch
from torch.nn import functional as F
import random
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class GaussianNoise:
    def __init__(self, weight=0.1):
        """ Adds Gaussian noise to input image

        Args:
            weight: max value of the Gaussian noise
        """
        self.weight = weight

    def mask(self, x):
        return torch.rand_like(x) * self.weight

    def apply(self, x, mask):
        return (x + mask).clamp(0, 1)

    def __call__(self, x):
        return self.apply(x, self.mask(x))


class SPNoise:
    def __init__(self, p=0.1):
        """ Adds salt and pepper noise to the input image

        Args:
            p: Probability of a pixel being noise
        """
        self.p = p

    def mask(self, x):
        return torch.rand_like(x) < self.p

    def apply(self, x, mask):
        return (x + mask).clamp(0, 1)

    def __call__(self, x):
        return self.apply(x, self.mask(x))


class Backgrounds:
    def __init__(self, ds=None, nbg=10):
        """ Adds backgrounds to input image (only works for black and white / binary images)

        Args:
            ds: Dataset from which backgrounds are sampled
            nbg: Number of backgrounds used
        """
        self.ds = ds if ds is not None else self._default_ds()
        self.nbg = nbg
        self.bgs = self.get_bgs()
        self._processed = False

    def _default_ds(self):
        path = './tmp/cifar'
        return CIFAR10(path, transform=ToTensor(), download=True)

    def get_bgs(self):
        bgs = []
        for i in range(self.nbg):
            id = torch.randint(len(self.ds), (1,)).long()
            bgs.append(self.ds[id][0])
        return bgs

    def process_bgs(self, x):
        bgs = []
        for bg in self.bgs:
            if x.shape[0] == 1:
                bg = bg.mean(0, keepdim=True)

            if bg.shape != x.shape:
                bg = F.interpolate(bg.unsqueeze(0), x.shape[-2:]).squeeze(0)
            bgs.append(bg)
        self.bgs = bgs
        self._processed = True

    def mask(self, x):
        if not self._processed:
            self.process_bgs(x)

        bg = random.choice(self.bgs)
        return bg

    def apply(self, x, mask):
        return (1 - ((1-x) * (1-(mask-0.1)))).clamp(0, 1)

    def __call__(self, x):
        return self.apply(x, self.mask(x))


class PairTransform:
    def __init__(self, transform_name):
        """ Applies a transform to two images in the same way

        Args:
            transform_name: One of ['Gaussian', 'Salt', 'BG']
        """
        transforms = {
            'Gaussian': GaussianNoise(),
            'Salt': SPNoise(),
            'BG': Backgrounds(),
        }
        self.transform = transforms[transform_name]
        self.same_mask = True if transform_name == 'BG' else False

    def __call__(self, x1, x2):
        if self.same_mask:
            mask = self.transform.mask(x1)
            return self.transform.apply(x1, mask), self.transform.apply(x2, mask)
        else:
            return self.transform(x1), self.transform(x2)


