import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda, ToTensor

from datasets.dsprites import PairSprites
from datasets.shapes3d import PairShapes3D


def sprites_transforms(_):
    return ToTensor(), ToTensor()

def shapes3d_transforms(_):
    return ToTensor(), ToTensor()

transforms = {
    'dsprites': sprites_transforms,
    'shapes3d': shapes3d_transforms,
}

def split(func):  # Splits a dataset into a train and val set
    def splitter(args):
        ds = func(args)
        lengths = int(len(ds) * (1 - args.split)), int(len(ds)) - int(len(ds) * (1 - args.split))
        train_ds, val_ds = random_split(ds, lengths) if args.split > 0 else (ds, None)
        return train_ds, val_ds

    return splitter


def fix_data_path(func):
    def fixer(args):
        args.data_path = args.data_path if args.data_path is not None else _default_paths[args.dataset]
        return func(args)

    return fixer


def set_to_loader(trainds, valds, args):
    trainloader = DataLoader(trainds, batch_size=args.batch_size, num_workers=7, shuffle=args.shuffle, drop_last=False,
                             pin_memory=True)
    if valds is not None:
        valloader = DataLoader(valds, batch_size=args.batch_size, num_workers=7, shuffle=False, drop_last=False,
                               pin_memory=True)
    else:
        valloader = None
    return trainloader, valloader


@split
@fix_data_path
def sprites(args):
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = False
    ds = PairSprites(args.data_path, download=True, transform=train_transform, wrapping=True,
                     output_targets=output_targets)
    return ds

@split
@fix_data_path
def shapes3d(args):
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = False
    ds = PairShapes3D(args.data_path, transform=train_transform, wrapping=True,
                     output_targets=output_targets)
    return ds


_default_paths = {
    'dsprites': '',
    'shapes3d': '',
}

datasets = {
    'dsprites': sprites,
    'shapes3d': shapes3d,
}

dataset_meta = {
    'dsprites': {'nc': 1, 'factors': 5, 'max_classes': 40},
    'shapes3d': {'nc': 3, 'factors': 6, 'max_classes': 40},
}
