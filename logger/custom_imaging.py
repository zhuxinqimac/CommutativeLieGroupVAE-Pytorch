#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: custom_imaging.py
# --- Creation Date: 31-12-2020
# --- Last Modified: Sat 12 Jun 2021 22:26:10 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Imaging.
"""
from torchvision.utils import save_image, make_grid
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw
import io
import numpy as np
from torchvision.transforms import ToTensor
import collections
import warnings
import os
from logger.imaging import Imager


class ShowReconX(Imager):
    def __init__(self,
                 logger,
                 n_per_row=8,
                 to_tb=True,
                 name_1='x1',
                 name_2='x_g_rec'):
        self.logger = logger
        self.n_per_row = n_per_row
        self.to_tb = to_tb
        self.name_1 = name_1
        self.name_2 = name_2

    def __call__(self, model, state, global_step=0):
        x, x_rec = state[self.name_1], state[self.name_2]
        imgs = torch.cat(
            [x[:self.n_per_row], x_rec[:self.n_per_row].sigmoid()], dim=0)
        if not self.to_tb:
            save_image(imgs,
                       './images/%s.png' % self.name_2,
                       nrow=self.n_per_row,
                       normalize=False,
                       pad_value=1)
        else:
            imgs = make_grid(imgs, nrow=self.n_per_row, pad_value=1)
            self.logger.writer.add_image('group_recons/%s' % self.name_2,
                                         imgs.cpu().numpy(), global_step)


class LatentWalkLie(Imager):
    def __init__(self,
                 logger,
                 latents,
                 dims_to_walk,
                 subgroup_sizes_ls,
                 limits=[-2, 2],
                 steps=8,
                 input_batch=None,
                 to_tb=False):
        self.input_batch = input_batch
        self.logger = logger
        self.latents = latents
        self.dims_to_walk = dims_to_walk
        self.subgroup_sizes_ls = subgroup_sizes_ls
        self.limits = limits
        self.steps = steps
        self.to_tb = to_tb

    def gfeats_to_text(self, gfeats):
        out_str = ''
        gf_idx = 0
        for k in range(self.latents):
            per_g_str_ls = []
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                per_g_str_ls.append('')
            for j in range(self.steps):
                b_idx = 0
                for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                    e_idx = b_idx + subgroup_size_i
                    # mat_dim = int(math.sqrt(subgroup_size_i))
                    per_g_str_ls[i] += (str(
                        gfeats[gf_idx][b_idx:e_idx].cpu().numpy().round(3)) +
                                        ', ')
                    b_idx = e_idx
                gf_idx += 1
            for str_i in per_g_str_ls:
                out_str += (str_i + '\n\n')
            out_str += ('\n\n' + '=' * 10 + '\n\n')
        return out_str

    def __call__(self, model, state, global_step=0):
        limits, steps, latents, dims_to_walk = self.limits, self.steps, self.latents, self.dims_to_walk
        linspace = torch.linspace(*limits, steps=steps)

        if self.input_batch is None:
            x = torch.zeros(len(dims_to_walk), steps, latents)
        else:
            x = model.rep_fn(self.input_batch)[0]
            x = x.view(1, 1, latents).repeat(len(dims_to_walk), steps, 1)

        x = x.view(len(dims_to_walk), steps, latents)
        ind = 0
        for i in dims_to_walk:
            x[ind, :, i] = linspace
            ind += 1

        x = x.flatten(0, 1)
        imgs, group_feats_G = model.decode_full(x)
        imgs = imgs.sigmoid()
        group_feats_G_text = self.gfeats_to_text(group_feats_G)
        if not self.to_tb:
            save_image(imgs,
                       './images/linspace.png',
                       steps,
                       normalize=False,
                       pad_value=1)
        else:
            img = make_grid(imgs, self.steps, pad_value=1)
            self.logger.writer.add_image('linspaces/linspace',
                                         img.cpu().numpy(), global_step)
            self.logger.writer.add_text('linspaces/group_feats_G',
                                        group_feats_G_text, global_step)
