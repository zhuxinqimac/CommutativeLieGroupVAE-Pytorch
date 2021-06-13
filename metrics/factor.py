#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: factor.py
# --- Creation Date: 16-01-2021
# --- Last Modified: Sat 12 Jun 2021 22:24:28 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""Implementation of FactorVAE Metric.

Based on "Disentangling by Factorising" (https://arxiv.org/abs/1802.05983).
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import torch
from sklearn import linear_model


class FactorVAEMetric:
    def __init__(self, ds, num_train=10000, num_eval=5000, bs=64, paired=False, fixed_shape=True, n_var_est=10000):
        """ FactorVAE Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_train (int): Number of points to train on
            num_eval (int): Number of points to evaluate on
            bs (int): batch size
            paired (bool): If True expect the dataset to output symmetry paired images
            fixed_shape (bool): If fix shape in dsprites.
            n_var_est (int): Number of examples to estimate global variance.
        """
        super().__init__()
        self.ds = ds
        self.num_train = num_train
        self.num_eval = num_eval
        self.bs = bs
        self.paired = paired
        self.fixed_shape = fixed_shape
        self.n_var_est = n_var_est

        if 'flatland' in str(type(self.ds)):
            self.num_factors = len(self.ds.latents_sizes)
        elif 'dsprites' in str(type(self.ds)):
            if self.fixed_shape:
                self.num_factors = len(self.ds.latents_sizes) - 1
            else:
                self.num_factors = len(self.ds.latents_sizes)
        else:
            self.num_factors = len(self.ds.latents_sizes)

    def __call__(self, pymodel):
        rep_fn = lambda x: pymodel.unwrap(pymodel.encode(x))[0]
        global_var = self._compute_variances(rep_fn, self.n_var_est)
        active_dims = self._prune_dims(global_var)
        scores_dict = {}

        if not active_dims.any():
            scores_dict["dmetric/fac_train"] = 0.
            scores_dict["dmetric/fac_eval"] = 0.
            scores_dict["dmetric/fac_num_act"] = 0
            return scores_dict

        train_votes = self._get_train_votes(rep_fn, self.bs, self.num_train,
                                            global_var, active_dims)
        print('train_votes:', train_votes)
        classifier = np.argmax(train_votes, axis=0)
        other_index = np.arange(train_votes.shape[1])
        train_accuracy = np.sum(
            train_votes[classifier, other_index]) * 1. / np.sum(train_votes)

        eval_votes = self._get_train_votes(rep_fn, self.bs, self.num_eval,
                                           global_var, active_dims)
        print('eval_votes:', eval_votes)
        eval_accuracy = np.sum(
            eval_votes[classifier, other_index]) * 1. / np.sum(eval_votes)
        scores_dict["dmetric/fac_train"] = train_accuracy
        scores_dict["dmetric/fac_eval"] = eval_accuracy
        scores_dict["dmetric/fac_num_act"] = active_dims.astype(int).sum()
        return scores_dict

    def _get_train_votes(self, rep_fn, bs, num_points, global_var, active_dims):
        votes = np.zeros((self.num_factors, global_var.shape[0]),
                         dtype=np.int64)
        for _ in range(num_points):
            factor_index, argmin = self._generate_training_sample(rep_fn, bs,
                                                             global_var, active_dims)
            votes[factor_index, argmin] += 1
        return votes

    def _generate_training_sample(self, rep_fn, bs, global_var, active_dims):
        # Select random coordinate to keep fixed.
        factor_index_metric = np.random.randint(self.num_factors)
        if 'dsprites' in str(type(self.ds)) and self.fixed_shape:
            factor_index = factor_index_metric + 1
        else:
            factor_index = factor_index_metric
        obs = []
        for i in range(bs):
            # Sample two mini batches of latent variables.
            factor = self.ds.sample_latent()
            # Fix the selected factor across mini-batch.
            if i == 0:
                fac_mem = factor[factor_index]
            else:
                factor[factor_index] = fac_mem
            # Obtain the observations.
            ob = self.ds.get_img_by_latent(factor)[0]
            if not torch.is_tensor(ob):
                ob = ob[0]
            obs.append(ob)
        obs = torch.stack(obs)
        reps = rep_fn(obs.to('cuda')).cpu().numpy()
        local_variances = np.var(reps, axis=0, ddof=1)
        argmin = np.argmin(local_variances[active_dims] / global_var[active_dims])
        return factor_index_metric, argmin

    def _prune_dims(self, variances, threshold=0.1):
        """Mask for dimensions collapsed to the prior."""
        scale_z = np.sqrt(variances)
        return scale_z >= threshold

    def _compute_variances(self, rep_fn,
                           bs,
                           eval_bs=64):
        obs = []
        for i in range(bs):
            obs_i = self.ds.get_img_by_latent(self.ds.sample_latent())[0]
            if not torch.is_tensor(obs_i):
                obs_i = obs_i[0]
            obs.append(obs_i)
        obs = torch.stack(obs)
        reps = self._obtain_representation(obs, rep_fn, eval_bs)

        assert reps.shape[0] == bs
        return np.var(reps, axis=0, ddof=1)

    def _obtain_representation(self, obs, rep_fn, bs):
        reps = None
        num_points = obs.shape[0]
        i = 0
        while i < num_points:
            num_points_iter = min(num_points - i, bs)
            cur_obs = obs[i:i + num_points_iter]
            if i == 0:
              reps = rep_fn(cur_obs.to('cuda')).cpu().numpy()
            else:
              reps = np.vstack((reps, rep_fn(cur_obs.to('cuda')).cpu().numpy()))
            i += num_points_iter
        return reps
