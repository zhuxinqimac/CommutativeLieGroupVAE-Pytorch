# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of the SAP score.

Based on "Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations" (https://openreview.net/forum?id=H1kG7GZAW), Section 3.
Implementation based on https://github.com/google-research/disentanglement_lib

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from metrics import utils
import numpy as np
from six.moves import range
from sklearn import svm


class SapMetric:
    def __init__(self, ds, num_points=1000, paired=False):
        """ SAP Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to evaluate on
            paired (bool): If True expect the dataset to output symmetry paired images
        """
        super().__init__()
        self.ds = ds
        self.num_points = num_points
        self.paired = paired

    def __call__(self, model):
        rep_fn = lambda x: model.unwrap(model.encode(x))[0]
        # logger.info("Generating training set.")
        mus, ys = utils.sample_batch(rep_fn, self.num_points, self.ds, paired=self.paired)
        mus_test, ys_test = utils.sample_batch(rep_fn, self.num_points, self.ds, paired=self.paired)

        if (ys[0] == 0).all():
            ys = ys[1:]
            ys_test = ys_test[1:]

        # logger.info("Computing score matrix.")
        return _compute_sap(mus, ys, mus_test, ys_test, True)


def _compute_sap(mus, ys, mus_test, ys_test, continuous_factors):
    """Computes score based on both training and testing codes and factors."""
    score_matrix = _compute_score_matrix(mus, ys, mus_test,
                                         ys_test, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]
    scores_dict = {}
    scores_dict["dmetric/SAP_score"] = _compute_avg_diff_top_two(score_matrix)
    # logger.info("SAP score: %.2g", scores_dict["SAP_score"])

    return scores_dict


def _compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
    """Compute score matrix as described in Section 3."""
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            if continuous_factors:
                # Attribute is considered continuous.
                cov_mu_i_y_j = np.cov(mu_i, np.transpose(y_j), ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                mu_i_test = mus_test[i, :]
                y_j_test = ys_test[j, :]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(mu_i[:, np.newaxis],  y_j)
                pred = classifier.predict(mu_i_test[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix


def _compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
