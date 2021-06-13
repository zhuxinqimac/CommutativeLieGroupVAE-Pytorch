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

"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
Implementation based on https://github.com/google-research/disentanglement_lib

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
from metrics import utils


class DciMetric:
    def __init__(self, ds, num_points=1000, paired=False):
        """ DCI Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to use in metric calculation
            paired (bool): If True expect the dataset to output symmetry paired images
        """
        self.ds = ds
        self.num_points = num_points
        self.paired = paired

    def __call__(self, model):
        try:
            rep_fn = lambda x: model.unwrap(model.encode(x))[0]
            mus_train, ys_train = utils.sample_batch(rep_fn, self.num_points, self.ds, paired=self.paired)
            assert mus_train.shape[1] == self.num_points
            assert ys_train.shape[1] == self.num_points
            mus_test, ys_test = utils.sample_batch(rep_fn, self.num_points, self.ds, paired=self.paired)

            if not (ys_train[0] > 0).any() or not (ys_test[0] > 0).any():
                ys_train = ys_train[1:]
                ys_test = ys_test[1:]

            scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)
        except:
            scores = {}
        return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err = compute_importance_gbt(
      mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["dmetric/dci_informativeness_train"] = train_err
  scores["dmetric/dci_informativeness_test"] = test_err
  scores["dmetric/dci_disentanglement"] = disentanglement(importance_matrix)
  scores["dmetric/dci_completeness"] = completeness(importance_matrix)
  return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    model = GradientBoostingClassifier()
    model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
  return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_code(importance_matrix):
  """Compute completeness of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)
