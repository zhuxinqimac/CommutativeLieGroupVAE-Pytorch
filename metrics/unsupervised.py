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

"""Unsupervised scores based on code covariance and mutual information."""
import numpy as np
import scipy
from metrics import utils


class UnsupervisedMetrics:

    def __init__(self, ds, num_points=1000, paired=False):
        """ Unsupervised Metrics by https://github.com/google-research/disentanglement_lib

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to evaluate on
            paired (bool): If True expect the dataset to output symmetry paired images
        """
        super().__init__()
        self.num_points = num_points
        self.ds = ds
        self.paired = paired

    def __call__(self, model):
        print('calc UnsupervisedMetrics.')
        rep_fn = lambda x: model.unwrap(model.encode(x))[0]
        scores = {}

        mus_train, _ = utils.sample_batch(rep_fn, self.num_points, self.ds, paired=self.paired)
        num_codes = mus_train.shape[0]
        cov_mus = np.cov(mus_train)
        assert num_codes == cov_mus.shape[0]

        # Gaussian total correlation.
        scores["dmetric/gaussian_total_correlation"] = gaussian_total_correlation(cov_mus)

        # Gaussian Wasserstein correlation.
        scores["dmetric/gaussian_wasserstein_correlation"] = gaussian_wasserstein_correlation(
            cov_mus)
        scores["dmetric/gaussian_wasserstein_correlation_norm"] = (
                scores["dmetric/gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_mus)))

        # Compute average mutual information between different factors.
        mus_discrete = utils.histogram_discretize(mus_train)
        mutual_info_matrix = utils.discrete_mutual_info(mus_discrete, mus_discrete)
        np.fill_diagonal(mutual_info_matrix, 0)
        mutual_info_score = np.sum(mutual_info_matrix) / (num_codes ** 2 - num_codes)
        scores["dmetric/mutual_info_score"] = mutual_info_score
        print('scores:', scores)
        return scores


def gaussian_total_correlation(cov):
  """Computes the total correlation of a Gaussian with covariance matrix cov.

  We use that the total correlation is the KL divergence between the Gaussian
  and the product of its marginals. By design, the means of these two Gaussians
  are zero and the covariance matrix of the second Gaussian is equal to the
  covariance matrix of the first Gaussian with off-diagonal entries set to zero.

  Args:
    cov: Numpy array with covariance matrix.

  Returns:
    Scalar with total correlation.
  """
  return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])


def gaussian_wasserstein_correlation(cov):
  """Wasserstein L2 distance between Gaussian and the product of its marginals.

  Args:
    cov: Numpy array with covariance matrix.

  Returns:
    Scalar with score.
  """
  sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
  return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
