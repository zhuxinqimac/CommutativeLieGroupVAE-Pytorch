"""Implementation of MIG Metric.

Based on "Isolating Sources of Disentanglement in VAEs" .
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import sklearn
import torch


def _histogram_discretize(target, num_bins=20):
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


class MigMetric:
    def __init__(self, ds, num_points=1000, paired=False):
        """ MIG Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to evaluate on
            paired (bool): If True expect the dataset to output symmetry paired images
        """
        super().__init__()
        self.ds = ds
        self.num_points = num_points
        self.paired = paired

    def _sample_one_representation(self, rep_fn):
        latent_1 = self.ds.sample_latent()
        img1 = self.ds.get_img_by_latent(latent_1)[0]
        if not torch.is_tensor(img1):
            img1 = img1[0]
        z1 = rep_fn(img1.cuda().unsqueeze(0))

        return z1.detach().cpu(), latent_1

    def _sample_batch(self, rep_fn):
        reps, factors = None, None
        for i in range(self.num_points):
            rep, fac = self._sample_one_representation(rep_fn)
            fac = fac[1:]

            if i == 0:
                reps, factors = rep, fac
            else:
              factors = np.vstack((factors, fac))
              reps = np.vstack((reps, rep))
        return np.transpose(reps), np.transpose(factors)

    def __call__(self, pymodel):
        rep_fn = lambda x: pymodel.unwrap(pymodel.encode(x))[0]
        reps, facs = self._sample_batch(rep_fn)
        discretized_mus = _histogram_discretize(reps)
        m = discrete_mutual_info(discretized_mus, facs)
        assert m.shape[0] == reps.shape[0]
        assert m.shape[1] == facs.shape[0]
        entropy = discrete_entropy(facs)
        sorted_m = np.sort(m, axis=0)[::-1]

        return {'dmetric/discrete_mig': np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))}

