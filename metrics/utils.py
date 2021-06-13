import numpy as np
import sklearn
import torch


def _get_random_latent(ds):
    f = []
    for factor in ds.latents_sizes:
        f.append(np.random.randint(0, factor))
    return np.array(f)


def _sample_one_representation(rep_fn, ds, paired=False):
    latent_1 = ds.sample_latent()
    img1 = ds.get_img_by_latent(latent_1)[0]
    if not torch.is_tensor(img1):
        img1 = img1[0]

    z = rep_fn(img1.to('cuda').unsqueeze(0))

    return z.detach().cpu(), latent_1


def sample_batch(model, num_points, ds, paired=False):
    reps, factors = None, None
    for i in range(num_points):
        rep, fac = _sample_one_representation(model, ds, paired=paired)
        # fac = fac[1:]

        if i == 0:
            reps, factors = rep, fac
        else:
            factors = np.vstack((factors, fac))
            reps = np.vstack((reps, rep))
    return np.transpose(reps), np.transpose(factors)


def histogram_discretize(target, num_bins=20):
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


def normalize_data(data, mean=None, stddev=None):
    if mean is None:
        mean = np.mean(data, axis=1)
    if stddev is None:
        stddev = np.std(data, axis=1)
    return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev
