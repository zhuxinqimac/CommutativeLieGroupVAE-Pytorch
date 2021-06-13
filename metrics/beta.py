"""Implementation of BetVAE Metric.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" .
Implementation based on https://github.com/google-research/disentanglement_lib
"""

import numpy as np
import torch
from sklearn import linear_model


class BetaVAEMetric:
    def __init__(self, ds, num_points=10000, bs=5, paired=False, fixed_shape=True):
        """ BetaVAE Metric

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to evaluate on
            bs (int): batch size
            paired (bool): If True expect the dataset to output symmetry paired images
            fixed_shape (bool): If fix shape in dsprites.
        """
        super().__init__()
        self.ds = ds
        self.num_points = num_points
        self.bs = bs
        self.paired = paired
        self.fixed_shape = fixed_shape

    def _get_sample_difference(self, rep_fn, bs):
        with torch.no_grad():

            if 'flatland' in str(type(self.ds)):
                K = np.random.randint(0, len(self.ds.latents_sizes))
            elif 'dsprites' in str(type(self.ds)):
                if self.fixed_shape:
                    K = np.random.randint(2, len(self.ds.latents_sizes))
                else:
                    K = np.random.randint(0, len(self.ds.latents_sizes))
            else:
                K = np.random.randint(0, len(self.ds.latents_sizes))

            diffs = []

            for i in range(bs):
                latent_1 = self.ds.sample_latent()
                latent_2 = self.ds.sample_latent()

                latent_1[K] = latent_2[K]

                img1 = self.ds.get_img_by_latent(latent_1)[0]
                img2 = self.ds.get_img_by_latent(latent_2)[0]
                if not torch.is_tensor(img1):
                    img1, img2 = img1[0], img2[0]

                z1 = rep_fn(img1.to('cuda').unsqueeze(0))
                z2 = rep_fn(img2.to('cuda').unsqueeze(0))

                diffs.append(torch.abs(z1 - z2))
            diffs = tuple(diffs)
        return torch.mean(torch.cat(diffs), 0).cpu().numpy(), K

    def _get_sample_batch(self, rep_fn, bs):
        labels = np.zeros(self.num_points, dtype=np.int64)
        points = None  # Dimensionality depends on the representation function.

        for i in range(self.num_points):
            feats, labels[i] = self._get_sample_difference(rep_fn, bs)
            if points is None:
              points = np.zeros((self.num_points, feats.shape[0]))
            points[i, :] = feats
        return points, labels

    def __call__(self, pymodel):
        rep_fn = lambda x: pymodel.unwrap(pymodel.encode(x))[0]
        train_points, train_labels = self._get_sample_batch(rep_fn, self.bs)

        model = linear_model.LogisticRegression(penalty='none', multi_class='multinomial', solver='newton-cg')
        model.fit(train_points, train_labels)

        train_accuracy = model.score(train_points, train_labels)
        train_accuracy = np.mean(model.predict(train_points) == train_labels)

        eval_points, eval_labels = self._get_sample_batch(rep_fn, self.bs)
        eval_accuracy = model.score(eval_points, eval_labels)

        return {'dmetric/hig_acc': train_accuracy, 'dmetric/val_hig_acc': eval_accuracy}
