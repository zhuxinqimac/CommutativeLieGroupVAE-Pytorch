import numpy as np
from metrics import utils


class FLMetric:
    def __init__(self, ds, num_points=1000, paired=False):
        """ Factor Leakage metric

        Args:
            ds (Dataset): torch Dataset object on which to train metrics which need training
            num_points (int): Number of points to use in metric calculation
            paired (bool): If True expect the dataset to output symmetry paired images
        """
        super().__init__()
        self.ds = ds
        self.num_points = num_points
        self.paired = paired

    def _sample_batch(self, model):
        rep_fn = lambda x: model.unwrap(model.encode(x))[0]
        reps, factors = None, None
        for i in range(self.num_points):
            rep, fac = utils._sample_one_representation(rep_fn, self.ds, paired=self.paired)
            fac = fac[1:]

            if i == 0:
                reps, factors = rep, fac
            else:
              factors = np.vstack((factors, fac))
              reps = np.vstack((reps, rep))
        return np.transpose(reps), np.transpose(factors)

    def __call__(self, model):
        reps, facs = self._sample_batch(model)
        discretized_mus = utils.histogram_discretize(reps)
        m = utils.discrete_mutual_info(discretized_mus, facs)
        entropy = utils.discrete_entropy(facs)
        sorted_m = np.sort(m, axis=0)[::-1]

        return {
                'dmetric/factor_leakage_mean': self.fl_mean_fn(sorted_m, 1),
                'dmetric/factor_leakage_norm_mean': self.fl_normalised_mean_fn(sorted_m, 1),
                'dmetric/factor_leakage_auc': self.fl_auc_fn(sorted_m),
                'dmetric/factor_leakage_nm_auc': self.fl_nm_auc_fn(sorted_m),
                # 'dmetric/factor_leakage_list': self.fl_auc_list_fn(sorted_m)
        }

    def mig_fn(self, sorted_m, entropy):
        return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))

    def fl_sum_fn(self, sorted_m, entropy, N):
        return np.mean(np.divide(np.sum(sorted_m[N:, :]), entropy[:]))

    def fl_mean_fn(self, sorted_m, N):
        return np.mean(np.divide(np.mean(sorted_m[N:, :]), 1))

    def fl_normalised_mean_fn(self, sorted_m, N):
        normalised_m = np.abs(sorted_m / np.max(sorted_m, axis=0))

        return np.mean(np.divide(np.mean(normalised_m[N:, :]), 1))

    def fl_auc_fn(self, sorted_m):
        area = 0
        normalised_m = np.abs(sorted_m/np.max(sorted_m, axis=0))

        for i in range(sorted_m.shape[0]):
            area += self.fl_mean_fn(normalised_m, i)
        return area / sorted_m.shape[0]

    def fl_auc_list_fn(self, sorted_m):
        values = []
        normalised_m = np.abs(sorted_m/np.max(sorted_m, axis=0))
        for i in range(sorted_m.shape[0]):
            values.append(self.fl_mean_fn(normalised_m, i))
        return values

    def fl_nm_auc_fn(self, sorted_m):
        area = 0
        for i in range(sorted_m.shape[0]-1):
            area += self.fl_mean_fn(np.abs(sorted_m/np.max(sorted_m, axis=0)), i+1)
        return area / (sorted_m.shape[0]-1)
