from metrics import *
import torch
import warnings


class MetricAggregator:
    def __init__(self, ds, val_ds, num_points, model, pair_ds=True, nactions=4, ntrue_actions=4, final=False, nindep_epochs=30, fixed_shape=True, verbose=True):
        """ Helper class to compute disentanglement metrics

        Args:
            ds (Dataset): torch Dataset object on which to train metrics which need training
            val_ds (Dataset): torch Dataset object on which to evaluate metrics
            num_points (int): Number of points to use in metric calculation
            model (VAE): PyTorch model to evaluate
            pair_ds (bool): If True expect the dataset to output symmetry paired images
            nactions (int): The number of actions/symmetries to expect from RGrVAE/ForwardVAE
            ntrue_actions (int): The true number of actions
            final (bool): If True also evaluate the true independence
            nindep_epochs (int): Number of epochs to train independence representations for
            fixed_shape (bool): If fix shape in dsprites.
            verbose (bool): If True print verbosely
        """
        self.ds = ds
        self.num_points = num_points
        self.model = model
        self.paired = pair_ds
        self.val_ds = val_ds
        self.nactions = nactions
        self.final = final
        self.ntrue_actions = ntrue_actions
        self.nindep_epochs = nindep_epochs
        self.fixed_shape = fixed_shape
        self.verbose = verbose
        self.metrics = self._init_metrics()

    def _init_metrics(self):
        fac = FactorVAEMetric(self.val_ds, num_train=10000, num_eval=5000, bs=64, paired=self.paired, fixed_shape=self.fixed_shape, n_var_est=10000)
        hig = BetaVAEMetric(self.val_ds, num_points=1000, paired=self.paired, fixed_shape=self.fixed_shape)
        mig = MigMetric(self.val_ds, num_points=1000, paired=self.paired)
        dci = DciMetric(self.val_ds, num_points=1000, paired=self.paired)
        mod = Modularity(self.val_ds, num_points=1000, paired=self.paired)
        sap = SapMetric(self.val_ds, num_points=1000, paired=self.paired)
        unsup = UnsupervisedMetrics(self.val_ds, num_points=1000, paired=self.paired)
        fl = FLMetric(self.val_ds, num_points=1000, paired=self.paired)
        ds = Downstream(self.val_ds, num_points=1000, paired=self.paired)

        metrics = [fac, hig, mig, dci, mod, sap, unsup, fl, ds]
        return metrics

    def __call__(self):
        import gc
        with torch.no_grad():
            outputs = {}
            for metric in self.metrics:
                if self.verbose:
                    print("Computing metric: {}".format(metric))
                # try:
                outputs.update(metric(self.model))
                # except:
                    # warnings.warn('Failed to compute metric: {}'.format(metric))
                gc.collect()
            print('outputs:', outputs)
            return outputs

