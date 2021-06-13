import torch
from sklearn import ensemble


class Downstream:
    def __init__(self, ds, num_points=1000, bs=5, paired=True, max_depth=15):
        """ Downstream performance task of predicting actions from symmetry image pair based on ForwardVAE.

        Args:
            ds (Dataset): torch dataset on which to evaluate
            num_points (int): Number of points to use in metric calculation
            bs (int): Batch size
            paired (bool): If True expect the dataset to output symmetry paired images
            max_depth: Max depth fo the tree classifier
        """
        super().__init__()
        self.ds = ds
        self.num_points = num_points
        self.bs = bs
        self.paired = paired
        self.max_depth = max_depth

    def _get_sample_difference(self, rep_fn):

        with torch.no_grad():
            x1s, x2s, labels = [], [], []

            for i in range(self.num_points):
                ind = torch.randint(0, len(self.ds), (1,))
                (x1, a), x2 = self.ds[ind]
                x1s.append(x1), x2s.append(x2), labels.append(a)
            x1s, x2s = torch.stack(x1s).cuda(), torch.stack(x2s).cuda()
            out = rep_fn(x1s, x2s)

            labels = torch.stack(labels)
            if labels.shape[-1] > 2:
                actions = torch.argmax(labels.abs(), -1)
                actions = actions * 2
                actions[(labels[:, actions // 2] < 0).any(dim=-1)] += 1
                labels = actions

        return out.cpu().numpy(), labels

    def __call__(self, pymodel):
        try:
            ds_pre_state = self.ds.output_targets
        except:
            ds_pre_state = None
        self.ds.output_targets = True

        rep_fn = lambda x, x2: torch.cat((pymodel.unwrap(pymodel.encode(x))[0], pymodel.unwrap(pymodel.encode(x2))[0]), 1)
        attn_fn = lambda x1, x2: pymodel.groups.get_attn(x1, x2)

        try:
            train_points, train_labels = self._get_sample_difference(rep_fn)
            model_rep = ensemble.RandomForestClassifier(max_depth=self.max_depth)
            model_rep.fit(train_points, train_labels)
            eval_points_rep, eval_labels_rep = self._get_sample_difference(rep_fn)
            eval_accuracy_rep = model_rep.score(eval_points_rep, eval_labels_rep)
        except:
            eval_accuracy_rep = None

        try:
            train_points, train_labels = self._get_sample_difference(attn_fn)
            model_attn = ensemble.RandomForestClassifier(max_depth=self.max_depth)
            model_attn.fit(train_points, train_labels)
            eval_points_attn, eval_labels_attn = self._get_sample_difference(attn_fn)
            eval_accuracy_attn = model_attn.score(eval_points_attn, eval_labels_attn)
        except:
            eval_accuracy_attn = None

        self.ds.output_targets = ds_pre_state
        return {'dmetric/downstream_rep': eval_accuracy_rep, 'dmetric/downstream_attn': eval_accuracy_attn}
