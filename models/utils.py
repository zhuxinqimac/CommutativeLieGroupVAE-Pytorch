import torch
import time
from torch.distributions import Categorical
from torch import nn
import collections
import os


def sprites_label_to_action(label):
    if label.shape[-1] > 1 and len(list(label.shape)) > 1:
        actions = torch.argmax(label.abs(), -1)
        actions[(label[:, actions] < 0).any(dim=-1)] += label.shape[-1]
        return actions
    return label


def attn_stats(attention, true_actions=None):
    state = {}
    state['attn/attn_argmax'] = torch.argmax(attention, -1)
    state['attn/attn'] = attention
    state['attn/attn_scaled_dists'] = torch.stack([attention[:, i]-0.5 + 10*i for i in range(attention.shape[1])], -1)
    state['attn/attn_norm'] = torch.norm(attention, dim=-1)
    state['attn/mean_entropy'] = Categorical(attention.mean(0)).entropy()
    state['attn/per_action_entropy'] = Categorical(attention).entropy().mean()

    per_action_dist = collections.defaultdict(list)
    if true_actions is not None:

        true_actions = sprites_label_to_action(true_actions)

        for i, a in enumerate(true_actions):
            per_action_dist[a.item()].append(attention[i])

        mean_true_action_entropy = []
        per_action_dist = {k: torch.stack(per_action_dist[k], 0).sum(0) for k in per_action_dist}
        for k in per_action_dist:
            dist = Categorical(per_action_dist[k]).entropy()
            mean_true_action_entropy.append(dist)

        # Pseudo estimate for the independence metric.
        per_action_dist = {k: v/v.norm(2) for k, v in per_action_dist.items()}
        overlap = [torch.dot(d1, d2) for k, d1 in per_action_dist.items() for kk, d2 in per_action_dist.items() if k != kk]
        state['attn/independence'] = 1 - (sum(overlap) / len(overlap))
        state['attn/true_action_entropy'] = (sum(mean_true_action_entropy) / len(mean_true_action_entropy))
    return state


def count_parameters(model):
    def _count(_model):
        return sum(p.numel() for p in _model.parameters() if p.requires_grad)

    out_str = ""

    for n,p in model.named_modules():
        if n == '':
            n = model.__class__.__name__
        out_str += str(n) + " - " + str(_count(p)) + '\n'
    return out_str


class ContextTimer:
    def __init__(self, tag):
        self.tag = tag

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start
        print('{} - Time: {}'.format(self.tag, self.interval))


class ParallelActions(nn.Module):
    def __init__(self, linear_module_list):
        super().__init__()
        self.linear_module_list = linear_module_list

    def _make_op(self, a, orders=None):
        if orders is not None:
            return self._slower_make_op(a, orders)

        A, B = [], []
        for mod in self.linear_module_list:
            A.append(mod.cob), B.append(mod.rep(orders))

        A, B = torch.stack(A), torch.stack(B).unsqueeze(1)

        if mod.use_cob:
            avail_weights = torch.matmul(torch.matmul(A.inverse(), B), A)
        else:
            avail_weights = B

        weights = avail_weights[a, 0]

        return weights

    def _slower_make_op(self, a, orders):
        A, B = [], []

        for i, ac in enumerate(a):
            A.append(self.linear_module_list[ac].cob)
            B.append(self.linear_module_list[ac].rep(orders[i]))

        A, B = torch.stack(A), torch.stack(B).unsqueeze(1)

        if self.linear_module_list[0].use_cob:
            avail_weights = torch.matmul(torch.matmul(A.inverse(), B), A)
        else:
            avail_weights = B

        return avail_weights.squeeze(1)

    def forward(self, x, a, orders=None):
        weight = self._make_op(a, orders)
        return torch.matmul(weight, x.unsqueeze(-1)).squeeze(-1)


def _safe_load(modelfile):
    state = torch.load(modelfile)
    if hasattr(state, 'state_dict'):
        return state.state_dict()
    else:
        return state


def model_loader(checkpoint_dir, model_label='model.pt'):
    model_path = os.path.join(checkpoint_dir, model_label)
    args_path = os.path.join(checkpoint_dir, 'args.pt')

    model = _safe_load(model_path)
    args = None
    try:
        args = torch.load(args_path)
    except:
        print('No args file associated...')
    return model, args


def clip_hook(grad_output):
    return grad_output.clamp(-10, 10)

