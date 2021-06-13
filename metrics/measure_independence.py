
from models.models import models
import os
import torch
from models.utils import model_loader, sprites_label_to_action
from datasets.datasets import datasets, dataset_meta, set_to_loader
from torch.nn import functional as F
from torch import nn, optim
import tqdm
import collections
import itertools
from torch.utils.data import DataLoader, Dataset
import random
import gc


class CoB(nn.Module):
    def __init__(self, subrep, repsize=(4, 4)):
        """ Change of Basis wrapper for representation

        Args:
            subrep (nn.Module): Sub-representation
            repsize (list[int]): Size of representation
        """
        super().__init__()
        self.subrep = subrep
        self.cob = nn.Parameter(torch.eye(*repsize).unsqueeze(0), requires_grad=True)

    def cyclic_angle(self):
        return self.subrep.cyclic_angle()

    def loss(self):
        return 0

    def forward(self, x, ac):
        cob = self.cob.squeeze(1)
        cob_inv = cob.inverse()
        y = torch.matmul(cob, x.unsqueeze(-1)).squeeze(-1)
        y2 = self.subrep(y, ac)
        return torch.matmul(cob_inv, y2.unsqueeze(-1)).squeeze(-1)

    def make_rep(self, ac):
        base = self.subrep.make_rep(ac)
        return self.cob, base


class NdRep(nn.Module):
    def __init__(self, dim=2, repsize=(4, 4)):
        """ N-dimensional representation

        Args:
            dim (int): Dimension of representation
            repsize (list[int]): Size of representation
        """
        super().__init__()
        self.angles = nn.Parameter(torch.ones(dim, dim), requires_grad=True)
        self.dim = dim
        self.repsize=repsize

    def __repr__(self):
        return str([self.angles[i] for i in range(4)])

    def loss(self):
        return abs(torch.det(self.angles) - 1)

    def cyclic_angle(self):
        m = self.angles.clamp(-1, 1)
        angles = [torch.acos(m[0,0]), torch.asin(m[0,1]), torch.acos(m[1, 1]), torch.asin(m[1, 0])]
        return torch.stack(angles)

    def make_rep(self, ac):
        eye = torch.eye(self.repsize[0], device=ac.device)
        angle = self.angles
        eye[0:self.dim, 0:self.dim] = angle
        return eye

    def forward(self, x, ac):
        rep = self.make_rep(ac).cuda()
        return torch.matmul(rep, x.unsqueeze(-1)).squeeze(-1)


class ParallelApply(nn.Module):
    def __init__(self, reps):
        """ Helper class to apply representations in parallel

        Args:
            reps (list[nn.Module]): List of representations
        """
        super().__init__()
        self.reps = reps

    def __iter__(self):
        return iter(self.reps)

    def forward(self, x, ac):
        if len(ac.shape) > 1:
            ac = ac.squeeze(-1)

        main_rep, cob_rep = [], []
        for r in self.reps:
            cob, subrep = r.make_rep(ac)
            main_rep.append(subrep)
            cob_rep.append(cob)

        main_rep = torch.stack(main_rep)[ac]
        cob_rep = torch.stack(cob_rep)[ac].squeeze(1)
        cob_inv = torch.inverse(cob_rep)

        return torch.matmul(torch.matmul(cob_rep, torch.matmul(main_rep, cob_inv)), x.unsqueeze(-1)).squeeze(-1)


class ListDs(Dataset):
    def __init__(self, list):
        """ Helper class for list of tensors dataset

        Args:
            list (list): List of tensors
        """
        self.items = list

    def __getitem__(self, item):
        return self.items[item]

    def __len__(self):
        return len(self.items)


class TrueIndep:
    def __init__(self, ds, val_ds=None, num_epochs=30, bs=1024, nactions=4, ntrue_actions=4, verbose=False):
        """ True Independence Metric

        Args:
            ds (Dataset): torch dataset on which to train
            val_ds (Dataset): torch dataset on which to evaluate:
            num_epochs (int): Number of epochs to train for
            bs (int): Batch size
            nactions (int): Number of actions to expect from model
            ntrue_actions (int): Number of true actions
            verbose (bool): If True print progress
        """
        self.ds = ds
        self.val_ds = val_ds if val_ds is not None else ds
        self.num_epochs = num_epochs
        self.bs = bs
        self.nactions = nactions
        self.ntrue_actions = ntrue_actions
        self.verbose = verbose

    def __call__(self, pymodel):
        gc.collect()
        pymodel_state = pymodel.training
        pymodel.eval()

        try:
            ds_pre_state = self.ds.output_targets
        except:
            ds_pre_state = None
        self.ds.output_targets = True
        self.val_ds.output_targets = True
        ds = torch.utils.data.ConcatDataset([self.ds, self.val_ds])

        trainloader = DataLoader(ds, self.bs, shuffle=True, num_workers=7)
        valloader = DataLoader(ds, self.bs, shuffle=True, num_workers=7)

        out = measure_indep(pymodel, self.nactions, 'cuda', self.ntrue_actions, self.num_epochs, self.verbose, trainloader, valloader, self.ds)
        out = {'dmetric/{}'.format(k): v for k, v in out.items()}
        out['dmetric/rep_mean_x2'] = torch.tensor(list(out['dmetric/x2_mse'].values())).mean()
        out['dmetric/rep_mean_z2'] = torch.tensor(list(out['dmetric/z_mse'].values())).mean()
        pymodel.training = pymodel_state
        self.ds.output_targets = ds_pre_state
        self.val_ds.output_targets = ds_pre_state
        print(out)
        gc.collect()
        return out


def measure_indep(vae, latents, device, ntrue_actions, epochs, verbose, trainloader, valloader, trainds):
    reps_init = [CoB(NdRep(2, repsize=(latents, latents)), repsize=(latents, latents)) for _ in range(ntrue_actions)]

    try:
        reps = list(itertools.chain.from_iterable(reps_init))
    except:
        reps = reps_init

    rep_list = nn.ModuleList(reps)
    rep_list.to(device)
    opt = optim.Adam(rep_list.parameters(), lr=0.1, weight_decay=0)
    from torch.optim.lr_scheduler import StepLR
    sched = StepLR(opt, 5, gamma=0.1)

    reps = ParallelApply(reps).cuda()

    epochs = epochs
    iterator = range(epochs)
    pb = trainloader
    if verbose:
        pb = tqdm.tqdm(pb)

    #### Get and store pre and post action latent representations from the dataset
    data = []
    for t, ((img, label), targets) in enumerate(pb):
        img, targets = img.to(device), targets.to(device)
        a = sprites_label_to_action(label).long().to(device)
        with torch.no_grad():
            z = vae.unwrap(vae.encode(img))[0].detach()
            zt = vae.unwrap(vae.encode(targets))[0].detach()

        data.extend(zip([zi for zi in z], [zti for zti in zt], [ai for ai in a]))
    if verbose:
        pb.close()

    gc.collect()
    data = ListDs(data)
    data = DataLoader(data, 1024, True, num_workers=0)

    if verbose:
        iterator = tqdm.tqdm(iterator)

    #### Optimise linear representations to approximate actions
    loss_history = []
    for i in iterator:
        vae.eval()
        mean_loss = 0

        for t, (z, zt, a) in enumerate(iter(data)):
            with torch.enable_grad():
                opt.zero_grad()

                zp = reps(z, a.view(z.shape[0]))
                loss = F.mse_loss(zp, zt, reduction='mean')
                loss += sum([r.loss() for r in reps])
                mean_loss += loss.item()

                loss.backward()
                opt.step()

                if verbose:
                    iterator.set_postfix_str('loss: {}'.format(loss.item()))
        sched.step()
        loss_history.append(mean_loss / len(data))


    #### Evaluate results
    with torch.no_grad():
        pb = iter(valloader)

        if verbose:
            pb = tqdm.tqdm(pb)

        expected_distance = []
        z_mse_per_action = collections.defaultdict(list)
        x2_mse_per_action = collections.defaultdict(list)
        action_rep_per_action = collections.defaultdict(list)
        indeps = []

        if 'PairSprites' in str(trainds):
            groups = [(2, 3), (4, 5), (6, 7), (8, 9)]
            symmetry = {2: 2.09, 3: 2.09, 4: 0.63, 5: 0.63, 6: 0.79, 7: 0.79, 8: 0.79, 9: 0.79}  #(2,3): 2.09. (4,5): 0.63. (6,7),(8,9): 0.79
        elif 'Forward' in str(trainds) or 'FlatLand' in str(trainds):
            groups = [(0, 2), (1, 3)]
            symmetry = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9}

        for (img, label), targets in pb:
            img, targets = img.to(device), targets.to(device)
            a = sprites_label_to_action(label).long().to(device)
            z = vae.unwrap(vae.encode(img))[0].data
            zt = vae.unwrap(vae.encode(targets))[0].data
            x2 = targets.data

            expected_distance.append(F.mse_loss(z, zt, reduction='none').mean(-1))

            zp = reps(z, a.view(z.shape[0]))
            losses = F.l1_loss(zp, zt, reduction='none').mean(-1)
            cur_x2_loss = F.l1_loss(vae.decode(zp).sigmoid(), x2, reduction='none').mean((1, 2, 3))
            action_rep = (zt - z).unsqueeze(1)

            tmp_action_rep_per_action = {}
            for action in a.unique():
                post_a = z - reps(z, action)
                tmp_action_rep_per_action[action.item()] = post_a / post_a.norm(2, dim=1).unsqueeze(-1)

            tmp = []
            for g1 in groups:
                for g2 in groups:
                    if g1 == g2:
                        continue
                    tmp.append(torch.max(torch.stack([(tmp_action_rep_per_action[ig1] * tmp_action_rep_per_action[ig2]).sum(-1) for ig1 in g1 for ig2 in g2], 1), dim=1).values)
            indeps.append(torch.cat(tmp, 0))

            for ia, ac in enumerate(a):
                z_mse_per_action[ac.item()].append(losses[ia].item())
                x2_mse_per_action[ac.item()].append(cur_x2_loss[ia].item())
                action_rep_per_action[ac.item()].append(action_rep[ia])

        mean_z_mse_per_action = {k: torch.tensor(v).mean().item() for k, v in z_mse_per_action.items()}
        mean_x2_mse_per_action = {k: torch.tensor(v).mean().item() for k, v in x2_mse_per_action.items()}
        indep = 1 - torch.cat(indeps).abs().mean().item()

        angles = {}
        for ir, rep in enumerate(reps):
            angles[ir] = rep.cyclic_angle().data

        orders = {k: (2*3.1415)/v.abs().data for k, v in angles.items()}
        expected_distance_between_z = torch.cat(expected_distance, 0).mean().item()
        symmetry_l1 = torch.tensor([(angles[k].mean() - symmetry[k]) for k in symmetry]).abs().mean().item()
        gc.collect()

        best_angle, best_sym_l1 = {}, {}
        best_x2, best_z2 = {}, {}

        for k, angle_set in angles.items():
            if k in symmetry:
                diff = (angle_set - symmetry[k]).abs()
                best = torch.argmin(diff)
                best_angle[k] = angle_set[best]
                best_sym_l1[k] = diff[best]
                best_x2[k] = x2_mse_per_action[k][best]
                best_z2[k] = z_mse_per_action[k][best]

        return {'true_independence': indep,
                'cyclic_angles': angles,
                'cyclic_orders': orders,
                'symmetry_l1': symmetry_l1,
                'symmetry_raw': {k: (angles[k].mean() - symmetry[k]).item() for k in symmetry},
                'x2_mse': mean_x2_mse_per_action,
                'z_mse': mean_z_mse_per_action,
                'expected_dist': expected_distance_between_z,

                'best_x2_mse': best_x2,
                'best_z_mse': best_z2,
                'best_angle': best_angle,
                'best_sym_l1': best_sym_l1,

                'mean_best_x2_mse': torch.tensor([v for v in best_x2.values()]).mean().item(),
                'mean_best_z_mse': torch.tensor([v for v in best_z2.values()]).mean().item(),
                'mean_best_sym_l1': torch.tensor([v for v in best_sym_l1.values()]).mean().item(),
                }
