from torchvision.utils import save_image, make_grid
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw
import io
import numpy as np
from torchvision.transforms import ToTensor
import collections
import warnings
import os
matplotlib.rcParams['figure.figsize'] = [10, 10]
warnings.filterwarnings("ignore", category=DeprecationWarning)


def text_to_tensor(size, text):
    img = Image.new('L', size)
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, fill=(100))
    return ToTensor()(img).float()


def text_list_to_tensor(size, text_list):
    tensors = []
    for t in text_list:
        tensors.append(text_to_tensor(size, t))
    return torch.stack(tensors, 0)


def pyplot_to_tensor():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    tensor = ToTensor()(im)
    buf.close()
    return tensor


def plt_to_tensor(fig):
    fig.canvas.draw()

    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    img = img / 255.0
    img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2) # if your TensorFlow + TensorBoard version are >= 1.8
    return img


class Imager:
    def __call__(self, model, state, global_step=0):
        pass


class TbLogger(Imager):
    def __init__(self, logger):
        self.writer = logger.writer


class ReconToTb(TbLogger):
    def __call__(self, model, state, step=0):
        recon = state['recon'].sigmoid()
        x = state['x']
        img = torch.cat([x[:8], recon[:8]], 0)
        img = make_grid(img, pad_value=1)

        self.writer.add_image('recons/recon', img.cpu().numpy(), step)


class ActionStepsToTb(TbLogger):
    def __call__(self, model, state, global_step=0):
        action_steps = state['action_sets']
        out_imgs = []
        for img in action_steps:
            out_imgs.append(img[:8])

        img = torch.cat(out_imgs, 0)
        img = make_grid(img, pad_value=1)
        self.writer.add_image('actions/steps', img.cpu().numpy(), global_step)


class ShowRecon(Imager):
    def __call__(self, model, state, global_step=0):
        os.makedirs('./images/', exist_ok=True)
        recon = state['recon']
        x = state['x']

        img = torch.cat([x[:8], recon[:8]], 0)
        save_image(img, './images/recon.png', normalize=False, pad_value=1)


class ShowLearntAction(Imager):
    def __init__(self, logger, to_tb=False):
        self.logger = logger
        self.to_tb = to_tb

    def __call__(self, model, state, global_step=0):
        x1 = state['x1']
        x2 = state['x2']
        recon = state['recon_hat'].sigmoid()
        true_recon = state['true_recon'].sigmoid()

        img = torch.cat([x1[:8], x2[:8], recon[:8], true_recon[:8]], 0)
        if 'action_examples' in state:
            actions = state['action_examples']
            true_actions = state['true_actions']

            try:
                strings = ["t{} - p{}".format(str(a.item()), str(pa.item())) for a, pa in zip(true_actions[:8], actions[:8])]
            except ValueError:
                strings = ["t{}x({}) - p{}".format(str(a.argmax().item()), str(a.max().item()), str(pa.item())) for a, pa in zip(true_actions[:8], actions[:8])]

            text_tensors = text_list_to_tensor(x1[0].shape[-2:], strings).to(x1.device)
            if img.shape[1] > 1:
                text_tensors = text_tensors.repeat(1, img.shape[1], 1, 1)
            img = torch.cat([img, text_tensors], 0)

        if not self.to_tb:
            save_image(img, './images/learnt_action.png', normalize=False, pad_value=1)
        else:
            img = make_grid(img, pad_value=1)
            self.logger.writer.add_image('actions/actions', img, global_step)


class LatentWalk(Imager):
    def __init__(self, logger, latents, dims_to_walk, limits=[-2, 2], steps=8, input_batch=None, to_tb=False):
        self.input_batch = input_batch
        self.logger = logger
        self.latents = latents
        self.dims_to_walk = dims_to_walk
        self.limits = limits
        self.steps = steps
        self.to_tb = to_tb

    def __call__(self, model, state, global_step=0):
        limits, steps, latents, dims_to_walk = self.limits, self.steps, self.latents, self.dims_to_walk
        linspace = torch.linspace(*limits, steps=steps)

        if self.input_batch is None:
            x = torch.zeros(len(dims_to_walk), steps, latents)
        else:
            x = model.rep_fn(self.input_batch)[0]
            x = x.view(1, 1, latents).repeat(len(dims_to_walk), steps, 1)

        x = x.view(len(dims_to_walk), steps, latents)
        ind = 0
        for i in dims_to_walk:
            x[ind, :, i] = linspace
            ind += 1

        x = x.flatten(0,1)
        imgs = model.decode(x).sigmoid()
        if not self.to_tb:
            save_image(imgs, './images/linspace.png', steps, normalize=False, pad_value=1)
        else:
            img = make_grid(imgs, self.steps, pad_value=1)
            self.logger.writer.add_image('linspaces/linspace', img.cpu().numpy(), global_step)


class RewardPlot(TbLogger):
    def __call__(self, model, state, global_step=0):
        super().__call__(model, state)
        try:
            self.writer.add_scalar('reinforce/reward_count', state['reward_count'], global_step)
        except:
            pass


class ActionListTbText(TbLogger):
    def __call__(self, model, state, global_step=0):
        super().__call__(model, state, global_step)
        try:
            self.writer.add_text('actions_examples/examples', str(state['action_examples']), global_step)
        except:
            pass


class ActionWiseRewardPlot(TbLogger):
    def __call__(self, model, state, global_step=0):
        try:
            import collections
            tmp = collections.defaultdict(list)
            for i, r in enumerate(state['rewards']):
                a = state['action_examples'][i]
                tmp[a.item()].append(r)

            action_wise_rewards = [torch.stack(tmp[i]).mean().cpu() for i in tmp]
            fig = plt.figure(figsize=(5, 5), dpi=150)
            fig.gca().plot(action_wise_rewards)
            im = plt_to_tensor(fig)
            self.writer.add_image('pyplot/actionwise_reward', im, global_step)
            plt.close()
        except:
            pass


class AttentionTb(Imager):
    def __init__(self, logger):
        self.logger = logger

    def __call__(self, model, state, global_step=0):

        try:
            self.logger.writer.add_histogram('attn/attn', state['attn/attn'], global_step)
        except:
            pass

        try:
            self.logger.writer.add_histogram('attn/attn_norm', state['attn/attn_norm'], global_step)
        except:
            pass

        try:
            self.logger.writer.add_histogram('attn/attn_scaled_dists', state['attn/attn_scaled_dists'], global_step)
        except:
            pass

        try:
            self.logger.writer.add_histogram('attn/attn_argmax', state['attn/attn_argmax'], global_step)
        except:
            pass

        try:
            histo = state['reinforce/per_action_reward']
            histo = histo / histo.norm(2)
            self.logger.writer.add_histogram('reinforce/per_action_reward', histo, bins=max(histo))
        except:
            pass

        try:
            histo = state['reinforce/optimal_rewards']
            buckets = range(0, len(state['reinforce/optimal_rewards']))
            self.logger.writer.add_histogram('reinforce/optimal_rewards', histo, bins=max(histo))
        except Exception as e:
            pass

        try:
            rewards = state['reinforce/batch_per_action_rewards']
            true_actions = state['true_actions']
            means = []
            for i, a in enumerate(true_actions.unique()):
                means.append(rewards[(true_actions == a).squeeze()].mean(0))

                buckets = range(1, rewards.shape[-1]+1)
                fig = plt.figure(figsize=(5, 5), dpi=150)
                fig.gca().bar(buckets, means[-1].cpu())
                im = plt_to_tensor(fig)
                self.logger.writer.add_image('pyplot/per_true_reward_{}'.format(i), im, global_step)
                plt.close(fig)

        except:
            pass

        try:
            self.logger.writer.add_scalar('attn/mean_entropy', state['attn/mean_entropy'], global_step)
        except:
            pass

        try:
            self.logger.writer.add_scalar('attn/per_action_entropy', state['attn/per_action_entropy'], global_step)
        except:
            pass

        try:
            self.logger.writer.add_scalar('attn/true_action_entropy', state['attn/true_action_entropy'], global_step)
        except:
            pass

        try:
            self.logger.writer.add_scalar('attn/independence', state['attn/independence'], global_step)
        except:
            pass

        try:
            means = {k: v for k, v in state['reinforce/paer'].items() if '_std' not in k}
            stds = {k: v for k, v in state['reinforce/paer'].items() if '_std' in k}
            self.logger.writer.add_text('reinforce/per_a_exp_reward', str(means) + '\n\n' + str(stds), global_step)
        except:
            pass


class ActionDensityPlot(TbLogger):
    def get_counts(self, predictions):
        return [sum(predictions == i).item() for i in list(range(8))]

    def __call__(self, model, state, global_step=0):

        try:
            pred, true = state['action_examples'], state['true_actions']

            counts = self.get_counts(pred)
            counts = [c / sum(counts) for c in counts]
            counts_str = ", ".join("{:.2f}".format(c) for c in counts)
            self.writer.add_text('actions_density/density', counts_str, global_step)
        except:
            pass


class ActionPredictionPlot(TbLogger):
    def get_counts(self, predictions):
        N = predictions.max() + 1
        return [sum(predictions == i).item() for i in list(range(N))]

    def __call__(self, model, state, global_step=0):
        try:
            pred, true = state['action_examples'], state['true_actions']
            outputs = collections.defaultdict(list)
            if true.shape[1] > 2:
                actions = torch.argmax(true.abs(), -1)
                actions = actions * 2
                actions[(true[:, actions // 2] < 0).any(dim=-1)] += 1
            
            for i in range(true.shape[0]):
                try:
                    outputs[true[i].item()].append(pred[i])
                except ValueError:
                    outputs[actions[i].item()].append(pred[i])

            counts = {k: self.get_counts(torch.stack(o)) for k, o in outputs.items()}
            counts = {"{}({})".format(k, torch.argmax(torch.tensor(v))): v for k, v in counts.items()}
            str_counts = ", ".join(["{}:{}".format(k, counts[k]) for k in sorted(counts.keys())])

            self.writer.add_text('actions_counts/counts', str_counts, global_step)
        except:
            pass


class GroupWiseActionPlot(Imager):
    def __init__(self, logger, groupwise_action, nlatents, nactions, to_tb=False, z1_start=True):
        self.groups = groupwise_action
        self.nlatents = nlatents
        self.nactions = nactions
        self.to_tb = to_tb
        self.logger = logger
        self.z1_start = z1_start

    def __call__(self, model, state, global_step=0):
        """ Should work """
        import numpy as np
        ims = []
        for ac in range(self.nactions):
            aux = []
            if self.z1_start:
                z = model.unwrap(model.encode(state['x1'])[0].unsqueeze(0))[0]
            else:
                z = torch.ones(self.nlatents).unsqueeze(0).cuda() * 1

            for i, action in enumerate(range(15)):
                im = model.decode(z.cuda()).sigmoid().detach().view(-1, state['x1'].shape[1], 64, 64).cpu().numpy()
                aux.append(im)

                next_z = self.groups.next_rep(z, torch.tensor([ac]).cuda())
                z = next_z
            ims.append(aux)

        if not self.to_tb:
            import matplotlib.pyplot as plt
            plt.close()
            fig, ax = plt.subplots(nrows=4, ncols=15, figsize=(15, 4))
            # fig.subplots_adjust(left=0.125, right=0.9, bottom=0.25, top=0.75, wspace=0.1, hspace=0.1)
            for k, i in enumerate(ax):
                for j, axis in enumerate(i):
                    axis.axis('off')
                    axis.imshow(ims[k][j])
                    axis.set_xticklabels([])
                    axis.set_yticklabels([])
                    # axis.set_aspect(1)
            plt.tight_layout()
            plt.savefig('./images/reconstruction_again.png')
        else:
            img = make_grid(torch.tensor(np.array(ims)).view(-1, state['x1'].shape[1], 64, 64), 15, pad_value=1)
            self.logger.writer.add_image('groups/groups', img, global_step)
            try:
                model.groups.to_tb(self.logger.writer, global_step)
            except:
                pass


def plot_classify_factors(outputs):
    classes = {}
    for k in outputs:
        if 'metric/' in k:
            classes[k[7:]] = outputs[k].item()

    x, y = [], []
    mean = None
    for k in classes:
        if k == 'val_acc':
            mean = classes[k]
        elif k == 'val_loss':
            pass
        else:
            x.append(float(k))
            y.append(classes[k])

    plt.plot(x, y, '.-', label='Accuracies')
    plt.gca().set_xticks(x)
    plt.gca().set_xticklabels(['Shape', 'Scale', 'Orientation', 'X', 'Y'])
    plt.axhline(mean, x[0], x[-1], color='r', linestyle='--', label='Mean')
    plt.xlabel('Factor')
    plt.ylabel('Prediction Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./images/factor_classification.pdf')
    plt.close()


def plot_recon_mse_for_generating_model(outputs, epoch_mses=[], file='./images/generating_mses.pdf'):
    mses = []

    const_vae_beta_1 = 0.0008
    const_vae_beta_5 = 0.0021
    const_vae_beta_10 = 0.0041

    for o in outputs:
        mses.append(o['mse'])
    epoch_mses.append(torch.stack(mses).mean())
    if len(epoch_mses) > 1:
        plt.close()
        plt.plot(epoch_mses[1:], '.-')
        plt.axhline(y=const_vae_beta_1, linestyle='--', color='r', label='beta_1')
        plt.axhline(y=const_vae_beta_5, linestyle='--', color='g', label='beta_5')
        plt.axhline(y=const_vae_beta_10, linestyle='--', color='b', label='beta_10')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.savefig(file)


def plot_train_actions_mse_per_factor(logs, file='./images/train_action_mse_per_factor.pdf', npz_file=None):
    items = []
    for k in logs:
        if 'action' in k and 'loss' not in k:
            items.append((k[-1:], logs[k]))
    items = [(x, y) for x,y in sorted(items)]

    if npz_file is not None:
        import numpy as np
        np.save(npz_file, np.array(items))

    plt.close()
    plt.plot([x for x,_ in items], [y for _,y in items], '.-')
    plt.gca().set_xticklabels(['Shape', 'Scale', 'Orientation', 'X', 'Y'])
    plt.xlabel('Factor')
    plt.ylabel('MSE')
    plt.savefig(file)


def combine_plots_train_actions_mse_per_factor(output_file='../images/train_action_mse_per_factor.pdf',
                                               files=('../images/version_5.npy', '../images/version_6.npy',
                                                      '../images/version_7.npy')):
    import numpy as np
    plt.close()
    labels = ['beta-', 'beta-', 'beta-']

    for i, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        plt.plot([x for x,_ in data], [y for _,y in data], '.-', label=labels[i])

    plt.gca().set_xticklabels(['Shape', 'Scale', 'Orientation', 'X', 'Y'])
    plt.xlabel('Factor')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(output_file)


def plot_train_actions_unsupervised_mse_per_factor(logs, file='./images/train_action_unsupervised_mse_per_factor.pdf', npz_file=None):
    items = []
    for k in logs:
        if 'actions_mse' in k and 'loss' not in k:
            items.append((k[-1:], logs[k]))
    items = [(x, y) for x, y in sorted(items)]

    if npz_file is not None:
        import numpy as np
        np.save(npz_file, np.array(items))

    plt.close()
    plt.plot([x for x, _ in items], [y for _, y in items], '.-')
    plt.gca().set_xticklabels(['Shape', 'Scale', 'Orientation', 'X', 'Y'])
    plt.xlabel('Factor')
    plt.ylabel('MSE')
    plt.savefig(file)


