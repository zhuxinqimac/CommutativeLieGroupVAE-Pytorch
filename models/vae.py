import torch
from torch.nn import Module
from models.losses import gaussian_kls
from logger.imaging import ShowRecon, LatentWalk, ReconToTb


class VAE(Module):
    def __init__(self, encoder, decoder, beta, max_capacity=None, capacity_leadin=None, anneal=1.):
        """ Base VAE class for other models

        Args:
            encoder (nn.Module): Encoder network, outputs size [bs, 2*latents]
            decoder (nn.Module): Decoder network, outputs size [bs, nc, N, N]
            beta (float): Beta value for KL divergence weight
            max_capacity (float): Max capacity for capactiy annealing
            capacity_leadin (int): Capacity leadin, linearly scale capacity up to max over leadin steps
            anneal (float): Annealing rate for KL weighting
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.capacity = max_capacity
        self.capacity_leadin = capacity_leadin
        self.global_step = 0
        self.anneal = anneal

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def load_vae_state(self, state):
        def key_map(s, enc_dec_str):
            idx = s.find(enc_dec_str)
            return s[(idx+len(enc_dec_str)+1):]

        encoder_state = {key_map(k, 'encoder'): v for k, v in state.items() if 'encoder' in k}
        decoder_state = {key_map(k, 'decoder'): v for k, v in state.items() if 'decoder' in k}
        self.encoder.load_state_dict(encoder_state)
        self.decoder.load_state_dict(decoder_state)

    def forward(self, x):
        mu, lv = self.unwrap(self.encode(x))
        return self.decode(self.reparametrise(mu, lv))

    def rep_fn(self, batch):
        x, y = batch
        mu, lv = self.unwrap(self.encode(x))
        return mu

    def imaging_cbs(self, args, logger, model, batch=None):
        return [
            ShowRecon(),
            ReconToTb(logger),
            LatentWalk(logger, args.latents, list(range(args.latents)), input_batch=batch, to_tb=True),
        ]

    def main_step(self, batch, batch_nb, loss_fn):

        x, y = batch

        mu, lv = self.unwrap(self.encode(x))
        z = self.reparametrise(mu, lv)
        x_hat = self.decode(z)

        loss = loss_fn(x_hat, x)
        total_kl = self.compute_kl(mu, lv, mean=False)
        beta_kl = self.control_capacity(total_kl, self.global_step, self.anneal)
        state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)
        self.global_step += 1

        tensorboard_logs = {'metric/loss': loss+beta_kl, 'metric/recon_loss': loss, 'metric/total_kl': total_kl,
                            'metric/beta_kl': beta_kl}
        return {'loss': loss+beta_kl, 'out': tensorboard_logs, 'state': state}

    def compute_kl(self, mu, lv, mean=False):
        total_kl, dimension_wise_kld, mean_kld = gaussian_kls(mu, lv, mean)
        return total_kl

    def make_state(self, batch_nb, x_hat, x, y, mu, lv, z):
        if batch_nb == 0:
            recon = x_hat[:8]
        else:
            recon = None
        state = {'x': x, 'y': y, 'x_hat': x_hat, 'recon': recon, 'mu': mu, 'lv': lv, 'x1': x, 'x2': y, 'z': z}
        return state

    def control_capacity(self, total_kl, global_step, anneal=1.):
        if self.capacity is not None:
            leadin = 1e5 if self.capacity_leadin is None else self.capacity_leadin
            delta = torch.tensor((self.capacity / leadin) * global_step).clamp(max=self.capacity)
            return (total_kl - delta).abs().clamp(min=0) * self.beta * (anneal ** global_step)
        else:
            return total_kl*self.beta

    def train_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def val_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def unwrap(self, x):
        return torch.split(x, x.shape[1]//2, dim=1)

    def reparametrise(self, mu, lv):
        if self.training:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu
