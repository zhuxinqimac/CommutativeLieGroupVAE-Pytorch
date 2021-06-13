from models.vae import VAE
import torch
from torch import nn, optim
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            # nn.LeakyReLU(True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            # nn.LeakyReLU(True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            # nn.LeakyReLU(True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            # nn.LeakyReLU(True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            # nn.LeakyReLU(True),
            nn.Linear(1000, 2),
            # nn.Softmax(-1)
        )

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE(VAE):
    def __init__(self, encoder, decoder, beta, latents, max_capacity=None, capacity_leadin=None, gamma=6.4, xav_init=False):
        super().__init__(encoder, decoder, beta, max_capacity, capacity_leadin)
        self.discriminator = [Discriminator(latents).cuda()]  # Exclude from register.
        self.disc_opt = optim.Adam(self.discriminator[0].parameters(), lr=1e-4, betas=(0.5, 0.9))
        # self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.gamma = float(gamma) if gamma is not None else 6.4
        if xav_init:
            for p in self.encoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.decoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.discriminator[0].modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)

    def permute_dims(self, z):
        assert z.dim() == 2

        # z = z.permute(1, 0)
        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        # return torch.cat(perm_z, 1).permute(1, 0)
        return torch.cat(perm_z, 1)

    def main_step(self, batch, batch_nb, loss_fn):
        out = super().main_step(batch, batch_nb, loss_fn)
        state = out['state']
        x, y, mu, lv, z, x_hat = state['x'], state['y'], state['mu'], state['lv'], state['z'], state['x_hat']

        D_z = self.discriminator[0](z.detach())
        z_perm = self.permute_dims(z)
        D_z_perm = self.discriminator[0](z_perm.detach())
        D_tc_loss = 0.5 * (F.cross_entropy(D_z, torch.zeros(D_z.shape[0], dtype=torch.long).to(D_z.device))
                           + F.cross_entropy(D_z_perm, torch.ones(D_z.shape[0], dtype=torch.long).to(D_z.device)))

        if self.training:
            self.disc_opt.zero_grad()
            D_tc_loss.backward()
            self.disc_opt.step()

        D_z = self.discriminator[0](z)
        vae_loss = out['loss']
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean() * self.gamma

        tensorboard_logs = out['out']
        tensorboard_logs.update({'metric/loss': vae_loss+vae_tc_loss, 'metric/tc_loss': vae_tc_loss.detach(), 'metric/disc_tc_loss': D_tc_loss.detach()})

        self.global_step += 1

        return {'loss': vae_loss + vae_tc_loss,
                'out': tensorboard_logs,
                'state': state}


def factor_vae(args):
    if args.dataset == 'forward':
        from models.forward_vae import ForwardEncoder, ForwardDecoder
        encoder, decoder = ForwardEncoder(args.latents), ForwardDecoder(args.latents)
    else:
        from models.beta import beta_shape_encoder, beta_shapes_decoder
        encoder, decoder = beta_shape_encoder(args), beta_shapes_decoder(args)

    return FactorVAE(encoder, decoder, args.beta, args.latents, args.capacity, args.capacity_leadin, args.factor_vae_gamma, args.xav_init)

def factor_conv_vae(args):
    from models.beta import beta_celeb_encoder, beta_celeb_decoder
    encoder, decoder = beta_celeb_encoder(args), beta_celeb_decoder(args)
    return FactorVAE(encoder, decoder, args.beta, args.latents, args.capacity, args.capacity_leadin, args.factor_vae_gamma, args.xav_init)
