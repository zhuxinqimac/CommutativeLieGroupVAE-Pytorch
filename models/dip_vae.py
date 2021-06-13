import torch
from models.vae import VAE


# lambda_d, lambda_od = 100, 10


def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def dip_vae_i_loss(mu, lambda_d, lambda_od):
    exp_mu = mu.mean(0)
    exp_mu_mu_t = (mu.unsqueeze(1) * mu.unsqueeze(2)).mean(0)

    cov = exp_mu_mu_t - exp_mu.unsqueeze(0) * exp_mu.unsqueeze(1)
    diag = torch.diagonal(cov, dim1=-2, dim2=-1)
    off_diag = cov - matrix_diag(diag)

    regulariser_od = lambda_od * (off_diag**2).sum()
    regulariser_d = lambda_d * ((diag-1)**2).sum()

    return regulariser_od + regulariser_d


def dip_vae_ii_loss(mu, lv, lambda_d, lambda_od):
    sigma = matrix_diag(lv.exp())
    exp_cov = sigma.mean(0)
    exp_mu = mu.mean(0)

    exp_mu_mu_t = (mu.unsqueeze(1) * mu.unsqueeze(2)).mean(0)
    cov_exp = exp_mu_mu_t - exp_mu.unsqueeze(0) * exp_mu.unsqueeze(1)
    cov_z = cov_exp + exp_cov

    diag = torch.diagonal(cov_z, dim1=-2, dim2=-1)
    off_diag = cov_z - matrix_diag(diag)

    regulariser_od = lambda_od * off_diag.pow(2).sum()
    regulariser_d = lambda_d * (diag - 1).pow(2).sum()

    return regulariser_d + regulariser_od


class DipVAE(VAE):
    def __init__(self, encoder, decoder, beta, lambda_d, lambda_od, dip_type='ii', max_capacity=None, capacity_leadin=None, xav_init=False):
        super().__init__(encoder, decoder, beta, max_capacity, capacity_leadin)
        self.type = dip_type

        if self.type == 'dip_vae_i':
            self.dip_loss = lambda mu, lv: dip_vae_i_loss(mu, lambda_d, lambda_od)
        else:
            self.dip_loss = lambda mu, lv: dip_vae_ii_loss(mu, lv, lambda_d, lambda_od)

        if xav_init:
            for p in self.encoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.decoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)

    def main_step(self, batch, batch_nb, loss_fn):
        out = super().main_step(batch, batch_nb, loss_fn)
        state = out['state']
        x, y, mu, lv, z, x_hat = state['x'], state['y'], state['mu'], state['lv'], state['z'], state['x_hat']

        dip_loss = self.dip_loss(mu, lv)
        vae_loss = out['loss']

        self.global_step += 1

        tensorboard_logs = out['out']
        tensorboard_logs['metric/dip_loss'] = dip_loss.detach()
        return {'loss': vae_loss + dip_loss, 'out': tensorboard_logs,
                'state': state}

def dip_vae(args):
    if args.dataset == 'forward':
        from models.forward_vae import ForwardDecoder, ForwardEncoder
        encoder, decoder = ForwardEncoder(args.latents), ForwardDecoder(args.latents)
    else:
        from models.beta import beta_shape_encoder, beta_shapes_decoder
        encoder, decoder = beta_shape_encoder(args), beta_shapes_decoder(args)

    dip_type = args.base_model if args.model in ['rl_group_vae'] else args.model

    return DipVAE(encoder, decoder, args.beta, args.lambda_d, args.lambda_od, dip_type, args.capacity, args.capacity_leadin, args.xav_init)

def dip_conv_vae(args):
    from models.beta import beta_celeb_encoder, beta_celeb_decoder
    encoder, decoder = beta_celeb_encoder(args), beta_celeb_decoder(args)

    dip_type = 'dip_vae_i' if args.model == 'dip_conv_vae_i' else 'dip_vae_ii'

    return DipVAE(encoder, decoder, args.beta, args.lambda_d, args.lambda_od, dip_type, args.capacity, args.capacity_leadin, args.xav_init)
