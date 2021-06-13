from torch import nn

from models.vae import VAE


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)


class View(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.view([x.shape[0], *self.size])


def beta_shape_encoder(args):
    return nn.Sequential(
        View(-1),
        nn.Linear(4096*args.nc, 1200),
        nn.ReLU(True),
        nn.Linear(1200, 1200),
        nn.ReLU(True),
        nn.Linear(1200, args.latents*2)
    )


def beta_shapes_decoder(args):
    return nn.Sequential(
        nn.Linear(args.latents, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, 4096*args.nc),
        View(args.nc, 64, 64),
    )


def beta_celeb_encoder(args):
    return nn.Sequential(
        nn.Conv2d(args.nc, 32, 4, 2, 1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 4, 2, 1),
        nn.ReLU(True),
        nn.Conv2d(32, 64, 4, 2, 1),
        nn.ReLU(True),
        nn.Conv2d(64, 64, 4, 2, 1),
        nn.ReLU(True),
        Flatten(),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Linear(256, args.latents*2)
    )


def beta_celeb_decoder(args):
    return nn.Sequential(
        nn.Linear(args.latents, 256),
        nn.ReLU(True),
        nn.Linear(256, 1024),
        nn.ReLU(True),
        View(64, 4, 4),
        nn.ConvTranspose2d(64, 64, 4, 2, 1),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 32, 4, 2, 1),
        nn.ReLU(True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1),
        nn.ReLU(True),
        nn.ConvTranspose2d(32, args.nc, 4, 2, 1),
        # nn.Sigmoid()
    )


class BetaShapes(VAE):
    def __init__(self, args):
        super().__init__(beta_shape_encoder(args), beta_shapes_decoder(args), args.beta, args.capacity, args.capacity_leadin)


class BetaCeleb(VAE):
    def __init__(self, args):
        super().__init__(beta_celeb_encoder(args), beta_celeb_decoder(args), args.beta, args.capacity, args.capacity_leadin)
