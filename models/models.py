from models.beta import BetaCeleb, BetaShapes
from models.factor_vae import factor_vae, factor_conv_vae
from models.dip_vae import dip_vae, dip_conv_vae
from models.lie_vae import LieCeleb

models = {
    'beta_shapes': BetaShapes,
    'beta_celeb': BetaCeleb,
    'factor_vae': factor_vae,
    'factor_conv_vae': factor_conv_vae,
    'dip_vae_i': dip_vae,
    'dip_vae_ii': dip_vae,
    'dip_conv_vae_i': dip_conv_vae,
    'dip_conv_vae_ii': dip_conv_vae,
    'lie_group': LieCeleb,
}
