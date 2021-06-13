import argparse
import ast
from trainer import run
from utils import _str_to_list_of_int, _str_to_list_of_str, _str_to_bool

parser = argparse.ArgumentParser('Lie Group Disentanglement')
# Basic Training Args
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='lie_group', type=str, choices=['beta_shapes', 'beta_celeb', 'dip_vae_i', 'dip_vae_ii',
                                                                       'lie_group', 'factor_conv_vae', 'dip_conv_vae_i', 'dip_conv_vae_ii'])
parser.add_argument('--dataset', default='shapes3d', type=str, choices=['dsprites', 'shapes3d'])
parser.add_argument('--fixed_shape', default=None, type=int, help='Fixed shape in dsprites. None for not fixed.')
parser.add_argument('--data-path', default=None, type=str, help='Path to dataset root')
parser.add_argument('--latents', default=10, type=int, help='Number of latents')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--split', default=0.1, type=float, help='Validation split fraction')
parser.add_argument('--shuffle', default=True, type=ast.literal_eval, help='Shuffle dataset')

# Model Loading
parser.add_argument('--base-model', default='beta_celeb', type=str, help='Base model')
parser.add_argument('--base-model-path', default=None, help='Path to base model state which is to be loaded')
parser.add_argument('--load-model', default=False, type=ast.literal_eval, help='Continue training by loading model')
parser.add_argument('--log-path', default=None, type=str, help='Path from which to load model')
parser.add_argument('--global-step', default=None, help='Set the initial logging step value', type=int)

# Learning Rates
parser.add_argument('--learning-rate', '-lr', default=1e-4, type=float, help='Learning rate')

# Hyperparams
parser.add_argument('--beta', default=1., type=float, help='Beta vae beta')
parser.add_argument('--capacity', default=None, type=float, help='KL Capacity')
parser.add_argument('--capacity-leadin', default=100000, type=int, help='KL capacity leadin')

# Metrics And Vis
parser.add_argument('--visualise', default=True, type=ast.literal_eval, help='Do visualisations')
parser.add_argument('--metrics', default=False, type=ast.literal_eval, help='Calculate disentanglement metrics at each step')
parser.add_argument('--end-metrics', default=True, type=ast.literal_eval, help='Calculate disentanglement metrics at end of run')
parser.add_argument('--evaluate', default=False, type=ast.literal_eval, help='Only run evalulation')
parser.add_argument('--eval_dataset', default=None, type=str, choices=['flatland', 'dsprites', 'teapot', 'teapot_nocolor', 'shapes3d'])
parser.add_argument('--eval_data_path', default=None, type=str)

# Lie Group Model
parser.add_argument('--subgroup_sizes_ls', default=[100], type=_str_to_list_of_int, help='Subgroup sizes list for subspace group vae')
parser.add_argument('--subspace_sizes_ls', default=[10], type=_str_to_list_of_int, help='Subspace sizes list for subspace group vae')
parser.add_argument('--lie_alg_init_scale', help='Hyper-param for lie_alg_init_scale.', default=0.001, type=float)
parser.add_argument('--hy_hes', default=40, type=float, help='Hyper-param for Hessian in LieVAE.')
parser.add_argument('--hy_rec', default=0.1, type=float, help='Hyper-param for gfeats reconstruction in GroupVAE and LieVAE.')
parser.add_argument('--hy_commute', default=0, type=float, help='Hyper-param for commutator in GroupVAE-v3.')
parser.add_argument('--forward_eg_prob', default=0.2, type=float, help='The prob to forward eg in LieGroupVAE.')
parser.add_argument('--recons_loss_type', default='l2', choices=['l2', 'bce'], type=str, help='The reconstruction type for x.')
parser.add_argument('--no_exp', default=False, type=_str_to_bool, help='If deactivate exp_mapping in LieGroupVAE.')

# FactorVAE
parser.add_argument('--factor_vae_gamma', default=6.4, type=float, help='The gamma in factor vae.')

# DIP-VAE
parser.add_argument('--lambda_d', default=1, type=float, help='The lambda_d in DIP vae.')
parser.add_argument('--lambda_od', default=10, type=float, help='The lambda_od in DIP vae.')

args = parser.parse_args()


if __name__ == '__main__':
    from models.utils import ContextTimer
    with ContextTimer('Run'):
        run(args)
