import gc
import os

import torch
from torch.nn import functional as F

from datasets.datasets import datasets, set_to_loader, dataset_meta
from logger.tb import Logger, write_args
from logger.utils import mean_log_list
from metrics.aggregator import MetricAggregator
from models.models import models
from models.utils import count_parameters, model_loader
from trainin_loop import train

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def run(args):
    if args.evaluate or args.load_model:
        checkpoint_path = os.path.join(args.log_path, 'checkpoints')
        model_state, old_args = model_loader(checkpoint_path)
    if args.evaluate:
        old_args.data_path, old_args.log_path = args.data_path, args.log_path
        old_args.evaluate, old_args.visualise, old_args.metrics = args.evaluate, args.visualise, args.metrics
        old_args.eval_dataset, old_args.eval_data_path = args.eval_dataset, args.eval_data_path
        old_args.split = args.split
        args = old_args

    args.nc, args.factors = dataset_meta[args.dataset]['nc'], dataset_meta[args.dataset]['factors']
    trainds, valds = datasets[args.dataset](args)
    trainloader, valloader = set_to_loader(trainds, valds, args)

    model = models[args.model](args)

    model.load_state_dict(model_state, strict=False) if args.evaluate or args.load_model else None
    model.cuda()

    if args.base_model_path is not None:
        model_state, _ = model_loader(args.base_model_path)
        model.load_vae_state(model_state)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.recons_loss_type == 'l2':
        loss_fn = lambda x_hat, x: (x_hat.sigmoid() - x).pow(2).sum() / x.shape[0]
    else:
        loss_fn = lambda x_hat, x: F.binary_cross_entropy_with_logits(x_hat.view(x_hat.size(0), -1), x.view(x.size(0), -1), reduction='sum') / x.shape[0]
    metric_list = MetricAggregator(
        trainds, trainds, 1000, model, False, args.latents, ntrue_actions=args.latents, final=True, fixed_shape=args.fixed_shape) if args.metrics else None

    version = None
    if args.log_path is not None and args.load_model:
        for a in args.log_path.split('/'):
            if 'version_' in a:
                version = a.split('_')[-1]

    logger = Logger(args.log_path, version)
    param_count = count_parameters(model)
    logger.writer.add_text('parameters/number_params', param_count.replace('\n', '\n\n'), 0)
    print(param_count)

    write_args(args, logger)
    if not args.evaluate:
        out = train(args, args.epochs, trainloader, valloader, model, optimiser, loss_fn, logger, metric_list, True)
    else:
        out = {}

    if args.evaluate or args.end_metrics:
        if args.eval_dataset and args.eval_data_path:
            del trainds
            del trainloader
            del valloader
            args.dataset = args.eval_dataset
            args.data_path = args.eval_data_path
            valds, _ = datasets[args.eval_dataset](args)
        if args.split > 0:
            valds = valds.dataset
        log_list = MetricAggregator(None, valds, 1000, model, False, args.latents, ntrue_actions=args.latents, final=True, fixed_shape=args.fixed_shape)()
        mean_logs = mean_log_list([log_list, ])
        logger.write_dict(mean_logs, model.global_step+1) if logger is not None else None

    gc.collect()
    return out
