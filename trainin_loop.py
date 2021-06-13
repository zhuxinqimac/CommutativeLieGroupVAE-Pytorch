from logger.utils import mean_log_list, save_model
from tqdm import tqdm
import torch


def train(args, epochs, trainloader, valloader, model, optimiser, loss_fn, logger=None, metric_list=None, cuda=True):
    pb = tqdm(total=epochs, unit_scale=True, smoothing=0.1, ncols=70)
    if valloader is not None:
        update_frac = 1./float(len(trainloader) + len(valloader))
    else:
        update_frac = 1./float(len(trainloader))
    global_step = 0 if not hasattr(args, 'global_step') or args.global_step is None else args.global_step
    loss, val_loss = torch.tensor(0), torch.tensor(0)
    mean_logs = {}

    for i in range(epochs):
        for t, data in enumerate(trainloader):
            optimiser.zero_grad()
            model.train()
            data = to_cuda(data) if cuda else data
            out = model.train_step(data, t, loss_fn)
            loss = out['loss']
            loss.backward()
            optimiser.step()
            pb.update(update_frac)
            pgs = [pg['lr'] for pg in optimiser.param_groups]
            pb.set_postfix_str('ver:{}, loss:{:.3f}, val_loss:{:.3f}, lr:{}'.format(logger.get_version(), loss.item(), val_loss.item(), pgs))
            global_step += 1

        if valloader is not None:
            valloader_tmp = valloader
        else:
            valloader_tmp = trainloader
        log_list = []
        with torch.no_grad():
            for t, data in enumerate(valloader_tmp):
                if t > len(valloader_tmp) * 0.1:
                    break
                model.eval()
                to_cuda(data) if cuda else None
                out = model.val_step(data, t, loss_fn)
                val_loss = out['loss']
                logs = out['out']
                log_list.append(parse_val_logs(t, args, model, data, logger, metric_list, logs, out['state'], global_step))
                pb.update(update_frac)
                pb.set_postfix_str(
                    'ver:{}, loss:{:.3f}, val_loss:{:.3f}'.format(logger.get_version(), loss.item(), val_loss.item()))
                global_step += 1

        mean_logs = mean_log_list(log_list)
        logger.write_dict(mean_logs, global_step) if logger is not None else None
        save_model(logger, model, args)
    return mean_logs


def parse_val_logs(t, args, model, batch, logger, metric_list, logs, state, global_step=0):
    if t == 0:
        if args.visualise:
            for vis in model.imaging_cbs(args, logger, model, batch):
                vis(model, state, global_step)
        logs.update(metric_list()) if metric_list is not None else None
    return logs


def to_cuda(x):
    if torch.is_tensor(x):
        x = x.cuda()
    else:
        for i, xi in enumerate(x):
            x[i] = to_cuda(xi)
    return x
