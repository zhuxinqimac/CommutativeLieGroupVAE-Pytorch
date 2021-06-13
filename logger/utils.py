import collections
import torch
import os
import warnings


def mean_log_list(log_list):
    main_log = collections.defaultdict(list)
    for logs in log_list:
        for k, v in logs.items():
            try:
                if isinstance(v, dict): continue
                if v is None: continue
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                main_log[k].append(v)
            except:
                warnings.warn('Failed to combine metric ({}) for tb'.format(k))

    # out = {}
    # for k, v in main_log.items():
        # if len(v)>0:
            # print(k+':', v)
            # out[k] = torch.stack(v, 0).float().mean(0)
    out = {k: torch.stack(v, 0).float().mean(0) for k, v in main_log.items() if len(v) > 0}
    return out


def save_model(logger, model, args, model_label='model.pt'):
    path = '{}/checkpoints/'.format(logger.log_dir)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, model_label))
    torch.save(args, os.path.join(path, 'args.pt'))
