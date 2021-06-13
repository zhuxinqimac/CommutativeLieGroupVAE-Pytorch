#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: collect_results.py
# --- Creation Date: 02-02-2021
# --- Last Modified: Sun 18 Apr 2021 16:21:19 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Collect results from a directory.
"""

import argparse
import torch
import os
import pdb
import glob
import pathlib
import numpy as np
import pickle
import pandas as pd
import re

from collections import OrderedDict

# Metrics entries of interest.
# If not shown here, means all entries of the metrics are of interest.
# moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim'], 'mig': ['discrete_mig']}
# moi = {'tpl': ['sum_dist'], 'fvm': ['eval_acc', 'act_dim'], 'tpl_large': ['sum_dist'], 'tpl_b4': ['sum_dist']}
moi = ['fac', 'fac_dim', 'beta', 'mig', 'dci_info',
       'dci_dis', 'dci_comp', 'mod', 'explicit', 'sap', 'fl']

# Brief dict of names.
brief = {'fac_eval': 'fac', 'fac_num_act': 'fac_dim', 'val_hig_acc': 'beta',
         'discrete_mig': 'mig', 'dci_informativeness_test': 'dci_info',
         'dci_disentanglement': 'dci_dis', 'dci_completeness': 'dci_comp',
         'modularity_score': 'mod', 'explicitness_score_test': 'explicit',
         'SAP_score': 'sap', 'factor_leakage_mean': 'fl'}

def get_mean(x):
    x = list(filter(None, x))
    return None if len(x) == 0 else np.mean(x)

def get_num(x):
    x = list(filter(None, x))
    return len(x)

def get_std(x):
    x = list(filter(None, x))
    return None if len(x) == 0 else np.std(x)

def _list_to_str(v):
    v2 = [str(x) for x in v]
    return ','.join(v2)

def get_mean_std(raw_dict):
    # raw_dict: {'fac': [0.5, 0.4], 'beta': [0.3, 0.4], ...}
    new_results = {}
    for k, v in raw_dict.items():
        k_mean, k_std = k+'.mean', k+'.std'
        v_mean = get_mean(v)
        v_std = get_std(v)
        n_samples = len(v)
        new_results[k_mean] = v_mean
        new_results[k_std] = v_std
    new_results['n_samples'] = n_samples
    return new_results

def get_config(dir_name, config_variables):
    last_version_dir = sorted(os.listdir(dir_name))[-1]
    args_file = os.path.join(dir_name, last_version_dir, 'checkpoints', 'args.pt')
    if not os.path.isfile(args_file):
        return None
    loaded_args = vars(torch.load(args_file))
    config_vars_ls = []
    for k in config_variables:
        if k in loaded_args:
            config_vars_ls.append(str(loaded_args[k]))
        else:
            config_vars_ls.append('*')
    # print(config_vars_ls)
    return '='.join(config_vars_ls)

def extract_this_results(dir_name):
    # this_results: {'fac.mean': 0.5, 'fac.std': 0.1, 'beta.mean': 4, ..., 'n_samples': 10}
    version_dirs = sorted(os.listdir(dir_name))
    raw_dict = {k: [] for k in moi}
    for v in version_dirs:
        metrics_file = os.path.join(dir_name, v, 'log_metrics.txt')
        if not os.path.isfile(metrics_file):
            continue
        with open(metrics_file, 'r') as f:
            data = f.readlines()
        tmp_dict = {}
        for line in data:
            if line.startswith('dmetric'):
                mname, mscore = [word.strip() for word in line.strip().split(':')]
                mname = mname.split('/')[1]
                if mname in brief.keys():
                    brief_mname = brief[mname]
                    if brief_mname in moi:
                        tmp_dict[brief_mname] = float(mscore)

        # Filter out incomplete entries.
        if len(tmp_dict) == len(raw_dict):
            for k in raw_dict.keys():
                raw_dict[k].append(tmp_dict[k])
    this_results = get_mean_std(raw_dict)
    return this_results

def parse_config_v(config):
    config = config[1:-1]
    config = [x.strip() for x in config.split(',')]
    # config: [['run_desc'], ['run_func_kwargs', 'G_args', 'module_G_list'], ...]
    return config

def main():
    parser = argparse.ArgumentParser(description='Collect results.')
    parser.add_argument('--in_dir', help='Parent directory of sub-result-dirs to collect results.',
                        type=str)
    parser.add_argument('--result_file', help='Results file.',
                        type=str, default='/mnt/hdd/repo_results/test.csv')
    parser.add_argument('--config_variables', help='Configs to extract from args.pt',
                        type=str, default=\
                        '[model, dataset, fixed_shape, latents, subgroup_sizes_ls, subspace_sizes_ls, lie_alg_init_type_ls, lie_alg_init_scale, hy_rec, hy_commute, hy_hes, forward_eg_prob, recons_loss_type]')
    args = parser.parse_args()

    args.config_variables = parse_config_v(args.config_variables)
    res_dirs = glob.glob(os.path.join(args.in_dir, '*/'))
    dir_ctimes = [pathlib.Path(dir_name[:-1]).stat().st_ctime for dir_name in res_dirs]
    dir_ctimes, res_dirs = (list(t) for t in zip(*sorted(zip(dir_ctimes, res_dirs))))
    # print('res_dirs:', res_dirs)
    # res_dirs.sort()
    results = {'_config': []}
    for k in moi:
        results[k+'.mean'] = []
        results[k+'.std'] = []
    results['n_samples'] = []
    results['path'] = []
    results['ctime'] = []
    # results: {'_config': ['model1', 'model2'], 'fac.mean': [0.5,0.8], 'fac.std': [0.1, 0.1], ..., 'n_samples': [10, 10]}

    for dir_name in res_dirs:
        config = get_config(dir_name, args.config_variables)
        if config is None:
            continue
        this_results = extract_this_results(dir_name)
        # this_results: {'fac.mean': 0.5, 'fac.std': 0.1, 'beta.mean': 4, ..., 'n_samples': 10}

        results['_config'].append(config)
        results['path'].append(os.path.basename(dir_name[:-1]))
        # fname = pathlib.Path(dir_name[:-1])
        # results['ctime'].append(fname.stat().st_ctime)
        for k in results.keys():
            if k != '_config' and k != 'path':
                if k in this_results.keys():
                    results[k].append(this_results[k])
                else:
                    results[k].append('-')

    for k in results.keys():
        print(k+': ', len(results[k]))
    save_results_to_csv(results, args)

def save_results_to_csv(results, args, sufix=''):
    results = OrderedDict(sorted(results.items()))
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.result_file[:-4]+sufix+'.csv', na_rep='-',
                      index=False, float_format='%.3f')


if __name__ == "__main__":
    main()
